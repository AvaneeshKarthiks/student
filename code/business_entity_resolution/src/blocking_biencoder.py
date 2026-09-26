"""
Candidate Generation (Blocking) Module.
Uses ModernBERT-base as a dense Bi-Encoder to generate embeddings,
followed by FAISS IVF similarity search to generate candidate pairs.

Memory strategy for 16 GB RAM / 4 GB VRAM
------------------------------------------
* Embedding cache files are loaded with ``np.load(..., mmap_mode='r')``.
  The OS memory-maps the .npy file so only the pages actually touched are
  brought into RAM.
* Similarity search uses FAISS IVF-Flat which builds a quantised index
  by streaming candidate chunks sequentially (no random mmap access).
  The index lives on GPU (~2.5 GB for 4.7M vectors) and all queries are
  searched in a single pass -- no O(q x c) double loop.
* Falls back to vectorised brute-force if FAISS is unavailable, using
  larger chunks and numpy fancy indexing (no Python per-row loops).
"""

from typing import List, Dict, Tuple, Optional
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc

from pathlib import Path
import sys

_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .config import BlockingConfig, CACHE_DIR
from .dataset import TextInferenceDataset


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------

def _embed_cache_path(texts: List[str], prefix: str = "embed") -> Path:
    """Return a deterministic .npy path in CACHE_DIR for the given text list."""
    digest = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{digest}.npy"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModernBERTBiEncoder(nn.Module):
    """ModernBERT-base bi-encoder with mean pooling and L2 normalisation."""

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(out.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)


# ---------------------------------------------------------------------------
# Blocking engine
# ---------------------------------------------------------------------------

class BiEncoderBlockingEngine:
    """
    Blocking engine: ModernBERT embeddings + FAISS IVF similarity search.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.device = torch.device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        model = ModernBERTBiEncoder(self.config.model_name).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs for BiEncoder!")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        self.model.eval()

    @property
    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # ------------------------------------------------------------------
    # Encoding  (pre-allocated write, memory-mapped read)
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        cache_prefix: str = "embed",
    ) -> np.ndarray:
        """
        Encode texts -> L2-normalised float32 embeddings.

        Returns a **memory-mapped** array when loading from cache so the OS
        pages in only the rows that are actually read.  On first run the
        result is written via a pre-allocated array (no np.vstack OOM).
        """
        cache_path = _embed_cache_path(texts, prefix=cache_prefix)

        if cache_path.exists():
            try:
                arr = np.load(cache_path, mmap_mode="r")
                print(f"  [cache hit] {cache_path.name}  ({arr.shape[0]} x {arr.shape[1]})")
                return arr
            except ValueError:
                # File exists but has no .npy header -- it was written as raw binary
                # by an older version of this code. Infer shape from file size.
                dim = self._unwrapped_model.encoder.config.hidden_size
                n   = cache_path.stat().st_size // (dim * 4)
                arr = np.memmap(cache_path, dtype=np.float32, mode="r", shape=(n, dim))
                print(f"  [cache hit] {cache_path.name}  (raw -> {n} x {dim})")
                return arr

        batch_size = batch_size or self.config.batch_size
        dataset    = TextInferenceDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        n   = len(texts)
        dim = self._unwrapped_model.encoder.config.hidden_size  # 768

        # np.lib.format.open_memmap writes a valid .npy file (header + data)
        # via a memory-mapped view -- the OS pages data to disk as rows are written,
        # so the full (n x dim x 4) array is NEVER resident in RAM at once.
        # np.load(mmap_mode='r') can then read it back without loading it either.
        result = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=(n, dim)
        )
        ptr = 0

        with torch.no_grad():
            for batch_texts in tqdm(dataloader, desc="ModernBERT Encoding", leave=False):
                encoded = self.tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                ).to(self.device)
                emb = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                batch_np = emb.cpu().numpy()
                result[ptr : ptr + len(batch_np)] = batch_np
                ptr += len(batch_np)

        result.flush()
        del result
        gc.collect()

        print(f"  [cache saved] {cache_path.name}  ({n} x {dim})")
        return np.load(cache_path, mmap_mode="r")

    # ------------------------------------------------------------------
    # Similarity search  (FAISS GPU fast-path + vectorised CPU fallback)
    # ------------------------------------------------------------------

    def _search_faiss_gpu(
        self,
        query_embeddings: np.ndarray,
        index_embeddings: np.ndarray,
        top_k: int,
        add_chunk_size: int = 500_000,
        query_chunk_size: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a FAISS IVF-Flat index on GPU by streaming candidate chunks,
        then search all queries.  Peak GPU mem ~2.5 GB for 4.7M x 768.

        The IVF quantiser avoids brute-force; nprobe controls quality/speed.
        """
        import faiss as _faiss

        n_cands = len(index_embeddings)
        dim     = index_embeddings.shape[1]
        actual_k = min(top_k, n_cands)

        # --- Build GPU index by streaming candidate chunks ----------------
        # IVF with sqrt(n) clusters; gives good recall at nprobe=64.
        n_clusters = min(int(n_cands ** 0.5), 4096)
        n_clusters = max(n_clusters, actual_k + 1)

        # Use inner-product (= cosine sim for L2-normalised vectors)
        cpu_quantiser = _faiss.IndexFlatIP(dim)
        cpu_index     = _faiss.IndexIVFFlat(cpu_quantiser, dim, n_clusters,
                                            _faiss.METRIC_INNER_PRODUCT)

        # Transfer to GPU
        if torch.cuda.device_count() > 1:
            print(f"  FAISS: Using {torch.cuda.device_count()} GPUs for index")
            gpu_index = _faiss.index_cpu_to_all_gpus(cpu_index)
            # Cannot easily set temp memory limits for index_cpu_to_all_gpus,
            # but multi-GPU systems usually have plenty of VRAM.
            gpu_res = None
        else:
            gpu_res   = _faiss.StandardGpuResources()
            # Limit temp memory to 512 MB so we don't OOM on 4 GB VRAM
            gpu_res.setTempMemory(512 * 1024 * 1024)
            gpu_index = _faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

        # Train on a subsample (max 256 K vectors, way enough for IVF)
        train_n  = min(n_cands, 256_000)
        step     = max(1, n_cands // train_n)
        train_data = np.array(
            index_embeddings[::step][:train_n], dtype=np.float32
        ).copy()
        print(f"  FAISS: training IVF ({n_clusters} clusters) on {len(train_data)} vectors ...")
        gpu_index.train(train_data)
        del train_data
        gc.collect()

        # Add candidates in sequential chunks (keeps mmap sequential -- no thrash)
        print(f"  FAISS: adding {n_cands} candidates in {add_chunk_size}-vector chunks ...")
        for start in tqdm(range(0, n_cands, add_chunk_size),
                          desc="  FAISS add", leave=False):
            end   = min(start + add_chunk_size, n_cands)
            chunk = np.array(
                index_embeddings[start:end], dtype=np.float32
            ).copy()
            gpu_index.add(chunk)
            del chunk

        gpu_index.nprobe = 64  # good recall-speed trade-off

        # --- Search queries in chunks (keep CPU-side memory bounded) ------
        n_queries   = len(query_embeddings)
        all_scores  = np.empty((n_queries, actual_k), dtype=np.float32)
        all_indices = np.empty((n_queries, actual_k), dtype=np.int64)

        print(f"  FAISS: searching {n_queries} queries (K={actual_k}) ...")
        for q_start in tqdm(range(0, n_queries, query_chunk_size),
                            desc="  FAISS search", leave=False):
            q_end = min(q_start + query_chunk_size, n_queries)
            q_np  = np.array(
                query_embeddings[q_start:q_end], dtype=np.float32
            ).copy()
            scores, indices = gpu_index.search(q_np, actual_k)
            all_scores[q_start:q_end]  = scores
            all_indices[q_start:q_end] = indices
            del q_np

        # Clean up GPU resources
        del gpu_index, cpu_index, cpu_quantiser, gpu_res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Sort descending by score (FAISS already does this, but be safe)
        order       = np.argsort(-all_scores,  axis=1)
        all_scores  = np.take_along_axis(all_scores,  order, axis=1)
        all_indices = np.take_along_axis(all_indices, order, axis=1)

        return all_scores, all_indices

    def _search_faiss_cpu(
        self,
        query_embeddings: np.ndarray,
        index_embeddings: np.ndarray,
        top_k: int,
        add_chunk_size: int = 500_000,
        query_chunk_size: int = 50_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        FAISS CPU IVF-Flat fallback (no GPU required).
        Slower than GPU but still >10x faster than brute-force matmul.
        """
        import faiss as _faiss

        n_cands  = len(index_embeddings)
        dim      = index_embeddings.shape[1]
        actual_k = min(top_k, n_cands)

        n_clusters = min(int(n_cands ** 0.5), 4096)
        n_clusters = max(n_clusters, actual_k + 1)

        quantiser = _faiss.IndexFlatIP(dim)
        index     = _faiss.IndexIVFFlat(quantiser, dim, n_clusters,
                                        _faiss.METRIC_INNER_PRODUCT)

        # Train
        train_n  = min(n_cands, 256_000)
        step     = max(1, n_cands // train_n)
        train_data = np.array(
            index_embeddings[::step][:train_n], dtype=np.float32
        ).copy()
        print(f"  FAISS-CPU: training IVF ({n_clusters} clusters) ...")
        index.train(train_data)
        del train_data
        gc.collect()

        # Add in chunks
        for start in tqdm(range(0, n_cands, add_chunk_size),
                          desc="  FAISS-CPU add", leave=False):
            end   = min(start + add_chunk_size, n_cands)
            chunk = np.array(
                index_embeddings[start:end], dtype=np.float32
            ).copy()
            index.add(chunk)
            del chunk

        index.nprobe = 64

        # Search
        n_queries   = len(query_embeddings)
        all_scores  = np.empty((n_queries, actual_k), dtype=np.float32)
        all_indices = np.empty((n_queries, actual_k), dtype=np.int64)

        for q_start in tqdm(range(0, n_queries, query_chunk_size),
                            desc="  FAISS-CPU search", leave=False):
            q_end = min(q_start + query_chunk_size, n_queries)
            q_np  = np.array(
                query_embeddings[q_start:q_end], dtype=np.float32
            ).copy()
            scores, indices = index.search(q_np, actual_k)
            all_scores[q_start:q_end]  = scores
            all_indices[q_start:q_end] = indices
            del q_np

        del index, quantiser
        gc.collect()

        order       = np.argsort(-all_scores,  axis=1)
        all_scores  = np.take_along_axis(all_scores,  order, axis=1)
        all_indices = np.take_along_axis(all_indices, order, axis=1)

        return all_scores, all_indices

    def _search_bruteforce_vectorised(
        self,
        query_embeddings: np.ndarray,
        index_embeddings: np.ndarray,
        top_k: int,
        query_chunk_size: int = 4_000,
        cand_chunk_size:  int = 200_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised brute-force fallback (no FAISS required).

        Improvements over original:
        - Larger candidate chunks (200K vs 50K) -> 5x fewer iterations -> 5x less mmap I/O
        - Larger query chunks (4K vs 2K) -> better GPU utilisation
        - Vectorised merge via numpy fancy indexing (no Python per-row loop)
        - Sequential mmap reads (always forward, never random)
        """
        n_queries = len(query_embeddings)
        n_cands   = len(index_embeddings)
        actual_k  = min(top_k, n_cands)

        all_scores  = np.full((n_queries, actual_k), -np.inf, dtype=np.float32)
        all_indices = np.full((n_queries, actual_k), -1,      dtype=np.int64)

        n_q_chunks = (n_queries + query_chunk_size - 1) // query_chunk_size
        n_c_chunks = (n_cands  + cand_chunk_size  - 1) // cand_chunk_size

        with tqdm(total=n_q_chunks * n_c_chunks,
                  desc="  Search (q-chunks x c-chunks)", leave=False) as pbar:

            for q_start in range(0, n_queries, query_chunk_size):
                q_end  = min(q_start + query_chunk_size, n_queries)
                q_size = q_end - q_start

                q_np  = np.array(query_embeddings[q_start:q_end], dtype=np.float32)
                q_gpu = torch.from_numpy(q_np).to(self.device)

                best_s = np.full((q_size, actual_k), -np.inf, dtype=np.float32)
                best_i = np.full((q_size, actual_k), -1,      dtype=np.int64)

                for c_start in range(0, n_cands, cand_chunk_size):
                    c_end  = min(c_start + cand_chunk_size, n_cands)
                    c_size = c_end - c_start

                    c_np  = np.array(index_embeddings[c_start:c_end], dtype=np.float32)
                    c_gpu = torch.from_numpy(c_np).to(self.device)

                    sims = torch.matmul(q_gpu, c_gpu.t())

                    chunk_k = min(actual_k, c_size)
                    top_vals, top_local = torch.topk(sims, k=chunk_k, dim=1)
                    chunk_s = top_vals.cpu().numpy()
                    chunk_i = top_local.cpu().numpy() + c_start

                    # Vectorised merge (no Python per-row loop)
                    merged_s = np.concatenate([best_s, chunk_s], axis=1)
                    merged_i = np.concatenate([best_i, chunk_i], axis=1)
                    top_pos  = np.argpartition(merged_s, -actual_k, axis=1)[:, -actual_k:]
                    rows     = np.arange(q_size)[:, None]
                    best_s   = merged_s[rows, top_pos]
                    best_i   = merged_i[rows, top_pos]

                    del c_gpu, c_np, sims, top_vals, top_local, merged_s, merged_i
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                    pbar.update(1)

                all_scores[q_start:q_end]  = best_s
                all_indices[q_start:q_end] = best_i

                del q_gpu, q_np, best_s, best_i
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        order       = np.argsort(-all_scores,  axis=1)
        all_scores  = np.take_along_axis(all_scores,  order, axis=1)
        all_indices = np.take_along_axis(all_indices, order, axis=1)

        return all_scores, all_indices

    def search_chunked(
        self,
        query_embeddings: np.ndarray,
        index_embeddings: np.ndarray,
        top_k: int,
        query_chunk_size: int = 2_000,    # ignored by FAISS paths
        cand_chunk_size:  int = 50_000,   # ignored by FAISS paths
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-K cosine similarity search with automatic backend selection.

        Priority:
          1. FAISS GPU  -- fastest, builds IVF index on GPU, ~2 min total
          2. FAISS CPU  -- ~10x faster than brute-force, no GPU needed
          3. Vectorised brute-force -- last resort, but with vectorised merge
             and larger chunks to minimise mmap thrashing
        """
        n_queries = len(query_embeddings)
        n_cands   = len(index_embeddings)

        # --- Try FAISS GPU -------------------------------------------------
        if HAS_FAISS and self.device.type == "cuda":
            try:
                import faiss as _faiss
                if hasattr(_faiss, 'StandardGpuResources'):
                    print(f"  Using FAISS GPU (IVF-Flat) for {n_queries} x {n_cands} search")
                    return self._search_faiss_gpu(
                        query_embeddings, index_embeddings, top_k
                    )
            except Exception as e:
                print(f"  FAISS GPU failed ({e}), trying FAISS CPU ...")

        # --- Try FAISS CPU -------------------------------------------------
        if HAS_FAISS:
            try:
                print(f"  Using FAISS CPU (IVF-Flat) for {n_queries} x {n_cands} search")
                return self._search_faiss_cpu(
                    query_embeddings, index_embeddings, top_k
                )
            except Exception as e:
                print(f"  FAISS CPU failed ({e}), falling back to brute-force ...")

        # --- Vectorised brute-force fallback --------------------------------
        print(f"  Using vectorised brute-force for {n_queries} x {n_cands} search")
        return self._search_bruteforce_vectorised(
            query_embeddings, index_embeddings, top_k,
            query_chunk_size=query_chunk_size,
            cand_chunk_size=cand_chunk_size,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
        candidate_pool_df: pd.DataFrame,
        top_k: Optional[int] = None,
        block_by_country: Optional[bool] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate candidate pairs for all Source 1 entities.
        Uses chunked search so the full candidate matrix is never in RAM.
        """
        top_k = top_k or self.config.top_k_candidates
        block_by_country = (
            block_by_country
            if block_by_country is not None
            else self.config.block_by_country
        )

        s1_ids   = source1_df["entity_id"].tolist()
        cand_ids = candidate_pool_df["entity_id"].tolist()

        if not candidate_pool_df.shape[0]:
            return {s1_id: [] for s1_id in s1_ids}

        candidate_results: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_ids}

        if block_by_country:
            countries = set(source1_df["country"].unique()) | set(
                candidate_pool_df["country"].unique()
            )

            for country in countries:
                s1_sub   = source1_df[source1_df["country"] == country]
                cand_sub = candidate_pool_df[candidate_pool_df["country"] == country]

                if not s1_sub.shape[0]:
                    continue
                if not cand_sub.shape[0]:
                    cand_sub = candidate_pool_df

                sub_s1_ids   = s1_sub["entity_id"].tolist()
                sub_cand_ids = cand_sub["entity_id"].tolist()

                print(
                    f"  Blocking [{country}]: "
                    f"{len(sub_s1_ids)} S1 vs {len(sub_cand_ids)} candidates"
                )

                # encode_texts returns mmap arrays -- not loaded into RAM until sliced
                s1_embeds   = self.encode_texts(
                    s1_sub["biencoder_text"].tolist(),
                    cache_prefix=f"s1_{country}",
                )
                cand_embeds = self.encode_texts(
                    cand_sub["biencoder_text"].tolist(),
                    cache_prefix=f"cand_{country}",
                )

                print(
                    f"  Searching (chunked) -- "
                    f"S1:{len(s1_embeds)} x Cand:{len(cand_embeds)}, top-K={top_k}"
                )
                _, indices = self.search_chunked(s1_embeds, cand_embeds, top_k=top_k)

                for i, s1_id in enumerate(sub_s1_ids):
                    retrieved = [sub_cand_ids[idx] for idx in indices[i] if idx >= 0]
                    candidate_results[s1_id] = [
                        cid for cid in retrieved
                        if cid.startswith(("S2-", "S3-")) and cid != s1_id
                    ]

                # Free embeddings after each country to reclaim RAM
                del s1_embeds, cand_embeds, indices
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        else:
            s1_embeds   = self.encode_texts(
                source1_df["biencoder_text"].tolist(), cache_prefix="s1_all"
            )
            cand_embeds = self.encode_texts(
                candidate_pool_df["biencoder_text"].tolist(), cache_prefix="cand_all"
            )
            _, indices = self.search_chunked(s1_embeds, cand_embeds, top_k=top_k)

            for i, s1_id in enumerate(s1_ids):
                retrieved = [cand_ids[idx] for idx in indices[i] if idx >= 0]
                candidate_results[s1_id] = [
                    cid for cid in retrieved
                    if cid.startswith(("S2-", "S3-")) and cid != s1_id
                ]

        return candidate_results
