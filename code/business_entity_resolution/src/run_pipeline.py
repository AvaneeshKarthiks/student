"""
Master Pipeline Script for Business Entity Resolution.
Executes the two-stage Filter-and-Refine architecture:
  1. Data Preprocessing & Country Normalization
  2. ModernBERT Bi-Encoder Candidate Generation  → output/candidate_pairs.tsv
  3. DeBERTa-v3-large Ditto Cross-Encoder Matching → cache/scored_pairs_*.json
  4. Verified Merge Graph Clustering              → output/matching_results.tsv

All intermediate results are persisted to disk so a crash never loses work:
  • CACHE_DIR/preprocessed_<tag>_<hash>.parquet  – cleaned DataFrames
  • CACHE_DIR/s1_<country>_<hash>.npy            – S1 ModernBERT embeddings
  • CACHE_DIR/cand_<country>_<hash>.npy          – Candidate embeddings
  • output/candidate_pairs.tsv                   – Written right after blocking
  • CACHE_DIR/scored_pairs_<hash>.json           – DeBERTa scores (resumed on crash)
  • output/scored_pairs_debug.tsv               – Human-readable score dump
  • output/matching_results.tsv                  – Final output

Memory strategy (16 GB RAM / 4 GB VRAM):
  • Raw DataFrames are deleted as soon as preprocessed copies are ready.
  • The bi-encoder model + all embeddings are explicitly deleted (and CUDA
    cache flushed) before the cross-encoder model is loaded.
  • The cross-encoder model is deleted before clustering.
  • gc.collect() is called after every major deletion.
"""

import argparse
import gc
import hashlib
import time
from pathlib import Path

import pandas as pd
import torch

from .config import BlockingConfig, CrossEncoderConfig, CACHE_DIR
from .preprocess import preprocess_dataframe
from .blocking_biencoder import BiEncoderBlockingEngine
from .cross_encoder_ditto import CrossEncoderMatchingEngine
from .graph_clustering import VerifiedMergeClusterer
from .utils import load_tsv, save_candidate_pairs, save_matching_results


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _free(*objects, label: str = "") -> None:
    """Delete objects, run gc.collect(), and flush CUDA cache."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if label:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [mem] After {label}: CUDA {used:.2f} GB used / {reserved:.2f} GB reserved")
        else:
            print(f"  [mem] After {label}: freed (CPU-only mode)")


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    test_dir: Path,
    output_dir: Path,
    top_k: int = 30,
    threshold: float = 0.85,
    device: str = "cuda",
):
    """
    Run end-to-end entity resolution inference on test files.
    Every intermediate result is written to disk and large objects are freed
    before the next stage loads its model.
    """
    start_time = time.time()
    print("=" * 70)
    print("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"  Test dir  : {test_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Top-K: {top_k}  |  Threshold: {threshold}  |  Device: {device}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matching_tsv_path   = output_dir / "matching_results.tsv"
    candidate_tsv_path  = output_dir / "candidate_pairs.tsv"

    # ──────────────────────────────────────────────────────────────────
    # Step 1 – Load & Preprocess
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 1/5] Loading and Preprocessing Data...")

    s1_df_raw = load_tsv(test_dir / "test_source1.tsv")
    s2_df_raw = load_tsv(test_dir / "test_source2.tsv")
    s3_df_raw = load_tsv(test_dir / "test_source3.tsv")

    all_s1_ids = s1_df_raw["entity_id"].tolist()
    print(f"  Loaded {len(s1_df_raw)} S1 | {len(s2_df_raw)} S2 | {len(s3_df_raw)} S3 records.")

    def _preprocess_cached(df_raw: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Preprocess with a parquet cache keyed by row count + first/last IDs."""
        key     = f"{tag}_{len(df_raw)}_{df_raw['entity_id'].iloc[0]}_{df_raw['entity_id'].iloc[-1]}"
        digest  = hashlib.md5(key.encode()).hexdigest()
        cache_p = CACHE_DIR / f"preprocessed_{tag}_{digest}.parquet"
        if cache_p.exists():
            print(f"  [cache hit] Preprocessed {tag} ← {cache_p.name}")
            return pd.read_parquet(cache_p)
        print(f"  Cleaning {tag} ({len(df_raw)} records)...")
        result = preprocess_dataframe(df_raw)
        result.to_parquet(cache_p, index=False)
        print(f"  [cache saved] {cache_p.name}")
        return result

    s1_df = _preprocess_cached(s1_df_raw, "s1")
    s2_df = _preprocess_cached(s2_df_raw, "s2")
    s3_df = _preprocess_cached(s3_df_raw, "s3")

    # Raw TSV DataFrames are no longer needed — free them now
    _free(s1_df_raw, s2_df_raw, s3_df_raw, label="raw TSVs freed")

    candidate_pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    _free(s2_df, s3_df, label="s2/s3 source frames freed (pool built)")
    print(f"  Candidate pool: {len(candidate_pool_df)} records.  RAM: {_ram_gb():.1f} GB")

    # ──────────────────────────────────────────────────────────────────
    # Step 2 – Blocking (ModernBERT embeddings + FAISS)
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 2/5] Candidate Generation (ModernBERT Bi-Encoder)...")
    blocking_cfg    = BlockingConfig(top_k_candidates=top_k, device=device)
    blocking_engine = BiEncoderBlockingEngine(config=blocking_cfg)

    candidates_dict = blocking_engine.generate_candidates(
        source1_df=s1_df,
        candidate_pool_df=candidate_pool_df,
        top_k=top_k,
        block_by_country=True,
    )

    # Save candidate pairs immediately — crash-safe checkpoint
    print(f"\n  Saving candidate pairs → {candidate_tsv_path}")
    save_candidate_pairs(candidates_dict, candidate_tsv_path, all_s1_ids)
    print(
        f"  Wrote {sum(len(v) for v in candidates_dict.values())} links "
        f"for {len(all_s1_ids)} S1 entities."
    )

    # ★ Free the bi-encoder model + all associated GPU memory before
    #   loading the much larger DeBERTa cross-encoder.
    del blocking_engine          # drops model weights from VRAM
    del blocking_cfg
    _free(label="bi-encoder model + CUDA cache cleared")
    print(f"  RAM after blocking cleanup: {_ram_gb():.1f} GB")

    # ──────────────────────────────────────────────────────────────────
    # Step 3 – Deep Matching (DeBERTa cross-encoder)
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 3/5] Deep Precision Matching (DeBERTa-v3-large Ditto)...")

    # Build lookup dicts with ONLY the columns actually used downstream:
    #   • score_candidates_dict  → "ditto_text"
    #   • verify_and_merge       → "clean_name", "postal_code"
    # Converting ALL columns on 3.8M rows via to_dict(orient="index") is
    # extremely slow; restricting to 3 columns is ~3× faster and uses less RAM.
    _SCORE_COLS = ["ditto_text", "clean_name", "postal_code"]

    print("  Building S1 record lookup dict...")
    s1_records_dict = (
        s1_df.set_index("entity_id")[_SCORE_COLS].to_dict(orient="index")
    )

    print(f"  Building candidate record lookup dict ({len(candidate_pool_df)} rows)...")
    candidate_records_dict = (
        candidate_pool_df.set_index("entity_id")[_SCORE_COLS].to_dict(orient="index")
    )

    # Free the large DataFrames — dicts are sufficient from here on
    _free(s1_df, candidate_pool_df, label="source DataFrames freed (dicts built)")
    print(f"  RAM after DF→dict conversion: {_ram_gb():.1f} GB")

    matching_cfg    = CrossEncoderConfig(device=device)
    matching_engine = CrossEncoderMatchingEngine(config=matching_cfg)

    scored_pairs = matching_engine.score_candidates_dict(
        candidate_pairs_dict=candidates_dict,
        source1_records=s1_records_dict,
        candidate_records=candidate_records_dict,
    )

    # Save human-readable score dump for debugging
    scored_tsv_path = output_dir / "scored_pairs_debug.tsv"
    pd.DataFrame(scored_pairs).to_csv(scored_tsv_path, sep="\t", index=False)
    print(f"  Saved all scored pairs → {scored_tsv_path}")

    # ★ Free the cross-encoder model before clustering
    del matching_engine
    del matching_cfg
    del candidates_dict          # large dict no longer needed
    del s1_records_dict          # ditto_text strings freed
    _free(label="cross-encoder model + score inputs freed")
    print(f"  RAM before clustering: {_ram_gb():.1f} GB")

    # ──────────────────────────────────────────────────────────────────
    # Steps 4 & 5 – Verified Merge Clustering + Save
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 4/5] Verified Merge Clustering (threshold={threshold:.2f})...")
    clusterer = VerifiedMergeClusterer(
        min_probability_threshold=threshold,
        min_pairwise_agreement=0.40,
        enable_consistency_check=True,
    )

    final_matches = clusterer.verify_and_merge(
        scored_pairs=scored_pairs,
        all_s1_ids=all_s1_ids,
        entity_records=candidate_records_dict,
    )

    # Free scored pairs list now that clustering is done
    _free(scored_pairs, candidate_records_dict, label="scored pairs + record dicts freed")

    print(f"\n[Step 5/5] Saving Results → {matching_tsv_path}")
    save_matching_results(final_matches, matching_tsv_path, all_s1_ids)

    non_empty  = sum(1 for m in final_matches.values() if m)
    singletons = len(final_matches) - non_empty
    print(
        f"  Summary: {len(final_matches)} entities | "
        f"{non_empty} matched | {singletons} singletons"
    )
    print(f"  Pipeline finished in {time.time() - start_time:.1f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Business Entity Resolution Pipeline."
    )
    parser.add_argument("--test-dir",   type=str,   default="dataset/test",
                        help="Folder containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", type=str,   default="output",
                        help="Folder where output TSVs are saved")
    parser.add_argument("--top-k",      type=int,   default=30,
                        help="Candidates per S1 entity in blocking stage")
    parser.add_argument("--threshold",  type=float, default=0.85,
                        help="Probability cutoff for cross-encoder matching")
    parser.add_argument("--device",     type=str,   default="cuda",
                        help="Execution device: cuda or cpu")

    args = parser.parse_args()
    run_pipeline(
        test_dir=Path(args.test_dir),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
