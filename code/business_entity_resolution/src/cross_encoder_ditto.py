"""
Deep Precision Matching Module using DeBERTa-v3-large and the Ditto Framework.
Performs token-level joint semantic comparison between business entity pairs.

Crash-safe: scored pairs are written to a JSON progress file under CACHE_DIR
every `chunk_size` pairs. A restart reloads the cache and skips already-scored
pairs — no work is ever lost.
"""

from typing import List, Dict, Tuple, Optional, Any
import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

from .config import CrossEncoderConfig, CACHE_DIR
from .dataset import CrossEncoderDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeBERTaDittoCrossEncoder(nn.Module):
    """
    DeBERTa-v3-large Cross-Encoder matching model.
    Takes joint token representations of entity pairs and outputs a binary match logit.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-large"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.model(**kwargs)
        return outputs.logits.view(-1)


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

class CrossEncoderMatchingEngine:
    """
    Matching engine for scoring candidate pairs using DeBERTa-v3-large in Ditto format.
    """

    def __init__(self, config: Optional[CrossEncoderConfig] = None):
        self.config = config or CrossEncoderConfig()
        self.device = torch.device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        model = DeBERTaDittoCrossEncoder(self.config.model_name).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs for CrossEncoder!")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        self.model.eval()

    # ------------------------------------------------------------------
    # Low-level batch scorer
    # ------------------------------------------------------------------

    def predict_pair_probabilities(
        self,
        candidate_pairs: List[Tuple[str, str, str, str]],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """
        Score a list of entity pairs.

        Args:
            candidate_pairs: List of (s1_id, cand_id, s1_ditto_text, cand_ditto_text).
            batch_size: Evaluation batch size (defaults to config value).

        Returns:
            List of match probabilities in [0.0, 1.0].
        """
        if not candidate_pairs:
            return []

        batch_size = batch_size or self.config.batch_size
        text_pairs = [(item[2], item[3]) for item in candidate_pairs]

        dataset = CrossEncoderDataset(
            pairs=text_pairs,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probabilities = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DeBERTa-v3 Matching", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                probs = torch.sigmoid(logits).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                probabilities.extend(probs)

        return probabilities

    # ------------------------------------------------------------------
    # High-level crash-safe scorer
    # ------------------------------------------------------------------

    def score_candidates_dict(
        self,
        candidate_pairs_dict: Dict[str, List[str]],
        source1_records: Dict[str, Dict[str, Any]],
        candidate_records: Dict[str, Dict[str, Any]],
        chunk_size: int = 10_000,
    ) -> List[Dict[str, Any]]:
        """
        Flatten the candidate dict into pairs, score with DeBERTa, and return results.

        Crash-safe: results are written to CACHE_DIR/scored_pairs_<hash>.json after
        every `chunk_size` pairs. A restart reloads the cache and resumes from the
        last saved position — no scoring work is lost.
        """
        print("  Building candidate pair list...")
        # Build the flat pair list
        pair_tuples: List[Tuple[str, str, str, str]] = []
        for s1_id, cand_ids in candidate_pairs_dict.items():
            s1_info = source1_records.get(s1_id)
            if not s1_info:
                continue
            s1_text = s1_info["ditto_text"]
            for cid in cand_ids:
                cand_info = candidate_records.get(cid)
                if not cand_info:
                    continue
                pair_tuples.append((s1_id, cid, s1_text, cand_info["ditto_text"]))

        print(f"  Scoring {len(pair_tuples)} total candidate pairs with DeBERTa-v3-large...")

        # Crash-safe cache key: use a fast structural fingerprint — count + first/last
        # pair IDs — instead of joining ALL pair strings (which was allocating >50 MB
        # of string before even starting to hash).
        if pair_tuples:
            key_str = (
                f"{len(pair_tuples)}"
                f"|{pair_tuples[0][0]}:{pair_tuples[0][1]}"
                f"|{pair_tuples[-1][0]}:{pair_tuples[-1][1]}"
            )
        else:
            key_str = "empty"
        cache_key = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"scored_pairs_{cache_key}.json"

        scored_pairs: List[Dict[str, Any]] = []
        start_idx = 0

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                scored_pairs = json.load(f)
            start_idx = len(scored_pairs)
            print(f"  [cache hit] Resuming from pair {start_idx}/{len(pair_tuples)}")

        # Score in chunks and save after each chunk
        for chunk_start in range(start_idx, len(pair_tuples), chunk_size):
            chunk = pair_tuples[chunk_start : chunk_start + chunk_size]
            probs = self.predict_pair_probabilities(chunk)
            for (s1_id, cid, _, _), prob in zip(chunk, probs):
                scored_pairs.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "probability": float(prob),
                    }
                )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(scored_pairs, f)
            print(
                f"  [cache saved] scored_pairs_{cache_key[:8]}…json  "
                f"({len(scored_pairs)}/{len(pair_tuples)} pairs)"
            )

        return scored_pairs
