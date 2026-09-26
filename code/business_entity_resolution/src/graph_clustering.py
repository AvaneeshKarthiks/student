"""
Graph Clustering and Output Generation Module.
Implements the 'Verified Merge' strategy to prevent transitive chaining errors,
protect singletons, and generate valid competition TSV files.
"""

from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import pandas as pd


def jaccard_similarity(str_a: str, str_b: str) -> float:
    """Calculate token-level Jaccard similarity between two strings."""
    tokens_a = set(str_a.lower().split())
    tokens_b = set(str_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class VerifiedMergeClusterer:
    """
    Clusterer applying Verified Merge strategy to prevent false merge chaining.
    """

    def __init__(
        self,
        min_probability_threshold: float = 0.85,
        min_pairwise_agreement: float = 0.40,
        enable_consistency_check: bool = True,
    ):
        self.threshold = min_probability_threshold
        self.min_agreement = min_pairwise_agreement
        self.enable_consistency_check = enable_consistency_check

    def verify_and_merge(
        self,
        scored_pairs: List[Dict[str, Any]],
        all_s1_ids: List[str],
        entity_records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Group pairwise predictions for each Source 1 reference entity with Verified Merge.
        
        Args:
            scored_pairs: List of dicts with 'source1_entity_id', 'candidate_entity_id', 'probability'.
            all_s1_ids: Complete list of test Source 1 entity IDs.
            entity_records: Optional dict of entity details for consistency checks.
            
        Returns:
            Dict mapping source1_entity_id to list of verified matched entity IDs.
        """
        # Step 1: Collect candidates above threshold for each S1 entity
        candidates_by_s1 = defaultdict(list)
        for item in scored_pairs:
            s1 = item["source1_entity_id"]
            cid = item["candidate_entity_id"]
            prob = float(item["probability"])
            if prob >= self.threshold and cid.startswith(("S2-", "S3-")) and cid != s1:
                candidates_by_s1[s1].append((cid, prob))

        final_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        # Step 2: Process each S1 entity
        for s1_id in all_s1_ids:
            cand_list = candidates_by_s1.get(s1_id, [])
            if not cand_list:
                # Retain as singleton
                continue

            # Sort by probability descending
            cand_list.sort(key=lambda x: x[1], reverse=True)

            if len(cand_list) == 1 or not self.enable_consistency_check or not entity_records:
                final_matches[s1_id] = [c[0] for c in cand_list]
                continue

            # Verified Merge multi-match consistency validation
            # The top candidate is the anchor candidate
            anchor_cid, _ = cand_list[0]
            anchor_info = entity_records.get(anchor_cid, {})
            anchor_name = anchor_info.get("clean_name", "")
            anchor_postal = anchor_info.get("postal_code", "")

            verified = [anchor_cid]

            for other_cid, prob in cand_list[1:]:
                other_info = entity_records.get(other_cid, {})
                other_name = other_info.get("clean_name", "")
                other_postal = other_info.get("postal_code", "")

                # Check 1: If postal codes exist in both and differ, potential false merge
                if anchor_postal and other_postal and anchor_postal != other_postal:
                    continue

                # Check 2: Mutual name consistency
                sim = jaccard_similarity(anchor_name, other_name)
                if sim >= self.min_agreement:
                    verified.append(other_cid)

            # Deduplicate while preserving order
            final_matches[s1_id] = list(dict.fromkeys(verified))

        return final_matches
