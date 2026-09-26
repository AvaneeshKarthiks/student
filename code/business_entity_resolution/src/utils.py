"""
Utility functions for file I/O, dataset parsing, and submission output formatting.
Enforces strict tab-separated TSV operations.
"""

from pathlib import Path
from typing import Dict, Set, List, Optional, Union
import pandas as pd


def load_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a tab-separated (.tsv) file.
    
    Args:
        filepath: Path to the TSV file.
        
    Returns:
        Loaded pandas DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read with explicit tab separator and string types
    return pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )


def load_ground_truth(filepath: Union[str, Path]) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping from train_ground_truth.tsv.
    
    Args:
        filepath: Path to ground truth file.
        
    Returns:
        Dict mapping source1_entity_id to set of true matched entity IDs.
    """
    df = load_tsv(filepath)
    ground_truth = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched_str = str(row.get("matched_entity_ids", "")).strip()
        if matched_str:
            ids = {mid.strip() for mid in matched_str.split(",") if mid.strip()}
        else:
            ids = set()
        ground_truth[s1] = ids
    return ground_truth


def save_candidate_pairs(
    candidates_dict: Dict[str, Union[List[str], Set[str]]],
    output_path: Union[str, Path],
    all_source1_ids: List[str],
) -> None:
    """
    Save candidate pairs to candidate_pairs.tsv with exact required format.
    
    Args:
        candidates_dict: Mapping from source1_id to list/set of candidate IDs.
        output_path: Destination TSV file path.
        all_source1_ids: Ordered list of all Source 1 entity IDs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1 in all_source1_ids:
        cands = candidates_dict.get(s1, [])
        # Deduplicate while preserving order or sorting
        if isinstance(cands, set):
            cand_list = sorted(cands)
        else:
            cand_list = list(dict.fromkeys(cands))
        
        # S2 and S3 IDs only
        valid_cands = [c for c in cand_list if c.startswith(("S2-", "S3-")) and c != s1]
        cand_str = ",".join(valid_cands)
        rows.append({"source1_entity_id": s1, "candidate_entity_ids": cand_str})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


def save_matching_results(
    matches_dict: Dict[str, Union[List[str], Set[str]]],
    output_path: Union[str, Path],
    all_source1_ids: List[str],
) -> None:
    """
    Save final matched entity IDs to matching_results.tsv.
    
    Args:
        matches_dict: Mapping from source1_id to list/set of matched IDs.
        output_path: Destination TSV file path.
        all_source1_ids: Ordered list of all Source 1 entity IDs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1 in all_source1_ids:
        matches = matches_dict.get(s1, [])
        if isinstance(matches, set):
            match_list = sorted(matches)
        else:
            match_list = list(dict.fromkeys(matches))

        # Filter out self-matches or malformed prefixes
        valid_matches = [m for m in match_list if m.startswith(("S2-", "S3-")) and m != s1]
        matched_str = ",".join(valid_matches)
        rows.append({"source1_entity_id": s1, "matched_entity_ids": matched_str})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
