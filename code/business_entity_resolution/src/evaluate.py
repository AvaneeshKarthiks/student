"""
Evaluation Module for Macro F_0.5 Metric.
Computes per-entity precision, recall, and F_0.5 score, including singletons,
matching the exact competition evaluation criteria.
"""

from typing import Dict, Set, Tuple, List, Any
import numpy as np


def compute_entity_f05(
    pred_set: Set[str],
    true_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_beta for a single Source 1 entity.
    
    Args:
        pred_set: Set of predicted matched entity IDs (S2-/S3-).
        true_set: Set of ground truth matched entity IDs (S2-/S3-).
        beta: Beta value for F-score (0.5 for precision weighting).
    
    Returns:
        (precision, recall, f_score)
    """
    beta_sq = beta ** 2
    factor = 1.0 + beta_sq  # 1.25 for beta=0.5

    # Case 1: Ground truth is empty (Singleton)
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0  # Correctly predicted singleton
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    # Case 2: Ground truth has matches, prediction is empty
    if not pred_set:
        return 0.0, 0.0, 0.0  # Missed all matches

    # Case 3: Both have entries
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        f_score = 0.0
    else:
        f_score = (factor * precision * recall) / denom

    return precision, recall, f_score


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Compute official macro F_0.5 score across all Source 1 entities.
    
    Args:
        predictions: Dict mapping source1_entity_id to set of predicted IDs.
        ground_truth: Dict mapping source1_entity_id to set of true IDs.
        
    Returns:
        Dictionary of performance metrics.
    """
    all_s1_ids = list(ground_truth.keys())
    if not all_s1_ids:
        return {"macro_f05": 0.0, "total_entities": 0}

    f05_scores = []
    precisions = []
    recalls = []

    singleton_count = 0
    correct_singletons = 0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for s1_id in all_s1_ids:
        true_matches = ground_truth.get(s1_id, set())
        pred_matches = predictions.get(s1_id, set())

        p, r, f05 = compute_entity_f05(pred_matches, true_matches, beta=0.5)

        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)

        if not true_matches:
            singleton_count += 1
            if not pred_matches:
                correct_singletons += 1
        else:
            tp = len(pred_matches & true_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

    macro_f05 = float(np.mean(f05_scores))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    singleton_acc = (correct_singletons / singleton_count) if singleton_count > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_entities": len(all_s1_ids),
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
    }
