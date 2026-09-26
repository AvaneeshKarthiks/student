"""
Threshold Optimization Module using Optuna.
Finds the exact probability cutoff that maximizes the Macro F_0.5 score
on validation data, aggressively suppressing false merges.
"""

from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .evaluate import evaluate_predictions
from .config import OptimizationConfig


def predictions_from_threshold(
    scored_pairs: List[Dict[str, Any]],
    all_s1_ids: List[str],
    threshold: float,
) -> Dict[str, Set[str]]:
    """
    Filter scored pairs by probability threshold into entity prediction sets.
    """
    preds = {s1_id: set() for s1_id in all_s1_ids}
    for item in scored_pairs:
        s1 = item["source1_entity_id"]
        cid = item["candidate_entity_id"]
        prob = item["probability"]
        if prob >= threshold and s1 in preds:
            preds[s1].add(cid)
    return preds


def optimize_threshold_optuna(
    scored_pairs: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    config: OptimizationConfig,
) -> Tuple[float, float]:
    """
    Run Optuna Bayesian hyperparameter search to find threshold maximizing Macro F_0.5.
    
    Args:
        scored_pairs: List of dicts with 'source1_entity_id', 'candidate_entity_id', 'probability'.
        ground_truth: Dict mapping source1_entity_id to true set of matched IDs.
        config: Optimization configuration dataclass.
        
    Returns:
        (best_threshold, best_macro_f05)
    """
    all_s1_ids = list(ground_truth.keys())

    if not HAS_OPTUNA:
        print("Optuna not installed. Falling back to fine-grained grid search.")
        return optimize_threshold_grid(scored_pairs, ground_truth, config)

    def objective(trial: optuna.Trial) -> float:
        thresh = trial.suggest_float("threshold", config.min_threshold, config.max_threshold)
        preds = predictions_from_threshold(scored_pairs, all_s1_ids, thresh)
        metrics = evaluate_predictions(preds, ground_truth)
        return metrics["macro_f05"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    best_thresh = study.best_params["threshold"]
    best_score = study.best_value
    print(f"Optuna Best Threshold: {best_thresh:.4f} -> Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_score


def optimize_threshold_grid(
    scored_pairs: List[Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    config: OptimizationConfig,
) -> Tuple[float, float]:
    """
    Deterministic grid search fallback for threshold optimization.
    """
    all_s1_ids = list(ground_truth.keys())
    threshold_candidates = np.linspace(config.min_threshold, config.max_threshold, num=100)

    best_score = -1.0
    best_thresh = config.default_threshold

    for thresh in threshold_candidates:
        preds = predictions_from_threshold(scored_pairs, all_s1_ids, thresh)
        metrics = evaluate_predictions(preds, ground_truth)
        score = metrics["macro_f05"]
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    print(f"Grid Search Best Threshold: {best_thresh:.4f} -> Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_score
