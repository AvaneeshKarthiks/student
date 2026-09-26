"""
Configuration module for the Business Entity Resolution System.
Contains paths, model parameters, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CACHE_DIR = BASE_DIR / "cache"

# Ensure output and checkpoint directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessingConfig:
    """Settings for text normalization and country parsing."""
    lowercase: bool = True
    standardize_suffixes: bool = True
    normalize_addresses: bool = True
    remove_accents: bool = True


@dataclass
class BlockingConfig:
    """Settings for the ModernBERT bi-encoder blocking stage.

    Batch sizes tuned for 16 GB RAM / 4 GB VRAM:
      - batch_size=32: ModernBERT-base at seq_len=256 uses ~2.5 GB VRAM.
    """
    model_name: str = "answerdotai/ModernBERT-base"
    max_seq_length: int = 256
    batch_size: int = 256 if (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9) else 32
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    top_k_candidates: int = 30
    similarity_metric: str = "cosine"  # cosine or ip
    block_by_country: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = field(
        default_factory=lambda: CHECKPOINTS_DIR / "biencoder_modernbert"
    )


@dataclass
class CrossEncoderConfig:
    """Settings for the DeBERTa-v3-large precision matching stage.

    Batch sizes tuned for 16 GB RAM / 4 GB VRAM:
      - batch_size=4, max_seq_length=256 → ~3.5 GB VRAM.
      - gradient_accumulation_steps=8 keeps effective batch = 32.
    """
    model_name: str = "microsoft/deberta-v3-large"
    max_seq_length: int = 256     # halves VRAM vs 512; entity records are short
    batch_size: int = 64 if (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 20e9) else 4
    gradient_accumulation_steps: int = 8   # effective batch = 32 (was 2)
    learning_rate: float = 1.5e-5
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = field(
        default_factory=lambda: CHECKPOINTS_DIR / "crossencoder_deberta"
    )
    # Asymmetric loss weights (penalize false merges)
    asymmetric_gamma_pos: float = 1.0
    asymmetric_gamma_neg: float = 4.0
    asymmetric_weight_neg: float = 3.5
    clip_margin: float = 0.05


@dataclass
class OptimizationConfig:
    """Settings for Optuna threshold search and post-processing."""
    n_trials: int = 50
    default_threshold: float = 0.85
    min_threshold: float = 0.50
    max_threshold: float = 0.99
    # Graph clustering / Verified merge settings
    enable_verified_merge: bool = True
    min_pairwise_agreement: float = 0.60


@dataclass
class PipelineConfig:
    """Master configuration for the complete entity resolution pipeline.

    All nested dataclass fields use default_factory so Python's dataclass
    machinery does not complain about mutable defaults.
    """
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    matching: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    train_dir: Path = field(default_factory=lambda: TRAIN_DIR)
    test_dir: Path = field(default_factory=lambda: TEST_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    train_source1_path: Path = field(default_factory=lambda: TRAIN_DIR / "train_source1.tsv")
    train_source2_path: Path = field(default_factory=lambda: TRAIN_DIR / "train_source2.tsv")
    train_source3_path: Path = field(default_factory=lambda: TRAIN_DIR / "train_source3.tsv")
    train_ground_truth_path: Path = field(
        default_factory=lambda: TRAIN_DIR / "train_ground_truth.tsv"
    )

    test_source1_path: Path = field(default_factory=lambda: TEST_DIR / "test_source1.tsv")
    test_source2_path: Path = field(default_factory=lambda: TEST_DIR / "test_source2.tsv")
    test_source3_path: Path = field(default_factory=lambda: TEST_DIR / "test_source3.tsv")

    output_matching_path: Path = field(
        default_factory=lambda: OUTPUT_DIR / "matching_results.tsv"
    )
    output_candidate_path: Path = field(
        default_factory=lambda: OUTPUT_DIR / "candidate_pairs.tsv"
    )
