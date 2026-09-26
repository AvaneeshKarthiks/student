"""
Dataset module providing PyTorch Dataset classes for Bi-encoder and Cross-encoder training.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextInferenceDataset(Dataset):
    """Simple dataset for batched embedding extraction."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class BiEncoderTrainDataset(Dataset):
    """
    Dataset for training ModernBERT bi-encoder with contrastive learning.
    Yields (anchor_text, positive_text).
    """

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


class CrossEncoderDataset(Dataset):
    """
    Dataset for training or running inference on DeBERTa-v3 cross-encoder (Ditto style).
    Accepts text pairs (text_a, text_b) and optional binary labels.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
    ):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text_a, text_b = self.pairs[idx]
        
        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item
