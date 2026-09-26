"""
Training script for ModernBERT-base Bi-Encoder candidate generator.
Fine-tunes the bi-encoder using InfoNCE / Multiple Negatives Ranking Loss on true matching pairs.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import PipelineConfig, BlockingConfig
from .preprocess import preprocess_dataframe
from .dataset import BiEncoderTrainDataset
from .blocking_biencoder import ModernBERTBiEncoder
from .utils import load_tsv, load_ground_truth


class ContrastiveBiEncoderLoss(nn.Module):
    """Multiple Negatives Ranking Loss (InfoNCE) for symmetric contrastive training."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        # Cosine similarity matrix scaled by temperature
        sim_matrix = torch.matmul(embeddings_a, embeddings_b.t()) / self.temperature
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        loss = self.cross_entropy(sim_matrix, labels)
        return loss


def build_training_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth: dict,
) -> List[Tuple[str, str]]:
    """Create (anchor, positive) text pairs from ground truth matching labels."""
    s1_dict = s1_df.set_index("entity_id")["biencoder_text"].to_dict()
    s2_dict = s2_df.set_index("entity_id")["biencoder_text"].to_dict()
    s3_dict = s3_df.set_index("entity_id")["biencoder_text"].to_dict()

    pool_dict = {**s2_dict, **s3_dict}
    pairs = []

    for s1_id, matched_ids in ground_truth.items():
        if s1_id not in s1_dict:
            continue
        text_a = s1_dict[s1_id]
        for mid in matched_ids:
            if mid in pool_dict:
                text_b = pool_dict[mid]
                pairs.append((text_a, text_b))

    print(f"Constructed {len(pairs)} positive pairs for bi-encoder contrastive training.")
    return pairs


def train_biencoder(config: PipelineConfig):
    """Execute bi-encoder contrastive training."""
    b_conf = config.blocking
    device = torch.device(b_conf.device)

    print("Loading raw training files...")
    s1_raw = load_tsv(config.train_source1_path)
    s2_raw = load_tsv(config.train_source2_path)
    s3_raw = load_tsv(config.train_source3_path)
    ground_truth = load_ground_truth(config.train_ground_truth_path)

    print("Preprocessing records...")
    s1_prep = preprocess_dataframe(s1_raw)
    s2_prep = preprocess_dataframe(s2_raw)
    s3_prep = preprocess_dataframe(s3_raw)

    pairs = build_training_pairs(s1_prep, s2_prep, s3_prep, ground_truth)
    random.shuffle(pairs)

    tokenizer = AutoTokenizer.from_pretrained(b_conf.model_name)
    model = ModernBERTBiEncoder(b_conf.model_name).to(device)
    loss_fn = ContrastiveBiEncoderLoss(temperature=0.05)

    dataset = BiEncoderTrainDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=b_conf.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=b_conf.learning_rate, weight_decay=b_conf.weight_decay)
    total_steps = len(dataloader) * b_conf.num_epochs
    warmup_steps = int(total_steps * b_conf.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Starting Bi-Encoder training for {b_conf.num_epochs} epochs on {device}...")
    model.train()
    for epoch in range(1, b_conf.num_epochs + 1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{b_conf.num_epochs}")
        for batch_a, batch_b in pbar:
            optimizer.zero_grad()

            enc_a = tokenizer(
                list(batch_a),
                padding=True,
                truncation=True,
                max_length=b_conf.max_seq_length,
                return_tensors="pt",
            ).to(device)

            enc_b = tokenizer(
                list(batch_b),
                padding=True,
                truncation=True,
                max_length=b_conf.max_seq_length,
                return_tensors="pt",
            ).to(device)

            emb_a = model(enc_a["input_ids"], enc_a["attention_mask"])
            emb_b = model(enc_b["input_ids"], enc_b["attention_mask"])

            loss = loss_fn(emb_a, emb_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    b_conf.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = b_conf.checkpoint_dir / "modernbert_biencoder.pt"
    torch.save(model.state_dict(), save_path)
    print(f"ModernBERT Bi-Encoder checkpoint saved to {save_path}")


if __name__ == "__main__":
    cfg = PipelineConfig()
    train_biencoder(cfg)
