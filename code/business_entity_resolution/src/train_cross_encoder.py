"""
Training script for DeBERTa-v3-large Cross-Encoder with Ditto serialization and Asymmetric Loss.
Fine-tunes model on positive matches and hard negative candidate pairs.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import PipelineConfig, CrossEncoderConfig
from .preprocess import preprocess_dataframe
from .dataset import CrossEncoderDataset
from .cross_encoder_ditto import DeBERTaDittoCrossEncoder
from .loss import AsymmetricLoss
from .evaluate import evaluate_predictions
from .utils import load_tsv, load_ground_truth


def build_crossencoder_training_data(
    s1_dict: Dict[str, str],
    pool_dict: Dict[str, str],
    ground_truth: Dict[str, Set[str]],
    candidates_dict: Dict[str, List[str]],
    negatives_per_positive: int = 3,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Construct training dataset of (text_a, text_b) pairs with binary labels (1 or 0).
    Pairs include all ground truth positives and hard negatives sampled from blocking candidates.
    """
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []

    for s1_id, true_matches in ground_truth.items():
        if s1_id not in s1_dict:
            continue
        text_a = s1_dict[s1_id]

        # Add true positives
        for mid in true_matches:
            if mid in pool_dict:
                pairs.append((text_a, pool_dict[mid]))
                labels.append(1)

        # Sample hard negatives from blocking candidate list
        cand_list = candidates_dict.get(s1_id, [])
        hard_negs = [cid for cid in cand_list if cid not in true_matches and cid in pool_dict]

        if hard_negs:
            num_negs = min(len(hard_negs), max(len(true_matches) * negatives_per_positive, 2))
            sampled_negs = random.sample(hard_negs, num_negs)
            for cid in sampled_negs:
                pairs.append((text_a, pool_dict[cid]))
                labels.append(0)

    print(f"Created {len(pairs)} pairs ({sum(labels)} positives, {len(labels) - sum(labels)} hard negatives).")
    return pairs, labels


def train_crossencoder(config: PipelineConfig):
    """Execute DeBERTa-v3-large Cross-Encoder fine-tuning with Asymmetric Loss."""
    c_conf = config.matching
    device = torch.device(c_conf.device)

    print("Loading training data for Cross-Encoder...")
    s1_raw = load_tsv(config.train_source1_path)
    s2_raw = load_tsv(config.train_source2_path)
    s3_raw = load_tsv(config.train_source3_path)
    ground_truth = load_ground_truth(config.train_ground_truth_path)

    print("Preprocessing records for Ditto format...")
    s1_prep = preprocess_dataframe(s1_raw)
    s2_prep = preprocess_dataframe(s2_raw)
    s3_prep = preprocess_dataframe(s3_raw)

    s1_dict = s1_prep.set_index("entity_id")["ditto_text"].to_dict()
    s2_dict = s2_prep.set_index("entity_id")["ditto_text"].to_dict()
    s3_dict = s3_prep.set_index("entity_id")["ditto_text"].to_dict()
    pool_dict = {**s2_dict, **s3_dict}

    # Dummy/simple candidates for training if no precomputed candidate file
    dummy_cands = {s1: list(ground_truth.get(s1, set())) for s1 in s1_dict.keys()}

    pairs, labels = build_crossencoder_training_data(s1_dict, pool_dict, ground_truth, dummy_cands)
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined) if combined else ([], [])
    pairs, labels = list(pairs), list(labels)

    tokenizer = AutoTokenizer.from_pretrained(c_conf.model_name)
    model = DeBERTaDittoCrossEncoder(c_conf.model_name).to(device)

    loss_fn = AsymmetricLoss(
        gamma_pos=c_conf.asymmetric_gamma_pos,
        gamma_neg=c_conf.asymmetric_gamma_neg,
        weight_neg=c_conf.asymmetric_weight_neg,
        clip_margin=c_conf.clip_margin,
    )

    dataset = CrossEncoderDataset(pairs=pairs, labels=labels, tokenizer=tokenizer, max_length=c_conf.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=c_conf.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=c_conf.learning_rate, weight_decay=c_conf.weight_decay)
    total_steps = (len(dataloader) // c_conf.gradient_accumulation_steps) * c_conf.num_epochs
    warmup_steps = int(total_steps * c_conf.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training DeBERTa-v3-large on {device} with Asymmetric Loss...")
    model.train()
    step = 0

    for epoch in range(1, c_conf.num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{c_conf.num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, batch_labels)
            loss = loss / c_conf.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * c_conf.gradient_accumulation_steps

            if (batch_idx + 1) % c_conf.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            pbar.set_postfix({"loss": f"{loss.item() * c_conf.gradient_accumulation_steps:.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    c_conf.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = c_conf.checkpoint_dir / "deberta_crossencoder.pt"
    torch.save(model.state_dict(), save_path)
    print(f"DeBERTa-v3-large Cross-Encoder checkpoint saved to {save_path}")


if __name__ == "__main__":
    cfg = PipelineConfig()
    train_crossencoder(cfg)
