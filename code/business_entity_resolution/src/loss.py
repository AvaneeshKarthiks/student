"""
Loss Module for F_0.5 Optimization.
Implements Asymmetric Loss to penalize False Merges (False Positives)
much more heavily than Missed Matches (False Negatives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Entity Resolution.
    Penalizes false merges (false positives) aggressively compared to false negatives.
    
    Formula:
        For y = 1 (Positive / True Match):
            L+ = - (1 - p)^gamma_pos * log(max(p, eps))
        For y = 0 (Negative / Non-match):
            p_m = clamp(p - clip_margin, min=0.0)
            L- = - weight_neg * (p_m)^gamma_neg * log(max(1 - p_m, eps))
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        weight_neg: float = 3.5,
        clip_margin: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.weight_neg = weight_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Predicted logits from model of shape (batch_size,) or (batch_size, 1).
            targets: Binary ground truth labels (0.0 or 1.0) of matching shape.
        
        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Probabilities
        probs = torch.sigmoid(logits)

        # Positive loss (y = 1)
        pos_probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        loss_pos = -targets * ((1.0 - pos_probs) ** self.gamma_pos) * torch.log(pos_probs)

        # Negative loss (y = 0) with probability margin shift
        neg_probs = torch.clamp(probs - self.clip_margin, min=0.0, max=1.0)
        neg_log = torch.log(torch.clamp(1.0 - neg_probs, min=self.eps, max=1.0))
        loss_neg = -(1.0 - targets) * self.weight_neg * (neg_probs ** self.gamma_neg) * neg_log

        # Total loss
        total_loss = loss_pos + loss_neg
        return total_loss.mean()
