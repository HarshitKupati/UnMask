"""
Loss functions for the Deepfake Detection System.
Implements Focal Loss (§7.1), Label Smoothing (§7.2), and Consistency Loss (§7.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.

    Parameters (from spec §7.1):
        gamma: 2.0  — focusing parameter, down-weights easy examples
        alpha: 0.75 — weight for the positive (fake) class
        label_smoothing: 0.1 — fake targets 1.0→0.9, real targets 0.0→0.1

    Implementation note: Applied to raw logits using binary_cross_entropy_with_logits
    internally, NOT binary_cross_entropy.
    """

    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, 1) or (B,) — raw model output (no sigmoid)
            targets: (B, 1) or (B,) — binary labels (0=real, 1=fake)
        Returns:
            Scalar focal loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing (§7.2)
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal modulation
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ConsistencyLoss(nn.Module):
    """
    Consistency Regularization Loss (§7.3).
    Penalizes the model for producing very different confidence scores
    for augmented vs non-augmented versions of the same input.

    Compute: MSELoss(sigmoid(logits_original), sigmoid(logits_augmented).detach())
    Weighted at 0.1× the focal loss.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits_original, logits_augmented):
        return self.mse(
            torch.sigmoid(logits_original),
            torch.sigmoid(logits_augmented).detach()
        )
