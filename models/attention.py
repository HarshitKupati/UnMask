"""
Attention modules for the Deepfake Detection System.
Implements CBAM (§6.2), SE-Gate fusion (§6.3), and Temporal Attention (§6.6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module  (§6.2)
# Applied after the last EfficientNet block, before pooling.
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Channel attention: global avg pool + global max pool → shared MLP → sigmoid gate."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])                              # (B, C)
        max_pool = x.amax(dim=[2, 3])                              # (B, C)
        gate = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))  # (B, C)
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise avg + max pool → concat → conv 7×7 → sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)                     # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)                     # (B, 1, H, W)
        gate = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))  # (B, 1, H, W)
        return x * gate


class CBAM(nn.Module):
    """Channel + Spatial attention applied sequentially."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SEGate — Squeeze-and-Excitation gating for feature fusion  (§6.3)
# Operates on the concatenated RGB+FFT vector (B, 3584).
# ─────────────────────────────────────────────────────────────────────────────

class SEGate(nn.Module):
    """
    SE-style gating:  Linear(3584→224) → ReLU → Linear(224→3584) → Sigmoid → elementwise multiply.
    Then project down:  Linear(3584→1024) → LayerNorm → GELU.
    """

    def __init__(self, in_features=3584, bottleneck=224, out_features=1024):
        super().__init__()
        # SE gating
        self.gate = nn.Sequential(
            nn.Linear(in_features, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_features),
            nn.Sigmoid(),
        )
        # Projection
        self.project = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 3584)
        return self.project(x * self.gate(x))


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Attention Module  (§6.6)
# Aggregates T hidden states from the BiLSTM into a single fixed-size vector.
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttentionModule(nn.Module):
    """
    Architecture:
      Input: (B, T, 1024) LSTM output
      Linear(1024→256) → Tanh → Linear(256→1) → squeeze → mask padding with -inf → Softmax
      Weighted sum: sum(attention_weight × hidden_state) across time → (B, 1024)
      Concatenate with mean-pooled and max-pooled LSTM output → (B, 3072)
    """

    def __init__(self, hidden_dim=1024, attn_dim=256):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (B, T, 1024)
            mask:        (B, T) bool — True for valid frames, False for padding
        Returns:
            aggregated:  (B, 3072) — concat of [attended, mean_pooled, max_pooled]
            attn_weights:(B, T) — softmax attention weights
        """
        # Compute raw scores → (B, T, 1) → (B, T)
        scores = self.score_fn(lstm_output).squeeze(-1)

        # Mask padding positions with -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=1)                    # (B, T)

        # Weighted sum → (B, 1024)
        attended = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)

        # Mean and max pooling (masked)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()              # (B, T, 1)
            masked_output = lstm_output * mask_expanded
            lengths = mask_expanded.sum(dim=1).clamp(min=1)         # (B, 1)
            mean_pooled = masked_output.sum(dim=1) / lengths        # (B, 1024)
            # For max pool, fill padding with -inf
            masked_for_max = lstm_output.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            max_pooled = masked_for_max.max(dim=1).values           # (B, 1024)
        else:
            mean_pooled = lstm_output.mean(dim=1)
            max_pooled = lstm_output.max(dim=1).values

        # Concatenate → (B, 3072)
        aggregated = torch.cat([attended, mean_pooled, max_pooled], dim=1)
        return aggregated, attn_weights
