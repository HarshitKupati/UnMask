"""
DeepfakeDetector — Two-stream EfficientNet + BiLSTM with XAI hooks.
Reference: Specification Document §6

Architecture overview:
  1. Dual EfficientNet encoders (RGB + FFT) with CBAM
  2. SE-Gate feature fusion  (3584 → 1024 per frame)
  3. BiLSTM temporal processing  (1024 → 1024 per timestep)
  4. Temporal Attention aggregation  (B, T, 1024) → (B, 3072)
  5. Classification head  (3072 → 1)

XAI hooks are registered during __init__ on:
  - self.rgb_encoder.blocks[-1]   (forward + backward for GradCAM)
  - self.fft_encoder.blocks[-1]   (forward + backward for GradCAM)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import timm

from .attention import CBAM, SEGate, TemporalAttentionModule


class DeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── 1. Dual EfficientNet spatial encoders ─────────────────────────
        # Separate pretrained instances (do NOT share weights — §6.2)
        self.rgb_encoder = timm.create_model(
            config['efficientnet_variant'],
            pretrained=True,
            num_classes=0,
            global_pool='',
            drop_path_rate=config['drop_path_rate'],
        )
        self.fft_encoder = timm.create_model(
            config['efficientnet_variant'],
            pretrained=True,
            num_classes=0,
            global_pool='',
            drop_path_rate=config['drop_path_rate'],
        )

        # Feature dimension: dynamically get the encoder's output channels
        feat_dim = self.rgb_encoder.num_features

        # ── 2. CBAM after last block, before pooling (§6.2) ──────────
        self.rgb_cbam = CBAM(feat_dim)
        self.fft_cbam = CBAM(feat_dim)

        # Adaptive average pooling: (B, 1792, 7, 7) → (B, 1792)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── 3. SE-Gate feature fusion (§6.3) ──────────────────────────
        # Concat RGB(1792) + FFT(1792) = 3584 → SE gate → project to 1024
        self.se_gate = SEGate(
            in_features=feat_dim * 2,
            bottleneck=224,
            out_features=1024,
        )

        # ── 4. BiLSTM temporal processor (§6.5) ──────────────────────
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=config['lstm_hidden_size'],    # 512
            num_layers=config['lstm_num_layers'],       # 2
            bidirectional=True,
            batch_first=True,
            dropout=config['lstm_dropout'],             # 0.4 (between layers only)
        )
        # Output: (B, T, 512*2) = (B, T, 1024)

        # Apply proper weight initialization (§6.5)
        self._init_lstm_weights()

        # ── 5. Temporal Attention (§6.6) ──────────────────────────────
        lstm_out_dim = config['lstm_hidden_size'] * 2   # 1024
        self.temporal_attention = TemporalAttentionModule(
            hidden_dim=lstm_out_dim,
            attn_dim=256,
        )
        # Output: (B, 3072) — concat of [attended, mean_pooled, max_pooled]

        # ── 6. Classification head (§6.7) ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(config['classifier_dropout_1']),  # 0.5
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config['classifier_dropout_2']),  # 0.3
            nn.Linear(256, 1),
            # No sigmoid — use BCEWithLogitsLoss / focal loss on raw logits
        )

        # ── 7. XAI hook storage (§6.8) ────────────────────────────────
        self.rgb_activations = None
        self.rgb_gradients = None
        self.fft_activations = None
        self.fft_gradients = None

        self._register_xai_hooks()

    # ─────────────────────────────────────────────────────────────────
    # Weight initialization
    # ─────────────────────────────────────────────────────────────────
    def _init_lstm_weights(self):
        """Explicit LSTM weight initialization (§6.5)."""
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                init.orthogonal_(param)           # recurrent weights
            elif 'weight_ih' in name:
                init.xavier_uniform_(param)       # input weights
            elif 'bias' in name:
                init.zeros_(param)                # biases

    # ─────────────────────────────────────────────────────────────────
    # XAI hooks — capture activations and gradients for GradCAM
    # ─────────────────────────────────────────────────────────────────
    def _register_xai_hooks(self):
        """Register forward + backward hooks on the last EfficientNet block (§6.8)."""

        def rgb_forward_hook(module, input, output):
            self.rgb_activations = output

        def rgb_backward_hook(module, grad_input, grad_output):
            if grad_output and len(grad_output) > 0:
                self.rgb_gradients = grad_output[0]
            else:
                self.rgb_gradients = None

        def fft_forward_hook(module, input, output):
            self.fft_activations = output

        def fft_backward_hook(module, grad_input, grad_output):
            if grad_output and len(grad_output) > 0:
                self.fft_gradients = grad_output[0]
            else:
                self.fft_gradients = None

        # Access last block of each encoder
        rgb_last_block = self.rgb_encoder.blocks[-1]
        fft_last_block = self.fft_encoder.blocks[-1]

        rgb_last_block.register_forward_hook(rgb_forward_hook)
        rgb_last_block.register_full_backward_hook(rgb_backward_hook)
        fft_last_block.register_forward_hook(fft_forward_hook)
        fft_last_block.register_full_backward_hook(fft_backward_hook)

    # ─────────────────────────────────────────────────────────────────
    # Forward pass  (§6.9)
    # ─────────────────────────────────────────────────────────────────
    def forward(self, rgb, fft, mask=None, return_attention=False):
        """
        Args:
            rgb:  (B, T, 3, 224, 224)
            fft:  (B, T, 3, 224, 224)
            mask: (B, T) bool — True for valid frames, False for padding
            return_attention: bool — if True, also return attention_weights (B, T)
        Returns:
            logits: (B, 1) — raw logits (no sigmoid)
            attention_weights: (B, T) — only if return_attention=True
        """
        B, T, C, H, W = rgb.shape

        # ── Collapse batch and time for efficient CNN processing ─────
        rgb_flat = rgb.view(B * T, C, H, W)
        fft_flat = fft.view(B * T, C, H, W)

        # ── Spatial encoding ─────────────────────────────────────────
        rgb_features = self.rgb_encoder(rgb_flat)       # (B*T, 1792, 7, 7)
        fft_features = self.fft_encoder(fft_flat)       # (B*T, 1792, 7, 7)

        # ── CBAM ─────────────────────────────────────────────────────
        rgb_features = self.rgb_cbam(rgb_features)
        fft_features = self.fft_cbam(fft_features)

        # ── Adaptive pooling → (B*T, 1792) ──────────────────────────
        rgb_vec = self.pool(rgb_features).flatten(1)
        fft_vec = self.pool(fft_features).flatten(1)

        # ── Feature fusion (§6.3) ────────────────────────────────────
        combined = torch.cat([rgb_vec, fft_vec], dim=1)  # (B*T, 3584)
        fused = self.se_gate(combined)                    # (B*T, 1024)

        # ── Restore temporal dimension (§6.4) ────────────────────────
        fused = fused.view(B, T, -1)                      # (B, T, 1024)

        # ── BiLSTM with packed sequences for masking (§6.4, §6.5) ───
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(
                fused, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(fused)
        # lstm_out: (B, T, 1024)

        # ── Temporal attention aggregation (§6.6) ────────────────────
        aggregated, attn_weights = self.temporal_attention(lstm_out, mask)
        # aggregated: (B, 3072)

        # ── Classification (§6.7) ────────────────────────────────────
        logits = self.classifier(aggregated)              # (B, 1)

        if return_attention:
            return logits, attn_weights
        return logits
