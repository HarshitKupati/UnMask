"""
Optimizer and LR Scheduler for the Deepfake Detection System.
Reference: Specification Document §9.2, §9.3

Implements:
  - AdamW with differential learning rates per parameter group (§9.2)
  - Two-phase scheduler: LinearWarmup + CosineAnnealingWarmRestarts (§9.3)
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
)


def build_optimizer_and_scheduler(model, config):
    """
    Build AdamW with differential learning rates and two-phase scheduler.

    Parameter groups (§9.2):
      - EfficientNet blocks 0-3: lr_backbone × 0.1 = 1e-6
      - EfficientNet blocks 4-6: lr_backbone × 0.5 = 5e-6
      - EfficientNet blocks 7-8: lr_backbone = 1e-5
      - CBAM modules:           lr_lstm = 1e-4
      - SE fusion module:       lr_lstm = 1e-4
      - LSTM all layers:        lr_lstm = 1e-4
      - Temporal attention:     lr_lstm = 1e-4
      - Classifier head:        lr_head = 3e-4

    Weight decay (§9.2): 1e-2, excluded from BatchNorm and bias params.
    """
    lr_backbone = config['lr_backbone']
    lr_lstm = config['lr_lstm']
    lr_head = config['lr_head']
    weight_decay = config['weight_decay']

    # Helper to check if param should skip weight decay
    def no_decay(name):
        return 'bias' in name or 'bn' in name or 'norm' in name

    param_groups = []

    # ── EfficientNet-B4 blocks (both RGB and FFT encoders) ──────────
    for encoder_name in ['rgb_encoder', 'fft_encoder']:
        encoder = getattr(model, encoder_name)

        # Blocks 0-3: early layers (lr_backbone × 0.1)
        for block_idx in range(min(4, len(encoder.blocks))):
            for name, param in encoder.blocks[block_idx].named_parameters():
                lr = lr_backbone * 0.1
                wd = 0.0 if no_decay(name) else weight_decay
                param_groups.append({
                    'params': [param], 'lr': lr, 'weight_decay': wd,
                    'name': f'{encoder_name}.blocks.{block_idx}.{name}'
                })

        # Blocks 4-6: middle layers (lr_backbone × 0.5)
        for block_idx in range(4, min(7, len(encoder.blocks))):
            for name, param in encoder.blocks[block_idx].named_parameters():
                lr = lr_backbone * 0.5
                wd = 0.0 if no_decay(name) else weight_decay
                param_groups.append({
                    'params': [param], 'lr': lr, 'weight_decay': wd,
                    'name': f'{encoder_name}.blocks.{block_idx}.{name}'
                })

        # Blocks 7+: late layers (lr_backbone)
        for block_idx in range(7, len(encoder.blocks)):
            for name, param in encoder.blocks[block_idx].named_parameters():
                lr = lr_backbone
                wd = 0.0 if no_decay(name) else weight_decay
                param_groups.append({
                    'params': [param], 'lr': lr, 'weight_decay': wd,
                    'name': f'{encoder_name}.blocks.{block_idx}.{name}'
                })

        # Stem and other encoder parameters (same as early blocks)
        for name, param in encoder.named_parameters():
            if 'blocks' not in name:
                wd = 0.0 if no_decay(name) else weight_decay
                param_groups.append({
                    'params': [param], 'lr': lr_backbone * 0.1, 'weight_decay': wd,
                    'name': f'{encoder_name}.{name}'
                })

    # ── CBAM attention modules ──────────────────────────────────────
    for module_name in ['rgb_cbam', 'fft_cbam']:
        for name, param in getattr(model, module_name).named_parameters():
            wd = 0.0 if no_decay(name) else weight_decay
            param_groups.append({
                'params': [param], 'lr': lr_lstm, 'weight_decay': wd,
                'name': f'{module_name}.{name}'
            })

    # ── SE-Gate fusion ──────────────────────────────────────────────
    for name, param in model.se_gate.named_parameters():
        wd = 0.0 if no_decay(name) else weight_decay
        param_groups.append({
            'params': [param], 'lr': lr_lstm, 'weight_decay': wd,
            'name': f'se_gate.{name}'
        })

    # ── LSTM ────────────────────────────────────────────────────────
    for name, param in model.lstm.named_parameters():
        wd = 0.0 if no_decay(name) else weight_decay
        param_groups.append({
            'params': [param], 'lr': lr_lstm, 'weight_decay': wd,
            'name': f'lstm.{name}'
        })

    # ── Temporal Attention ──────────────────────────────────────────
    for name, param in model.temporal_attention.named_parameters():
        wd = 0.0 if no_decay(name) else weight_decay
        param_groups.append({
            'params': [param], 'lr': lr_lstm, 'weight_decay': wd,
            'name': f'temporal_attention.{name}'
        })

    # ── Classifier head ────────────────────────────────────────────
    for name, param in model.classifier.named_parameters():
        wd = 0.0 if no_decay(name) else weight_decay
        param_groups.append({
            'params': [param], 'lr': lr_head, 'weight_decay': wd,
            'name': f'classifier.{name}'
        })

    optimizer = AdamW(param_groups)

    # ── Two-phase LR scheduler (§9.3) ──────────────────────────────
    warmup_epochs = config['warmup_epochs']  # 3

    # Phase 1: Linear warmup from 0 to target LR
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )

    # Phase 2: CosineAnnealingWarmRestarts
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],          # 10 epochs
        T_mult=config['T_mult'],    # 2 (period doubles: 10, 20, 40)
        eta_min=config['lr_min'],   # 1e-7
    )

    # Combine with SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler
