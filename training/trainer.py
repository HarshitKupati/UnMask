"""
Trainer — Main training loop for the Deepfake Detection System.
Reference: Specification Document §10

Implements:
  - Backbone freezing protocol (§10.1)
  - Step-by-step training epoch with AMP (§10.2)
  - Gradient accumulation and clipping (§9.4, §9.5)
  - Mixup augmentation at sequence level (§8.4)
  - Validation epoch (§10.3)
  - Checkpoint saving (§10.4)
  - Early stopping (§10.5)
  - Memory cleanup (§14.3)
"""

import os
import gc
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.losses import FocalLoss, ConsistencyLoss
from training.metrics import compute_metrics
from config import CONFIG

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader,
                 config=None, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or CONFIG
        self.device = device

        # Loss functions (§7)
        self.focal_loss = FocalLoss(
            gamma=self.config['focal_gamma'],
            alpha=self.config['focal_alpha'],
            label_smoothing=self.config['label_smoothing'],
        )
        self.consistency_loss = ConsistencyLoss()

        # AMP (§9.6)
        self.scaler = GradScaler()

        # Training state
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.top_checkpoints = []   # list of (val_auc, path) sorted ascending

        # Directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        # Training log
        self.log_path = os.path.join(self.config['log_dir'], 'training_log.csv')
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write('epoch,train_loss,train_auc,val_loss,val_auc,val_f1,lr\n')

    # ─────────────────────────────────────────────────────────────────
    # Backbone Freezing Protocol  (§10.1)
    # ─────────────────────────────────────────────────────────────────
    def update_trainable_params(self, epoch):
        """
        Colab T4 Optimized Protocol:
        Epochs 1-4: Freeze all EfficientNet parameters. Only train CBAM, SE, LSTM, etc.
        Epoch 5+:   Partial unfreeze — only unfreeze the very last block of the CNN.
                    Never fully unfreeze to save GPU memory and compute time.
        """
        if epoch < self.config['unfreeze_epoch']:
            # Freeze all backbone
            for encoder_name in ['rgb_encoder', 'fft_encoder']:
                encoder = getattr(self.model, encoder_name)
                for param in encoder.parameters():
                    param.requires_grad = False
            logger.info(f"Epoch {epoch}: Backbone FROZEN (warmup phase)")

        else:
            # Partial unfreeze: only the last block of the EfficientNet
            for encoder_name in ['rgb_encoder', 'fft_encoder']:
                encoder = getattr(self.model, encoder_name)
                
                # First freeze everything in the encoder
                for param in encoder.parameters():
                    param.requires_grad = False
                    
                # Then unfreeze only the last block
                for param in encoder.blocks[-1].parameters():
                    param.requires_grad = True
                    
            logger.info(f"Epoch {epoch}: Backbone PARTIALLY unfrozen (last block only)")

    # ─────────────────────────────────────────────────────────────────
    # Mixup  (§8.4)
    # ─────────────────────────────────────────────────────────────────
    def apply_mixup(self, rgb, fft, labels):
        """
        Sequence-level mixup (§8.4).
        Sample lambda from Beta(0.4, 0.4).
        """
        alpha = self.config['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        batch_size = rgb.size(0)
        index = torch.randperm(batch_size, device=rgb.device)

        mixed_rgb = lam * rgb + (1 - lam) * rgb[index]
        mixed_fft = lam * fft + (1 - lam) * fft[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_rgb, mixed_fft, mixed_labels

    # ─────────────────────────────────────────────────────────────────
    # Training Epoch  (§10.2)
    # ─────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch):
        """Execute one training epoch following spec §10.2 step by step."""
        self.model.train()
        self.update_trainable_params(epoch)

        accum_steps = self.config['gradient_accumulation_steps']
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            rgb = batch['rgb'].to(self.device)       # (B, T, 3, 224, 224)
            fft = batch['fft'].to(self.device)
            labels = batch['label'].to(self.device)   # (B, 1)
            mask = batch['mask'].to(self.device)      # (B, T)

            # Mixup (§8.4) — apply with probability, not in last 5 epochs
            use_mixup = (
                random.random() < self.config['mixup_probability']
                and epoch < self.config['num_epochs'] - 5
            )
            if use_mixup:
                rgb, fft, labels = self.apply_mixup(rgb, fft, labels)

            # Forward pass under autocast (§9.6)
            with autocast():
                logits = self.model(rgb, fft, mask=mask)  # (B, 1)
                loss = self.focal_loss(logits, labels)

            # Divide by accumulation steps (§9.4)
            loss = loss / accum_steps

            # Backward (§10.2g)
            self.scaler.scale(loss).backward()

            # Accumulation boundary (§10.2h)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            all_logits.extend(logits.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
            num_batches += 1

            pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})

        # Epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_logits)
        )

        return avg_loss, metrics

    # ─────────────────────────────────────────────────────────────────
    # Validation Epoch  (§10.3)
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Validation with full-set metric computation (§10.3)."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            fft = batch['fft'].to(self.device)
            labels = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)

            with autocast():
                logits = self.model(rgb, fft, mask=mask)
                loss = self.focal_loss(logits, labels)

            total_loss += loss.item()
            # Store as CPU tensors (§14.3)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        # Compute metrics on full validation set (§10.3 step 5)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_logits)
        )

        return avg_loss, metrics

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint Management  (§10.4)
    # ─────────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch, val_auc):
        """Save checkpoint, keeping only top-K (§10.4)."""
        ckpt_path = os.path.join(
            self.config['checkpoint_dir'],
            f'epoch_{epoch:02d}_auc_{val_auc:.4f}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'val_auc': val_auc,
            'config': self.config,
        }, ckpt_path)

        self.top_checkpoints.append((val_auc, ckpt_path))
        self.top_checkpoints.sort(key=lambda x: x[0])

        # Keep only top-K
        while len(self.top_checkpoints) > self.config['save_top_k']:
            _, remove_path = self.top_checkpoints.pop(0)
            if os.path.exists(remove_path):
                os.remove(remove_path)

        # Also save as best_model.pth
        if val_auc >= self.best_val_auc:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'val_auc': val_auc,
                'config': self.config,
            }, best_path)

        logger.info(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, checkpoint_path):
        """Resume from checkpoint on Colab reconnect (§10.4)."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.scaler.load_state_dict(ckpt['scaler_state'])
        self.best_val_auc = ckpt.get('val_auc', 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']} (val_auc={self.best_val_auc:.4f})")
        return ckpt['epoch']

    # ─────────────────────────────────────────────────────────────────
    # Full Training Loop
    # ─────────────────────────────────────────────────────────────────
    def fit(self, start_epoch=0):
        """
        Full training loop (§10 + §17).

        Follows the training phases:
          Epoch 1-3:  Warmup + frozen backbone
          Epoch 4:    Transition
          Epoch 5-9:  Partial unfreeze
          Epoch 10+:  Full unfreeze + cosine annealing
          Epoch 35:   Disable mixup
        """
        num_epochs = self.config['num_epochs']
        logger.info(f"Starting training: {num_epochs} epochs, device={self.device}")
        eff_batch = self.config['batch_size'] * self.config['gradient_accumulation_steps']
        logger.info(f"Effective batch size: {eff_batch}")

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            # ── Train ────────────────────────────────────────────────
            train_loss, train_metrics = self.train_epoch(epoch)

            # Step scheduler (§10.2 step 5)
            self.scheduler.step()

            # Memory cleanup (§14.3)
            torch.cuda.empty_cache()
            gc.collect()

            # ── Validate ─────────────────────────────────────────────
            val_loss, val_metrics = self.validate_epoch(epoch)

            # ── Logging (§10.2 step 6) ───────────────────────────────
            current_lr = self.optimizer.param_groups[-1]['lr']
            elapsed = time.time() - epoch_start

            log_msg = (
                f"Epoch {epoch}/{num_epochs-1} "
                f"| train_loss={train_loss:.4f} train_auc={train_metrics['auc']:.4f} "
                f"| val_loss={val_loss:.4f} val_auc={val_metrics['auc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"| lr={current_lr:.2e} | time={elapsed:.1f}s"
            )
            logger.info(log_msg)
            print(log_msg)

            # Append to CSV log
            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_metrics['auc']:.6f},"
                        f"{val_loss:.6f},{val_metrics['auc']:.6f},"
                        f"{val_metrics['f1']:.6f},{current_lr:.8e}\n")

            # ── Checkpoint (§10.4) ───────────────────────────────────
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics['auc'])
            else:
                self.patience_counter += 1

            # ── Early stopping (§10.5) ───────────────────────────────
            if self.patience_counter >= self.config['patience']:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_auc={self.best_val_auc:.4f}"
                )
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Training complete. Best val_auc={self.best_val_auc:.4f}")
        return self.best_val_auc
