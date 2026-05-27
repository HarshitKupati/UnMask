"""
Central configuration file for the Deepfake Detection System.
All hyperparameters are defined here — never hardcode values elsewhere.
Reference: Specification Document §9.1
"""

import os

# Auto-detect project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    'sequence_length':          10,
    'image_size':               224,
    'batch_size':               8,
    'num_workers':              2,

    # ── Model ─────────────────────────────────────────────────────────
    'efficientnet_variant':     'efficientnet_b0',
    'lstm_hidden_size':         512,
    'lstm_num_layers':          2,
    'lstm_dropout':             0.4,
    'drop_path_rate':           0.3,
    'classifier_dropout_1':     0.5,
    'classifier_dropout_2':     0.3,

    # ── Training ──────────────────────────────────────────────────────
    'num_epochs':               40,
    'warmup_epochs':            3,
    'gradient_clip_norm':       1.0,
    'gradient_accumulation_steps': 4,
    'mixup_alpha':              0.4,
    'mixup_probability':        0.3,
    'label_smoothing':          0.1,

    # ── Loss ──────────────────────────────────────────────────────────
    'focal_gamma':              2.0,
    'focal_alpha':              0.75,
    'consistency_weight':       0.1,

    # ── Optimizer ─────────────────────────────────────────────────────
    'weight_decay':             1e-2,
    'lr_backbone':              1e-5,
    'lr_lstm':                  1e-4,
    'lr_head':                  3e-4,
    'lr_min':                   1e-7,

    # ── Scheduler ─────────────────────────────────────────────────────
    'T_0':                      10,
    'T_mult':                   2,

    # ── Regularization ────────────────────────────────────────────────
    'unfreeze_epoch':           5,
    'unfreeze_partial_epoch':   5,   # same as above per spec
    'full_unfreeze_epoch':      10,

    # ── Early stopping ────────────────────────────────────────────────
    'patience':                 7,
    'monitor_metric':           'val_auc',

    # ── Checkpointing ─────────────────────────────────────────────────
    'checkpoint_dir':           os.path.join(PROJECT_DIR, 'checkpoints'),
    'save_top_k':               3,

    # ── XAI ───────────────────────────────────────────────────────────
    'gradcam_layer':            'rgb_encoder.blocks.6',

    # ── Logging ───────────────────────────────────────────────────────
    'use_wandb':                False,   # set True when ready
    'project_name':             'deepfake-detection',
    'log_dir':                  os.path.join(PROJECT_DIR, 'logs'),

    # ── Paths ─────────────────────────────────────────────────────────
    'project_dir':              PROJECT_DIR,
    'data_dir':                 os.path.join(PROJECT_DIR, 'data'),
    'faces_dir':                os.path.join(PROJECT_DIR, 'data', 'faces'),
    'faces_fft_dir':            os.path.join(PROJECT_DIR, 'data', 'faces_fft'),
    'splits_dir':               os.path.join(PROJECT_DIR, 'data', 'splits'),
    'results_dir':              os.path.join(PROJECT_DIR, 'results'),
    'xai_samples_dir':          os.path.join(PROJECT_DIR, 'xai_samples'),

    # ── Face Extraction ───────────────────────────────────────────────
    'face_size':                224,
    'face_margin':              20,
    'face_confidence_threshold': 0.90,
    'min_face_size':            80,
    'frames_per_video':         30,
    'blur_threshold':           100,

    # ── Inference ─────────────────────────────────────────────────────
    'tta_enabled':              True,
    'temperature':              1.0,      # calibrated later on val set
    'calibrated_threshold':     0.5,      # calibrated later on val set

    # ── Normalization ─────────────────────────────────────────────────
    'rgb_mean':                 (0.485, 0.456, 0.406),
    'rgb_std':                  (0.229, 0.224, 0.225),
    'fft_mean':                 (0.5, 0.5, 0.5),
    'fft_std':                  (0.5, 0.5, 0.5),
}
