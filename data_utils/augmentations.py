"""
Augmentation pipelines for the Deepfake Detection System.
Uses albumentations with ReplayCompose for temporal consistency (§8).

Critical rule (§8.1): Apply the exact SAME random augmentation to ALL frames
within a single sequence. Use ReplayCompose to achieve this.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations():
    """
    Training augmentations (§8.2).
    Applied in the exact order specified with exact parameters.
    Uses ReplayCompose so we can replay on subsequent frames.
    """
    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.3,
        ),
        A.ImageCompression(
            quality_lower=60,
            quality_upper=100,
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.2,
        ),
        A.ColorJitter(
            hue=10 / 360,           # albumentations uses 0-0.5 range
            saturation=20 / 255,
            brightness=0,            # handled by RandomBrightnessContrast
            contrast=0,
            p=0.3,
        ),
        A.CoarseDropout(
            max_holes=4,
            max_height=24,
            max_width=24,
            fill_value=0,
            p=0.2,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=0,           # cv2.BORDER_REFLECT
            p=0.3,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_validation_augmentations():
    """
    Validation / Test augmentations (§8.3).
    Only normalize + convert to tensor. No random transforms during evaluation.
    """
    return A.ReplayCompose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_fft_augmentations():
    """
    FFT-stream augmentations.
    Only normalize with FFT-specific stats (§5.2). No spatial augmentations
    on FFT maps as they would destroy frequency information.
    """
    return A.ReplayCompose([
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        ToTensorV2(),
    ])
