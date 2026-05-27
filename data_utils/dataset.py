"""
DeepfakeDataset — PyTorch Dataset class for the Deepfake Detection System.
Reference: Specification Document §5

Handles:
  - Loading sequences of face crops per video
  - Padding short sequences / truncating long sequences to fixed length T
  - On-the-fly FFT magnitude map computation
  - Applying augmentation consistently across all frames (ReplayCompose)
  - Returning both RGB and FFT tensors with masks

Return format (§5.3):
    {
        'rgb':      FloatTensor of shape (T, 3, 224, 224),
        'fft':      FloatTensor of shape (T, 3, 224, 224),
        'label':    FloatTensor of shape (1,),
        'video_id': str,
        'mask':     BoolTensor of shape (T,)   # True=valid, False=padding
    }
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmentations import get_fft_augmentations


class DeepfakeDataset(Dataset):
    """
    Unified dataset for both image-folder and video-sequence data.

    Args:
        data_list: list of dicts with keys:
            - 'video_id': str
            - 'frames_dir': str  — path to directory containing face crop PNGs
            - 'label': int (0=real, 1=fake)
        sequence_length: int (T=20)
        transform: ReplayCompose for RGB augmentation
        fft_dir: str or None — if provided, load pre-computed FFT maps from disk.
                                Otherwise, compute FFT on-the-fly.
    """

    def __init__(self, data_list, sequence_length=20, transform=None, fft_dir=None):
        self.data_list = data_list
        self.T = sequence_length
        self.transform = transform
        self.fft_dir = fft_dir
        self.fft_transform = get_fft_augmentations()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        video_id = entry['video_id']
        label = entry['label']
        frames_dir = entry['frames_dir']

        # ── Load frame paths, sorted by index ────────────────────────
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if not frame_paths:
            frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))

        num_frames = len(frame_paths)

        # ── Sample / pad to fixed sequence length T (§5.2) ───────────
        if num_frames == 0:
            # Edge case: no frames — return all-zero padded sequence
            indices = []
        elif num_frames >= self.T:
            # Truncation: uniform subsample to exactly T frames
            indices = np.linspace(0, num_frames - 1, self.T, dtype=int).tolist()
        else:
            # Use all frames, will pad below
            indices = list(range(num_frames))

        # ── Load and augment frames ──────────────────────────────────
        rgb_frames = []
        fft_frames = []
        replay_data = None

        for i, frame_idx in enumerate(indices):
            # Load RGB face crop
            img = cv2.imread(frame_paths[frame_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            # Apply augmentation with ReplayCompose (§8.1)
            if self.transform is not None:
                if i == 0:
                    # First frame: apply normally, capture replay data
                    augmented = self.transform(image=img)
                    replay_data = augmented['replay']
                    img_aug = augmented['image']
                else:
                    # Subsequent frames: replay exact same augmentation
                    augmented = A.ReplayCompose.replay(replay_data, image=img)
                    img_aug = augmented['image']
            else:
                # No transform — just normalize manually
                img_aug = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            rgb_frames.append(img_aug)

            # ── Compute FFT magnitude map (§4.2) ────────────────────
            if self.fft_dir is not None:
                # Load pre-computed FFT
                fft_path = os.path.join(
                    self.fft_dir,
                    video_id,
                    os.path.basename(frame_paths[frame_idx])
                )
                fft_img = cv2.imread(fft_path)
                if fft_img is not None:
                    fft_img = cv2.cvtColor(fft_img, cv2.COLOR_BGR2RGB)
                else:
                    fft_img = self._compute_fft(cv2.imread(frame_paths[frame_idx]))
            else:
                fft_img = self._compute_fft(
                    cv2.imread(frame_paths[frame_idx])
                )

            fft_img = cv2.resize(fft_img, (224, 224))
            fft_augmented = self.fft_transform(image=fft_img)
            fft_frames.append(fft_augmented['image'])

        # ── Padding (§5.2): repeat last valid frame ──────────────────
        valid_count = len(rgb_frames)
        if valid_count == 0:
            # All-zero fallback
            zero_rgb = torch.zeros(3, 224, 224)
            zero_fft = torch.zeros(3, 224, 224)
            rgb_frames = [zero_rgb] * self.T
            fft_frames = [zero_fft] * self.T
            valid_count = 0
        else:
            while len(rgb_frames) < self.T:
                rgb_frames.append(rgb_frames[-1].clone())
                fft_frames.append(fft_frames[-1].clone())

        # ── Build mask (§5.3) ────────────────────────────────────────
        mask = torch.zeros(self.T, dtype=torch.bool)
        mask[:valid_count] = True

        # ── Stack tensors ────────────────────────────────────────────
        rgb_tensor = torch.stack(rgb_frames[:self.T])   # (T, 3, 224, 224)
        fft_tensor = torch.stack(fft_frames[:self.T])   # (T, 3, 224, 224)

        return {
            'rgb':      rgb_tensor,
            'fft':      fft_tensor,
            'label':    torch.tensor([label], dtype=torch.float32),
            'video_id': video_id,
            'mask':     mask,
        }

    @staticmethod
    def _compute_fft(bgr_image):
        """
        Compute 2D FFT magnitude map from a BGR image (§4.2).
        Process:
          1. Convert to grayscale
          2. Compute 2D FFT
          3. Shift zero-frequency to center
          4. Log magnitude: log(|FFT| + 1e-8)
          5. Normalize to [0, 1]
          6. Replicate to 3 channels
        Returns:
          numpy array (H, W, 3) uint8
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-8)

        # Normalize to [0, 1]
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max - mag_min > 0:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude = np.zeros_like(magnitude)

        # Convert to uint8 and replicate to 3 channels
        magnitude_uint8 = (magnitude * 255).astype(np.uint8)
        fft_3ch = np.stack([magnitude_uint8] * 3, axis=-1)
        return fft_3ch
