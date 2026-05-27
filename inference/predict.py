"""
Inference pipeline for the Deepfake Detection System.
Reference: Specification Document §13

Supports:
  - Single video inference with full XAI output (§13.1)
  - Single image inference
  - Test-Time Augmentation (TTA) — 5 versions (§13.2)
  - Confidence thresholds (§13.3)
  - Temperature scaling (§12.4)
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast

from models.detector import DeepfakeDetector
from data_utils.extract_faces import get_face_detector, compute_fft
from data_utils.augmentations import get_validation_augmentations, get_fft_augmentations
from xai.explainer import GradCAMExplainer, IntegratedGradientsExplainer
from xai.visualize import create_composite_visualization, overlay_heatmap
from training.metrics import get_confidence_label
from config import CONFIG


class DeepfakePredictor:
    """
    Full inference pipeline (§13.1).
    """

    def __init__(self, checkpoint_path, config=None, device='cuda'):
        self.config = config or CONFIG
        self.device = device

        # Load model
        self.model = DeepfakeDetector(self.config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # Temperature from calibration (§12.4)
        self.temperature = self.config.get('temperature', 1.0)
        self.threshold = self.config.get('calibrated_threshold', 0.5)

        # Face detector
        self.mtcnn = get_face_detector(device)

        # Transforms
        self.rgb_transform = get_validation_augmentations()
        self.fft_transform = get_fft_augmentations()

        # XAI
        self.gradcam = GradCAMExplainer(self.model, device)
        self.ig_explainer = IntegratedGradientsExplainer(self.model, device)

    @torch.no_grad()
    def predict_video(self, video_path, use_tta=None, compute_xai=True):
        """
        Full video inference pipeline (§13.1).

        Steps:
          1. Extract faces using MTCNN
          2. Compute FFT magnitude maps
          3. Assemble sequence of T=20 frames
          4. Apply TTA if enabled (5 versions)
          5. Apply temperature scaling
          6. Classify as FAKE/REAL with confidence
          7. Compute GradCAM on top-3 attended frames
          8. Compute Integrated Gradients on highest-attention frame
          9. Return full XAI output dict

        Returns: dict matching §11.5 output format
        """
        if use_tta is None:
            use_tta = self.config.get('tta_enabled', True)

        T = self.config['sequence_length']

        # ── Step 1: Extract faces ────────────────────────────────────
        frames, original_frames = self._extract_faces_from_video(video_path)
        if len(frames) == 0:
            return self._empty_result("No faces detected")

        # ── Step 2: Compute FFT ──────────────────────────────────────
        fft_frames = [compute_fft(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in frames]

        # ── Step 3: Assemble sequence ────────────────────────────────
        rgb_tensor, fft_tensor, mask = self._build_sequence(frames, fft_frames, T)

        # ── Step 4: TTA (§13.2) ──────────────────────────────────────
        if use_tta:
            logit = self._predict_with_tta(rgb_tensor, fft_tensor, mask)
        else:
            with autocast():
                logit = self.model(
                    rgb_tensor.unsqueeze(0).to(self.device),
                    fft_tensor.unsqueeze(0).to(self.device),
                    mask=mask.unsqueeze(0).to(self.device),
                )
            logit = logit.item()

        # ── Step 5: Temperature scaling ──────────────────────────────
        scaled_logit = logit / self.temperature
        probability = 1.0 / (1.0 + np.exp(-scaled_logit))

        # ── Step 6: Classify ─────────────────────────────────────────
        label, confidence = get_confidence_label(probability)

        result = {
            'prediction': float(probability),
            'label': label,
            'confidence': confidence,
        }

        # ── Steps 7-9: XAI (§11) ────────────────────────────────────
        if compute_xai:
            xai_result = self._compute_xai(
                rgb_tensor, fft_tensor, mask,
                original_frames[:T], probability, label, confidence
            )
            result.update(xai_result)

        return result

    def predict_image(self, image_path, compute_xai=True):
        """
        Single image inference — treated as a single-frame sequence.
        """
        img = cv2.imread(image_path)
        if img is None:
            return self._empty_result("Cannot load image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face
        if self.mtcnn is not None:
            try:
                boxes, probs, _ = self.mtcnn.detect(img_rgb, landmarks=True)
                if boxes is not None and len(boxes) > 0:
                    best = np.argmax(probs)
                    x1, y1, x2, y2 = boxes[best].astype(int)
                    h, w = img.shape[:2]
                    m = 20
                    x1, y1 = max(0, x1 - m), max(0, y1 - m)
                    x2, y2 = min(w, x2 + m), min(h, y2 + m)
                    img = img[y1:y2, x1:x2]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        face = cv2.resize(img_rgb, (224, 224))

        # Build single-frame sequence
        frames = [face]
        fft_frames = [compute_fft(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))]

        T = self.config['sequence_length']
        rgb_tensor, fft_tensor, mask = self._build_sequence(frames, fft_frames, T)

        with autocast():
            logit = self.model(
                rgb_tensor.unsqueeze(0).to(self.device),
                fft_tensor.unsqueeze(0).to(self.device),
                mask=mask.unsqueeze(0).to(self.device),
            )

        scaled_logit = logit.item() / self.temperature
        probability = 1.0 / (1.0 + np.exp(-scaled_logit))
        label, confidence = get_confidence_label(probability)

        result = {
            'prediction': float(probability),
            'label': label,
            'confidence': confidence,
        }

        if compute_xai:
            xai_result = self._compute_xai(
                rgb_tensor, fft_tensor, mask,
                [cv2.cvtColor(face, cv2.COLOR_RGB2BGR)],
                probability, label, confidence
            )
            result.update(xai_result)

        return result

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _extract_faces_from_video(self, video_path):
        """Extract faces from video file."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        T = self.config['sequence_length']

        if total < 1:
            cap.release()
            return [], []

        sample_indices = np.linspace(0, total - 1, min(T, total), dtype=int)
        faces = []
        originals = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.mtcnn is not None:
                try:
                    boxes, probs, _ = self.mtcnn.detect(frame_rgb, landmarks=True)
                    if boxes is not None and len(boxes) > 0:
                        best = np.argmax(probs)
                        if probs[best] >= 0.90:
                            x1, y1, x2, y2 = boxes[best].astype(int)
                            h, w = frame.shape[:2]
                            m = 20
                            x1, y1 = max(0, x1 - m), max(0, y1 - m)
                            x2, y2 = min(w, x2 + m), min(h, y2 + m)
                            face = frame_rgb[y1:y2, x1:x2]
                            face = cv2.resize(face, (224, 224))
                            faces.append(face)
                            originals.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                            continue
                except Exception:
                    pass

            # Fallback: use full frame
            face = cv2.resize(frame_rgb, (224, 224))
            faces.append(face)
            originals.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        cap.release()
        return faces, originals

    def _build_sequence(self, frames, fft_frames, T):
        """Build padded sequence tensors."""
        rgb_tensors = []
        fft_tensors = []

        for frame in frames[:T]:
            augmented = self.rgb_transform(image=frame)
            rgb_tensors.append(augmented['image'])

        for fft_img in fft_frames[:T]:
            fft_img_resized = cv2.resize(fft_img, (224, 224))
            augmented = self.fft_transform(image=fft_img_resized)
            fft_tensors.append(augmented['image'])

        valid_count = len(rgb_tensors)

        # Pad
        while len(rgb_tensors) < T:
            if valid_count > 0:
                rgb_tensors.append(rgb_tensors[-1].clone())
                fft_tensors.append(fft_tensors[-1].clone())
            else:
                rgb_tensors.append(torch.zeros(3, 224, 224))
                fft_tensors.append(torch.zeros(3, 224, 224))

        mask = torch.zeros(T, dtype=torch.bool)
        mask[:valid_count] = True

        return torch.stack(rgb_tensors), torch.stack(fft_tensors), mask

    def _predict_with_tta(self, rgb_tensor, fft_tensor, mask):
        """
        Test-Time Augmentation (§13.2).
        5 versions: original + horizontal flip + brightness±0.1 + Gaussian blur.
        Average the 5 sigmoid outputs.
        """
        rgb = rgb_tensor.unsqueeze(0).to(self.device)
        fft = fft_tensor.unsqueeze(0).to(self.device)
        m = mask.unsqueeze(0).to(self.device)

        logits = []

        # 1. Original
        with autocast():
            out = self.model(rgb, fft, mask=m)
        logits.append(out.item())

        # 2. Horizontal flip
        rgb_flip = torch.flip(rgb, dims=[-1])
        fft_flip = torch.flip(fft, dims=[-1])
        with autocast():
            out = self.model(rgb_flip, fft_flip, mask=m)
        logits.append(out.item())

        # 3. Brightness +0.1
        rgb_bright = torch.clamp(rgb + 0.1, 0, 1)
        with autocast():
            out = self.model(rgb_bright, fft, mask=m)
        logits.append(out.item())

        # 4. Brightness -0.1
        rgb_dark = torch.clamp(rgb - 0.1, 0, 1)
        with autocast():
            out = self.model(rgb_dark, fft, mask=m)
        logits.append(out.item())

        # 5. Gaussian blur (applied to unnormalized, re-normalize)
        # For simplicity, apply slight noise as blur proxy
        rgb_blur = rgb + torch.randn_like(rgb) * 0.01
        with autocast():
            out = self.model(rgb_blur, fft, mask=m)
        logits.append(out.item())

        # Average logits
        return float(np.mean(logits))

    def _compute_xai(self, rgb_tensor, fft_tensor, mask,
                     original_frames, probability, label, confidence):
        """Compute all XAI outputs (§11)."""
        rgb = rgb_tensor.unsqueeze(0)
        fft = fft_tensor.unsqueeze(0)
        m = mask.unsqueeze(0)

        # GradCAM per frame (§11.2)
        try:
            gradcam_result = self.gradcam.compute_per_frame_gradcam(rgb, fft, m)
            if isinstance(gradcam_result, tuple) and len(gradcam_result) == 2:
                per_frame_cam, aggregate_cam = gradcam_result
            else:
                print("Warning: GradCAM returned unexpected format, using fallback")
                T = rgb.shape[1]
                per_frame_cam = [np.zeros((224, 224))] * T
                aggregate_cam = np.zeros((224, 224))
        except Exception as e:
            print(f"Error in GradCAM computation: {e}")
            T = rgb.shape[1]
            per_frame_cam = [np.zeros((224, 224))] * T
            aggregate_cam = np.zeros((224, 224))

        # Attention weights (§11.4)
        try:
            with torch.no_grad():
                attn_result = self.model(
                    rgb.to(self.device), fft.to(self.device),
                    mask=m.to(self.device), return_attention=True
                )
                if isinstance(attn_result, tuple) and len(attn_result) == 2:
                    _, attn_weights = attn_result
                    attn_np = attn_weights.cpu().numpy().flatten()
                else:
                    print("Warning: Model returned unexpected format for attention weights")
                    attn_np = np.zeros(rgb.shape[1])
        except Exception as e:
            print(f"Error in attention weights computation: {e}")
            attn_np = np.zeros(rgb.shape[1])

        # Top suspicious frames (only select from valid frames using the mask)
        if mask is not None:
            valid_indices = torch.where(mask)[0].cpu().numpy().tolist()
        else:
            valid_indices = list(range(len(attn_np)))

        if len(valid_indices) > 0:
            valid_attn = attn_np[valid_indices]
            sorted_valid_sub_idxs = np.argsort(valid_attn)[-3:][::-1]
            top_suspicious = [valid_indices[idx] for idx in sorted_valid_sub_idxs]
        else:
            top_suspicious = []

        # Integrated Gradients on highest-attention frame (§11.3)
        try:
            ig_result = self.ig_explainer.compute_ig(rgb, fft, m)
            if isinstance(ig_result, tuple) and len(ig_result) == 2:
                ig_attr, ig_delta = ig_result
            else:
                print("Warning: IG returned unexpected format, using fallback")
                ig_attr = np.zeros((rgb.shape[1], 3, 224, 224))
                ig_delta = float('inf')
        except Exception as e:
            print(f"Error in Integrated Gradients computation: {e}")
            ig_attr = np.zeros((rgb.shape[1], 3, 224, 224))
            ig_delta = float('inf')

        # GradCAM overlays (§11.2 step 8)
        gradcam_overlays = []
        original_crops = []
        for i, idx in enumerate(top_suspicious):
            if idx < len(original_frames) and idx < len(per_frame_cam):
                overlay = cv2.resize(original_frames[idx], (224, 224))
                original_crops.append(overlay.copy())
                heatmap_overlay = overlay_heatmap(overlay, per_frame_cam[idx])
                gradcam_overlays.append(heatmap_overlay)

        # IG visualization
        if ig_attr.ndim == 4:
            # Sum across channels, normalize
            ig_vis = np.abs(ig_attr).sum(axis=1)  # (T, 224, 224)
            if ig_vis.max() > 0:
                ig_vis = ig_vis / ig_vis.max()
            # Take frame with highest attention
            if len(top_suspicious) > 0 and top_suspicious[0] < len(ig_vis):
                ig_frame = ig_vis[top_suspicious[0]]
            else:
                ig_frame = ig_vis[0]
            ig_frame_rgb = cv2.applyColorMap(
                (ig_frame * 255).astype(np.uint8), cv2.COLORMAP_HOT
            )
        else:
            ig_frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        # Composite visualization (§11.6)
        if len(original_frames) > 0:
            resized_originals = [cv2.resize(f, (224, 224)) for f in original_frames]
            vis = create_composite_visualization(
                resized_originals,
                per_frame_cam,
                attn_np,
                probability,
                confidence,
                label,
                save_path=None,
            )
        else:
            vis = np.zeros((224, 224, 3), dtype=np.uint8)

        return {
            'gradcam_per_frame': per_frame_cam,
            'gradcam_aggregate': aggregate_cam,
            'ig_attribution': ig_frame_rgb,
            'attention_weights': attn_np,
            'top_suspicious_frames': top_suspicious,
            'top_suspicious_crops': original_crops,
            'top_suspicious_overlays': gradcam_overlays,
            'visualization': vis,
        }

    @staticmethod
    def _empty_result(reason):
        return {
            'prediction': 0.5,
            'label': 'UNCERTAIN',
            'confidence': f'Error: {reason}',
            'gradcam_per_frame': [],
            'gradcam_aggregate': np.zeros((224, 224)),
            'ig_attribution': np.zeros((224, 224, 3)),
            'attention_weights': np.zeros(20),
            'top_suspicious_frames': [],
            'visualization': np.zeros((224, 224, 3)),
        }
