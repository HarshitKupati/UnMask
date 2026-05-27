"""
GradCAM and Integrated Gradients for the Deepfake Detection System.
Reference: Specification Document §11.2, §11.3

GradCAM implementation:
  - Uses custom hooks (NOT captum's GradCAM wrapper — §11.2)
  - Hooks are already registered in DeepfakeDetector.__init__
  - Accesses model.rgb_activations, model.rgb_gradients, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMExplainer:
    """
    Custom GradCAM for the dual-stream architecture (§11.2).
    
    Steps:
      1. Forward pass populates model.rgb_activations (B, 1792, 7, 7)
      2. Backward hook captures model.rgb_gradients (B, 1792, 7, 7)
      3. weights = global_avg_pool(gradients) → (B, 1792, 1, 1)
      4. cam = relu(sum(weights × activations, dim=C)) → (B, 7, 7)
      5. Upsample to 224×224 with bilinear interpolation
      6. Normalize to [0, 1]
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    @torch.enable_grad()
    def compute_gradcam(self, rgb, fft, mask=None, stream='rgb'):
        """
        Compute GradCAM heatmap.

        Args:
            rgb: (B, T, 3, 224, 224) or (1, T, 3, 224, 224)
            fft: (B, T, 3, 224, 224)
            mask: (B, T) bool
            stream: 'rgb' or 'fft'

        Returns:
            cam: numpy array (B*T, 224, 224) normalized [0, 1]
        """
        self.model.eval()
        rgb = rgb.to(self.device).requires_grad_(True)
        fft = fft.to(self.device).requires_grad_(True)

        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass (populates activations via hooks)
        logits = self.model(rgb, fft, mask=mask)

        # Backward to populate gradients (§11.2 step 2)
        self.model.zero_grad()
        logits.sum().backward()

        # Select stream
        if stream == 'rgb':
            activations = self.model.rgb_activations  # (B*T, 1792, 7, 7)
            gradients = self.model.rgb_gradients       # (B*T, 1792, 7, 7)
        else:
            activations = self.model.fft_activations
            gradients = self.model.fft_gradients

        if activations is None or gradients is None:
            return None

        # Step 3: Global average pool of gradients → (B*T, C, 1, 1)
        weights = gradients.mean(dim=[2, 3], keepdim=True)

        # Step 4: Weighted combination → (B*T, 7, 7)
        cam = F.relu((weights * activations).sum(dim=1))

        # Step 5: Upsample to 224×224
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

        # Step 6: Normalize per sample
        cam = cam.detach().cpu().numpy()
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 0:
                cam[i] = (c - c_min) / (c_max - c_min)
            else:
                cam[i] = np.zeros_like(c)

        return cam

    def compute_per_frame_gradcam(self, rgb, fft, mask=None):
        """
        Compute GradCAM per frame independently (§11.2).
        Returns heatmaps for each of the T frames plus an aggregate.

        Args:
            rgb: (1, T, 3, 224, 224)
            fft: (1, T, 3, 224, 224)

        Returns:
            per_frame: list of T numpy arrays, each (224, 224)
            aggregate: numpy array (224, 224) — max across frames
        """
        cam = self.compute_gradcam(rgb, fft, mask, stream='rgb')
        if cam is None:
            T = rgb.shape[1]
            return [np.zeros((224, 224))] * T, np.zeros((224, 224))

        T = rgb.shape[1]
        per_frame = [cam[i] for i in range(min(len(cam), T))]

        # Aggregate: max across frames (§11.2)
        aggregate = np.max(np.stack(per_frame), axis=0)

        return per_frame, aggregate


class IntegratedGradientsExplainer:
    """
    Integrated Gradients using captum (§11.3).

    Configuration:
      - Target: output logit
      - Baseline: black image (zeros)
      - Steps: 50
      - Return convergence delta: True (delta should be < 0.05)
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    @torch.enable_grad()
    def compute_ig(self, rgb, fft, mask=None, n_steps=50):
        """
        Compute Integrated Gradients on the RGB stream.

        Args:
            rgb: (1, T, 3, 224, 224)
            fft: (1, T, 3, 224, 224)
            mask: (1, T)
            n_steps: number of interpolation steps

        Returns:
            attribution: numpy array (T, 3, 224, 224)
            convergence_delta: float
        """
        try:
            from captum.attr import IntegratedGradients

            self.model.eval()
            rgb = rgb.to(self.device)
            fft = fft.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            # Wrapper function for captum
            def forward_fn(rgb_input):
                B_in = rgb_input.shape[0]
                fft_in = fft.expand(B_in, -1, -1, -1, -1).contiguous()
                mask_in = mask.expand(B_in, -1).contiguous() if mask is not None else None
                logits = self.model(rgb_input, fft_in, mask=mask_in)
                return logits.squeeze(1)

            ig = IntegratedGradients(forward_fn)
            baseline = torch.zeros_like(rgb)

            attributions, delta = ig.attribute(
                rgb,
                baselines=baseline,
                n_steps=n_steps,
                internal_batch_size=2,
                return_convergence_delta=True,
            )

            attr_np = attributions.squeeze(0).detach().cpu().numpy()  # (T, 3, 224, 224)

            return attr_np, delta.item()

        except ImportError:
            print("captum not installed. Skipping Integrated Gradients.")
            T = rgb.shape[1]
            return np.zeros((T, 3, 224, 224)), float('inf')
