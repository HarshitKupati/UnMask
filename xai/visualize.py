"""
Visualization utilities for XAI outputs.
Reference: Specification Document §11.5, §11.6

Produces:
  - GradCAM heatmap overlays (JET colormap, alpha=0.4)
  - Temporal attention bar chart
  - Composite visualization combining all XAI outputs
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay GradCAM heatmap on original image (§11.2 step 8).

    Args:
        image:    numpy array (H, W, 3) BGR, uint8
        heatmap:  numpy array (H, W) float [0, 1]
        alpha:    overlay transparency
        colormap: OpenCV colormap

    Returns:
        overlay: numpy array (H, W, 3) BGR, uint8
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    if image.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def create_attention_bar_chart(attention_weights, top_k=3):
    """
    Create temporal attention weight bar chart (§11.4).

    Args:
        attention_weights: numpy array (T,)
        top_k: number of top frames to highlight

    Returns:
        chart_image: numpy array (H, W, 3) BGR
    """
    T = len(attention_weights)
    top_indices = np.argsort(attention_weights)[-top_k:]

    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    colors = ['#e74c3c' if i in top_indices else '#3498db' for i in range(T)]
    ax.bar(range(T), attention_weights, color=colors)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Temporal Attention Weights')
    ax.set_xticks(range(T))
    plt.tight_layout()

    # Convert to numpy
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    chart_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return chart_bgr


def create_composite_visualization(
    original_frames,
    gradcam_heatmaps,
    attention_weights,
    prediction,
    confidence,
    label,
    ig_attribution=None,
    top_k=3,
    save_path=None,
):
    """
    Composite visualization (§11.6).

    Layout:
      Row 1: Top-3 most attended frames (original)
      Row 2: GradCAM overlays on those 3 frames
      Row 3: Attention weight bar chart + prediction label + confidence score

    Args:
        original_frames: list of numpy arrays (224, 224, 3) BGR
        gradcam_heatmaps: list of numpy arrays (224, 224) float [0, 1]
        attention_weights: numpy array (T,)
        prediction: float (sigmoid probability)
        confidence: str (confidence level)
        label: str ('FAKE' or 'REAL')
        ig_attribution: optional numpy array
        top_k: number of top frames to show
        save_path: if provided, save to this path

    Returns:
        composite: numpy array (H, W, 3) BGR
    """
    # Find top-k attended frames
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]

    # Ensure we have enough frames
    top_indices = [i for i in top_indices if i < len(original_frames)]

    if len(top_indices) == 0:
        top_indices = [0]

    # ── Row 1: Original top frames ───────────────────────────────────
    row1_frames = []
    for idx in top_indices:
        frame = original_frames[idx].copy()
        # Add frame index text
        cv2.putText(frame, f"Frame {idx}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        row1_frames.append(frame)

    # ── Row 2: GradCAM overlays on top frames ────────────────────────
    row2_frames = []
    for idx in top_indices:
        if idx < len(gradcam_heatmaps):
            overlay = overlay_heatmap(original_frames[idx], gradcam_heatmaps[idx])
        else:
            overlay = original_frames[idx].copy()
        row2_frames.append(overlay)

    # ── Assemble rows ────────────────────────────────────────────────
    # Pad to same number of images
    while len(row1_frames) < top_k:
        row1_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        row2_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    row1 = np.hstack(row1_frames[:top_k])
    row2 = np.hstack(row2_frames[:top_k])

    # ── Row 3: Attention bar chart + prediction info ─────────────────
    chart = create_attention_bar_chart(attention_weights, top_k)
    chart_resized = cv2.resize(chart, (row1.shape[1], 150))

    # Add prediction text
    color = (0, 0, 255) if label == 'FAKE' else (0, 255, 0)
    info_bar = np.zeros((40, row1.shape[1], 3), dtype=np.uint8)
    text = f"Prediction: {label} | Probability: {prediction:.4f} | {confidence}"
    cv2.putText(info_bar, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── Stack everything ─────────────────────────────────────────────
    composite = np.vstack([row1, row2, chart_resized, info_bar])

    if save_path:
        cv2.imwrite(save_path, composite)

    return composite
