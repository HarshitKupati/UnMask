"""
Face extraction and FFT preprocessing pipeline.
Reference: Specification Document §4

Pipeline:
  1. Sample frames from videos uniformly (§4.1)
  2. Detect and crop faces using MTCNN (§4.1)
  3. Apply blur filtering via Laplacian variance (§4.1)
  4. Compute FFT magnitude maps (§4.2)
  5. Save as PNG: faces/{label}/{video_id}/frame_{index:03d}.png
                   faces_fft/{label}/{video_id}/frame_{index:03d}.png
"""

import os
import sys
import csv
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    FACE_DETECTOR = 'mtcnn'
except ImportError:
    FACE_DETECTOR = None

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FACE_SIZE = 224
MARGIN = 20
CONFIDENCE_THRESHOLD = 0.90
MIN_FACE_SIZE = 80
FRAMES_PER_VIDEO = 30
BLUR_THRESHOLD = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def get_face_detector(device='cuda'):
    """Initialize MTCNN face detector (§4.1)."""
    if FACE_DETECTOR == 'mtcnn':
        return MTCNN(
            image_size=FACE_SIZE,
            margin=MARGIN,
            min_face_size=MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, CONFIDENCE_THRESHOLD],
            post_process=False,
            device=device,
        )
    else:
        logger.warning("facenet-pytorch not installed. Install with: pip install facenet-pytorch")
        return None


def compute_fft(bgr_image):
    """Compute 2D FFT magnitude map (§4.2)."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1e-8)
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)
    magnitude_uint8 = (magnitude * 255).astype(np.uint8)
    return np.stack([magnitude_uint8] * 3, axis=-1)


def is_blurry(image, threshold=BLUR_THRESHOLD):
    """Check if image is too blurry via Laplacian variance (§4.1)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


def extract_faces_from_video(video_path, mtcnn, output_dir, fft_output_dir,
                              video_id, label, device='cuda'):
    """
    Extract face crops from a single video file (§4.1).

    Args:
        video_path: path to video file
        mtcnn: MTCNN detector instance
        output_dir: base directory for RGB face crops
        fft_output_dir: base directory for FFT maps
        video_id: unique identifier for this video
        label: 'real' or 'fake'
        device: torch device

    Returns:
        dict with extraction statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return {'status': 'failed', 'reason': 'cannot_open'}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return {'status': 'failed', 'reason': 'no_frames'}

    # Create output directories
    face_dir = os.path.join(output_dir, label, video_id)
    fft_dir = os.path.join(fft_output_dir, label, video_id)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(fft_dir, exist_ok=True)

    # Frame sampling (§4.1): uniform sampling of FRAMES_PER_VIDEO frames
    sample_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
    
    valid_faces = 0
    blur_rejected = 0
    saved_index = 0

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces with MTCNN
        try:
            boxes, probs, landmarks = mtcnn.detect(
                frame_rgb, landmarks=True
            )
        except Exception:
            continue

        if boxes is None or len(boxes) == 0:
            continue

        # Take highest-confidence face
        best_idx = np.argmax(probs)
        if probs[best_idx] < CONFIDENCE_THRESHOLD:
            continue

        box = boxes[best_idx].astype(int)
        x1, y1, x2, y2 = box

        # Check minimum face size
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            continue

        # Add margin
        h, w = frame.shape[:2]
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(w, x2 + MARGIN)
        y2 = min(h, y2 + MARGIN)

        # Crop and resize
        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

        # Blur filtering (§4.1)
        blurry, variance = is_blurry(face_crop)
        if blurry:
            blur_rejected += 1
            continue

        # Save RGB face
        face_path = os.path.join(face_dir, f'frame_{saved_index:03d}.png')
        cv2.imwrite(face_path, face_crop)

        # Compute and save FFT (§4.2)
        fft_map = compute_fft(face_crop)
        fft_path = os.path.join(fft_dir, f'frame_{saved_index:03d}.png')
        cv2.imwrite(fft_path, fft_map)

        valid_faces += 1
        saved_index += 1

    cap.release()

    # If fewer than 10 valid faces, try again with 60 samples (§4.1)
    if valid_faces < 10:
        logger.info(f"Video {video_id}: only {valid_faces} faces from {FRAMES_PER_VIDEO} samples, retrying with 60")
        return _retry_extraction(video_path, mtcnn, face_dir, fft_dir,
                                 video_id, saved_index, device)

    return {
        'status': 'success',
        'valid_faces': valid_faces,
        'blur_rejected': blur_rejected,
    }


def _retry_extraction(video_path, mtcnn, face_dir, fft_dir,
                       video_id, start_index, device):
    """Retry with increased sampling if < 10 faces found (§4.1)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, 60, dtype=int)

    valid_faces = start_index
    saved_index = start_index

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
        except Exception:
            continue

        if boxes is None or len(boxes) == 0:
            continue

        best_idx = np.argmax(probs)
        if probs[best_idx] < CONFIDENCE_THRESHOLD:
            continue

        box = boxes[best_idx].astype(int)
        x1, y1, x2, y2 = box

        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            continue

        h, w = frame.shape[:2]
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(w, x2 + MARGIN)
        y2 = min(h, y2 + MARGIN)

        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

        blurry, _ = is_blurry(face_crop)
        if blurry:
            continue

        face_path = os.path.join(face_dir, f'frame_{saved_index:03d}.png')
        cv2.imwrite(face_path, face_crop)

        fft_map = compute_fft(face_crop)
        fft_path = os.path.join(fft_dir, f'frame_{saved_index:03d}.png')
        cv2.imwrite(fft_path, fft_map)

        valid_faces += 1
        saved_index += 1

    cap.release()

    if valid_faces < 10:
        return {'status': 'skipped', 'valid_faces': valid_faces, 'reason': 'insufficient_faces'}

    return {'status': 'success', 'valid_faces': valid_faces}


def extract_faces_from_images(image_dir, output_dir, fft_output_dir,
                               video_id, label, mtcnn=None, device='cuda'):
    """
    Extract faces from a directory of images (for image-based datasets).
    Each image is treated as a single frame.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    face_dir = os.path.join(output_dir, label, video_id)
    fft_dir = os.path.join(fft_output_dir, label, video_id)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(fft_dir, exist_ok=True)

    valid_faces = 0

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue

        if mtcnn is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                boxes, probs, _ = mtcnn.detect(img_rgb, landmarks=True)
            except Exception:
                boxes, probs = None, None

            if boxes is not None and len(boxes) > 0:
                best_idx = np.argmax(probs)
                if probs[best_idx] >= CONFIDENCE_THRESHOLD:
                    box = boxes[best_idx].astype(int)
                    x1, y1, x2, y2 = box
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - MARGIN)
                    y1 = max(0, y1 - MARGIN)
                    x2 = min(w, x2 + MARGIN)
                    y2 = min(h, y2 + MARGIN)
                    img = img[y1:y2, x1:x2]

        face_crop = cv2.resize(img, (FACE_SIZE, FACE_SIZE))

        face_path = os.path.join(face_dir, f'frame_{i:03d}.png')
        cv2.imwrite(face_path, face_crop)

        fft_map = compute_fft(face_crop)
        fft_path = os.path.join(fft_dir, f'frame_{i:03d}.png')
        cv2.imwrite(fft_path, fft_map)

        valid_faces += 1

    return {'status': 'success', 'valid_faces': valid_faces}


def extract_single_image(image_path, output_dir, fft_output_dir,
                          image_id, label, mtcnn=None):
    """
    Process a single standalone image file.
    Saves the face crop and FFT map into:
      output_dir/{label}/{image_id}/frame_000.png
      fft_output_dir/{label}/{image_id}/frame_000.png

    This is the path used for datasets of individual image files
    (not videos or directories of frames).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'status': 'failed', 'reason': 'cannot_read'}

    # Create output directories
    face_dir = os.path.join(output_dir, label, image_id)
    fft_dir = os.path.join(fft_output_dir, label, image_id)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(fft_dir, exist_ok=True)

    # Face detection
    if mtcnn is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            boxes, probs, _ = mtcnn.detect(img_rgb, landmarks=True)
        except Exception:
            boxes, probs = None, None

        if boxes is not None and len(boxes) > 0:
            best_idx = np.argmax(probs)
            if probs[best_idx] >= CONFIDENCE_THRESHOLD:
                box = boxes[best_idx].astype(int)
                x1, y1, x2, y2 = box
                h, w = img.shape[:2]
                x1 = max(0, x1 - MARGIN)
                y1 = max(0, y1 - MARGIN)
                x2 = min(w, x2 + MARGIN)
                y2 = min(h, y2 + MARGIN)
                img = img[y1:y2, x1:x2]
            else:
                # Low confidence — skip this image
                return {'status': 'skipped', 'reason': 'low_confidence'}
        else:
            # No face detected — skip
            return {'status': 'skipped', 'reason': 'no_face'}

    # Resize to standard face size
    face_crop = cv2.resize(img, (FACE_SIZE, FACE_SIZE))

    # Blur check
    blurry, variance = is_blurry(face_crop)
    if blurry:
        return {'status': 'skipped', 'reason': 'blurry', 'variance': variance}

    # Save face crop
    face_path = os.path.join(face_dir, 'frame_000.png')
    cv2.imwrite(face_path, face_crop)

    # Compute and save FFT
    fft_map = compute_fft(face_crop)
    fft_path = os.path.join(fft_dir, 'frame_000.png')
    cv2.imwrite(fft_path, fft_map)

    return {'status': 'success', 'valid_faces': 1}


def build_data_list_from_csv(csv_path, faces_dir):
    """
    Build a data_list from a split CSV file (§3.5).
    CSV columns: video_id, video_path, label, manipulation_type, dataset_source, split
    """
    data_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = 'fake' if int(row['label']) == 1 else 'real'
            frames_dir = os.path.join(faces_dir, label_str, row['video_id'])
            if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                data_list.append({
                    'video_id': row['video_id'],
                    'frames_dir': frames_dir,
                    'label': int(row['label']),
                    'split': row.get('split', 'train'),
                    'manipulation_type': row.get('manipulation_type', 'unknown'),
                    'dataset_source': row.get('dataset_source', 'unknown'),
                })
    return data_list


def build_data_list_from_directory(faces_dir):
    """
    Build a data_list by scanning the faces directory structure:
      faces_dir/real/video_id/frame_*.png
      faces_dir/fake/video_id/frame_*.png
    """
    data_list = []
    for label_name, label_int in [('real', 0), ('fake', 1)]:
        label_dir = os.path.join(faces_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for video_id in sorted(os.listdir(label_dir)):
            frames_dir = os.path.join(label_dir, video_id)
            if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                data_list.append({
                    'video_id': video_id,
                    'frames_dir': frames_dir,
                    'label': label_int,
                })
    return data_list


if __name__ == '__main__':
    """
    Usage:
      python -m data_utils.extract_faces --input_dir data/raw --output_dir data/faces \\
             --fft_output_dir data/faces_fft --mode video
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from videos/images")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing raw videos or images")
    parser.add_argument('--output_dir', type=str, default='data/faces',
                        help="Output directory for face crops")
    parser.add_argument('--fft_output_dir', type=str, default='data/faces_fft',
                        help="Output directory for FFT maps")
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default='video')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    mtcnn = get_face_detector(device)

    if mtcnn is None:
        logger.error("No face detector available. Install facenet-pytorch.")
        sys.exit(1)

    skipped_videos = []

    # Process real and fake subdirectories
    for label in ['real', 'fake']:
        label_dir = os.path.join(args.input_dir, label)
        if not os.path.isdir(label_dir):
            logger.warning(f"Directory not found: {label_dir}")
            continue

        items = sorted(os.listdir(label_dir))
        for item in tqdm(items, desc=f"Processing {label}"):
            item_path = os.path.join(label_dir, item)
            video_id = os.path.splitext(item)[0]

            if args.mode == 'video':
                result = extract_faces_from_video(
                    item_path, mtcnn, args.output_dir,
                    args.fft_output_dir, video_id, label, device
                )
            else:
                result = extract_faces_from_images(
                    item_path, args.output_dir,
                    args.fft_output_dir, video_id, label, mtcnn, device
                )

            if result['status'] == 'skipped':
                skipped_videos.append(f"{video_id},{result.get('reason', 'unknown')}")

    # Log skipped videos (§4.1)
    if skipped_videos:
        with open('skipped_videos.txt', 'w') as f:
            f.write('\n'.join(skipped_videos))
        logger.info(f"Skipped {len(skipped_videos)} videos. See skipped_videos.txt")
