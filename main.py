"""
Main entry point for the Deepfake Detection System.
Reference: Specification Document

Usage:
  Train:
    python main.py --mode train --data_dir data/faces --epochs 40

  Predict (video):
    python main.py --mode predict --input path/to/video.mp4 --checkpoint checkpoints/best_model.pth

  Predict (image):
    python main.py --mode predict --input path/to/image.jpg --checkpoint checkpoints/best_model.pth

  Extract faces:
    python main.py --mode extract --input_dir data/raw --output_dir data/faces

  Evaluate:
    python main.py --mode evaluate --checkpoint checkpoints/best_model.pth --data_dir data/faces
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import CONFIG
from models.detector import DeepfakeDetector
from data_utils.dataset import DeepfakeDataset
from data_utils.augmentations import get_training_augmentations, get_validation_augmentations
from data_utils.extract_faces import build_data_list_from_directory
from training.scheduler import build_optimizer_and_scheduler
from training.trainer import Trainer
from training.metrics import compute_metrics, find_optimal_threshold
from inference.predict import DeepfakePredictor


# ── Reproducibility ──────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Logging ──────────────────────────────────────────────────────────
os.makedirs(CONFIG['log_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CONFIG['log_dir'], 'main.log')),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection System — EfficientNet-B4 + BiLSTM + XAI'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'predict', 'extract', 'evaluate'],
                        help='Operation mode')
    parser.add_argument('--data_dir', type=str, default=CONFIG['faces_dir'],
                        help='Path to face crops directory (for train/evaluate)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input path for prediction (video or image file)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory for face extraction')
    parser.add_argument('--output_dir', type=str, default=CONFIG['faces_dir'],
                        help='Output directory for face extraction')
    parser.add_argument('--fft_output_dir', type=str, default=CONFIG['faces_fft_dir'],
                        help='Output directory for FFT maps')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=CONFIG['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--no_xai', action='store_true',
                        help='Disable XAI computation during prediction')
    parser.add_argument('--save_dir', type=str, default=CONFIG['xai_samples_dir'],
                        help='Directory to save prediction results')
    parser.add_argument('--split_csv', type=str, default=os.path.join(CONFIG['splits_dir'], 'dataset_splits.csv'),
                        help='Path to dataset splits CSV file (for reproducible training)')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Training mode
# ─────────────────────────────────────────────────────────────────────

def run_training(args):
    """Full training pipeline."""
    seed_everything(args.seed)

    # Update config
    config = CONFIG.copy()
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size

    logger.info("=" * 70)
    logger.info("DEEPFAKE DETECTION SYSTEM — TRAINING")
    logger.info("=" * 70)
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {config['num_epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    eff_batch = config['batch_size'] * config['gradient_accumulation_steps']
    logger.info(f"Effective batch size: {eff_batch}")

    # ── Build data lists ─────────────────────────────────────────────
    if os.path.exists(args.split_csv):
        logger.info(f"Using split CSV: {args.split_csv}")
        from data_utils.extract_faces import build_data_list_from_csv
        data_list = build_data_list_from_csv(args.split_csv, args.data_dir)
        train_list = [d for d in data_list if d.get('split') == 'train']
        val_list = [d for d in data_list if d.get('split') == 'val']
    else:
        logger.warning(f"Split CSV not found at {args.split_csv}. Falling back to dynamic directory scan.")
        logger.warning("It is highly recommended to run `python data_utils/create_splits.py` for reproducible training.")
        data_list = build_data_list_from_directory(args.data_dir)
        if len(data_list) == 0:
            logger.error(f"No data found in {args.data_dir}. Run face extraction first.")
            return

        logger.info(f"Total samples: {len(data_list)}")

        # ── Train/Val split (Dynamic fallback) ──────────────────────
        random.shuffle(data_list)
        split_idx = int(len(data_list) * (1 - args.val_split))
        train_list = data_list[:split_idx]
        val_list = data_list[split_idx:]
        
    logger.info(f"Train: {len(train_list)}, Val: {len(val_list)}")

    # ── Datasets ─────────────────────────────────────────────────────
    train_dataset = DeepfakeDataset(
        train_list,
        sequence_length=config['sequence_length'],
        transform=get_training_augmentations(),
    )
    val_dataset = DeepfakeDataset(
        val_list,
        sequence_length=config['sequence_length'],
        transform=get_validation_augmentations(),
    )

    # ── WeightedRandomSampler for class balance ──────────────────────
    labels = [entry['label'] for entry in train_list]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = DeepfakeDetector(config).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model, optimizer, scheduler,
        train_loader, val_loader,
        config=config, device=args.device,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ── Train ────────────────────────────────────────────────────────
    best_auc = trainer.fit(start_epoch=start_epoch)
    logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Prediction mode
# ─────────────────────────────────────────────────────────────────────

def run_prediction(args):
    """Single video/image inference with XAI."""
    if args.checkpoint is None:
        logger.error("Must provide --checkpoint for prediction")
        return
    if args.input is None:
        logger.error("Must provide --input for prediction")
        return

    predictor = DeepfakePredictor(
        checkpoint_path=args.checkpoint,
        config=CONFIG,
        device=args.device,
    )

    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    if ext in video_extensions:
        result = predictor.predict_video(
            input_path,
            use_tta=not args.no_tta,
            compute_xai=not args.no_xai,
        )
    elif ext in image_extensions:
        result = predictor.predict_image(
            input_path,
            compute_xai=not args.no_xai,
        )
    else:
        logger.error(f"Unsupported file type: {ext}")
        return

    # Print results
    print("\n" + "=" * 50)
    print(f"  Prediction: {result['label']}")
    print(f"  Probability: {result['prediction']:.4f}")
    print(f"  Confidence: {result['confidence']}")
    print("=" * 50)

    # Save visualization
    if 'visualization' in result and result['visualization'] is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        save_path = os.path.join(args.save_dir, f'{base_name}_prediction.png')
        import cv2
        cv2.imwrite(save_path, result['visualization'])
        print(f"  Visualization saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# Face extraction mode
# ─────────────────────────────────────────────────────────────────────

def run_extraction(args):
    """Extract faces from raw videos/images."""
    if args.input_dir is None:
        logger.error("Must provide --input_dir for extraction")
        return

    from data_utils.extract_faces import (
        extract_faces_from_video,
        extract_faces_from_images,
        extract_single_image,
        get_face_detector,
    )
    from tqdm import tqdm

    mtcnn = get_face_detector(args.device)
    if mtcnn is None:
        logger.error("Face detector not available. Install with: pip install facenet-pytorch")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

    stats = {'processed': 0, 'skipped': 0, 'failed': 0}

    for label in ['real', 'fake']:
        label_dir = os.path.join(args.input_dir, label)
        if not os.path.isdir(label_dir):
            logger.warning(f"Directory not found: {label_dir}")
            continue

        items = sorted(os.listdir(label_dir))
        logger.info(f"Processing {len(items)} items in {label}/")

        for item in tqdm(items, desc=f"Extracting {label}"):
            item_path = os.path.join(label_dir, item)
            video_id = os.path.splitext(item)[0]
            ext = os.path.splitext(item)[1].lower()

            try:
                if os.path.isdir(item_path):
                    # Directory of frames (video already split into frames)
                    result = extract_faces_from_images(
                        item_path, args.output_dir, args.fft_output_dir,
                        video_id, label, mtcnn, args.device
                    )
                elif ext in image_extensions:
                    # Single image file — process directly
                    result = extract_single_image(
                        item_path, args.output_dir, args.fft_output_dir,
                        video_id, label, mtcnn
                    )
                elif ext in video_extensions:
                    # Video file
                    result = extract_faces_from_video(
                        item_path, mtcnn, args.output_dir, args.fft_output_dir,
                        video_id, label, args.device
                    )
                else:
                    continue

                if result.get('status') == 'success':
                    stats['processed'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                logger.warning(f"Failed on {item}: {e}")
                stats['failed'] += 1

    logger.info(f"Face extraction complete. "
                f"Processed: {stats['processed']}, "
                f"Skipped: {stats['skipped']}, "
                f"Failed: {stats['failed']}")


# ─────────────────────────────────────────────────────────────────────
# Evaluation mode
# ─────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    """Evaluate model on test set."""
    if args.checkpoint is None:
        logger.error("Must provide --checkpoint for evaluation")
        return

    config = CONFIG.copy()
    model = DeepfakeDetector(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    data_list = build_data_list_from_directory(args.data_dir)
    if len(data_list) == 0:
        logger.error("No data found for evaluation")
        return

    dataset = DeepfakeDataset(
        data_list,
        sequence_length=config['sequence_length'],
        transform=get_validation_augmentations(),
    )
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )

    all_logits = []
    all_labels = []

    from torch.cuda.amp import autocast

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            rgb = batch['rgb'].to(args.device)
            fft = batch['fft'].to(args.device)
            mask = batch['mask'].to(args.device)
            labels = batch['label']

            with autocast():
                logits = model(rgb, fft, mask=mask)

            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    metrics = compute_metrics(np.array(all_labels), np.array(all_logits))
    threshold, best_f1 = find_optimal_threshold(
        np.array(all_labels), np.array(all_logits)
    )

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  AUC-ROC:  {metrics['auc']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  AP:       {metrics['ap']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  FNR:      {metrics.get('fnr', 'N/A')}")
    print(f"  Optimal threshold: {threshold:.3f} (F1={best_f1:.4f})")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    # Create necessary directories
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(CONFIG['xai_samples_dir'], exist_ok=True)

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'predict':
        run_prediction(args)
    elif args.mode == 'extract':
        run_extraction(args)
    elif args.mode == 'evaluate':
        run_evaluation(args)
