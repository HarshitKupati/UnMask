"""
Script to create a static train/val split CSV for the Deepfake Detection System.
This ensures reproducibility and avoids data leakage across training sessions.
"""

import os
import csv
import random
import argparse
from pathlib import Path

def create_splits(faces_dir, output_csv, val_split=0.2, seed=42, max_samples=None):
    random.seed(seed)
    
    data_list = []
    
    # Scan extracted faces directory
    for label_name, label_int in [('real', 0), ('fake', 1)]:
        label_dir = os.path.join(faces_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
            
        video_ids = sorted(os.listdir(label_dir))
        for video_id in video_ids:
            frames_dir = os.path.join(label_dir, video_id)
            if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                data_list.append({
                    'video_id': video_id,
                    'label': label_int,
                })
                
    if not data_list:
        print(f"No data found in {faces_dir}. Please run extraction first.")
        return

    print(f"Found {len(data_list)} total samples.")
    
    # ── Experiment Mode: Limit Dataset Size ──
    if max_samples is not None and max_samples > 0 and len(data_list) > max_samples:
        random.shuffle(data_list)
        data_list = data_list[:max_samples]
        print(f"Limiting dataset to {max_samples} samples for experiment mode.")
    
    # Shuffle and split
    random.shuffle(data_list)
    split_idx = int(len(data_list) * (1 - val_split))
    
    # Assign split tags
    for i, item in enumerate(data_list):
        item['split'] = 'train' if i < split_idx else 'val'
        
    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'label', 'split'])
        writer.writeheader()
        for item in data_list:
            writer.writerow(item)
            
    print(f"Created split: {split_idx} train, {len(data_list) - split_idx} val.")
    print(f"Saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create reproducible train/val splits")
    parser.add_argument('--faces_dir', type=str, default='data/faces', help="Path to extracted faces")
    parser.add_argument('--output_csv', type=str, default='data/splits/dataset_splits.csv', help="Output CSV path")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split ratio")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--max_samples', type=int, default=0, help="Limit total videos for quick training (0 for all)")
    
    args = parser.parse_args()
    
    # Handle max_samples (convert 0 to None)
    limit = args.max_samples if args.max_samples > 0 else None
    
    create_splits(args.faces_dir, args.output_csv, args.val_split, args.seed, max_samples=limit)
