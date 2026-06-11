"""
MS COCO 2017 Vehicle Subset Download and Preparation Script.

Downloads the COCO 2017 dataset (or uses existing files) and filters
it to vehicle-only classes for ATMS-Net Phase 1 training.

Vehicle classes extracted from COCO:
    - car (COCO id=3)      → ATMS-Net class 0
    - motorcycle (COCO id=4) → ATMS-Net class 1
    - bus (COCO id=6)       → ATMS-Net class 2
    - truck (COCO id=8)     → ATMS-Net class 3

Output:
    - Per-image YOLO format label files (.txt): class cx cy w h (normalized)
    - train.txt and val.txt listing image paths
    - Dataset statistics summary

Usage:
    python data/coco/download_coco.py --data-dir data/coco --download
    python data/coco/download_coco.py --data-dir data/coco  # (if COCO already exists)
"""

import os
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path


# COCO category ID → ATMS-Net class mapping
COCO_VEHICLE_CATEGORIES = {
    3: 0,   # car → 0
    4: 1,   # motorcycle → 1
    6: 2,   # bus → 2
    8: 3,   # truck → 3
}

ATMS_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']


def download_coco(data_dir):
    """
    Download COCO 2017 train images and annotations.

    Note: This downloads ~18GB of images. Run this on a machine with
    sufficient disk space and bandwidth. On Kaggle, COCO is pre-available.
    """
    import urllib.request
    import zipfile

    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    }

    os.makedirs(data_dir, exist_ok=True)

    for name, url in urls.items():
        zip_path = os.path.join(data_dir, f'{name}.zip')
        if not os.path.exists(zip_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
            print(f"  → Saved to {zip_path}")
        else:
            print(f"  → {name} already exists, skipping download")

        # Extract
        print(f"  Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        print(f"  → Done")


def process_coco_annotations(data_dir, split='train', val_fraction=0.2):
    """
    Filter COCO annotations to vehicle classes and convert to YOLO format.

    YOLO label format (per line in .txt file):
        class_id  cx  cy  w  h
        All values normalized to [0, 1] relative to image dimensions.

    Args:
        data_dir: Root COCO data directory
        split: 'train' to process train2017 annotations
        val_fraction: Fraction of vehicle images to use for validation

    Returns:
        stats: Dict with dataset statistics
    """
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}2017.json')
    img_dir = os.path.join(data_dir, f'{split}2017')
    label_dir = os.path.join(data_dir, 'labels', f'{split}2017')

    if not os.path.exists(ann_file):
        print(f"ERROR: Annotation file not found: {ann_file}")
        print("Run with --download flag first, or point --data-dir to existing COCO.")
        return None

    print(f"\nProcessing COCO {split}2017 annotations...")
    os.makedirs(label_dir, exist_ok=True)

    # Load COCO annotations
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # Build image ID → image info lookup
    images = {img['id']: img for img in coco['images']}

    # Gather vehicle annotations per image
    vehicle_anns = defaultdict(list)
    class_counts = defaultdict(int)
    skipped_small = 0

    for ann in coco['annotations']:
        cat_id = ann['category_id']
        if cat_id not in COCO_VEHICLE_CATEGORIES:
            continue

        # Skip crowd annotations and very small boxes
        if ann.get('iscrowd', 0):
            continue

        bbox = ann['bbox']  # [x, y, width, height] in absolute pixels
        if bbox[2] < 5 or bbox[3] < 5:  # Skip tiny annotations
            skipped_small += 1
            continue

        img_id = ann['image_id']
        img_info = images[img_id]
        img_w, img_h = img_info['width'], img_info['height']

        # Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] normalized
        cx = (bbox[0] + bbox[2] / 2) / img_w
        cy = (bbox[1] + bbox[3] / 2) / img_h
        w = bbox[2] / img_w
        h = bbox[3] / img_h

        # Clip to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        atms_class = COCO_VEHICLE_CATEGORIES[cat_id]
        vehicle_anns[img_id].append(f"{atms_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        class_counts[ATMS_CLASS_NAMES[atms_class]] += 1

    # Write label files
    vehicle_images = []
    for img_id, labels in vehicle_anns.items():
        img_info = images[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)

        # Only include images that actually exist on disk
        if not os.path.exists(img_path):
            continue

        # Write YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

        vehicle_images.append(img_path)

    # Train/val split
    random.seed(42)
    random.shuffle(vehicle_images)
    n_val = int(len(vehicle_images) * val_fraction)
    val_images = vehicle_images[:n_val]
    train_images = vehicle_images[n_val:]

    # Write split files
    train_txt = os.path.join(data_dir, 'train.txt')
    val_txt = os.path.join(data_dir, 'val.txt')

    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_images))

    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_images))

    stats = {
        'total_images': len(vehicle_images),
        'train_images': len(train_images),
        'val_images': len(val_images),
        'class_counts': dict(class_counts),
        'total_annotations': sum(class_counts.values()),
        'skipped_small': skipped_small,
    }

    return stats


def print_stats(stats):
    """Pretty-print dataset statistics."""
    if stats is None:
        return

    print("\n" + "=" * 50)
    print("COCO Vehicle Subset — Dataset Statistics")
    print("=" * 50)
    print(f"  Total images with vehicles: {stats['total_images']:,}")
    print(f"  Training images:            {stats['train_images']:,}")
    print(f"  Validation images:          {stats['val_images']:,}")
    print(f"  Total vehicle annotations:  {stats['total_annotations']:,}")
    print(f"  Skipped (too small):        {stats['skipped_small']:,}")
    print("-" * 50)
    print("  Per-class counts:")
    for cls_name, count in sorted(stats['class_counts'].items()):
        bar = '█' * int(count / max(stats['class_counts'].values()) * 30)
        print(f"    {cls_name:12s}: {count:6,}  {bar}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Prepare COCO vehicle subset for ATMS-Net')
    parser.add_argument('--data-dir', type=str, default='data/coco',
                        help='Root directory for COCO data')
    parser.add_argument('--download', action='store_true',
                        help='Download COCO 2017 dataset (requires ~20GB disk)')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Fraction of images for validation (default: 0.2)')
    args = parser.parse_args()

    data_dir = args.data_dir

    if args.download:
        download_coco(data_dir)

    stats = process_coco_annotations(data_dir, split='train', val_fraction=args.val_fraction)
    print_stats(stats)

    if stats:
        print(f"\n✓ Label files saved to: {os.path.join(data_dir, 'labels', 'train2017')}")
        print(f"✓ Train split: {os.path.join(data_dir, 'train.txt')}")
        print(f"✓ Val split:   {os.path.join(data_dir, 'val.txt')}")
        print(f"\nReady for training!")


if __name__ == '__main__':
    main()
