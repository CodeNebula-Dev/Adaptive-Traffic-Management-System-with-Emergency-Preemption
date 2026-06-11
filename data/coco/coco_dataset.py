"""
PyTorch Dataset for MS COCO Vehicle Subset.

Loads images and YOLO-format labels, applies augmentations (including
mosaic), and returns tensors ready for training the ATMS-Net detector.

This dataset works with the output of download_coco.py:
    - Image paths listed in train.txt / val.txt
    - YOLO label files in data/coco/labels/train2017/

Usage:
    from data.coco.coco_dataset import COCOVehicleDataset, detection_collate_fn

    dataset = COCOVehicleDataset(
        img_list='data/coco/train.txt',
        label_dir='data/coco/labels/train2017',
        img_size=416,
        augment=True,
    )

    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        collate_fn=detection_collate_fn, num_workers=4,
    )
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import (
    mosaic_augmentation,
    apply_augmentations,
    letterbox,
    letterbox_labels,
)


class COCOVehicleDataset(Dataset):
    """
    PyTorch Dataset for COCO vehicle subset with YOLO-format labels.

    Supports mosaic augmentation (which requires access to other images
    in the dataset via the dataset reference in mosaic_augmentation).

    Args:
        img_list: Path to .txt file listing image paths (one per line)
        label_dir: Directory containing YOLO .txt label files
        img_size: Target image size (default: 416)
        augment: Whether to apply training augmentations
        mosaic_prob: Probability of applying mosaic augmentation (default: 0.5)
    """

    def __init__(self, img_list, label_dir, img_size=416, augment=True, mosaic_prob=0.5):
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.label_dir = label_dir

        # Load image paths
        with open(img_list, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines() if line.strip()]

        print(f"  → Loaded {len(self.img_paths)} images from {img_list}")

    def __len__(self):
        return len(self.img_paths)

    def load_image_and_labels(self, index):
        """
        Load raw image and labels without augmentation.
        Used by mosaic augmentation to access other images.

        Args:
            index: Dataset index

        Returns:
            img: (H, W, 3) BGR numpy array
            labels: (N, 5) numpy array — [class, cx, cy, w, h] normalized
        """
        # Load image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: return a blank image if file is corrupted/missing
            img = np.full((416, 416, 3), 114, dtype=np.uint8)
            return img, np.zeros((0, 5), dtype=np.float32)

        # Load labels
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_filename)

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, dtype=np.float32)
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)  # Single label → (1, 5)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return img, labels

    def __getitem__(self, index):
        """
        Get a training sample.

        Returns:
            img_tensor: (3, img_size, img_size) float32 tensor in [0, 1]
            targets: (N, 6) float32 tensor — [batch_idx(0), class, cx, cy, w, h]
                in absolute pixel coordinates of the target image
        """
        # Decide whether to use mosaic
        use_mosaic = self.augment and np.random.random() < self.mosaic_prob

        if use_mosaic:
            img, labels = mosaic_augmentation(self, index, self.img_size)
            # Mosaic labels are already in absolute pixel coords

            # Apply additional augmentations (HSV, flip, cutout) but NOT letterbox
            # since mosaic already produces the right size
            from utils.augmentations import hsv_jitter, random_horizontal_flip, cutout
            img = hsv_jitter(img)
            img, labels = random_horizontal_flip(img, labels, p=0.5)
            if np.random.random() < 0.5:
                img = cutout(img, labels)
        else:
            # Standard pipeline
            img, labels = self.load_image_and_labels(index)
            img, labels = apply_augmentations(
                img, labels, self.img_size, augment=self.augment
            )

        # Convert image: BGR → RGB, HWC → CHW, [0,255] → [0,1]
        img = img[:, :, ::-1].copy()  # BGR to RGB
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC to CHW
        img_tensor = torch.from_numpy(img).float() / 255.0

        # Build targets tensor: [batch_idx, class, cx, cy, w, h]
        # batch_idx is set to 0 here and corrected in collate_fn
        if len(labels) > 0:
            targets = torch.zeros((len(labels), 6), dtype=torch.float32)
            targets[:, 0] = 0  # batch_idx placeholder
            targets[:, 1:] = torch.from_numpy(labels[:, :5])
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return img_tensor, targets


def detection_collate_fn(batch):
    """
    Custom collate function for detection dataloader.

    Standard collate can't handle variable-length target tensors (each image
    has a different number of objects). This function:
    1. Stacks images normally (all same size due to letterbox)
    2. Concatenates targets and sets correct batch indices

    Args:
        batch: List of (img_tensor, targets) tuples from __getitem__

    Returns:
        images: (B, 3, H, W) float32 tensor
        targets: (N_total, 6) tensor — [batch_idx, class, cx, cy, w, h]
    """
    images = []
    targets = []

    for batch_idx, (img, target) in enumerate(batch):
        images.append(img)
        if target.shape[0] > 0:
            target[:, 0] = batch_idx  # Set correct batch index
            targets.append(target)

    images = torch.stack(images, dim=0)

    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, targets
