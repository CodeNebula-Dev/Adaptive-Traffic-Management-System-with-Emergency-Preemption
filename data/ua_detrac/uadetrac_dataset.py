"""
UA-DETRAC Traffic Surveillance Dataset Loader for ATMS-Net.

Features:
    - 5-Class Support: [car, motorcycle, bus, truck, unknown_vehicle]
    - High-Resolution Support: 512x512 / 640x640
    - Boundary Cut-Off Augmentation: Simulates vehicles entering/exiting intersection camera edges
    - Rain/Night Surveillance Lighting Jitter: HSV adjustments + Cutout
    - Multi-scale Mosaic & Letterbox transformations
"""

import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import letterbox, hsv_jitter, random_horizontal_flip, cutout


class UADetracDataset(Dataset):
    """
    PyTorch Dataset for UA-DETRAC Traffic Surveillance Benchmark.

    Args:
        img_list: Path to text file listing absolute image paths (train.txt or val.txt)
        label_dir: Directory containing YOLO .txt label files
        img_size: Target square image resolution (default: 512)
        augment: Whether to apply data augmentation (True for training)
        mosaic_prob: Probability of applying 4-image mosaic augmentation (default: 0.4)
        num_classes: Number of target classes (default: 5)
    """

    CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'unknown_vehicle']

    def __init__(self, img_list, label_dir='data/ua_detrac/labels', img_size=512,
                 augment=True, mosaic_prob=0.4, num_classes=5):
        super().__init__()
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.num_classes = num_classes
        self.label_dir = label_dir

        # Read image list directly (instant load)
        with open(img_list, 'r') as f:
            self.img_paths = [line.strip() for line in f if line.strip()]

        print(f"  → Loaded {len(self.img_paths):,} images from {img_list}", flush=True)

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, index):
        """Load BGR image from disk."""
        path = self.img_paths[index]
        img = cv2.imread(path)
        if img is None:
            # Fallback canvas if missing
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return img, path

    def _load_labels(self, img_path):
        """
        Load YOLO format labels for a given image directly.
        Format per line: class_id cx cy w h (all in [0, 1])
        """
        p = Path(img_path)
        label_file = f"{p.parent.name}_{p.stem}.txt"
        label_path = os.path.join(self.label_dir, label_file)

        if not os.path.exists(label_path):
            # Fallback to direct stem name
            label_path = os.path.join(self.label_dir, f"{p.stem}.txt")
            if not os.path.exists(label_path):
                return np.zeros((0, 5), dtype=np.float32)

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    if w > 0.001 and h > 0.001:
                        boxes.append([cls_id, cx, cy, w, h])

        if not boxes:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def __getitem__(self, index):
        """
        Returns:
            img_tensor: Float tensor (3, img_size, img_size) in [0, 1] (RGB)
            targets: Float tensor (N, 6) — [batch_idx, class_id, cx, cy, w, h] (normalized)
        """
        img, img_path = self._load_image(index)
        labels = self._load_labels(img_path)

        orig_h, orig_w = img.shape[:2]

        # 1. Augmentation: Mosaic (combine 4 surveillance frames)
        if self.augment and random.random() < self.mosaic_prob and len(self) >= 4:
            img, labels = self._load_mosaic(index)
            orig_h, orig_w = img.shape[:2]

        # 2. Letterbox resize to square target_size
        img, ratio, (dw, dh) = letterbox(img, target_size=self.img_size)

        # 3. Adjust normalized bbox coordinates after letterbox padding
        if len(labels) > 0:
            # Convert normalized cxcywh -> absolute pixel xyxy on original image
            boxes = labels[:, 1:].copy()
            x1 = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w
            y1 = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h
            x2 = (boxes[:, 0] + boxes[:, 2] / 2) * orig_w
            y2 = (boxes[:, 1] + boxes[:, 3] / 2) * orig_h

            # Scale and shift to letterboxed canvas
            x1 = x1 * ratio + dw
            y1 = y1 * ratio + dh
            x2 = x2 * ratio + dw
            y2 = y2 * ratio + dh

            # Clip to [0, img_size]
            x1 = np.clip(x1, 0, self.img_size)
            y1 = np.clip(y1, 0, self.img_size)
            x2 = np.clip(x2, 0, self.img_size)
            y2 = np.clip(y2, 0, self.img_size)

            # Convert back to normalized cxcywh
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            # Keep only valid non-collapsed boxes
            valid = (w > 2.0) & (h > 2.0)
            labels = labels[valid]
            if len(labels) > 0:
                labels[:, 1] = cx[valid] / self.img_size
                labels[:, 2] = cy[valid] / self.img_size
                labels[:, 3] = w[valid] / self.img_size
                labels[:, 4] = h[valid] / self.img_size
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

        # 4. Color / Lighting / Adverse Weather Augmentations
        if self.augment:
            img = hsv_jitter(img, h_gain=0.015, s_gain=0.6, v_gain=0.4)
            img, labels = random_horizontal_flip(img, labels, p=0.5)
            if random.random() < 0.2:
                img = cutout(img, labels, n_holes=4, max_size=0.15)

        # BGR -> RGB -> Float Tensor in [0, 1]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img)

        # Target tensor: [0 (placeholder batch idx), class_id, cx, cy, w, h]
        if len(labels) > 0:
            targets = np.zeros((len(labels), 6), dtype=np.float32)
            targets[:, 1:] = labels
            targets_tensor = torch.from_numpy(targets)
        else:
            targets_tensor = torch.zeros((0, 6), dtype=torch.float32)

        return img_tensor, targets_tensor

    def _load_mosaic(self, index):
        """Mosaic 4 surveillance images into one 2x2 grid."""
        indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]
        s = self.img_size
        xc, yc = int(random.uniform(s * 0.4, s * 0.6)), int(random.uniform(s * 0.4, s * 0.6))

        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, idx in enumerate(indices):
            img, path = self._load_image(idx)
            h, w = img.shape[:2]
            labels = self._load_labels(path)

            # Coordinates on mosaic canvas
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # Bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(labels) > 0:
                boxes = labels[:, 1:].copy()
                bx1 = (boxes[:, 0] - boxes[:, 2] / 2) * w + padw
                by1 = (boxes[:, 1] - boxes[:, 3] / 2) * h + padh
                bx2 = (boxes[:, 0] + boxes[:, 2] / 2) * w + padw
                by2 = (boxes[:, 1] + boxes[:, 3] / 2) * h + padh

                bx1 = np.clip(bx1, 0, s * 2)
                by1 = np.clip(by1, 0, s * 2)
                bx2 = np.clip(bx2, 0, s * 2)
                by2 = np.clip(by2, 0, s * 2)

                bw = bx2 - bx1
                bh = by2 - by1
                bcx = bx1 + bw / 2.0
                bcy = by1 + bh / 2.0

                valid = (bw > 2.0) & (bh > 2.0)
                if np.any(valid):
                    l = labels[valid].copy()
                    l[:, 1] = bcx[valid] / (s * 2)
                    l[:, 2] = bcy[valid] / (s * 2)
                    l[:, 3] = bw[valid] / (s * 2)
                    l[:, 4] = bh[valid] / (s * 2)
                    mosaic_labels.append(l)

        if mosaic_labels:
            mosaic_labels = np.vstack(mosaic_labels)
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)

        # Resize mosaic down to target resolution
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        return mosaic_img, mosaic_labels
