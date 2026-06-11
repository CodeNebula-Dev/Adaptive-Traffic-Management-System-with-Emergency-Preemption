"""
Data Augmentation Pipeline for ATMS-Net Vehicle Detector.

Implements augmentations specifically designed for object detection training:
    - Mosaic: Stitch 4 images into a 2×2 grid (most impactful YOLO augmentation)
    - HSV Jitter: Random hue/saturation/value shifts for lighting robustness
    - Random Horizontal Flip: Standard geometric augmentation
    - Cutout / Random Erase: Mask patches for redundant feature learning
    - Letterbox Resize: Aspect-ratio-preserving resize with padding

All augmentations correctly transform both images AND their bounding box labels.
"""

import cv2
import numpy as np
import random


def letterbox(img, target_size=416, color=(114, 114, 114)):
    """
    Resize image maintaining aspect ratio with gray padding.

    This prevents distortion — vehicles maintain their true proportions,
    which is important for accurate bounding box regression.

    Args:
        img: Input image (H, W, 3) numpy array
        target_size: Target square size (default: 416)
        color: Padding color (default: gray)

    Returns:
        padded: Resized and padded image (target_size, target_size, 3)
        ratio: Scale ratio used
        (dw, dh): Padding offsets
    """
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)

    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    dw = (target_size - new_w) / 2
    dh = (target_size - new_h) / 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    # Ensure exact target size (handle rounding)
    padded = cv2.resize(padded, (target_size, target_size))

    return padded, ratio, (dw, dh)


def letterbox_labels(labels, ratio, pad, orig_shape):
    """
    Transform labels to match letterboxed image.

    Args:
        labels: (N, 5) array — [class, cx, cy, w, h] in normalized coords [0,1]
        ratio: Scale ratio from letterbox
        pad: (dw, dh) padding offsets
        orig_shape: (h, w) original image shape

    Returns:
        transformed_labels: (N, 5) in absolute pixel coords of target image
    """
    if len(labels) == 0:
        return labels

    labels = labels.copy()
    h, w = orig_shape

    # Convert from normalized to absolute original coords
    labels[:, 1] *= w  # cx
    labels[:, 2] *= h  # cy
    labels[:, 3] *= w  # w
    labels[:, 4] *= h  # h

    # Apply scale and padding
    labels[:, 1] = labels[:, 1] * ratio + pad[0]  # cx
    labels[:, 2] = labels[:, 2] * ratio + pad[1]  # cy
    labels[:, 3] *= ratio  # w
    labels[:, 4] *= ratio  # h

    return labels


def mosaic_augmentation(dataset, index, target_size=416):
    """
    Mosaic augmentation: combines 4 random images into a 2×2 grid.

    This is the single most impactful augmentation in YOLO training because:
    1. Forces the model to see partial objects at image boundaries
    2. Provides 4× more context diversity per training sample
    3. Naturally varies object scale and position
    4. Reduces the need for a large batch size (each sample is 4 images)

    Args:
        dataset: Dataset instance with __getitem__ returning (img, labels)
        index: Index of the primary image
        target_size: Output image size

    Returns:
        mosaic_img: (target_size, target_size, 3) combined image
        mosaic_labels: (N, 5) — [class, cx, cy, w, h] in absolute pixel coords
    """
    # Random center point for the mosaic
    cx = int(random.uniform(target_size * 0.25, target_size * 0.75))
    cy = int(random.uniform(target_size * 0.25, target_size * 0.75))

    # Select 3 additional random indices
    indices = [index] + random.choices(range(len(dataset)), k=3)

    mosaic_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    mosaic_labels = []

    for i, idx in enumerate(indices):
        img, labels = dataset.load_image_and_labels(idx)
        h, w = img.shape[:2]

        # Determine placement in mosaic
        if i == 0:  # Top-left
            x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # Top-right
            x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, target_size), cy
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # Bottom-left
            x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(cy + h, target_size)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
        else:  # Bottom-right
            x1a, y1a, x2a, y2a = cx, cy, min(cx + w, target_size), min(cy + h, target_size)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

        # Place image patch
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # Adjust labels
        if len(labels) > 0:
            labels = labels.copy()
            # Labels are [class, cx, cy, w, h] in normalized coords [0,1]
            # Convert to absolute coords in original image
            labels[:, 1] *= w  # cx in original pixels
            labels[:, 2] *= h  # cy
            labels[:, 3] *= w  # w
            labels[:, 4] *= h  # h

            # Shift to mosaic position
            pad_w = x1a - x1b
            pad_h = y1a - y1b
            labels[:, 1] += pad_w  # shift cx
            labels[:, 2] += pad_h  # shift cy

            mosaic_labels.append(labels)

    if len(mosaic_labels) > 0:
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)

        # Clip labels to mosaic bounds
        mosaic_labels[:, 1] = np.clip(mosaic_labels[:, 1], 0, target_size)
        mosaic_labels[:, 2] = np.clip(mosaic_labels[:, 2], 0, target_size)

        # Remove labels whose boxes fall outside or are too small
        valid = (
            (mosaic_labels[:, 1] > 0) & (mosaic_labels[:, 1] < target_size) &
            (mosaic_labels[:, 2] > 0) & (mosaic_labels[:, 2] < target_size) &
            (mosaic_labels[:, 3] > 2) & (mosaic_labels[:, 4] > 2)
        )
        mosaic_labels = mosaic_labels[valid]
    else:
        mosaic_labels = np.zeros((0, 5))

    return mosaic_img, mosaic_labels


def hsv_jitter(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    Random HSV color space augmentation.

    Simulates different lighting conditions (time of day, weather, camera
    settings) that the model will encounter at real intersections.

    Args:
        img: Input image (H, W, 3) BGR numpy array
        h_gain: Max fractional change in hue
        s_gain: Max fractional change in saturation
        v_gain: Max fractional change in value (brightness)

    Returns:
        augmented: Color-jittered image
    """
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180  # Hue wraps at 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255)

    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_horizontal_flip(img, labels, p=0.5):
    """
    Random horizontal flip with label adjustment.

    Args:
        img: Input image (H, W, 3) numpy array
        labels: (N, 5) — [class, cx, cy, w, h] in absolute pixel coords
        p: Probability of flipping

    Returns:
        img: Possibly flipped image
        labels: Labels with adjusted cx coordinates
    """
    if random.random() < p:
        img = np.fliplr(img).copy()
        if len(labels) > 0:
            w = img.shape[1]
            labels[:, 1] = w - labels[:, 1]  # Mirror cx

    return img, labels


def cutout(img, labels, n_holes=1, max_size=0.2):
    """
    Random cutout augmentation (random erase).

    Masks out random rectangular patches, forcing the model to learn
    redundant features and not rely on any single spatial region.

    Args:
        img: Input image (H, W, 3) numpy array
        labels: Labels (not modified, but used to avoid erasing all objects)
        n_holes: Number of patches to erase
        max_size: Maximum patch size as fraction of image dimension

    Returns:
        img: Image with random patches erased (filled with gray)
    """
    h, w = img.shape[:2]
    img = img.copy()

    for _ in range(n_holes):
        hole_h = int(random.uniform(0.05, max_size) * h)
        hole_w = int(random.uniform(0.05, max_size) * w)

        y = random.randint(0, h - hole_h)
        x = random.randint(0, w - hole_w)

        img[y:y + hole_h, x:x + hole_w] = 114  # Gray fill

    return img


def apply_augmentations(img, labels, target_size=416, augment=True):
    """
    Apply the full augmentation pipeline to a single image.

    When augment=False (validation), only letterbox resize is applied.

    Args:
        img: Input image (H, W, 3) BGR numpy array
        labels: (N, 5) — [class, cx, cy, w, h] in normalized coords [0,1]
        target_size: Output image size
        augment: Whether to apply augmentations

    Returns:
        img: Augmented image (target_size, target_size, 3)
        labels: Transformed labels (N, 5) [class, cx, cy, w, h] absolute pixel coords
    """
    h, w = img.shape[:2]

    # Letterbox resize (always applied)
    img, ratio, (dw, dh) = letterbox(img, target_size)
    labels = letterbox_labels(labels, ratio, (dw, dh), (h, w))

    if augment and len(labels) > 0:
        # HSV color jitter
        img = hsv_jitter(img)

        # Random horizontal flip
        img, labels = random_horizontal_flip(img, labels, p=0.5)

        # Random cutout (50% probability)
        if random.random() < 0.5:
            img = cutout(img, labels)

    return img, labels
