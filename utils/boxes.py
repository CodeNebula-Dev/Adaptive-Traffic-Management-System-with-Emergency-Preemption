"""
Bounding box utilities for ATMS-Net detector.

Provides format conversions, IoU computation (standard, GIoU, DIoU, CIoU),
box clipping, and letterbox coordinate rescaling.

Box Formats:
    - xyxy: [x1, y1, x2, y2] — top-left and bottom-right corners
    - xywh: [x, y, w, h]     — top-left corner + width/height
    - cxcywh: [cx, cy, w, h] — center + width/height (YOLO format)

All functions operate on torch tensors with shape (..., 4).
"""

import torch
import math


def xyxy_to_xywh(boxes):
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def xywh_to_xyxy(boxes):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_cxcywh(boxes):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1
    ], dim=-1)


def cxcywh_to_xyxy(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ], dim=-1)


def clip_boxes(boxes, img_shape):
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        img_shape: (height, width) tuple
    """
    boxes[..., 0].clamp_(0, img_shape[1])  # x1
    boxes[..., 1].clamp_(0, img_shape[0])  # y1
    boxes[..., 2].clamp_(0, img_shape[1])  # x2
    boxes[..., 3].clamp_(0, img_shape[0])  # y2
    return boxes


def box_area(boxes):
    """Compute area of boxes in xyxy format. Shape: (N, 4) -> (N,)."""
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)

    Returns:
        iou: Tensor of shape (N, M) with pairwise IoU values
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Intersection coordinates
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # (N, M)
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-7)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Compute IoU and its variants between box1 and box2.

    This is the per-element version used in loss computation — both inputs
    must have the same shape (N, 4).

    Args:
        box1: Tensor of shape (N, 4)
        box2: Tensor of shape (N, 4)
        x1y1x2y2: If True, inputs are in xyxy format. If False, cxcywh.
        GIoU, DIoU, CIoU: Enable respective IoU variant

    Returns:
        iou: Tensor of shape (N,)
    """
    if not x1y1x2y2:
        # Convert from center format to corner format
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area + eps

    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        # Smallest enclosing box
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)

        if GIoU:
            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + eps
            return iou - (enclose_area - union_area) / enclose_area

        # Diagonal distance of enclosing box
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        # Center distance
        rho2 = (
            ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2
            + ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)) ** 2
        ) / 4

        if DIoU:
            return iou - rho2 / c2

        if CIoU:
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
            v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
            with torch.no_grad():
                alpha = v / (1 - iou + v + eps)
            return iou - (rho2 / c2 + v * alpha)

    return iou


def rescale_boxes(boxes, orig_shape, target_shape):
    """
    Rescale boxes from target_shape coordinates back to orig_shape.
    Used to undo letterbox padding during inference.

    Args:
        boxes: Tensor (N, 4) in xyxy format, in target_shape coords
        orig_shape: (orig_h, orig_w)
        target_shape: (target_h, target_w)
    """
    gain = min(target_shape[0] / orig_shape[0], target_shape[1] / orig_shape[1])
    pad_x = (target_shape[1] - orig_shape[1] * gain) / 2
    pad_y = (target_shape[0] - orig_shape[0] * gain) / 2

    boxes[..., 0] -= pad_x  # x1
    boxes[..., 2] -= pad_x  # x2
    boxes[..., 1] -= pad_y  # y1
    boxes[..., 3] -= pad_y  # y2
    boxes[..., :4] /= gain

    return clip_boxes(boxes, orig_shape)
