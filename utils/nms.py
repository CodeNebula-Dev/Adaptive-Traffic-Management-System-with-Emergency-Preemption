"""
Non-Maximum Suppression (NMS) for ATMS-Net Vehicle Detector.

Filters overlapping detections by keeping only the highest-confidence
prediction for each detected object. Provides both standard NMS and
class-aware NMS (which applies NMS per class independently).
"""

import torch
import torchvision


def nms(boxes, scores, iou_threshold=0.45):
    """
    Standard non-maximum suppression.

    Args:
        boxes: Tensor (N, 4) in xyxy format
        scores: Tensor (N,) confidence scores
        iou_threshold: IoU threshold above which overlapping boxes are suppressed

    Returns:
        keep: Tensor of indices to keep
    """
    return torchvision.ops.nms(boxes, scores, iou_threshold)


def class_aware_nms(predictions, conf_threshold=0.25, iou_threshold=0.45, max_detections=300):
    """
    Class-aware NMS: applies NMS per class independently.

    This prevents a high-confidence car detection from suppressing a
    nearby motorcycle detection, which would happen with class-agnostic NMS.

    Args:
        predictions: Tensor (N, 5+C) — [cx, cy, w, h, obj_conf, cls1, ..., clsC]
            Decoded predictions from the detection head (eval mode output).
        conf_threshold: Minimum confidence to consider a detection
        iou_threshold: NMS IoU threshold
        max_detections: Maximum number of detections to return

    Returns:
        detections: Tensor (M, 7) — [x1, y1, x2, y2, conf, cls_conf, cls_id]
            or None if no detections pass threshold
    """
    # Filter by objectness confidence
    obj_conf = predictions[:, 4]
    mask = obj_conf > conf_threshold
    predictions = predictions[mask]

    if predictions.shape[0] == 0:
        return None

    # Compute per-class confidence: obj_conf × cls_conf
    cls_conf, cls_id = predictions[:, 5:].max(dim=1)
    conf = predictions[:, 4] * cls_conf

    # Filter by combined confidence
    mask = conf > conf_threshold
    if mask.sum() == 0:
        return None

    predictions = predictions[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]
    cls_conf = cls_conf[mask]

    # Convert from cxcywh to xyxy
    boxes = torch.zeros_like(predictions[:, :4])
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2

    # Class-aware NMS: offset boxes by class to prevent cross-class suppression
    # Each class gets its own "space" by adding a large offset per class
    max_coord = boxes.max()
    class_offset = cls_id.float() * (max_coord + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, 0] += class_offset
    boxes_for_nms[:, 2] += class_offset

    # Apply NMS
    keep = torchvision.ops.nms(boxes_for_nms, conf, iou_threshold)

    # Limit detections
    keep = keep[:max_detections]

    # Build output: [x1, y1, x2, y2, conf, cls_conf, cls_id]
    detections = torch.cat([
        boxes[keep],
        conf[keep].unsqueeze(1),
        cls_conf[keep].unsqueeze(1),
        cls_id[keep].float().unsqueeze(1),
    ], dim=1)

    return detections


def batch_nms(batch_predictions, conf_threshold=0.25, iou_threshold=0.45, max_detections=300):
    """
    Apply class-aware NMS to a batch of predictions.

    Args:
        batch_predictions: Tensor (B, N, 5+C) from detector eval mode
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Max detections per image

    Returns:
        results: List of B tensors, each (M_i, 7) or None
    """
    results = []
    for i in range(batch_predictions.shape[0]):
        det = class_aware_nms(
            batch_predictions[i],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        results.append(det)
    return results
