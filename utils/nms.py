"""
Non-Maximum Suppression (NMS) for ATMS-Net Vehicle Detector.

Filters overlapping detections by keeping only the highest-confidence
prediction for each detected object. Provides:
    - Standard NMS
    - Class-aware NMS (evaluates NMS per class independently)
    - Cross-class suppression (eliminates duplicate multi-class boxes on same vehicle)
    - Unknown vehicle detection (flags high-objectness vehicles with ambiguous class)
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


def class_aware_nms(predictions, conf_threshold=0.25, iou_threshold=0.45,
                    max_detections=300, cross_class_suppress=True, cross_class_iou=0.50,
                    unknown_obj_thresh=0.75, unknown_cls_thresh=0.45):
    """
    Class-aware NMS with optional cross-class duplicate suppression and unknown vehicle handling.

    Args:
        predictions: Tensor (N, 5+C) — [cx, cy, w, h, obj_conf, cls1, ..., clsC]
            Decoded predictions from the detection head (eval mode output).
        conf_threshold: Minimum confidence to consider a detection
        iou_threshold: NMS IoU threshold per class
        max_detections: Maximum number of detections to return
        cross_class_suppress: If True, suppress overlapping detections of DIFFERENT classes
            on the same physical object (e.g. keeps truck if confidence > car on same box)
        cross_class_iou: Overlap threshold for cross-class suppression
        unknown_obj_thresh: Objectness threshold for detecting an unknown/unclassified vehicle
        unknown_cls_thresh: Maximum class confidence threshold below which high-objectness
            detections are categorized as unknown_vehicle

    Returns:
        detections: Tensor (M, 7) — [x1, y1, x2, y2, conf, cls_conf, cls_id]
            or None if no detections pass threshold.
            (cls_id = 4 indicates an 'unknown_vehicle')
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

    # Unknown vehicle detection:
    # High objectness (real physical vehicle) but ambiguous class score (e.g. rickshaw)
    is_unknown = (predictions[:, 4] >= unknown_obj_thresh) & (cls_conf < unknown_cls_thresh)
    if is_unknown.any():
        cls_id[is_unknown] = 4  # Class index 4 = unknown_vehicle
        conf[is_unknown] = predictions[is_unknown, 4]  # Use objectness as conf

    # Use max of obj_conf and combined conf for thresholding
    max_score = torch.maximum(predictions[:, 4], conf)
    mask = max_score > conf_threshold
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
    max_coord = boxes.max()
    class_offset = cls_id.float() * (max_coord + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, 0] += class_offset
    boxes_for_nms[:, 2] += class_offset

    # Apply intra-class NMS
    keep = torchvision.ops.nms(boxes_for_nms, conf, iou_threshold)

    # Optional cross-class duplicate suppression:
    # If two surviving boxes from different classes overlap heavily on the same object,
    # keep only the one with higher confidence score
    if cross_class_suppress and len(keep) > 1:
        surviving_boxes = boxes[keep]
        surviving_confs = conf[keep]
        cross_keep = torchvision.ops.nms(surviving_boxes, surviving_confs, cross_class_iou)
        keep = keep[cross_keep]

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


def batch_nms(batch_predictions, conf_threshold=0.25, iou_threshold=0.45,
              max_detections=300, cross_class_suppress=True, cross_class_iou=0.50):
    """
    Apply NMS to a batch of predictions.

    Args:
        batch_predictions: Tensor (B, N, 5+C) from detector eval mode
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Max detections per image
        cross_class_suppress: Whether to suppress duplicate multi-class overlaps

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
            cross_class_suppress=cross_class_suppress,
            cross_class_iou=cross_class_iou,
        )
        results.append(det)
    return results
