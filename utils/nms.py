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

    # Spatial Containment De-duplication:
    # If a smaller box is largely (>70%) nested inside a larger higher-scoring box
    # (e.g. windshield/wheel sub-box on a car), suppress the internal sub-box.
    if len(keep) > 1:
        s_boxes = boxes[keep]
        s_confs = conf[keep]
        s_areas = (s_boxes[:, 2] - s_boxes[:, 0]) * (s_boxes[:, 3] - s_boxes[:, 1])
        
        # Sort by confidence descending
        s_order = torch.argsort(s_confs, descending=True)
        valid_mask = torch.ones(len(keep), dtype=torch.bool, device=boxes.device)
        
        for i in range(len(s_order)):
            idx_i = s_order[i]
            if not valid_mask[idx_i]:
                continue
            box_i = s_boxes[idx_i]
            
            for j in range(i + 1, len(s_order)):
                idx_j = s_order[j]
                if not valid_mask[idx_j]:
                    continue
                box_j = s_boxes[idx_j]
                
                # Intersection
                ix1 = torch.maximum(box_i[0], box_j[0])
                iy1 = torch.maximum(box_i[1], box_j[1])
                ix2 = torch.minimum(box_i[2], box_j[2])
                iy2 = torch.minimum(box_i[3], box_j[3])
                
                iw = torch.clamp(ix2 - ix1, min=0.0)
                ih = torch.clamp(iy2 - iy1, min=0.0)
                inter = iw * ih
                
                area_j = s_areas[idx_j]
                # If smaller box j is >70% inside higher-confidence box i, suppress j
                if area_j > 0 and (inter / area_j) > 0.70:
                    valid_mask[idx_j] = False
                    
        keep = keep[valid_mask]

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
              max_detections=300, cross_class_suppress=True, cross_class_iou=0.35,
              unknown_obj_thresh=0.75, unknown_cls_thresh=0.50):
    """
    Apply NMS to a batch of predictions.

    Args:
        batch_predictions: Tensor (B, N, 5+C) from detector eval mode
        conf_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
        max_detections: Max detections per image
        cross_class_suppress: Whether to suppress duplicate multi-class overlaps
        cross_class_iou: Overlap threshold for cross-class suppression (default: 0.35)

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
