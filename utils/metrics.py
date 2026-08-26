"""
Evaluation Metrics for ATMS-Net Vehicle Detector.

Computes:
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: COCO-style mAP (averaged over 0.5 to 0.95 in 0.05 steps)
    - Per-class AP: Individual Average Precision per vehicle class
    - Precision and Recall at various thresholds
"""

import torch
import numpy as np
from utils.boxes import box_iou


class DetectionMetrics:
    """
    Accumulates detection results across batches and computes mAP.

    Usage:
        metrics = DetectionMetrics(num_classes=4)

        for images, targets in val_loader:
            predictions = model(images)
            detections = batch_nms(predictions, ...)
            metrics.update(detections, targets)

        results = metrics.compute()
        print(f"mAP@0.5: {results['mAP50']:.4f}")
    """

    CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']

    def __init__(self, num_classes=4, iou_thresholds=None):
        """
        Args:
            num_classes: Number of object classes
            iou_thresholds: List of IoU thresholds for mAP computation
                Default: [0.5] for mAP@0.5, [0.5:0.05:0.95] for COCO mAP
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5]

        # Accumulate per-image results: list of dicts
        self.all_detections = []  # List[Tensor(M, 7)]  [x1,y1,x2,y2,conf,cls_conf,cls_id]
        self.all_targets = []     # List[Tensor(N, 5)]   [cls_id, x1,y1,x2,y2]

    def reset(self):
        """Reset accumulated statistics."""
        self.all_detections.clear()
        self.all_targets.clear()

    def update(self, detections, targets, img_size=None):
        """
        Add a batch of detection results.

        Args:
            detections: List of B tensors (M_i, 7) or None from batch_nms
            targets: Tensor (N_total, 6) [batch_idx, cls, cx, cy, w, h]
                in absolute pixel coordinates
        """
        batch_size = len(detections)

        for batch_idx in range(batch_size):
            det = detections[batch_idx]

            # Get targets for this image
            img_targets = targets[targets[:, 0] == batch_idx]

            if img_targets.shape[0] > 0:
                # Convert targets from cxcywh to xyxy
                gt_cls = img_targets[:, 1]
                gt_cx, gt_cy, gt_w, gt_h = img_targets[:, 2], img_targets[:, 3], img_targets[:, 4], img_targets[:, 5]
                gt_boxes = torch.stack([
                    gt_cx - gt_w / 2, gt_cy - gt_h / 2,
                    gt_cx + gt_w / 2, gt_cy + gt_h / 2,
                ], dim=1)
                gt = torch.cat([gt_cls.unsqueeze(1), gt_boxes], dim=1)  # (N, 5)
            else:
                gt = torch.zeros((0, 5))

            if det is not None:
                det = det.cpu()

            self.all_detections.append(det)
            self.all_targets.append(gt)

    def compute(self, skip_coco_map=False):
        """
        Compute mAP and per-class AP.

        Args:
            skip_coco_map: If True, skip the expensive mAP@0.5:0.95 computation
                (useful during early training when detections are noisy and plentiful)

        Returns:
            dict with keys:
                'mAP50': float — mAP at IoU=0.5
                'mAP50_95': float — COCO-style mAP (0.0 if skipped)
                'per_class_ap50': dict mapping class_name → AP@0.5
        """
        # Compute AP at IoU=0.5
        ap50_per_class = self._compute_ap_at_threshold(0.5)

        # Compute COCO-style mAP (average over 0.5 to 0.95)
        # Skip during early epochs to avoid 10x slowdown on noisy detections
        if skip_coco_map:
            mAP50_95 = 0.0
        else:
            coco_thresholds = np.arange(0.5, 1.0, 0.05)
            ap_coco = []
            for thresh in coco_thresholds:
                ap_at_thresh = self._compute_ap_at_threshold(thresh)
                ap_coco.append(np.mean(list(ap_at_thresh.values())))
            mAP50_95 = np.mean(ap_coco)

        mAP50 = np.mean(list(ap50_per_class.values()))

        # Per-class results
        per_class = {}
        for cls_idx in range(self.num_classes):
            name = self.CLASS_NAMES[cls_idx] if cls_idx < len(self.CLASS_NAMES) else f'class_{cls_idx}'
            per_class[name] = ap50_per_class.get(cls_idx, 0.0)

        return {
            'mAP50': float(mAP50),
            'mAP50_95': float(mAP50_95),
            'per_class_ap50': per_class,
        }

    def _compute_ap_at_threshold(self, iou_threshold):
        """
        Compute per-class AP at a specific IoU threshold.

        Uses vectorized IoU computation: computes the full (D, G) IoU matrix
        in one call per image instead of looping per detection.

        Returns:
            dict mapping class_idx → AP value
        """
        ap_per_class = {}

        for cls_idx in range(self.num_classes):
            # Gather all predictions and GTs for this class
            all_pred_conf = []
            all_pred_tp = []
            total_gt = 0

            for img_idx in range(len(self.all_detections)):
                det = self.all_detections[img_idx]
                gt = self.all_targets[img_idx]

                # Count GT boxes of this class
                if gt.shape[0] > 0:
                    gt_cls_mask = gt[:, 0] == cls_idx
                    gt_boxes_cls = gt[gt_cls_mask, 1:5]
                    n_gt = gt_boxes_cls.shape[0]
                else:
                    gt_boxes_cls = torch.zeros((0, 4))
                    n_gt = 0
                total_gt += n_gt

                if det is None or det.shape[0] == 0:
                    continue

                # Filter detections of this class
                det_cls_mask = det[:, 6] == cls_idx
                det_cls = det[det_cls_mask]

                if det_cls.shape[0] == 0:
                    continue

                # Sort by confidence (descending)
                sorted_idx = det_cls[:, 4].argsort(descending=True)
                det_cls = det_cls[sorted_idx]

                confs = det_cls[:, 4].tolist()
                all_pred_conf.extend(confs)

                if n_gt == 0:
                    # All detections are false positives
                    all_pred_tp.extend([0] * det_cls.shape[0])
                    continue

                # Vectorized IoU: compute full (D, G) matrix in one call
                iou_matrix = box_iou(det_cls[:, :4], gt_boxes_cls)  # (D, G)

                # Greedy matching: iterate detections (sorted by conf) and
                # assign each to its best unmatched GT
                gt_matched = torch.zeros(n_gt, dtype=torch.bool)

                for d_idx in range(det_cls.shape[0]):
                    ious = iou_matrix[d_idx]  # (G,) — precomputed, no box_iou call

                    # Mask already-matched GTs
                    ious_available = ious.clone()
                    ious_available[gt_matched] = 0.0

                    best_iou, best_gt = ious_available.max(dim=0)

                    if best_iou >= iou_threshold:
                        all_pred_tp.append(1)
                        gt_matched[best_gt] = True
                    else:
                        all_pred_tp.append(0)

            # Compute AP from precision-recall curve
            if total_gt == 0:
                ap_per_class[cls_idx] = 0.0
                continue

            if len(all_pred_conf) == 0:
                ap_per_class[cls_idx] = 0.0
                continue

            # Sort all predictions by confidence
            sorted_indices = np.argsort(-np.array(all_pred_conf))
            tp = np.array(all_pred_tp)[sorted_indices]

            # Cumulative TP and FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(1 - tp)

            precision = cum_tp / (cum_tp + cum_fp + 1e-7)
            recall = cum_tp / (total_gt + 1e-7)

            # Compute AP using all-point interpolation (COCO style)
            ap = self._compute_ap_from_pr(precision, recall)
            ap_per_class[cls_idx] = float(ap)

        return ap_per_class

    @staticmethod
    def _compute_ap_from_pr(precision, recall):
        """
        Compute Average Precision from precision-recall curve.

        Uses the all-point interpolation method (COCO style):
        the precision at each recall level is the maximum precision
        at any recall level ≥ that recall.
        """
        # Prepend sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the envelope (monotonically decreasing precision)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Find points where recall changes
        recall_changes = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum (Δrecall × precision) at those points
        ap = np.sum((mrec[recall_changes + 1] - mrec[recall_changes]) * mpre[recall_changes + 1])

        return ap
