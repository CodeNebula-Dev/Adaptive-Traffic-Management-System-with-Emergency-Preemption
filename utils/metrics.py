"""
Evaluation Metrics for ATMS-Net Vehicle Detector.

Computes:
    - mAP@0.5: Mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95: COCO-style mAP (averaged over 0.5 to 0.95 in 0.05 steps)
    - Per-class AP: Individual Average Precision per vehicle class
"""

import torch
import numpy as np
from utils.boxes import box_iou


class DetectionMetrics:
    """
    Accumulates detection results across batches and computes mAP.
    Uses fully vectorized matching across all 10 IoU thresholds (0.5:0.95)
    for sub-second evaluation even with 200,000+ candidate detections.

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

    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.iouv = torch.linspace(0.5, 0.95, 10)  # 10 IoU thresholds for COCO evaluation

        # Accumulate per-image results
        self.all_detections = []  # List[Tensor(M, 7)] [x1,y1,x2,y2,conf,cls_conf,cls_id]
        self.all_targets = []     # List[Tensor(N, 5)] [cls_id, x1,y1,x2,y2]

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
        Compute mAP@0.5 and mAP@0.5:0.95 using vectorized matching.

        Returns:
            dict with keys:
                'mAP50': float — mAP at IoU=0.5
                'mAP50_95': float — COCO-style mAP
                'per_class_ap50': dict mapping class_name → AP@0.5
        """
        tp_list = []
        conf_list = []
        pred_cls_list = []
        target_cls_list = []

        iouv = self.iouv if not skip_coco_map else self.iouv[:1]  # [0.5] if skip_coco_map

        for img_idx in range(len(self.all_detections)):
            det = self.all_detections[img_idx]
            gt = self.all_targets[img_idx]

            if gt.shape[0] > 0:
                target_cls_list.append(gt[:, 0].numpy())

            if det is not None and det.shape[0] > 0:
                # Sort detections by confidence descending
                sorted_idx = det[:, 4].argsort(descending=True)
                det = det[sorted_idx]

                # Match detections to GT across all IoU thresholds
                tp = self._process_image(det, gt, iouv)
                tp_list.append(tp.numpy())
                conf_list.append(det[:, 4].numpy())
                pred_cls_list.append(det[:, 6].numpy())

        if len(tp_list) == 0 or len(target_cls_list) == 0:
            per_class = {self.CLASS_NAMES[i]: 0.0 for i in range(self.num_classes)}
            return {'mAP50': 0.0, 'mAP50_95': 0.0, 'per_class_ap50': per_class}

        tp = np.concatenate(tp_list, axis=0)
        conf = np.concatenate(conf_list, axis=0)
        pred_cls = np.concatenate(pred_cls_list, axis=0)
        target_cls = np.concatenate(target_cls_list, axis=0)

        # Compute AP for each class and IoU threshold
        ap_matrix = self._compute_ap_matrix(tp, conf, pred_cls, target_cls, self.num_classes)

        # AP@0.5 is index 0
        ap50 = ap_matrix[:, 0]
        mAP50 = float(np.mean(ap50))

        if skip_coco_map:
            mAP50_95 = 0.0
        else:
            mAP50_95 = float(np.mean(ap_matrix))

        per_class = {}
        for cls_idx in range(self.num_classes):
            name = self.CLASS_NAMES[cls_idx] if cls_idx < len(self.CLASS_NAMES) else f'class_{cls_idx}'
            per_class[name] = float(ap50[cls_idx])

        return {
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'per_class_ap50': per_class,
        }

    @staticmethod
    def _process_image(detections, labels, iouv):
        """
        Vectorized greedy matching of detections to ground truth boxes.
        Computes TP matrix of shape (num_detections, num_iou_thresholds).
        """
        num_dets = detections.shape[0]
        num_thresh = len(iouv)
        correct = np.zeros((num_dets, num_thresh), dtype=bool)

        if labels.shape[0] == 0:
            return torch.from_numpy(correct)

        # Compute pairwise IoU: (M_gt, N_det)
        iou = box_iou(labels[:, 1:5], detections[:, :4])
        # Class match mask: (M_gt, N_det)
        correct_class = (labels[:, 0:1] == detections[:, 6:7].T)

        for i, threshold in enumerate(iouv):
            # Candidate pairs where IoU >= threshold and class matches
            x = torch.where((iou >= threshold) & correct_class)
            if x[0].shape[0] > 0:
                # [gt_idx, det_idx, iou_val]
                matches = torch.cat((torch.stack(x, dim=1), iou[x[0], x[1]][:, None]), dim=1).numpy()
                if matches.shape[0] > 1:
                    # Sort by IoU descending
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    # 1 unique match per detection
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # 1 unique match per ground truth
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

        return torch.from_numpy(correct)

    @staticmethod
    def _compute_ap_matrix(tp, conf, pred_cls, target_cls, num_classes):
        """
        Compute AP for all classes across all IoU levels using standard interpolation.
        """
        # Sort all predictions across dataset by confidence descending
        sort_idx = np.argsort(-conf)
        tp = tp[sort_idx]
        pred_cls = pred_cls[sort_idx]

        num_thresh = tp.shape[1]
        ap = np.zeros((num_classes, num_thresh), dtype=np.float32)

        for ci in range(num_classes):
            cls_mask = (pred_cls == ci)
            n_gt = (target_cls == ci).sum()
            n_pred = cls_mask.sum()

            if n_pred == 0 or n_gt == 0:
                continue

            # Cumulative TP and FP for class ci
            tp_cls = tp[cls_mask]
            tpc = tp_cls.cumsum(axis=0)
            fpc = (1 - tp_cls).cumsum(axis=0)

            recall = tpc / (n_gt + 1e-16)
            precision = tpc / (tpc + fpc)

            for j in range(num_thresh):
                # Continuous interpolation (COCO style)
                mrec = np.concatenate(([0.0], recall[:, j], [1.0]))
                mpre = np.concatenate(([1.0], precision[:, j], [0.0]))

                # Monotonically decreasing precision envelope
                for k in range(len(mpre) - 1, 0, -1):
                    mpre[k - 1] = max(mpre[k - 1], mpre[k])

                # Integrate area under PR curve
                indices = np.where(mrec[1:] != mrec[:-1])[0]
                ap[ci, j] = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

        return ap
