"""
Loss Functions for ATMS-Net Vehicle Detector.

Implements the combined YOLO detection loss:
    L_total = λ_box × L_CIoU + λ_obj × L_obj + λ_cls × L_cls

Components:
    - CIoU Loss: Complete IoU for bounding box regression
    - BCE Loss: Binary cross-entropy for objectness and classification
    - SimOTA: Simplified Optimal Transport Assignment for matching
      ground-truth boxes to predictions

SimOTA (from YOLOX) replaces hand-crafted assignment rules with a
dynamic top-k matching strategy that adapts to each image's content.
This is significantly better than fixed IoU-threshold assignment for
training from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.boxes import bbox_iou, cxcywh_to_xyxy


class YOLOLoss(nn.Module):
    """
    Combined YOLO detection loss with SimOTA assignment.

    The loss is computed per feature-map scale and summed. For each scale:
    1. Decode raw predictions to absolute coordinates
    2. Run SimOTA to match GT boxes → predicted cells
    3. Compute CIoU loss on matched pairs (box regression)
    4. Compute BCE loss on objectness (positive + negative cells)
    5. Compute BCE loss on class labels (positive cells only)

    Args:
        num_classes: Number of object classes
        strides: Feature map strides (default: [8, 16, 32])
        box_weight: Weight for box regression loss
        obj_weight: Weight for objectness loss
        cls_weight: Weight for classification loss
    """

    def __init__(self, num_classes=4, strides=(8, 16, 32),
                 box_weight=0.05, obj_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        """
        Compute detection loss.

        Args:
            predictions: Dict from DetectionHead.forward() in training mode:
                {
                    'cls': List of (B, C, Hi, Wi) tensors per scale
                    'reg': List of (B, 4, Hi, Wi) tensors per scale
                    'obj': List of (B, 1, Hi, Wi) tensors per scale
                    'strides': [8, 16, 32]
                }
            targets: Tensor of shape (N_total, 6) where each row is:
                [batch_idx, class_id, cx, cy, w, h]
                All coordinates are in absolute pixel values (input image scale)

        Returns:
            loss_dict: Dict with 'loss', 'box_loss', 'obj_loss', 'cls_loss'
        """
        device = predictions['cls'][0].device
        dtype = predictions['cls'][0].dtype

        total_box_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_obj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_cls_loss = torch.tensor(0.0, device=device, dtype=dtype)

        batch_size = predictions['cls'][0].shape[0]

        # Process each scale
        for scale_idx in range(len(self.strides)):
            cls_pred = predictions['cls'][scale_idx]  # (B, C, H, W)
            reg_pred = predictions['reg'][scale_idx]  # (B, 4, H, W)
            obj_pred = predictions['obj'][scale_idx]  # (B, 1, H, W)
            stride = self.strides[scale_idx]

            h, w = cls_pred.shape[2], cls_pred.shape[3]

            # Generate grid for this scale
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

            # Reshape predictions to (B, H*W, ...)
            cls_pred_flat = cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            reg_pred_flat = reg_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            obj_pred_flat = obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)

            # Decode box predictions to absolute coords for assignment
            grid_flat = grid.reshape(-1, 2)  # (H*W, 2)
            decoded_xy = (reg_pred_flat[..., :2].sigmoid() * 2 - 0.5 + grid_flat) * stride
            decoded_wh = (reg_pred_flat[..., 2:4].sigmoid() * 2) ** 2 * stride
            decoded_boxes = torch.cat([decoded_xy, decoded_wh], dim=-1)  # (B, H*W, 4) cxcywh

            # Compute per-image losses
            for batch_idx in range(batch_size):
                # Get targets for this image
                img_targets = targets[targets[:, 0] == batch_idx]

                obj_target = torch.zeros_like(obj_pred_flat[batch_idx])  # (H*W, 1)

                if img_targets.shape[0] == 0:
                    # No targets in this image — only objectness loss
                    total_obj_loss += self.bce_obj(
                        obj_pred_flat[batch_idx], obj_target
                    ).sum()
                    continue

                gt_classes = img_targets[:, 1].long()  # (N_gt,)
                gt_boxes = img_targets[:, 2:6]  # (N_gt, 4) cxcywh

                # Run SimOTA matching
                matched_pred_idx, matched_gt_idx = self._simota_matching(
                    cls_pred_flat[batch_idx].detach(),
                    obj_pred_flat[batch_idx].detach(),
                    decoded_boxes[batch_idx].detach(),
                    gt_classes,
                    gt_boxes,
                    stride,
                )

                # Set objectness targets
                if len(matched_pred_idx) > 0:
                    # Compute IoU between matched pairs for soft objectness target
                    matched_decoded = decoded_boxes[batch_idx][matched_pred_idx]
                    matched_gt = gt_boxes[matched_gt_idx]

                    iou_values = bbox_iou(
                        cxcywh_to_xyxy(matched_decoded),
                        cxcywh_to_xyxy(matched_gt),
                    ).clamp(0, 1)

                    obj_target[matched_pred_idx, 0] = iou_values.detach()

                    # Box loss (CIoU) on matched pairs
                    ciou = bbox_iou(
                        cxcywh_to_xyxy(matched_decoded),
                        cxcywh_to_xyxy(matched_gt),
                        CIoU=True,
                    )
                    total_box_loss += (1.0 - ciou).mean()

                    # Classification loss on matched pairs
                    cls_targets_onehot = torch.zeros_like(
                        cls_pred_flat[batch_idx][matched_pred_idx]
                    )
                    cls_targets_onehot[
                        torch.arange(len(matched_gt_idx)), gt_classes[matched_gt_idx]
                    ] = 1.0

                    total_cls_loss += self.bce_cls(
                        cls_pred_flat[batch_idx][matched_pred_idx],
                        cls_targets_onehot,
                    ).sum() / max(len(matched_pred_idx), 1)

                # Objectness loss (all predictions)
                total_obj_loss += self.bce_obj(
                    obj_pred_flat[batch_idx], obj_target
                ).sum() / max(h * w, 1)

        # Normalize by batch size
        num_scales = len(self.strides)
        total_box_loss /= max(batch_size, 1)
        total_obj_loss /= max(batch_size * num_scales, 1)
        total_cls_loss /= max(batch_size, 1)

        # Weighted sum
        loss = (
            self.box_weight * total_box_loss
            + self.obj_weight * total_obj_loss
            + self.cls_weight * total_cls_loss
        )

        return {
            'loss': loss,
            'box_loss': total_box_loss.detach(),
            'obj_loss': total_obj_loss.detach(),
            'cls_loss': total_cls_loss.detach(),
        }

    def _simota_matching(self, cls_preds, obj_preds, decoded_boxes,
                         gt_classes, gt_boxes, stride):
        """
        Simplified Optimal Transport Assignment (SimOTA).

        Dynamically assigns ground-truth boxes to predicted cells based on
        a cost matrix combining classification cost and IoU cost. The number
        of positive assignments per GT box is determined dynamically based on
        how many predictions have high IoU with that GT box.

        Steps:
        1. Filter candidates: only cells whose centers are within GT boxes
        2. Compute cost matrix: cls_cost + 3.0 * iou_cost
        3. Dynamic top-k: for each GT, select top-k candidates where k is
           determined by the sum of top-10 IoU values
        4. Handle conflicts: if a cell is assigned to multiple GTs, keep
           the one with lowest cost

        Args:
            cls_preds: (H*W, C) class logits
            obj_preds: (H*W, 1) objectness logits
            decoded_boxes: (H*W, 4) decoded predictions in cxcywh
            gt_classes: (N_gt,) class indices
            gt_boxes: (N_gt, 4) GT boxes in cxcywh
            stride: Current feature map stride

        Returns:
            matched_pred_idx: Tensor of matched prediction indices
            matched_gt_idx: Tensor of matched GT indices
        """
        n_gt = gt_boxes.shape[0]
        n_pred = decoded_boxes.shape[0]

        if n_gt == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Step 1: Get candidate predictions (centers within expanded GT boxes)
        # Expand GT boxes by 2.5 * stride to allow nearby predictions
        pred_centers = decoded_boxes[:, :2]  # (N_pred, 2) — cx, cy
        gt_xyxy = cxcywh_to_xyxy(gt_boxes)  # (N_gt, 4)

        # Check if prediction centers fall within GT boxes (with margin)
        margin = 2.5 * stride
        is_in_box = (
            (pred_centers[:, None, 0] > gt_xyxy[None, :, 0] - margin) &
            (pred_centers[:, None, 1] > gt_xyxy[None, :, 1] - margin) &
            (pred_centers[:, None, 0] < gt_xyxy[None, :, 2] + margin) &
            (pred_centers[:, None, 1] < gt_xyxy[None, :, 3] + margin)
        )  # (N_pred, N_gt)

        is_candidate = is_in_box.any(dim=1)  # (N_pred,)
        candidate_idx = is_candidate.nonzero(as_tuple=False).squeeze(-1)

        if candidate_idx.numel() == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Step 2: Compute cost matrix for candidates
        candidate_boxes = decoded_boxes[candidate_idx]  # (N_cand, 4)
        candidate_cls = cls_preds[candidate_idx]  # (N_cand, C)
        candidate_obj = obj_preds[candidate_idx]  # (N_cand, 1)

        # IoU between candidates and all GT boxes
        pair_iou = self._pairwise_iou(candidate_boxes, gt_boxes)  # (N_cand, N_gt)

        # Classification cost
        gt_onehot = F.one_hot(gt_classes, self.num_classes).float()  # (N_gt, C)
        cls_prob = (candidate_cls.sigmoid() * candidate_obj.sigmoid())  # (N_cand, C)

        cls_cost = F.binary_cross_entropy(
            cls_prob.unsqueeze(1).expand(-1, n_gt, -1),
            gt_onehot.unsqueeze(0).expand(candidate_idx.numel(), -1, -1),
            reduction='none',
        ).sum(-1)  # (N_cand, N_gt)

        # Combined cost
        cost_matrix = cls_cost + 3.0 * (1.0 - pair_iou)

        # Step 3: Dynamic top-k per GT
        matching_matrix = torch.zeros_like(cost_matrix)  # (N_cand, N_gt)

        for gt_idx in range(n_gt):
            # Determine k from IoU distribution
            topk_iou, _ = torch.topk(pair_iou[:, gt_idx], min(10, pair_iou.shape[0]))
            dynamic_k = max(int(topk_iou.sum().item()), 1)
            dynamic_k = min(dynamic_k, candidate_idx.numel())

            # Select top-k lowest cost candidates
            _, topk_indices = torch.topk(cost_matrix[:, gt_idx], dynamic_k, largest=False)
            matching_matrix[topk_indices, gt_idx] = 1.0

        # Step 4: Handle conflicts (cell matched to multiple GTs)
        matched_gt_per_pred = matching_matrix.sum(dim=1)  # (N_cand,)
        conflict_mask = matched_gt_per_pred > 1

        if conflict_mask.any():
            # For conflicting cells, keep only the GT with minimum cost
            conflict_costs = cost_matrix[conflict_mask]  # (N_conflict, N_gt)
            _, min_cost_gt = conflict_costs.min(dim=1)

            matching_matrix[conflict_mask] = 0
            matching_matrix[conflict_mask, min_cost_gt] = 1.0

        # Extract matched pairs
        matched_mask = matching_matrix.sum(dim=1) > 0  # (N_cand,)
        matched_pred_local = matched_mask.nonzero(as_tuple=False).squeeze(-1)

        if matched_pred_local.numel() == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        matched_pred_idx = candidate_idx[matched_pred_local]
        matched_gt_idx = matching_matrix[matched_pred_local].argmax(dim=1)

        return matched_pred_idx, matched_gt_idx

    @staticmethod
    def _pairwise_iou(boxes1, boxes2):
        """
        Compute pairwise IoU between two sets of boxes in cxcywh format.

        Args:
            boxes1: (N, 4) in cxcywh
            boxes2: (M, 4) in cxcywh

        Returns:
            iou: (N, M) pairwise IoU matrix
        """
        b1 = cxcywh_to_xyxy(boxes1)  # (N, 4)
        b2 = cxcywh_to_xyxy(boxes2)  # (M, 4)

        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)

        inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union_area = area1[:, None] + area2[None, :] - inter_area

        return inter_area / (union_area + 1e-7)
