"""
Anchor-Free Decoupled Detection Head for ATMS-Net Vehicle Detector.

Uses a decoupled head design where classification and regression have
separate branches. This is empirically shown to improve convergence
over coupled heads (YOLOX, 2021).

For each of the 3 feature map scales, the head predicts:
    - Classification: C class probabilities per spatial cell (sigmoid)
    - Regression: 4 bounding box offsets per cell (x_off, y_off, w, h)
    - Objectness: 1 confidence score per cell (sigmoid)

The predictions are grid-relative during training and decoded to
absolute coordinates during inference using precomputed grid offsets.
"""

import torch
import torch.nn as nn
import math
from models.backbone.csp_darknet import ConvBnAct


class DecoupledHead(nn.Module):
    """
    Single-scale decoupled detection head.

    Applies to one feature map and produces classification, regression,
    and objectness predictions via separate branches.

    Architecture:
        input → shared stem (2× 3×3 conv)
            ├── cls branch → 3×3 conv → 1×1 conv → (C,) per cell
            ├── reg branch → 3×3 conv → 1×1 conv → (4,) per cell
            └── obj branch → 3×3 conv → 1×1 conv → (1,) per cell

    Args:
        in_channels: Number of input channels from the neck
        num_classes: Number of detection classes
        hidden_channels: Number of channels in the head branches (default: 256)
    """

    def __init__(self, in_channels, num_classes, hidden_channels=256):
        super().__init__()
        self.num_classes = num_classes

        # Shared stem: reduces channels and adds non-linearity
        self.stem = ConvBnAct(in_channels, hidden_channels, kernel_size=1)

        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBnAct(hidden_channels, hidden_channels, kernel_size=3),
            ConvBnAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        # Regression branch (bounding box: x_offset, y_offset, width, height)
        self.reg_conv = nn.Sequential(
            ConvBnAct(hidden_channels, hidden_channels, kernel_size=3),
            ConvBnAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.reg_pred = nn.Conv2d(hidden_channels, 4, kernel_size=1)

        # Objectness branch
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Initialize prediction layer biases
        self._init_biases()

    def _init_biases(self):
        """
        Initialize prediction layer biases for stable early training.

        Classification bias: Set so that initial sigmoid output ≈ 0.01
        (prevents early training from generating many false positives).

        Objectness bias: Same strategy.
        """
        # Prior probability for classification and objectness
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (B, C, H, W)

        Returns:
            cls_out: (B, num_classes, H, W) — raw logits, apply sigmoid for probs
            reg_out: (B, 4, H, W) — bbox offsets (x_off, y_off, w, h)
            obj_out: (B, 1, H, W) — objectness logits
        """
        x = self.stem(x)

        # Classification
        cls_feat = self.cls_conv(x)
        cls_out = self.cls_pred(cls_feat)

        # Regression + objectness share the same conv features
        reg_feat = self.reg_conv(x)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        return cls_out, reg_out, obj_out


class DetectionHead(nn.Module):
    """
    Multi-scale anchor-free detection head.

    Wraps DecoupledHead for each of the 3 feature map scales from the neck.
    Handles grid generation for decoding predictions to absolute coordinates.

    At training time: returns raw predictions for loss computation
    At inference time: decodes predictions and applies confidence thresholding

    Args:
        in_channels_list: List of input channel counts for each scale [N3_ch, F4_ch, F5_ch]
        num_classes: Number of detection classes (4 for ATMS-Net: car, truck, bus, motorcycle)
        strides: Feature map strides relative to input image
    """

    def __init__(self, in_channels_list, num_classes=4, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.num_scales = len(strides)

        # One decoupled head per scale (separate weights per scale)
        self.heads = nn.ModuleList([
            DecoupledHead(ch, num_classes)
            for ch in in_channels_list
        ])

        # Precomputed grids (lazily initialized on first forward)
        self.grids = [None] * self.num_scales

    def _make_grid(self, h, w, stride, device, dtype):
        """
        Create a grid of (x, y) coordinates for decoding box predictions.

        Each cell in the feature map corresponds to a (stride × stride) region
        in the original image. The grid stores the top-left corner of each cell
        in input-image coordinates.

        Returns:
            grid: Tensor of shape (1, 1, H, W, 2) with (x, y) coords
        """
        yv, xv = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij'
        )
        grid = torch.stack([xv, yv], dim=-1)  # (H, W, 2)
        return grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)

    def decode_predictions(self, cls_out, reg_out, obj_out, stride, grid):
        """
        Decode raw network outputs to absolute image coordinates.

        The network predicts offsets relative to grid cell centers. This
        function converts them to absolute bounding boxes:
            x_abs = (x_offset + grid_x) * stride
            y_abs = (y_offset + grid_y) * stride
            w_abs = exp(w_pred) * stride
            h_abs = exp(h_pred) * stride

        Args:
            cls_out: (B, C, H, W) — class logits
            reg_out: (B, 4, H, W) — bbox offset predictions
            obj_out: (B, 1, H, W) — objectness logits
            stride: Stride of this feature map
            grid: Precomputed (x, y) grid

        Returns:
            output: (B, H*W, 5+C) — [x, y, w, h, obj_conf, cls1, cls2, ...]
        """
        batch_size = cls_out.shape[0]
        h, w = cls_out.shape[2], cls_out.shape[3]

        # Reshape to (B, H*W, ...)
        cls_out = cls_out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        reg_out = reg_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        obj_out = obj_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)

        # Reshape grid
        grid = grid.reshape(1, -1, 2)  # (1, H*W, 2)

        # Decode center coordinates: sigmoid offset + grid position, scaled by stride
        xy = (reg_out[..., :2].sigmoid() * 2 - 0.5 + grid) * stride

        # Decode width/height: exp of prediction, scaled by stride
        wh = (reg_out[..., 2:4].sigmoid() * 2) ** 2 * stride

        # Apply sigmoid to objectness and class predictions
        obj_conf = obj_out.sigmoid()
        cls_conf = cls_out.sigmoid()

        # Concatenate: [cx, cy, w, h, obj_conf, cls_conf_1, ..., cls_conf_C]
        output = torch.cat([xy, wh, obj_conf, cls_conf], dim=-1)

        return output

    def forward(self, features, targets=None):
        """
        Multi-scale detection forward pass.

        Args:
            features: List of 3 feature maps from neck [(B,C3,H3,W3), (B,C4,H4,W4), (B,C5,H5,W5)]
            targets: Ground truth targets (only used to determine training mode)

        Returns:
            If training: dict with raw outputs per scale for loss computation
                {
                    'cls': [(B, C, H, W), ...],
                    'reg': [(B, 4, H, W), ...],
                    'obj': [(B, 1, H, W), ...],
                    'strides': [8, 16, 32]
                }
            If eval: Tensor of shape (B, total_predictions, 5+C) with decoded boxes
        """
        outputs_cls = []
        outputs_reg = []
        outputs_obj = []
        decoded_outputs = []

        for i, (feat, head) in enumerate(zip(features, self.heads)):
            cls_out, reg_out, obj_out = head(feat)
            outputs_cls.append(cls_out)
            outputs_reg.append(reg_out)
            outputs_obj.append(obj_out)

            if not self.training:
                h, w = feat.shape[2], feat.shape[3]
                stride = self.strides[i]

                # Lazily create or update grid
                if self.grids[i] is None or self.grids[i].shape[3] != h:
                    self.grids[i] = self._make_grid(
                        h, w, stride, feat.device, feat.dtype
                    )

                decoded = self.decode_predictions(
                    cls_out, reg_out, obj_out, stride, self.grids[i]
                )
                decoded_outputs.append(decoded)

        if self.training:
            return {
                'cls': outputs_cls,
                'reg': outputs_reg,
                'obj': outputs_obj,
                'strides': list(self.strides),
            }
        else:
            # Concatenate predictions from all scales
            return torch.cat(decoded_outputs, dim=1)
