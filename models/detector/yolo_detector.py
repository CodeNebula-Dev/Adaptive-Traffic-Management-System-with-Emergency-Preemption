"""
ATMS-Net Vehicle Detector — Full Model Assembly.

Composes the CSP-Darknet backbone, FPN+PANet neck, and anchor-free
detection head into a single nn.Module.

This is the Module 1 of ATMS-Net, trained from scratch in Phase 1
on MS COCO vehicle classes (car, truck, bus, motorcycle).

Usage:
    model = ATMSDetector(num_classes=4)
    # Training
    model.train()
    outputs = model(images)  # dict with raw predictions per scale
    loss = criterion(outputs, targets)

    # Inference
    model.eval()
    predictions = model(images)  # (B, N, 5+C) decoded predictions
"""

import torch
import torch.nn as nn
import yaml

from models.backbone.csp_darknet import CSPDarknet
from models.neck.fpn_panet import FPNPANet
from models.detector.detection_head import DetectionHead


class ATMSDetector(nn.Module):
    """
    ATMS-Net Custom YOLO-Style Vehicle Detector.

    End-to-end detector: raw image → bounding boxes + classes + confidence.

    Architecture:
        Input (B, 3, 416, 416)
            → CSPDarknet backbone → (P3, P4, P5) multi-scale features
            → FPNPANet neck → (N3, F4, F5) fused features
            → DetectionHead → predictions per scale

    In training mode, returns raw logits for loss computation.
    In eval mode, returns decoded absolute-coordinate predictions.

    Args:
        num_classes: Number of object classes to detect (default: 4)
        depth_mul: Depth multiplier for scaling network depth (default: 0.33)
        width_mul: Width multiplier for scaling network width (default: 0.5)
        in_channels: Input image channels (default: 3)
    """

    # Vehicle class names for ATMS-Net
    CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']

    def __init__(self, num_classes=4, depth_mul=0.33, width_mul=0.5, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: extracts multi-scale features
        self.backbone = CSPDarknet(
            in_channels=in_channels,
            depth_mul=depth_mul,
            width_mul=width_mul,
        )

        # Neck: bidirectional feature fusion
        self.neck = FPNPANet(
            in_channels=self.backbone.out_channels,
            depth_mul=depth_mul,
        )

        # Head: anchor-free detection predictions
        self.head = DetectionHead(
            in_channels_list=self.neck.out_channels,
            num_classes=num_classes,
            strides=(8, 16, 32),
        )

    def forward(self, x):
        """
        Full forward pass: image → detections.

        Args:
            x: Input tensor of shape (B, 3, H, W), values in [0, 1]

        Returns:
            Training: dict with 'cls', 'reg', 'obj' raw outputs per scale
            Eval: Tensor (B, N, 5+C) — [cx, cy, w, h, obj_conf, cls1..clsC]
        """
        # Backbone: extract multi-scale features
        features = self.backbone(x)  # (P3, P4, P5)

        # Neck: fuse features bidirectionally
        fused = self.neck(features)  # (N3, F4, F5)

        # Head: predict detections
        output = self.head(fused)

        return output

    @classmethod
    def from_config(cls, config_path):
        """
        Instantiate model from a YAML config file.

        Args:
            config_path: Path to detector.yaml

        Returns:
            ATMSDetector instance configured according to the YAML
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_cfg = config.get('model', {})
        return cls(
            num_classes=model_cfg.get('num_classes', 4),
            depth_mul=model_cfg.get('depth_mul', 0.33),
            width_mul=model_cfg.get('width_mul', 0.5),
            in_channels=model_cfg.get('in_channels', 3),
        )

    def get_param_count(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self):
        """Print a summary of the model architecture."""
        total, trainable = self.get_param_count()
        print("=" * 60)
        print("ATMS-Net Vehicle Detector")
        print("=" * 60)
        print(f"  Classes:           {self.num_classes} ({', '.join(self.CLASS_NAMES[:self.num_classes])})")
        print(f"  Total params:      {total:,}")
        print(f"  Trainable params:  {trainable:,}")
        print(f"  Size (MB):         {total * 4 / 1024**2:.1f}")
        print("-" * 60)
        print(f"  Backbone channels: {self.backbone.out_channels}")
        print(f"  Neck channels:     {self.neck.out_channels}")
        print(f"  Detection strides: {self.head.strides}")
        print("=" * 60)


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * current

    The EMA model typically produces smoother and more stable predictions
    than the raw training model, especially in later epochs. Used during
    validation and inference.

    Args:
        model: The model whose parameters to track
        decay: EMA decay rate (default: 0.9999)
    """

    def __init__(self, model, decay=0.9999):
        self.ema = self._copy_model(model)
        self.ema.eval()
        self.decay = decay
        self.updates = 0

    @staticmethod
    def _copy_model(model):
        """Create a deep copy of the model for EMA."""
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model

    def update(self, model):
        """Update EMA parameters with current model parameters."""
        self.updates += 1
        # Ramp up decay from 0 to target over first few thousand updates
        d = self.decay * (1 - math.exp(-self.updates / 2000))

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, ema_param in self.ema.named_parameters():
                if name in model_params:
                    ema_param.data.mul_(d).add_(model_params[name].data, alpha=1 - d)

            model_buffers = dict(model.named_buffers())
            for name, ema_buf in self.ema.named_buffers():
                if name in model_buffers:
                    ema_buf.data.copy_(model_buffers[name].data)


import math  # for ModelEMA
