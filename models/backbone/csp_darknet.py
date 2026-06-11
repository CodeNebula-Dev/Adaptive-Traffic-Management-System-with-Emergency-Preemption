"""
CSP-Darknet Backbone for ATMS-Net Vehicle Detector.

Architecture: Cross-Stage Partial Network with residual blocks.
Produces multi-scale feature maps at strides 8, 16, and 32 — feeding
into the FPN+PANet neck for bidirectional feature fusion.

Key components:
    - ConvBnAct: Conv2d → BatchNorm2d → SiLU activation
    - Bottleneck: Two-conv residual block with optional skip connection
    - CSPBlock: Cross-Stage Partial block that splits gradient flow
    - SPPBlock: Spatial Pyramid Pooling for multi-scale receptive field
    - CSPDarknet: Full backbone assembling stem + 4 stages + SPP

Reference: CSPNet (Wang et al., 2020) — gradient flow splitting reduces
redundant computation while preserving feature richness.
"""

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """
    Standard Conv → BatchNorm → SiLU block.

    This is the fundamental building block used throughout the backbone.
    SiLU (Swish) activation is used instead of ReLU/LeakyReLU as it
    provides smoother gradients and slightly better convergence in
    modern detection architectures.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        groups: Number of groups for grouped convolution (default: 1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # Same padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard bottleneck residual block.

    Two ConvBnAct layers with an optional skip connection:
        input → 1×1 conv (reduce) → 3×3 conv (process) → add input → output

    The skip connection enables gradient flow through deeper networks and
    is essential for training the backbone from scratch without pretrained weights.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        shortcut: Whether to use skip connection (default: True)
        expansion: Channel expansion ratio for the hidden layer (default: 0.5)
    """

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBnAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBnAct(hidden_channels, out_channels, kernel_size=3)
        self.use_shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.use_shortcut:
            out = out + x
        return out


class CSPBlock(nn.Module):
    """
    Cross-Stage Partial Block.

    Splits input into two paths:
    - Path 1: Goes through N bottleneck blocks (learns complex features)
    - Path 2: Direct 1×1 conv (preserves gradient flow, reduces redundancy)
    Then concatenates both paths and fuses with a final 1×1 conv.

    This design halves the computation compared to a full residual stack
    while maintaining equivalent representational capacity, because only
    half the channels go through the expensive bottleneck chain.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_bottlenecks: Number of bottleneck blocks in the deep path
        shortcut: Whether bottlenecks use skip connections
    """

    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2

        # Path 1: through bottleneck chain
        self.conv1 = ConvBnAct(in_channels, hidden_channels, kernel_size=1)

        # Path 2: direct pass
        self.conv2 = ConvBnAct(in_channels, hidden_channels, kernel_size=1)

        # Bottleneck chain on path 1
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
            for _ in range(num_bottlenecks)
        ])

        # Fusion after concatenation (hidden_channels * 2 → out_channels)
        self.conv3 = ConvBnAct(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        path1 = self.bottlenecks(self.conv1(x))
        path2 = self.conv2(x)
        return self.conv3(torch.cat([path1, path2], dim=1))


class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling Block.

    Applies max-pooling at multiple kernel sizes (5, 9, 13) and concatenates
    the results with the original feature map. This expands the effective
    receptive field without adding learnable parameters, which is critical
    for detecting vehicles at varying distances from the intersection camera.

    Placed at the end of the backbone (after the last CSP stage) to capture
    both local details and global context before feature fusion in the neck.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        pool_sizes: Tuple of max-pooling kernel sizes
    """

    def __init__(self, in_channels, out_channels, pool_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2

        self.conv1 = ConvBnAct(in_channels, hidden_channels, kernel_size=1)

        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in pool_sizes
        ])

        # After concat: hidden_channels * (1 + len(pool_sizes)) → out_channels
        self.conv2 = ConvBnAct(hidden_channels * (1 + len(pool_sizes)), out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        pool_outputs = [x] + [pool(x) for pool in self.pools]
        return self.conv2(torch.cat(pool_outputs, dim=1))


class CSPDarknet(nn.Module):
    """
    CSP-Darknet Backbone.

    Full backbone that processes an input image and produces three multi-scale
    feature maps for the detection neck:
        - P3 at stride 8  (52×52 for 416 input) — detects large/close objects
        - P4 at stride 16 (26×26) — detects medium objects
        - P5 at stride 32 (13×13) — detects small/distant objects

    Architecture:
        Stem (stride 2) → Stage1 (stride 4) → Stage2 → P3 (stride 8)
        → Stage3 → P4 (stride 16) → Stage4 + SPP → P5 (stride 32)

    The depth and width of the network are controlled by multipliers to
    allow scaling the model for different compute budgets.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB)
        depth_mul: Depth multiplier — scales number of bottlenecks per CSP block
        width_mul: Width multiplier — scales number of channels per layer
    """

    # Base channel widths for each stage
    BASE_CHANNELS = [64, 128, 256, 512, 1024]
    # Number of bottleneck blocks per CSP stage
    BASE_DEPTHS = [1, 2, 3, 1]

    def __init__(self, in_channels=3, depth_mul=0.33, width_mul=0.5):
        super().__init__()

        # Scale channels and depths
        channels = [max(round(c * width_mul), 1) for c in self.BASE_CHANNELS]
        depths = [max(round(d * depth_mul), 1) for d in self.BASE_DEPTHS]

        # Stem: aggressive 6×6 stride-2 downsample
        # Input: (B, 3, 416, 416) → Output: (B, channels[0], 208, 208)
        self.stem = ConvBnAct(in_channels, channels[0], kernel_size=6, stride=2)

        # Stage 1: stride-2 downsample + CSP block
        # (B, channels[0], 208, 208) → (B, channels[1], 104, 104)
        self.stage1 = nn.Sequential(
            ConvBnAct(channels[0], channels[1], kernel_size=3, stride=2),
            CSPBlock(channels[1], channels[1], num_bottlenecks=depths[0]),
        )

        # Stage 2: → P3 output at stride 8
        # (B, channels[1], 104, 104) → (B, channels[2], 52, 52)
        self.stage2 = nn.Sequential(
            ConvBnAct(channels[1], channels[2], kernel_size=3, stride=2),
            CSPBlock(channels[2], channels[2], num_bottlenecks=depths[1]),
        )

        # Stage 3: → P4 output at stride 16
        # (B, channels[2], 52, 52) → (B, channels[3], 26, 26)
        self.stage3 = nn.Sequential(
            ConvBnAct(channels[2], channels[3], kernel_size=3, stride=2),
            CSPBlock(channels[3], channels[3], num_bottlenecks=depths[2]),
        )

        # Stage 4: → P5 output at stride 32, with SPP
        # (B, channels[3], 26, 26) → (B, channels[4], 13, 13)
        self.stage4 = nn.Sequential(
            ConvBnAct(channels[3], channels[4], kernel_size=3, stride=2),
            CSPBlock(channels[4], channels[4], num_bottlenecks=depths[3]),
            SPPBlock(channels[4], channels[4]),
        )

        # Store output channel counts for the neck to use
        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass producing three multi-scale feature maps.

        Args:
            x: Input image tensor of shape (B, 3, H, W)

        Returns:
            Tuple of (P3, P4, P5) feature maps:
                P3: (B, C2, H/8,  W/8)  — stride 8
                P4: (B, C3, H/16, W/16) — stride 16
                P5: (B, C4, H/32, W/32) — stride 32
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32
        return p3, p4, p5
