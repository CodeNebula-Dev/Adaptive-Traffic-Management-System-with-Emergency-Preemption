"""
FPN + PANet Neck for ATMS-Net Vehicle Detector.

Implements bidirectional feature fusion:
    1. Top-down pathway (FPN): Propagates semantic information from deep layers
       to shallow layers via upsampling + concatenation + CSP fusion
    2. Bottom-up pathway (PANet): Propagates localisation-strong features from
       shallow layers back to deep layers via strided convolution + concatenation

This is critical for vehicle detection at intersections because vehicles near
the camera appear large (need fine-grained features from P3) while vehicles far
away appear small (need strong semantic features from P5). The bidirectional
fusion ensures both types of information are available at all scales.

Input:  P3 (stride 8), P4 (stride 16), P5 (stride 32) from CSPDarknet backbone
Output: N3, F4, F5 — three fused feature maps at the same strides
"""

import torch
import torch.nn as nn
from models.backbone.csp_darknet import ConvBnAct, CSPBlock


class FPNPANet(nn.Module):
    """
    Feature Pyramid Network with Path Aggregation Network.

    Architecture:
        Top-down (FPN):
            P5 → lateral conv → upsample → concat(P4) → CSP → N4
            N4 → lateral conv → upsample → concat(P3) → CSP → N3

        Bottom-up (PANet):
            N3 → downsample conv → concat(N4) → CSP → F4
            F4 → downsample conv → concat(P5) → CSP → F5

    The CSP blocks in the neck use fewer bottlenecks than the backbone
    (typically 1) since the neck's job is fusion, not feature extraction.

    Args:
        in_channels: List of 3 channel counts from backbone [P3_ch, P4_ch, P5_ch]
        depth_mul: Depth multiplier for CSP blocks in the neck
    """

    def __init__(self, in_channels, depth_mul=0.33):
        super().__init__()
        c3, c4, c5 = in_channels  # Channel counts for P3, P4, P5

        neck_depth = max(round(1 * depth_mul), 1)  # Typically 1 bottleneck

        # ---- Top-down pathway (FPN) ----

        # P5 lateral: reduce channels before upsampling
        self.lateral_p5 = ConvBnAct(c5, c4, kernel_size=1)

        # After concat with P4: c4 + c4 channels → CSP → c4 channels
        self.fpn_csp_p4 = CSPBlock(c4 * 2, c4, num_bottlenecks=neck_depth, shortcut=False)

        # N4 lateral: reduce channels before upsampling
        self.lateral_n4 = ConvBnAct(c4, c3, kernel_size=1)

        # After concat with P3: c3 + c3 channels → CSP → c3 channels
        self.fpn_csp_p3 = CSPBlock(c3 * 2, c3, num_bottlenecks=neck_depth, shortcut=False)

        # ---- Bottom-up pathway (PANet) ----

        # N3 → downsample to match N4 spatial size
        self.down_n3 = ConvBnAct(c3, c3, kernel_size=3, stride=2)

        # After concat with N4: c3 + c4 channels → CSP → c4 channels
        self.pan_csp_p4 = CSPBlock(c3 + c4, c4, num_bottlenecks=neck_depth, shortcut=False)

        # F4 → downsample to match P5 spatial size
        self.down_f4 = ConvBnAct(c4, c4, kernel_size=3, stride=2)

        # After concat with P5 lateral output: c4 + c5 channels → CSP → c5 channels
        # Note: we concat with the original P5 (before lateral), so it's c4 + c5
        self.pan_csp_p5 = CSPBlock(c4 + c5, c5, num_bottlenecks=neck_depth, shortcut=False)

        # Store output channels for the detection head
        self.out_channels = [c3, c4, c5]

        # Upsampling layer (nearest neighbor, no learnable params)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features):
        """
        Bidirectional feature fusion.

        Args:
            features: Tuple of (P3, P4, P5) from backbone

        Returns:
            Tuple of (N3, F4, F5) — fused feature maps at strides 8, 16, 32
        """
        p3, p4, p5 = features

        # ---- Top-down pathway ----
        # P5 → reduce channels → upsample → concat with P4 → CSP fuse
        p5_lateral = self.lateral_p5(p5)
        p5_up = self.upsample(p5_lateral)
        n4 = self.fpn_csp_p4(torch.cat([p5_up, p4], dim=1))

        # N4 → reduce channels → upsample → concat with P3 → CSP fuse
        n4_lateral = self.lateral_n4(n4)
        n4_up = self.upsample(n4_lateral)
        n3 = self.fpn_csp_p3(torch.cat([n4_up, p3], dim=1))

        # ---- Bottom-up pathway ----
        # N3 → downsample → concat with N4 → CSP fuse
        n3_down = self.down_n3(n3)
        f4 = self.pan_csp_p4(torch.cat([n3_down, n4], dim=1))

        # F4 → downsample → concat with P5 → CSP fuse
        f4_down = self.down_f4(f4)
        f5 = self.pan_csp_p5(torch.cat([f4_down, p5], dim=1))

        return n3, f4, f5
