"""
models/srcnn.py
---------------
SRCNN-based Super-Resolution model for SAR imagery.

Architecture inspired by "Learning a Deep Convolutional Network for Image
Super-Resolution" (Dong et al., 2015), extended with:
  - Configurable depth and filter counts
  - Pre-upsampling via bicubic interpolation
  - No batch normalization (preserves SAR amplitude statistics)
  - No dropout (inappropriate for SR tasks)
  - Residual global skip connection (helps gradient flow, improves convergence)

Two variants:
  - SRCNNBaseline: Classic 3-layer SRCNN
  - SRCNNDeep:    Deeper configurable version (recommended)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv(in_ch: int, out_ch: int, k: int, pad: int = 0) -> nn.Conv2d:
    """Conv2d with no bias (following SRCNN convention)."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=True)


class ResBlock(nn.Module):
    """
    Simple residual block without BatchNorm.

    Avoids BN because it alters amplitude statistics in SAR images
    and can introduce checkerboard artifacts in SR output.
    """

    def __init__(self, n_feats: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# Classic 3-layer SRCNN baseline
# ---------------------------------------------------------------------------

class SRCNNBaseline(nn.Module):
    """
    Original SRCNN: patch extraction → non-linear mapping → reconstruction.

    Input: bicubic-upscaled LR image (pre-upsampling)
    Output: SR image with same spatial size

    Args:
        n_channels: Number of input/output channels (1 for grayscale SAR)
        f1, f2, f3: Filter counts for each layer
        k1, k2, k3: Kernel sizes (original: 9, 5, 5)
    """

    def __init__(
        self,
        n_channels: int = 1,
        f1: int = 64,
        f2: int = 32,
        f3: int = 1,
        k1: int = 9,
        k2: int = 5,
        k3: int = 5,
    ):
        super().__init__()
        self.conv1 = _conv(n_channels, f1, k1, pad=k1 // 2)
        self.conv2 = _conv(f1, f2, k2, pad=k2 // 2)
        self.conv3 = _conv(f2, n_channels, k3, pad=k3 // 2)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR tensor (B, 1, H_lr, W_lr)

        Returns:
            SR tensor (B, 1, H_hr, W_hr) — bicubic upscale applied internally
        """
        # Pre-upsampling with bicubic (classic SRCNN approach)
        _, _, h, w = x.shape
        scale = 4  # Will be overridden by SRCNNDeep; here hardcoded for baseline
        x_up = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)

        out = F.relu(self.conv1(x_up))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out + x_up  # Global residual


# ---------------------------------------------------------------------------
# Deep SRCNN (recommended)
# ---------------------------------------------------------------------------

class SRCNNDeep(nn.Module):
    """
    Deeper SRCNN with configurable residual blocks.

    Architecture:
        1. Feature extraction (input conv)
        2. N residual blocks
        3. Pixel-shuffle upsampling (sub-pixel convolution)
        4. Reconstruction conv

    Pixel-shuffle upsampling avoids checkerboard artifacts and is
    more parameter-efficient than transposed convolutions.

    Args:
        n_channels: 1 for grayscale SAR
        n_feats:    Feature map channels
        n_layers:   Number of residual blocks
        scale:      Super-resolution upscale factor
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_feats: int = 64,
        n_layers: int = 8,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        # Feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(n_channels, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Residual body
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_layers)])

        # Upsampling via pixel shuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )

        # Reconstruction
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1, bias=True)

        # Global skip (bicubic upscale of input, added to output)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR tensor (B, 1, H, W)

        Returns:
            SR tensor (B, 1, H*scale, W*scale)
        """
        # Global skip connection (bicubic reference)
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        feat = self.head(x)
        feat = self.body(feat)
        feat = self.upsample(feat)
        out = self.tail(feat)

        return out + x_up  # Residual learning w.r.t. bicubic baseline

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_srcnn(cfg) -> SRCNNDeep:
    """
    Instantiate SRCNN from config.

    Args:
        cfg: Full config DotDict

    Returns:
        SRCNNDeep model
    """
    model_cfg = cfg.srcnn
    model = SRCNNDeep(
        n_channels=model_cfg.get("n_channels", 1),
        n_feats=model_cfg.get("n_feats", 64),
        n_layers=model_cfg.get("n_layers", 8),
        scale=cfg.data.get("scale_factor", 4),
    )
    return model
