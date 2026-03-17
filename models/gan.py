"""
models/gan.py
-------------
Lightweight SRGAN-style model for SAR super-resolution.

Design choices for SAR / small dataset:
  - Generator uses residual blocks without BN (avoids amplitude distortion)
  - Discriminator is a PatchGAN (evaluates local patches, less prone to
    global mode collapse, more efficient)
  - Adversarial weight kept very low (0.01) to avoid hallucinating SAR features
  - Content loss (L1) is the dominant training signal

Generator architecture: ResNet-style (modified SRGAN generator)
Discriminator: PatchGAN discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Generator building blocks (no BN for SAR)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block without BatchNorm."""

    def __init__(self, n_feats: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution upsampling (avoids checkerboard)."""

    def __init__(self, n_feats: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1, bias=True)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SRGenerator(nn.Module):
    """
    SRGAN-style generator for SAR super-resolution.

    Architecture:
        Head → N residual blocks → long skip → upsampling → tail

    Args:
        n_channels:  1 for grayscale SAR
        n_feats:     Feature map channels
        n_res_blocks: Number of residual blocks in body
        scale:       SR upscale factor (2, 4)
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_feats: int = 64,
        n_res_blocks: int = 8,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(n_channels, n_feats, 9, padding=4, bias=True),
            nn.PReLU(n_feats),
        )

        # Body: residual blocks
        self.body = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_res_blocks)])

        # Body tail (before long skip addition)
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)

        # Upsampling
        # For scale=4: two 2x upsample blocks; for scale=2: one 2x block
        upsample_blocks = []
        if scale == 4:
            upsample_blocks += [UpsampleBlock(n_feats, 2), UpsampleBlock(n_feats, 2)]
        elif scale == 2:
            upsample_blocks += [UpsampleBlock(n_feats, 2)]
        elif scale == 8:
            upsample_blocks += [UpsampleBlock(n_feats, 2)] * 3
        else:
            raise ValueError(f"Unsupported scale: {scale}. Use 2, 4, or 8.")
        self.upsample = nn.Sequential(*upsample_blocks)

        # Reconstruction
        self.tail = nn.Conv2d(n_feats, n_channels, 9, padding=4, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR tensor (B, 1, H, W)

        Returns:
            SR tensor (B, 1, H*scale, W*scale)
        """
        # Global skip (bicubic reference)
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        head_out = self.head(x)
        body_out = self.body_tail(self.body(head_out)) + head_out  # Long skip
        sr = self.tail(self.upsample(body_out))

        return sr + x_up  # Residual w.r.t. bicubic

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Classifies 70×70 (receptive-field) patches as real or fake.
    More stable than global discriminator for small datasets.

    Args:
        n_channels:   Input channels (1 for grayscale SAR)
        n_feats:      Base feature count (doubled each layer)
        n_layers:     Number of convolutional layers (default 3)
    """

    def __init__(self, n_channels: int = 1, n_feats: int = 64, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []

        # First layer: no instance norm
        layers += [
            nn.Conv2d(n_channels, n_feats, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(n_feats * mult_prev, n_feats * mult, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(n_feats * mult),  # Instance norm: stable without BN
                nn.LeakyReLU(0.2, inplace=True),
            ]

        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(n_feats * mult_prev, n_feats * mult, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(n_feats * mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output: 1-channel map (real/fake per patch)
        layers += [nn.Conv2d(n_feats * mult, 1, 4, stride=1, padding=1, bias=True)]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W)

        Returns:
            Patch-wise real/fake scores (B, 1, H', W')
        """
        return self.model(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gan(cfg):
    """
    Build Generator and Discriminator from config.

    Returns:
        (generator, discriminator)
    """
    gan_cfg = cfg.gan
    generator = SRGenerator(
        n_channels=gan_cfg.get("n_channels", 1),
        n_feats=gan_cfg.get("n_feats", 64),
        n_res_blocks=gan_cfg.get("n_res_blocks", 8),
        scale=cfg.data.get("scale_factor", 4),
    )
    discriminator = PatchDiscriminator(
        n_channels=gan_cfg.get("n_channels", 1),
        n_feats=gan_cfg.get("discriminator_feats", 64),
    )
    return generator, discriminator
