"""
evaluation/losses.py
--------------------
Loss functions for SRCNN and GAN training.

SRCNN losses:
  - MSELoss
  - L1Loss
  - TV loss (optional, configurable weight)

GAN losses:
  - Adversarial loss (LSGAN: MSE on discriminator output, more stable)
  - Content / pixel loss (L1, dominant for SAR to avoid hallucinations)
  - TV loss (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Total Variation Loss
# ---------------------------------------------------------------------------

class TVLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness.

    Penalizes rapid pixel-to-pixel variation, gently suppressing
    checkerboard artifacts without blurring meaningful SAR edges.

    Weight should be kept very small (e.g., 1e-6 to 1e-5).
    """

    def __init__(self, weight: float = 1e-6):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight == 0:
            return torch.tensor(0.0, device=x.device)
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return self.weight * (diff_h.abs().mean() + diff_w.abs().mean())


# ---------------------------------------------------------------------------
# SRCNN Loss
# ---------------------------------------------------------------------------

class SRCNNLoss(nn.Module):
    """
    Combined loss for SRCNN training.

    = pixel_loss + tv_loss

    Args:
        loss_type:     "l1" or "mse"
        tv_loss_weight: Weight for TV regularization (0 disables it)
    """

    def __init__(self, loss_type: str = "l1", tv_loss_weight: float = 0.0):
        super().__init__()
        self.pixel_loss = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
        self.tv_loss = TVLoss(tv_loss_weight)
        self.loss_type = loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        """
        Args:
            pred:   SR output (B, C, H, W)
            target: HR ground truth (B, C, H, W)

        Returns:
            (total_loss, {component_losses})
        """
        pixel = self.pixel_loss(pred, target)
        tv = self.tv_loss(pred)
        total = pixel + tv
        return total, {"pixel_loss": pixel.item(), "tv_loss": tv.item()}


# ---------------------------------------------------------------------------
# GAN Losses
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    """
    Least Squares GAN adversarial loss (Mao et al., 2017).

    More stable than vanilla GAN; avoids vanishing gradients.
    Real labels = 1.0, Fake labels = 0.0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


class GeneratorLoss(nn.Module):
    """
    Total generator loss:
        L_G = w_pixel * L_pixel + w_adv * L_adv + L_tv

    SAR-specific: L_pixel (L1) dominates to preserve true structure.
    Low adversarial weight prevents hallucination of speckle features.

    Args:
        pixel_weight:      Weight for pixel-wise L1 content loss
        adversarial_weight: Weight for LSGAN adversarial loss (keep low: ~0.01)
        tv_weight:         Total variation regularization weight
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        adversarial_weight: float = 0.01,
        tv_weight: float = 1e-6,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight

        self.pixel_loss = nn.L1Loss()
        self.adv_loss = LSGANLoss()
        self.tv_loss = TVLoss(tv_weight)

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> tuple:
        """
        Args:
            sr:     Generator output (B, C, H, W)
            hr:     Ground truth HR (B, C, H, W)
            d_fake: Discriminator output on fake SR images

        Returns:
            (total_loss, {component_losses})
        """
        pixel = self.pixel_loss(sr, hr)
        adv = self.adv_loss(d_fake, is_real=True)  # Generator wants D(G(x)) → 1
        tv = self.tv_loss(sr)

        total = (
            self.pixel_weight * pixel
            + self.adversarial_weight * adv
            + tv
        )
        return total, {
            "pixel_loss": pixel.item(),
            "adv_loss": adv.item(),
            "tv_loss": tv.item(),
        }


class DiscriminatorLoss(nn.Module):
    """
    Standard LSGAN discriminator loss:
        L_D = 0.5 * [L(D(HR), 1) + L(D(SR), 0)]
    """

    def __init__(self):
        super().__init__()
        self.adv_loss = LSGANLoss()

    def forward(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> tuple:
        """
        Args:
            d_real: D output on real HR images
            d_fake: D output on fake SR images (detached)

        Returns:
            (total_loss, {component_losses})
        """
        real_loss = self.adv_loss(d_real, is_real=True)
        fake_loss = self.adv_loss(d_fake, is_real=False)
        total = 0.5 * (real_loss + fake_loss)
        return total, {"d_real_loss": real_loss.item(), "d_fake_loss": fake_loss.item()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_srcnn_criterion(cfg) -> SRCNNLoss:
    train_cfg = cfg.train_srcnn
    return SRCNNLoss(
        loss_type=train_cfg.get("loss", "l1"),
        tv_loss_weight=train_cfg.get("tv_loss_weight", 0.0),
    )


def build_gan_criteria(cfg):
    """Returns (generator_criterion, discriminator_criterion)."""
    train_cfg = cfg.train_gan
    gen_criterion = GeneratorLoss(
        pixel_weight=train_cfg.get("pixel_loss_weight", 1.0),
        adversarial_weight=train_cfg.get("adversarial_loss_weight", 0.01),
        tv_weight=train_cfg.get("tv_loss_weight", 1e-6),
    )
    disc_criterion = DiscriminatorLoss()
    return gen_criterion, disc_criterion
