"""
train/train_gan.py
------------------
Training pipeline for the GAN super-resolution model (SRGAN-style).

Training protocol for SAR / small datasets:
  1. Warmup phase: Train generator only with L1 loss (no adversarial)
     → Establishes a stable initialization before the discriminator is added
  2. Adversarial phase: Jointly train G and D
     → Very low adversarial weight (0.01) keeps content loss dominant

Key design decisions:
  - Alternating G/D updates (1:1 ratio)
  - Gradient penalty is not used (lightweight design for limited compute)
  - LSGAN loss for discriminator stability
  - Early stopping on val PSNR
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from models.gan import SRGenerator, PatchDiscriminator, build_gan
from evaluation.losses import build_gan_criteria
from evaluation.metrics import compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, EarlyStopping
from utils.logger import get_logger, TrainingLogger
from utils.device import get_device


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-step helpers
# ---------------------------------------------------------------------------

def _generator_step(
    generator: SRGenerator,
    discriminator: PatchDiscriminator,
    gen_criterion,
    g_optimizer: torch.optim.Optimizer,
    lr: torch.Tensor,
    hr: torch.Tensor,
    adversarial: bool,
    grad_clip: float = 1.0,
) -> tuple:
    """One generator update step."""
    generator.train()
    discriminator.eval()  # Freeze D stats during G update

    g_optimizer.zero_grad()
    sr = generator(lr)

    if adversarial:
        d_fake = discriminator(sr)
        g_loss, loss_dict = gen_criterion(sr, hr, d_fake)
    else:
        # Warmup: pixel loss only
        pixel_loss = nn.L1Loss()(sr, hr)
        g_loss = pixel_loss
        loss_dict = {"pixel_loss": pixel_loss.item(), "adv_loss": 0.0, "tv_loss": 0.0}

    g_loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
    g_optimizer.step()

    return sr.detach(), g_loss.item(), loss_dict


def _discriminator_step(
    generator: SRGenerator,
    discriminator: PatchDiscriminator,
    disc_criterion,
    d_optimizer: torch.optim.Optimizer,
    lr: torch.Tensor,
    hr: torch.Tensor,
    sr_detached: torch.Tensor,
    grad_clip: float = 1.0,
) -> tuple:
    """One discriminator update step."""
    discriminator.train()

    d_optimizer.zero_grad()
    d_real = discriminator(hr)
    d_fake = discriminator(sr_detached)

    d_loss, loss_dict = disc_criterion(d_real, d_fake)
    d_loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
    d_optimizer.step()

    return d_loss.item(), loss_dict


# ---------------------------------------------------------------------------
# Epoch-level helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    generator, discriminator,
    gen_criterion, disc_criterion,
    g_optimizer, d_optimizer,
    loader, device, adversarial: bool,
) -> dict:
    total_g, total_d = 0.0, 0.0
    total_psnr, total_ssim = 0.0, 0.0
    n = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Generator step
        sr, g_loss, _ = _generator_step(
            generator, discriminator, gen_criterion, g_optimizer,
            lr, hr, adversarial,
        )
        total_g += g_loss

        # Discriminator step (only in adversarial phase)
        if adversarial:
            d_loss, _ = _discriminator_step(
                generator, discriminator, disc_criterion, d_optimizer,
                lr, hr, sr,
            )
            total_d += d_loss

        metrics = compute_metrics(sr.clamp(0, 1), hr)
        total_psnr += metrics["psnr"]
        total_ssim += metrics["ssim"]
        n += 1

    return {
        "g_loss": total_g / max(n, 1),
        "d_loss": total_d / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def _val_epoch(generator, loader, device) -> dict:
    generator.eval()
    total_psnr, total_ssim = 0.0, 0.0
    n = 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = generator(lr).clamp(0, 1)
        metrics = compute_metrics(sr, hr)
        total_psnr += metrics["psnr"]
        total_ssim += metrics["ssim"]
        n += 1
    return {
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


# ---------------------------------------------------------------------------
# GAN Trainer
# ---------------------------------------------------------------------------

class GANTrainer:
    """
    Full training manager for the GAN super-resolution model.

    Args:
        cfg:          Full config DotDict
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        resume_g:     Optional checkpoint path for generator
        resume_d:     Optional checkpoint path for discriminator
    """

    def __init__(
        self,
        cfg,
        train_loader,
        val_loader,
        resume_g: Optional[str] = None,
        resume_d: Optional[str] = None,
    ):
        self.cfg = cfg
        self.train_cfg = cfg.train_gan
        self.device = get_device(cfg.get("device", "auto"))

        # Models
        self.generator, self.discriminator = build_gan(cfg)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        logger.info(f"Generator params: {self.generator.n_parameters():,}")
        logger.info(f"Discriminator params: {self.discriminator.n_parameters():,}")

        # Criteria
        self.gen_criterion, self.disc_criterion = build_gan_criteria(cfg)

        # Optimizers
        g_lr = self.train_cfg.get("g_learning_rate", 1e-4)
        d_lr = self.train_cfg.get("d_learning_rate", 1e-4)
        wd = self.train_cfg.get("weight_decay", 1e-4)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, weight_decay=wd, betas=(0.9, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, weight_decay=wd, betas=(0.9, 0.999)
        )

        # Schedulers
        total_epochs = self.train_cfg.get("epochs", 300)
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=total_epochs, eta_min=1e-7
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=total_epochs, eta_min=1e-7
        )

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logging
        log_dir = Path(self.train_cfg.get("checkpoint_dir", "checkpoints/gan"))
        file_logger = get_logger("gan_training", log_file=str(log_dir / "train.log"))
        self.tlogger = TrainingLogger(file_logger)

        # Early stopping on val PSNR
        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 30),
            min_delta=es_cfg.get("min_delta", 1e-5),
            mode="max",
        ) if es_cfg.get("enabled", True) else None

        # State
        self.start_epoch = 0
        self.best_psnr = 0.0
        self.warmup_epochs = self.train_cfg.get("warmup_epochs", 10)

        # Optional resume
        if resume_g:
            ckpt = load_checkpoint(resume_g, self.generator, self.g_optimizer, self.device)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_psnr = ckpt.get("best_psnr", 0.0)
        if resume_d:
            load_checkpoint(resume_d, self.discriminator, self.d_optimizer, self.device)

    def train(self) -> dict:
        """
        Run full GAN training loop (warmup + adversarial phases).

        Returns:
            Dict with best_psnr
        """
        total_epochs = self.train_cfg.get("epochs", 300)
        log_interval = self.train_cfg.get("log_interval", 10)
        ckpt_dir = self.train_cfg.get("checkpoint_dir", "checkpoints/gan")

        logger.info(
            f"Starting GAN training: {self.warmup_epochs} warmup + "
            f"{total_epochs - self.warmup_epochs} adversarial epochs"
        )

        history = {
            "epoch": [],
            "train_loss_g_loss": [],
            "train_loss_d_loss": [],
            "train_psnr": [],
            "train_ssim": [],
            "val_psnr": [],
            "val_ssim": [],
        }

        for epoch in range(self.start_epoch, total_epochs):
            adversarial = epoch >= self.warmup_epochs

            if adversarial and epoch == self.warmup_epochs:
                logger.info("Switching to adversarial training phase.")

            train_metrics = _train_epoch(
                self.generator, self.discriminator,
                self.gen_criterion, self.disc_criterion,
                self.g_optimizer, self.d_optimizer,
                self.train_loader, self.device, adversarial,
            )
            val_metrics = _val_epoch(self.generator, self.val_loader, self.device)

            self.g_scheduler.step()
            self.d_scheduler.step()

            is_best = val_metrics["psnr"] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics["psnr"]

            if epoch % log_interval == 0 or is_best:
                self.tlogger.log_epoch(epoch, train_metrics, "train")
                self.tlogger.log_epoch(epoch, val_metrics, "val")

            # Save checkpoints
            g_state = {
                "epoch": epoch,
                "model_state_dict": self.generator.state_dict(),
                "optimizer_state_dict": self.g_optimizer.state_dict(),
                "best_psnr": self.best_psnr,
            }
            d_state = {
                "epoch": epoch,
                "model_state_dict": self.discriminator.state_dict(),
                "optimizer_state_dict": self.d_optimizer.state_dict(),
            }
            save_checkpoint(g_state, ckpt_dir, "generator_latest.pth", is_best, "generator_best.pth")
            save_checkpoint(d_state, ckpt_dir, "discriminator_latest.pth")

            history["epoch"].append(epoch)
            history["train_loss_g_loss"].append(train_metrics["g_loss"])
            history["train_loss_d_loss"].append(train_metrics["d_loss"])
            history["train_psnr"].append(train_metrics["psnr"])
            history["train_ssim"].append(train_metrics["ssim"])
            history["val_psnr"].append(val_metrics["psnr"])
            history["val_ssim"].append(val_metrics["ssim"])

            if self.early_stopping and self.early_stopping(val_metrics["psnr"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"GAN training complete. Best val PSNR: {self.best_psnr:.4f} dB")
        return {"best_psnr": self.best_psnr, "history": history}
