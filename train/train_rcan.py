import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

import time
import os
import sys

#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rcan import build_rcan
from evaluation.losses import build_rcan_criterion
from evaluation.metrics import compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, EarlyStopping
from utils.logger import get_logger, TrainingLogger
from utils.device import get_device


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    train: bool = True,
    grad_clip: float = 1.0,
    log_every_n_steps: int = 50,
    phase: str = "train",
) -> dict:
    """Run one training or validation epoch."""
    model.train(train)
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    start_t = time.perf_counter()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, (lr, hr) in enumerate(loader, start=1):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            if use_amp and scaler is not None and scaler.is_enabled():
                with torch.amp.autocast(device_type=device.type):
                    sr = model(lr)
                    loss, _ = criterion(sr, hr)

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                sr = model(lr)
                loss, _ = criterion(sr, hr)

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            metrics = compute_metrics(sr.clamp(0, 1), hr)
            total_loss += loss.item()
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]
            n_batches += 1
            if log_every_n_steps > 0 and step % log_every_n_steps == 0:
                elapsed = time.perf_counter() - start_t
                it_s = elapsed / max(step, 1)
                logger.info(
                    f"{phase}: step {step}/{len(loader)} | "
                    f"loss {total_loss / max(n_batches, 1):.5f} | "
                    f"psnr {total_psnr / max(n_batches, 1):.3f} | "
                    f"{it_s:.3f}s/it"
                )

    return {
        "loss": total_loss / max(n_batches, 1),
        "psnr": total_psnr / max(n_batches, 1),
        "ssim": total_ssim / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class RCANTrainer:
    """
    Full training manager for the RCAN model.

    Args:
        cfg:          Full config DotDict
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        resume:       Optional path to checkpoint to resume from
    """

    def __init__(self, cfg, train_loader, val_loader, resume: Optional[str] = None):
        self.cfg = cfg
        self.train_cfg = cfg.train_rcan
        self.device = get_device(cfg.get("device", "auto"))

        # Model
        self.model = build_rcan(cfg).to(self.device)
        logger.info(f"RCAN params: {self.model.n_parameters():,}")

        # Criterion
        self.criterion = build_rcan_criterion(cfg)

        # Optimizer
        # 1. Improved Optimizer: AdamW with lower weight decay for better generalization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.get("learning_rate", 1e-4),
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        """
        # LR scheduler: Cosine annealing for stable convergence
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_cfg.get("epochs", 200),
            eta_min=1e-7,
        )
        """

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(self.train_cfg.get("epochs", 200) * 0.5), int(self.train_cfg.get("epochs", 200) * 0.75)],
            gamma=0.5
        )

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logging
        log_dir = Path(self.train_cfg.get("checkpoint_dir", "checkpoints/rcan"))
        file_logger = get_logger(
            "rcan_training",
            log_file=str(log_dir / "train.log"),
        )
        self.tlogger = TrainingLogger(file_logger)

        # Early stopping
        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 1e-5),
            mode="max",  # Monitor PSNR
        ) if es_cfg.get("enabled", True) else None

        # Resume
        self.start_epoch = 0
        self.best_psnr = 0.0
        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optimizer, self.device)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_psnr = ckpt.get("best_psnr", 0.0)

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dict with final best_psnr and best_ssim
        """
        total_epochs = self.train_cfg.get("epochs", 200)
        log_interval = self.train_cfg.get("log_interval", 10)
        ckpt_dir = self.train_cfg.get("checkpoint_dir", "checkpoints/rcan")

        logger.info(f"Starting RCAN training for {total_epochs} epochs")

        history = {
            "epoch": [],
            "train_loss": [],
            "train_psnr": [],
            "train_ssim": [],
            "val_loss": [],
            "val_psnr": [],
            "val_ssim": [],
        }

        for epoch in range(self.start_epoch, total_epochs):
            # Train
            train_metrics = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer,
                self.device,
                use_amp=self.use_amp,
                scaler=self.scaler,
                train=True,
                log_every_n_steps=self.train_cfg.get("log_every_n_steps", 50),
                phase="train",
            )
            # Validate
            val_metrics = _run_epoch(
                self.model, self.val_loader, self.criterion,
                None,
                self.device,
                use_amp=self.use_amp,
                scaler=self.scaler,
                train=False,
                log_every_n_steps=0,
                phase="val",
            )

            self.scheduler.step()
            is_best = val_metrics["psnr"] > self.best_psnr

            if is_best:
                self.best_psnr = val_metrics["psnr"]

            # Log
            if epoch % log_interval == 0 or is_best:
                self.tlogger.log_epoch(epoch, train_metrics, phase="train")
                self.tlogger.log_epoch(epoch, val_metrics, phase="val")

            # Save checkpoint
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_psnr": self.best_psnr,
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
            }
            save_checkpoint(
                state,
                ckpt_dir,
                filename="latest.pth",
                is_best=is_best,
                best_filename="best_model.pth",
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_psnr"].append(train_metrics["psnr"])
            history["train_ssim"].append(train_metrics["ssim"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_psnr"].append(val_metrics["psnr"])
            history["val_ssim"].append(val_metrics["ssim"])

            # Early stopping (monitor val PSNR)
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["psnr"]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Training complete. Best val PSNR: {self.best_psnr:.4f} dB")
        return {"best_psnr": self.best_psnr , "history": history }
