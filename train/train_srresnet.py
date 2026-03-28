import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from models.srresnet import build_srresnet
from evaluation.losses import build_srresnet_criterion
from evaluation.metrics import compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, EarlyStopping
from utils.logger import get_logger, TrainingLogger
from utils.device import get_device


logger = get_logger(__name__)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool = True,
    grad_clip: float = 1.0,
) -> dict:
    model.train(train)
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for lr, hr in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            sr = model(lr)
            loss, _ = criterion(sr, hr)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            metrics = compute_metrics(sr.clamp(0, 1), hr)
            total_loss += loss.item()
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "psnr": total_psnr / max(n_batches, 1),
        "ssim": total_ssim / max(n_batches, 1),
    }


class SRResNetTrainer:
    def __init__(self, cfg, train_loader, val_loader, resume: Optional[str] = None):
        self.cfg = cfg
        self.train_cfg = cfg.train_srresnet
        self.device = get_device(cfg.get("device", "auto"))

        self.model = build_srresnet(cfg).to(self.device)
        logger.info(f"SRResNet params: {self.model.n_parameters():,}")

        self.criterion = build_srresnet_criterion(cfg)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.get("learning_rate", 1e-4),
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_cfg.get("epochs", 200),
            eta_min=1e-7,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        log_dir = Path(self.train_cfg.get("checkpoint_dir", "checkpoints/srresnet"))
        file_logger = get_logger(
            "srresnet_training",
            log_file=str(log_dir / "train.log"),
        )
        self.tlogger = TrainingLogger(file_logger)

        es_cfg = self.train_cfg.get("early_stopping", {}) or {}
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 1e-5),
            mode="max",
        ) if es_cfg.get("enabled", True) else None

        self.start_epoch = 0
        self.best_psnr = 0.0
        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optimizer, self.device)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_psnr = ckpt.get("best_psnr", 0.0)

    def train(self) -> dict:
        total_epochs = self.train_cfg.get("epochs", 200)
        log_interval = self.train_cfg.get("log_interval", 10)
        ckpt_dir = self.train_cfg.get("checkpoint_dir", "checkpoints/srresnet")

        logger.info(f"Starting SRResNet training for {total_epochs} epochs")

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
            train_metrics = _run_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                train=True,
            )
            val_metrics = _run_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                None,
                self.device,
                train=False,
            )

            self.scheduler.step()
            is_best = val_metrics["psnr"] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics["psnr"]

            if epoch % log_interval == 0 or is_best:
                self.tlogger.log_epoch(epoch, train_metrics, phase="train")
                self.tlogger.log_epoch(epoch, val_metrics, phase="val")

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

            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["psnr"]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Training complete. Best val PSNR: {self.best_psnr:.4f} dB")
        return {"best_psnr": self.best_psnr, "history": history}
