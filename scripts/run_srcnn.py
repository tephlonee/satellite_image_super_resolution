#!/usr/bin/env python3
"""
scripts/run_srcnn.py
--------------------
Entry point for training the SRCNN super-resolution model on SAR images.

Usage:
    python scripts/run_srcnn.py --config configs/config.yaml
    python scripts/run_srcnn.py --config configs/config.yaml --resume checkpoints/srcnn/latest.pth
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.device import set_seed
from utils.logger import get_logger
from data.dataset import build_dataloaders
from train.train_srcnn import SRCNNTrainer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRCNN for SAR Super-Resolution")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SAR Super-Resolution | SRCNN Training")
    logger.info("=" * 60)

    cfg = load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    # Build data loaders
    logger.info("Building data loaders...")
    train_loader, val_loader, test_loader, preprocessor = build_dataloaders(cfg)

    # Train
    trainer = SRCNNTrainer(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
    )
    results = trainer.train()

    # Final test evaluation
    logger.info("Running test set evaluation...")
    from evaluation.metrics import compute_metrics
    from utils.device import get_device
    import torch

    device = get_device(cfg.get("device", "auto"))
    model = trainer.model

    # Load best model for test eval
    best_ckpt = f"{cfg.train_srcnn.checkpoint_dir}/best_model.pth"
    if os.path.exists(best_ckpt):
        from utils.checkpoint import load_checkpoint
        load_checkpoint(best_ckpt, model, device=device)

    model.eval()
    test_psnr, test_ssim = 0.0, 0.0
    n = 0
    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr).clamp(0, 1)
            metrics = compute_metrics(sr, hr)
            test_psnr += metrics["psnr"]
            test_ssim += metrics["ssim"]
            n += 1

    test_psnr /= max(n, 1)
    test_ssim /= max(n, 1)

    logger.info("=" * 60)
    logger.info(f"Test Results → PSNR: {test_psnr:.4f} dB | SSIM: {test_ssim:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
