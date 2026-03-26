#!/usr/bin/env python3
"""
scripts/run_gan.py
------------------
Entry point for training the GAN super-resolution model on SAR images.

Usage:
    python scripts/run_gan.py --config configs/config.yaml
    python scripts/run_gan.py --config configs/config.yaml \
        --resume_g checkpoints/gan/generator_latest.pth \
        --resume_d checkpoints/gan/discriminator_latest.pth
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.device import set_seed, get_device
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.history import save_train_info
from data.dataset import build_dataloaders
from train.train_gan import GANTrainer
from evaluation.metrics import compute_metrics
import torch

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN for SAR Super-Resolution")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume_g", type=str, default=None,
                        help="Path to generator checkpoint to resume from")
    parser.add_argument("--resume_d", type=str, default=None,
                        help="Path to discriminator checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SAR Super-Resolution | GAN Training")
    logger.info("=" * 60)

    cfg = load_config(args.config)
    logger.info(f"Config loaded: {args.config}")

    train_loader, val_loader, test_loader, preprocessor = build_dataloaders(cfg)

    trainer = GANTrainer(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_g=args.resume_g,
        resume_d=args.resume_d,
    )
    results = trainer.train()

    # Test evaluation
    device = get_device(cfg.get("device", "auto"))
    generator = trainer.generator

    best_ckpt = f"{cfg.train_gan.checkpoint_dir}/generator_best.pth"
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, generator, device=device)

    generator.eval()
    test_psnr, test_ssim = 0.0, 0.0
    n = 0
    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr).clamp(0, 1)
            metrics = compute_metrics(sr, hr)
            test_psnr += metrics["psnr"]
            test_ssim += metrics["ssim"]
            n += 1

    test_psnr /= max(n, 1)
    test_ssim /= max(n, 1)

    logger.info("=" * 60)
    logger.info(f"Test Results → PSNR: {test_psnr:.4f} dB | SSIM: {test_ssim:.4f}")
    logger.info("=" * 60)

    results["test_metrics"] = {"psnr": test_psnr, "ssim": test_ssim}
    save_train_info(results, cfg.train_gan.train_log_dir , model_name="gan")


if __name__ == "__main__":
    main()
