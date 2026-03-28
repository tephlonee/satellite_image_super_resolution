#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.device import set_seed
from utils.logger import get_logger
from data.dataset import build_dataloaders
from utils.history import save_train_info
from train.train_srresnet import SRResNetTrainer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRResNet for SAR Super-Resolution")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SAR Super-Resolution | SRResNet Training")
    logger.info("=" * 60)

    cfg = load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    logger.info("Building data loaders...")
    train_loader, val_loader, test_loader, preprocessor = build_dataloaders(
        cfg,
        batch_size=cfg.train_srresnet.batch_size,
    )

    trainer = SRResNetTrainer(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=args.resume,
    )
    results = trainer.train()

    logger.info("Running test set evaluation...")
    from evaluation.metrics import compute_metrics
    from utils.device import get_device
    import torch

    device = get_device(cfg.get("device", "auto"))
    model = trainer.model

    best_ckpt = f"{cfg.train_srresnet.checkpoint_dir}/best_model.pth"
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

    results["test_psnr"] = test_psnr
    results["test_ssim"] = test_ssim

    save_train_info(results, cfg.train_srresnet.train_log_dir, model_name="srresnet")


if __name__ == "__main__":
    main()
