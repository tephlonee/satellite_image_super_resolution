#!/usr/bin/env python3
"""
scripts/run_tuning.py
---------------------
Optuna hyperparameter search for SRCNN or GAN.

Usage:
    python scripts/run_tuning.py --config configs/config.yaml --model srcnn
    python scripts/run_tuning.py --config configs/config.yaml --model gan \
        --n_trials 30 --epochs_per_trial 15
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device import set_seed
from utils.logger import get_logger
from tuning.tune import run_tuning

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SAR SR models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "gan"],
                        help="Model to tune")
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Number of Optuna trials (overrides config)")
    parser.add_argument("--epochs_per_trial", type=int, default=30,
                        help="Training epochs per trial (keep low for speed)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info(f"SAR SR Hyperparameter Tuning | model={args.model}")
    logger.info("=" * 60)

    results = run_tuning(
        cfg_path=args.config,
        model=args.model,
        n_trials_override=args.n_trials,
        n_epochs_per_trial=args.epochs_per_trial,
        seed=args.seed,
    )

    logger.info("\n--- Best Hyperparameters ---")
    logger.info(f"  PSNR: {results['best_value']:.4f} dB")
    for k, v in results["best_params"].items():
        logger.info(f"  {k}: {v}")

    # Save best params to JSON
    out_path = f"tuning_results_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"best_psnr": results["best_value"], "best_params": results["best_params"]},
            f, indent=2,
        )
    logger.info(f"Best params saved to: {out_path}")


if __name__ == "__main__":
    main()
