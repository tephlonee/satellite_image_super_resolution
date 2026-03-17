"""
utils/logger.py
---------------
Unified logging setup for training, evaluation, and tuning.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with console (and optionally file) handlers.

    Args:
        name:     Logger name (typically __name__ of the calling module)
        log_file: Optional path to a log file
        level:    Logging level (default: INFO)

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TrainingLogger:
    """
    Lightweight metrics tracker for training loops.
    Accumulates losses and metrics per epoch and logs summaries.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._history: dict = {}

    def log_epoch(self, epoch: int, metrics: dict, phase: str = "train") -> None:
        """
        Log a dict of metrics for one epoch.

        Args:
            epoch:   Current epoch number
            metrics: {metric_name: value} dict
            phase:   "train" or "val"
        """
        parts = [f"Epoch {epoch:04d} [{phase.upper()}]"]
        for k, v in metrics.items():
            key = f"{phase}/{k}"
            self._history.setdefault(key, []).append((epoch, v))
            parts.append(f"{k}: {v:.6f}")
        self.logger.info("  |  ".join(parts))

    def get_history(self, key: str) -> list:
        """Retrieve logged history for a metric key, e.g. 'val/psnr'."""
        return self._history.get(key, [])

    def best_value(self, key: str, mode: str = "max") -> float:
        """Return the best recorded value for a metric."""
        history = self.get_history(key)
        if not history:
            return float("-inf") if mode == "max" else float("inf")
        values = [v for _, v in history]
        return max(values) if mode == "max" else min(values)
