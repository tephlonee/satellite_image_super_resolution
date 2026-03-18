"""
utils/checkpoint.py
-------------------
Checkpoint saving and loading utilities.
"""

import torch
from pathlib import Path
from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    best_filename: str = "best_model.pth",
) -> None:
    """
    Save a training checkpoint.

    Args:
        state:           Dict containing model state_dict, optimizer, epoch, metrics, etc.
        checkpoint_dir:  Directory to save checkpoints
        filename:        Checkpoint filename
        is_best:         If True, also save a copy as best_model.pth
        best_filename:   Filename for the best checkpoint
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_path = ckpt_dir / filename
    torch.save(state, save_path)
    logger.debug(f"Checkpoint saved: {save_path}")

    if is_best:
        best_path = ckpt_dir / best_filename
        torch.save(state, best_path)
        logger.info(f"Best model updated: {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path:      Path to checkpoint file
        model:     Model to restore weights into
        optimizer: (Optional) Optimizer to restore state into
        device:    Target device

    Returns:
        The full checkpoint dict (contains epoch, metrics, etc.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {path} (epoch {checkpoint.get('epoch', '?')})")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    return checkpoint


class EarlyStopping:
    """
    Monitor a validation metric and signal when training should stop.

    Args:
        patience:  Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode:      'min' for loss-based, 'max' for metric-based (PSNR/SSIM)
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Update state with new metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement."
                )

        return self.should_stop

    @property
    def is_best(self) -> bool:
        """True if the last call registered a new best value."""
        return self.counter == 0
