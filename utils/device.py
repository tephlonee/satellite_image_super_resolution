"""
utils/device.py
---------------
Device selection and reproducibility helpers.
"""

import random
import numpy as np
import torch
from typing import Union
from utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Select compute device.

    Args:
        preference: "auto" | "cuda" | "cpu" | "mps"

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            f"  GPU: {torch.cuda.get_device_name(0)} | "
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")
