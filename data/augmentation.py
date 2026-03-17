"""
data/augmentation.py
--------------------
Paired augmentation for LR-HR image patches.

All transforms are applied identically to both LR and HR images
to maintain correspondence. The LR image is processed at 1/scale
spatial resolution, so spatial transforms must account for this.

Transforms implemented:
  - Horizontal flip
  - Vertical flip
  - 90 / 180 / 270 degree rotations
  - Random crop (applied to HR; LR crop computed from scale factor)
"""

import random
import numpy as np
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Basic spatial transforms (numpy-level, applied before tensor conversion)
# ---------------------------------------------------------------------------

def hflip_pair(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal flip."""
    return np.fliplr(lr).copy(), np.fliplr(hr).copy()


def vflip_pair(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vertical flip."""
    return np.flipud(lr).copy(), np.flipud(hr).copy()


def rot90_pair(lr: np.ndarray, hr: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate 90 * k degrees counter-clockwise."""
    return np.rot90(lr, k=k).copy(), np.rot90(hr, k=k).copy()


def random_crop_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    hr_patch_size: int,
    scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random aligned crop of HR patch and corresponding LR patch.

    Args:
        lr:            LR image (H/scale x W/scale)
        hr:            HR image (H x W)
        hr_patch_size: Output HR patch size in pixels
        scale:         Downscaling factor

    Returns:
        (lr_patch, hr_patch) with shapes (hr_patch_size//scale, ...) and (hr_patch_size, ...)
    """
    lr_patch_size = hr_patch_size // scale
    hr_h, hr_w = hr.shape[:2]

    if hr_h < hr_patch_size or hr_w < hr_patch_size:
        raise ValueError(
            f"Image ({hr_h}x{hr_w}) smaller than patch size ({hr_patch_size}). "
            "Reduce patch_size or use larger images."
        )

    # Random top-left corner in HR space
    top = random.randint(0, hr_h - hr_patch_size)
    left = random.randint(0, hr_w - hr_patch_size)

    hr_patch = hr[top : top + hr_patch_size, left : left + hr_patch_size]
    lr_patch = lr[top // scale : top // scale + lr_patch_size,
                  left // scale : left // scale + lr_patch_size]

    return lr_patch, hr_patch


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

class PairedAugmentation:
    """
    Applies randomized spatial augmentations consistently to LR-HR pairs.

    Args:
        cfg:   Augmentation config dict (from config.yaml 'augmentation' section)
        scale: Super-resolution scale factor
    """

    def __init__(self, cfg: dict, scale: int = 4):
        self.enabled = cfg.get("enabled", True)
        self.hflip = cfg.get("horizontal_flip", True)
        self.vflip = cfg.get("vertical_flip", True)
        self.rotation = cfg.get("rotation", True)
        self.do_random_crop = cfg.get("random_crop", True)
        self.scale = scale

    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        hr_patch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to an LR-HR pair.

        Args:
            lr:            LR image array (float32, HxW)
            hr:            HR image array (float32, HxW)
            hr_patch_size: Target HR patch size for cropping

        Returns:
            Augmented (lr_patch, hr_patch) arrays
        """
        if not self.enabled:
            # Just crop if augmentation disabled
            if self.do_random_crop:
                lr, hr = random_crop_pair(lr, hr, hr_patch_size, self.scale)
            return lr, hr

        # 1. Random crop (always do this to get uniform patch size)
        if self.do_random_crop:
            lr, hr = random_crop_pair(lr, hr, hr_patch_size, self.scale)

        # 2. Horizontal flip
        if self.hflip and random.random() < 0.5:
            lr, hr = hflip_pair(lr, hr)

        # 3. Vertical flip
        if self.vflip and random.random() < 0.5:
            lr, hr = vflip_pair(lr, hr)

        # 4. Random rotation (90/180/270)
        if self.rotation:
            k = random.randint(0, 3)
            if k > 0:
                lr, hr = rot90_pair(lr, hr, k)

        return lr, hr


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert HxW float32 numpy array to (1, H, W) float32 tensor.

    Args:
        image: 2-D or 3-D (HxWxC) numpy array

    Returns:
        Tensor with shape (C, H, W)
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # Add channel dim
    elif image.ndim == 3:
        image = image.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(image.copy()).float()
