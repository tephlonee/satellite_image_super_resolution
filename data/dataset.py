"""
data/dataset.py
---------------
SAR Super-Resolution dataset with:
  - Automatic HR/LR pair generation via bicubic downsampling
  - Patch-based training (random crops)
  - Train/val/test splits
  - SAR preprocessing (log transform, normalization, optional speckle filtering)
  - Paired augmentation

Supports images in: .tif, .tiff, .png, .jpg
For GeoTIFF with multiple bands, only the first band is used.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import SARPreprocessor
from data.augmentation import PairedAugmentation, to_tensor
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional: rasterio / tifffile for .tif support
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str, window: tuple = None) -> np.ndarray:
    """
    Load a SAR image as a 2-D float32 numpy array.
    Optionally load only a window (y, x, height, width).
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext in (".tif", ".tiff"):
        if RASTERIO_AVAILABLE:
            with rasterio.open(path) as src:
                if window is not None:
                    y, x, h, w = window
                    arr = src.read(1, window=rasterio.windows.Window(x, y, w, h)).astype(np.float32)
                else:
                    arr = src.read(1).astype(np.float32)
            return arr
        if TIFFFILE_AVAILABLE:
            img = tifffile.imread(path)
            if img.ndim == 3:
                img = img[0]  # First band
            return img.astype(np.float32)

    # Fallback: PIL for .png / .jpg or unhandled .tif
    from PIL import Image
    img = Image.open(path).convert("L")  # Grayscale
    return np.array(img, dtype=np.float32)


def generate_lr(hr: np.ndarray, scale: int) -> np.ndarray:
    """
    Downsample HR image to create LR pair using bicubic interpolation.

    Args:
        hr:    High-resolution image (H x W)
        scale: Downscaling factor

    Returns:
        Low-resolution image (H//scale x W//scale)
    """
    from PIL import Image
    h, w = hr.shape
    lr_h, lr_w = h // scale, w // scale

    # Normalize to [0, 255] for PIL, then downsample, then restore scale
    hr_min, hr_max = hr.min(), hr.max()
    if hr_max - hr_min < 1e-8:
        return np.zeros((lr_h, lr_w), dtype=np.float32)

    hr_uint8 = ((hr - hr_min) / (hr_max - hr_min) * 255).astype(np.uint8)
    lr_pil = Image.fromarray(hr_uint8).resize((lr_w, lr_h), Image.BICUBIC)
    lr_norm = np.array(lr_pil, dtype=np.float32) / 255.0
    lr = lr_norm * (hr_max - hr_min) + hr_min
    return lr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SARDataset(Dataset):
    """
    Patch-based SAR super-resolution dataset.

    Loads images on demand, applies preprocessing, then generates random patch
    pairs (LR + HR) during __getitem__.

    Args:
        image_paths:  List of paths to HR SAR images
        cfg:          Full config DotDict
        augment:      Whether to apply augmentation
        preprocessor: Fitted SARPreprocessor instance
    """

    def __init__(
        self,
        image_paths: List[str],
        cfg,
        augment: bool = True,
        preprocessor: Optional[SARPreprocessor] = None,
    ):
        self.image_paths = image_paths
        self.scale = cfg.data.scale_factor
        self.patch_size = cfg.data.patch_size
        self.augment = augment

        # Preprocessing
        self.preprocessor = preprocessor or SARPreprocessor(cfg.preprocessing)

        # Augmentation
        self.augmentation = PairedAugmentation(
            cfg.augmentation if augment else {"enabled": False},
            scale=self.scale,
        )

    def __len__(self) -> int:
        # Return a virtual length large enough for good epoch coverage
        return len(self.image_paths) * 50

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx % len(self.image_paths)
        path = self.image_paths[img_idx]
        ext = Path(path).suffix.lower()
        # Patch size in HR
        ps = self.patch_size * self.scale
        if ext in (".tif", ".tiff") and RASTERIO_AVAILABLE:
            with rasterio.open(path) as src:
                h, w = src.height, src.width
                if h < ps or w < ps:
                    raise ValueError(f"Image {path} too small for patch size {ps}")
                y = np.random.randint(0, h - ps + 1)
                x = np.random.randint(0, w - ps + 1)
                hr_patch = src.read(1, window=rasterio.windows.Window(x, y, ps, ps)).astype(np.float32)
        else:
            raw = load_image(path)
            h, w = raw.shape
            if h < ps or w < ps:
                raise ValueError(f"Image {path} too small for patch size {ps}")
            y = np.random.randint(0, h - ps + 1)
            x = np.random.randint(0, w - ps + 1)
            hr_patch = raw[y:y+ps, x:x+ps]
        # Generate LR patch
        lr_patch = generate_lr(hr_patch, self.scale)
        # Preprocess
        hr_patch = self.preprocessor(hr_patch)
        lr_patch = self.preprocessor(lr_patch)
        # Augment
        lr_patch, hr_patch = self.augmentation(lr_patch, hr_patch, hr_patch_size=self.patch_size)
        # Convert to tensors (C=1)
        return to_tensor(lr_patch), to_tensor(hr_patch)


# ---------------------------------------------------------------------------
# Dataset splitting and DataLoader factory
# ---------------------------------------------------------------------------

def split_image_paths(
    image_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Discover image files and split into train / val / test sets.
    Only include .tif images that do NOT have '.preview' in their filename.
    """
    extensions = {".tif"}
    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in extensions and "preview" not in p.stem
    )

    if not paths:
        raise FileNotFoundError(f"No valid .tif images found in: {image_dir}")

    logger.info(f"Found {len(paths)} valid .tif images in {image_dir}")

    random.seed(seed)
    random.shuffle(paths)
    paths = [str(p) for p in paths]

    n = len(paths)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]

    logger.info(f"Split: {len(train)} train | {len(val)} val | {len(test)} test")
    return train, val, test


def worker_init_fn(wid , seed):
    import numpy as np
    np.random.seed(seed + wid)


def build_dataloaders(cfg, seed: int = 42):
    """
    Build train, val, and test DataLoader objects from config.
    Uses downsampled proxies for preprocessor.fit to avoid OOM.
    """
    train_paths, val_paths, test_paths = split_image_paths(
        image_dir=cfg.data.image_dir,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=seed,
    )

    # Fit preprocessor on downsampled training images only
    preprocessor = SARPreprocessor(cfg.preprocessing)
    raw_train = []
    if RASTERIO_AVAILABLE:
        for p in train_paths:
            with rasterio.open(p) as src:
                # Target ~1024px on longest side
                longest = max(src.height, src.width)
                scale = max(1, longest // 1024)
                out_h = max(1, src.height // scale)
                out_w = max(1, src.width // scale)
                arr = src.read(1, out_shape=(1, out_h, out_w)).astype(np.float32)
                raw_train.append(arr)
    else:
        from PIL import Image
        for p in train_paths:
            arr = load_image(p)
            longest = max(arr.shape)
            scale = max(1, longest // 1024)
            small = np.array(Image.fromarray(arr).resize((arr.shape[1]//scale, arr.shape[0]//scale)), dtype=np.float32)
            raw_train.append(small)
    preprocessor.fit(raw_train)

    train_ds = SARDataset(train_paths, cfg, augment=True, preprocessor=preprocessor)
    val_ds = SARDataset(val_paths, cfg, augment=False, preprocessor=preprocessor)
    test_ds = SARDataset(test_paths, cfg, augment=False, preprocessor=preprocessor)

    loader_kwargs = dict(
        batch_size=cfg.train_srcnn.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        worker_init_fn=partial(worker_init_fn, seed=seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, preprocessor
