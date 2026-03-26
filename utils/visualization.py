from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import get_logger  

logger = get_logger(__name__)


def _save_gray_image(ax, img, title: str = "", cmap: str = "gray"):
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)


def save_pair_visualization(lr: np.ndarray, hr: np.ndarray, out_dir: str, basename: str) -> str:
    """Save side-by-side LR / HR (SR) comparison figure and return path."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _save_gray_image(axes[0], lr, "LR (input)")
    _save_gray_image(axes[1], hr, "HR / SR (target)")
    plt.tight_layout()
    out_path = out_dir / f"{basename}_pair.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved pair visualization: {out_path}")
    return str(out_path)


def save_preprocessing_visualization(image: np.ndarray, steps : dict, out_dir: str, basename: str) -> dict:
    """Save figures for each preprocessing step: log, speckle, normalization."""
  
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    # Log transform

    for key, items in steps.items():
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        arr = np.asarray(items)
        if arr.dtype.kind not in {'f', 'i'}:
            arr = arr.astype(np.float32)
        _save_gray_image(ax, arr, f"{key} transform")
        log_path = out_dir / f"{basename}_{key}.png"
        fig.savefig(log_path, dpi=150)
        plt.close(fig)
        paths[key] = str(log_path)

        """

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _save_gray_image(ax, steps["speckle_filtered"], f"Speckle filtered ({getattr(preprocessor, 'speckle_method', 'lee')})")
        speckle_path = out_dir / f"{basename}_speckle.png"
        fig.savefig(speckle_path, dpi=150)
        plt.close(fig)
        paths["speckle"] = str(speckle_path)
        # Normalization

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _save_gray_image(ax, steps["normalized"], "Normalized")
        norm_path = out_dir / f"{basename}_norm.png"
        fig.savefig(norm_path, dpi=150)
        plt.close(fig)
        paths["norm"] = str(norm_path)

        """
    return paths

def save_patch_and_steps(lr: np.ndarray, hr: np.ndarray, out_dir: str, basename: str , steps : dict) -> dict:
    """Save LR/HR pair figure and preprocessing effect figures for each patch."""
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["pair"] = save_pair_visualization(lr, hr, out_dir, basename)
    results["lr_steps"] = save_preprocessing_visualization(lr, steps ,  out_dir, f"{basename}_lr")
    results["hr_steps"] = save_preprocessing_visualization(hr, steps ,  out_dir, f"{basename}_hr")
    return results