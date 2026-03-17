"""
evaluation/metrics.py
---------------------
Evaluation metrics for super-resolution:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)

All metrics operate on:
  - Tensors or numpy arrays in [0, 1]
  - Single-channel (grayscale) or multi-channel inputs
"""

import numpy as np
import torch
from typing import Union

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        pred:       Predicted SR image (B, C, H, W) or (H, W)
        target:     Ground-truth HR image, same shape
        data_range: Maximum value of the image (1.0 for normalized)

    Returns:
        Mean PSNR over the batch in dB
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Uses skimage if available, otherwise falls back to a pure-numpy
    implementation.

    Args:
        pred:       Predicted SR image (B, C, H, W) or (H, W)
        target:     Ground-truth HR image, same shape
        data_range: Maximum value range (1.0 for [0,1] images)

    Returns:
        Mean SSIM over the batch
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Handle batch dimension
    if pred.ndim == 4:  # (B, C, H, W)
        scores = []
        for b in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                scores.append(_compute_ssim(pred[b, c], target[b, c], data_range))
        return float(np.mean(scores))
    elif pred.ndim == 3:  # (C, H, W)
        scores = [_compute_ssim(pred[c], target[c], data_range) for c in range(pred.shape[0])]
        return float(np.mean(scores))
    else:  # (H, W)
        return _compute_ssim(pred, target, data_range)


def _compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """Internal 2-D SSIM computation."""
    if SKIMAGE_AVAILABLE:
        return float(_skimage_ssim(pred, target, data_range=data_range))
    return _ssim_numpy(pred, target, data_range)


def _ssim_numpy(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Pure-numpy SSIM (Wang et al., 2004).
    Fallback when skimage is not available.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    from scipy.ndimage import uniform_filter
    try:
        mu1 = uniform_filter(pred, size=11)
        mu2 = uniform_filter(target, size=11)
    except Exception:
        # Ultra-minimal fallback
        mu1 = np.mean(pred)
        mu2 = np.mean(target)
        sigma1_sq = np.var(pred)
        sigma2_sq = np.var(target)
        sigma12 = np.cov(pred.ravel(), target.ravel())[0, 1]
        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(num / (den + 1e-10))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(pred ** 2, size=11) - mu1_sq
    sigma2_sq = uniform_filter(target ** 2, size=11) - mu2_sq
    sigma12 = uniform_filter(pred * target, size=11) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) /
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Batch metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> dict:
    """
    Compute all SR metrics for a batch.

    Args:
        pred:       Predicted images (B, C, H, W)
        target:     Ground-truth images (B, C, H, W)
        data_range: Image value range

    Returns:
        Dict with keys 'psnr' and 'ssim'
    """
    return {
        "psnr": psnr(pred, target, data_range),
        "ssim": ssim(pred, target, data_range),
    }
