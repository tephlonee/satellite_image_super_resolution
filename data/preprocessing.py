"""
data/preprocessing.py
---------------------
SAR-specific preprocessing:
  - Log transformation to convert multiplicative speckle to additive
  - Normalization to [0, 1]
  - Optional speckle reduction filters (Lee, Frost, Median)

Design principles:
  - Preserve edges and structural scatterers
  - Never oversmooth
  - All filters are optional and kernel-size configurable
"""

import numpy as np
from typing import Optional
import warnings

import torch



try:
    from scipy.ndimage import uniform_filter, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available; Lee/Frost filters will fall back to median.")

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Log Transform
# ---------------------------------------------------------------------------

def log_transform(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply log1p transform to SAR amplitude/intensity image.

    Converts multiplicative speckle noise → additive noise in log domain.

    Args:
        image: 2-D or 3-D numpy array (HxW or HxWxC), non-negative values
        eps:   Small offset to avoid log(0)

    Returns:
        Log-transformed array with same shape
    """
    image = np.asarray(image, dtype=np.float32)

    """
    if np.any(image < 0):
        warnings.warn("Negative values found; clipping to 0 before log transform.")
        image = np.clip(image, 0, None)

    """
    return np.log1p(np.abs(image))


def inverse_log_transform(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Invert the log1p transform: expm1(x) - eps.

    Args:
        image: Log-transformed array

    Returns:
        Reconstructed linear-scale array
    """
    return np.expm1(np.asarray(image, dtype=np.float32)) - eps


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(
    image: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> tuple:
    """
    Normalize image to [0, 1].

    Args:
        image:   Input array
        min_val: Known minimum (use training set statistics if available)
        max_val: Known maximum

    Returns:
        (normalized_image, min_val, max_val) tuple so inverse can be applied
    """
    image = np.asarray(image, dtype=np.float32)
    min_val = image.min()  if min_val is None else min_val
    max_val = image.max()   if max_val is None else max_val
    denom = max_val - min_val

    
    if denom < 1e-8:
        return np.zeros_like(image), min_val, max_val

    return (image - min_val) / denom, min_val, max_val


def denormalize(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Invert [0, 1] normalization."""
    return np.asarray(image, dtype=np.float32) * (max_val - min_val) + min_val


def add_speckle_noise(img, L=1.0):
    """
    Apply multiplicative speckle noise to SAR image.

    Args:
        img (torch.Tensor): Image in linear scale [0, 1] or positive float
                            Shape: (C, H, W) or (H, W)
        L (float): Number of looks (controls noise level)

    Returns:
        torch.Tensor: Noisy image
    """
    if not torch.is_tensor(img):
        raise TypeError("Input must be a torch.Tensor")

    # Gamma noise with mean = 1, variance = 1/L
    noise = torch.distributions.Gamma(L, L).sample(img.shape).to(img.device)

    noisy_img = img * noise

    return noisy_img


# ---------------------------------------------------------------------------
# Speckle Filters (optional, mild)
# ---------------------------------------------------------------------------

def lee_filter(image: np.ndarray, kernel_size: int = 3, looks: int = 1) -> np.ndarray:
    """
    Lee filter for multiplicative SAR speckle suppression.

    Adaptive: smooth homogeneous areas, preserve edges/scatterers.

    Args:
        image:       2-D float array (single band)
        kernel_size: Size of the local window (odd)
        looks:       Number of looks (controls noise variance estimate)

    Returns:
        Filtered array
    """
    if not SCIPY_AVAILABLE:
        return median_filter_sar(image, kernel_size)

    img = np.asarray(image, dtype=np.float32)
    img_sq = img ** 2

    local_mean = uniform_filter(img, size=kernel_size)
    local_mean_sq = uniform_filter(img_sq, size=kernel_size)
    local_var = local_mean_sq - local_mean ** 2

    # Noise variance estimate from number of looks
    noise_var = (local_mean ** 2) / looks

    # Lee weight
    weight = local_var / (local_var + noise_var + 1e-10)
    filtered = local_mean + weight * (img - local_mean)
    return filtered.astype(np.float32)


def frost_filter(image: np.ndarray, kernel_size: int = 3, damping: float = 2.0) -> np.ndarray:
    """
    Frost filter (exponentially damped convolution) for SAR speckle.

    Adaptively weighs neighbors by local coefficient of variation.

    Args:
        image:       2-D float array
        kernel_size: Window size (odd)
        damping:     Damping factor (higher = more smoothing in homogeneous areas)

    Returns:
        Filtered array
    """
    if not SCIPY_AVAILABLE:
        return median_filter_sar(image, kernel_size)

    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    half = kernel_size // 2
    output = np.zeros_like(img)

    # Pre-compute distance weights template
    y_idx, x_idx = np.mgrid[-half : half + 1, -half : half + 1]
    dist = np.sqrt(x_idx ** 2 + y_idx ** 2)

    # Pad image
    padded = np.pad(img, half, mode="reflect")

    for i in range(h):
        for j in range(w):
            patch = padded[i : i + kernel_size, j : j + kernel_size]
            mean = patch.mean()
            std = patch.std()
            cv = std / (mean + 1e-10)  # Coefficient of variation
            k = damping * cv ** 2
            weights = np.exp(-k * dist)
            weights /= weights.sum()
            output[i, j] = (weights * patch).sum()

    return output.astype(np.float32)


def median_filter_sar(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Light median filter for SAR speckle.

    Use with small kernel sizes only (3 recommended).

    Args:
        image:       2-D float array
        kernel_size: Kernel size

    Returns:
        Median-filtered array
    """
    if SCIPY_AVAILABLE:
        from scipy.ndimage import median_filter
        return median_filter(image.astype(np.float32), size=kernel_size)
    # Pure numpy fallback (slow, for testing only)
    import numpy.lib.stride_tricks as nst
    img = np.asarray(image, dtype=np.float32)
    half = kernel_size // 2
    padded = np.pad(img, half, mode="reflect")
    shape = img.shape + (kernel_size, kernel_size)
    strides = padded.strides * 2
    patches = nst.as_strided(padded, shape=shape, strides=strides)
    return np.median(patches.reshape(*img.shape, -1), axis=-1).astype(np.float32)


SPECKLE_FILTER_MAP = {
    "lee": lee_filter,
    "frost": frost_filter,
    "median": median_filter_sar,
}


def apply_speckle_filter(
    image: np.ndarray,
    method: str = "lee",
    kernel_size: int = 3,
    **kwargs,
) -> np.ndarray:
    """
    Dispatch speckle filtering.

    Args:
        image:       2-D SAR image array
        method:      "lee" | "frost" | "median"
        kernel_size: Filter window size
        **kwargs:    Additional filter-specific args

    Returns:
        Filtered array
    """
    if method not in SPECKLE_FILTER_MAP:
        raise ValueError(f"Unknown speckle filter: {method}. Choose from {list(SPECKLE_FILTER_MAP)}")
    fn = SPECKLE_FILTER_MAP[method]
    return fn(image, kernel_size=kernel_size, **kwargs)


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

class SARPreprocessor:
    """
    Configurable end-to-end SAR preprocessing pipeline.

    Usage:
        preprocessor = SARPreprocessor(cfg.preprocessing)
        lr_tensor, hr_tensor = preprocessor(lr_img, hr_img)
    """

    def __init__(self, cfg):
        self.log_transform = cfg.get("log_transform", True)
        self.normalize = cfg.get("normalize", True)
        self.speckle_noise = cfg.get("speckle_noise", False)
        speckle_cfg = cfg.get("speckle_filter", {}) or {}
        self.speckle_enabled = speckle_cfg.get("enabled", False)
        self.speckle_method = speckle_cfg.get("method", "lee")
        self.speckle_kernel = speckle_cfg.get("kernel_size", 3)
        self.speckle_looks = speckle_cfg.get("looks", 1)

        # Statistics for consistent normalization (set after fitting on training set)
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def fit(self, images: list) -> None:
        """Compute global min/max from a list of images for consistent normalization."""
        min_val = None
        max_val = None
        for im in images:
            vals = np.asarray(im, dtype=np.float32)
            if self.log_transform:
                vals = log_transform(vals)
            im_min = float(vals.min())
            im_max = float(vals.max())
            if min_val is None:
                min_val, max_val = im_min, im_max
            else:
                min_val = min(min_val, im_min)
                max_val = max(max_val, im_max)

        if min_val is None or max_val is None:
            min_val, max_val = 0.0, 1.0

        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to a single image.

        Args:
            image: 2-D float32 SAR array

        Returns:
            Preprocessed float32 array in [0, 1]
        """
        img = np.asarray(image, dtype=np.float32)

        if self.speckle_noise and not self.speckle_enabled:
            
            img = add_speckle_noise(img)

        speckled_image = None
        # Optional speckle filter (apply BEFORE log transform for multiplicative model)
        if self.speckle_enabled and not self.speckle_noise:
            
            img = apply_speckle_filter(
                img,
                method=self.speckle_method,
                kernel_size=self.speckle_kernel,
                looks=self.speckle_looks,
            )

            speckled_image = img

        # Log transform
        log_transformed_image = None
        if self.log_transform:
            img = log_transform(img)


            log_transformed_image = img

        # Normalize
        normalized_image = None
        if self.normalize:
            img, self.min_val, self.max_val = normalize(img , self.min_val, self.max_val)

            normalized_image = img

        return img, log_transformed_image, normalized_image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)
