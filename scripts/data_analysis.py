import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from utils.config import load_config
from data.preprocessing import SARPreprocessor, apply_speckle_filter, log_transform, normalize

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False





def list_image_paths(image_dir: str):
    exts = {".tif", ".tiff"}
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    paths = sorted(
        x for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in exts and "preview" not in x.stem
    )
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff images found in: {image_dir}")
    return [str(x) for x in paths]


def read_meta(path: str):
    fp = Path(path)
    out = {
        "path": str(fp),
        "name": fp.name,
        "size_bytes": fp.stat().st_size,
        "size_mb": fp.stat().st_size / (1024 * 1024),
        "height": None,
        "width": None,
        "bands": None,
        "dtype": None,
        "is_complex": None,
    }
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            out["height"] = int(src.height)
            out["width"] = int(src.width)
            out["bands"] = int(src.count)
            out["dtype"] = str(src.dtypes[0])
            out["is_complex"] = ("complex" in str(src.dtypes[0]).lower())
    return out


def load_downsampled(path: str, target_max_dim: int = 1024):
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required for this analysis script (install rasterio).")
    with rasterio.open(path) as src:
        longest = max(int(src.height), int(src.width))
        down = max(1, longest // int(target_max_dim))
        out_h = max(1, int(src.height) // down)
        out_w = max(1, int(src.width) // down)
        arr = src.read(1, out_shape=(1, out_h, out_w))
        arr = np.squeeze(arr)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        return arr.astype(np.float32)


def sample_pixels(arr: np.ndarray, n: int, rng: np.random.Generator):
    flat = np.asarray(arr, dtype=np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.empty((0,), dtype=np.float32)
    if flat.size <= n:
        return flat
    idx = rng.integers(0, flat.size, size=n, endpoint=False)
    return flat[idx]


def basic_stats(arr: np.ndarray):
    a = np.asarray(arr, dtype=np.float32).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    p = np.percentile(a, [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist()
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p0": p[0],
        "p1": p[1],
        "p5": p[2],
        "p50": p[3],
        "p95": p[4],
        "p99": p[5],
        "p100": p[6],
    }


def gradient_energy(img: np.ndarray):
    a = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(a)
    return float(np.mean(gx * gx + gy * gy))


def estimate_enl_intensity(amplitude_img: np.ndarray, rng: np.random.Generator, window: int = 128, n_samples: int = 64, keep_low_grad: int = 8):
    img = np.asarray(amplitude_img, dtype=np.float32)
    h, w = img.shape
    if h < window or w < window:
        return None
    candidates = []
    for _ in range(int(n_samples)):
        y = int(rng.integers(0, h - window + 1))
        x = int(rng.integers(0, w - window + 1))
        patch = img[y:y + window, x:x + window]
        ge = gradient_energy(patch)
        candidates.append((ge, patch))
    candidates.sort(key=lambda t: t[0])
    chosen = candidates[:max(1, int(keep_low_grad))]
    enls = []
    for _, patch in chosen:
        inten = patch * patch
        mu = float(np.mean(inten))
        var = float(np.var(inten))
        if var > 1e-12 and mu > 0:
            enls.append((mu * mu) / var)
    if not enls:
        return None
    return float(np.median(np.array(enls, dtype=np.float32)))


def patch_eligibility_count(h: int, w: int, hr_patch: int, stride: int):
    if h is None or w is None:
        return None
    if h < hr_patch or w < hr_patch:
        return 0
    ny = (h - hr_patch) // stride + 1
    nx = (w - hr_patch) // stride + 1
    return int(ny * nx)


def save_hist_plot(values: np.ndarray, out_path: Path, title: str, bins: int = 256):
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    plt.figure(figsize=(8, 4.5))
    plt.hist(v, bins=bins, density=True)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_scatter(xs, ys, out_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(6.5, 5.2))
    plt.scatter(xs, ys, s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--out_dir", type=str, default="analysis_out")
    ap.add_argument("--max_images", type=int, default=50)
    ap.add_argument("--pixels_per_image", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(cfg.data.image_dir)
    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    metas = []
    for p in image_paths:
        metas.append(read_meta(p))

    total_bytes = int(sum(m["size_bytes"] for m in metas))
    dtype_counts = Counter([m["dtype"] for m in metas if m["dtype"] is not None])
    complex_count = int(sum(1 for m in metas if m.get("is_complex") is True))

    hr_patch = int(cfg.data.patch_size) * int(cfg.data.scale_factor)
    stride = int(getattr(cfg.data, "stride", 32))

    for m in metas:
        m["eligible_patches_est"] = patch_eligibility_count(m["height"], m["width"], hr_patch, stride)

    metas_sorted_by_patches = sorted(
        [m for m in metas if m["eligible_patches_est"] is not None],
        key=lambda x: x["eligible_patches_est"],
        reverse=True,
    )

    with open(out_dir / "dataset_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name", "path", "size_bytes", "size_mb", "height", "width", "bands",
                "dtype", "is_complex", "eligible_patches_est"
            ],
        )
        w.writeheader()
        for m in metas:
            w.writerow(m)

    heights = [m["height"] for m in metas if m["height"] is not None]
    widths = [m["width"] for m in metas if m["width"] is not None]
    sizes_mb = [m["size_mb"] for m in metas]

    if heights and widths:
        save_scatter(
            widths, heights,
            out_dir / "resolution_scatter.png",
            "Image Resolutions",
            "width (px)", "height (px)",
        )

    save_hist_plot(np.array(sizes_mb, dtype=np.float32), out_dir / "file_size_hist.png", "File Size (MB)", bins=60)

    sample_paths = image_paths[:]
    rng.shuffle(sample_paths)
    sample_paths = sample_paths[:max(1, min(int(args.max_images), len(sample_paths)))]

    pre = SARPreprocessor(cfg.preprocessing)
    fit_imgs = []
    for p in sample_paths:
        arr = load_downsampled(p, target_max_dim=1024)
        fit_imgs.append(arr)
    pre.fit(fit_imgs)

    raw_samples = []
    log_samples = []
    norm_samples = []
    filt_samples = []
    grad_samples = []

    enl_rows = []
    speckle_cfg = cfg.preprocessing.get("speckle_filter", {}) or {}
    speckle_enabled = bool(speckle_cfg.get("enabled", False))
    speckle_method = str(speckle_cfg.get("method", "lee"))
    speckle_kernel = int(speckle_cfg.get("kernel_size", 3))
    speckle_looks = int(speckle_cfg.get("looks", 1))

    for p in sample_paths:
        arr = load_downsampled(p, target_max_dim=1024)
        raw = np.asarray(arr, dtype=np.float32)
        raw_samples.append(sample_pixels(raw, int(args.pixels_per_image), rng))

        proc, log_img, norm_img = pre.process(raw)

        if log_img is not None:
            log_samples.append(sample_pixels(log_img, int(args.pixels_per_image), rng))
        if norm_img is not None:
            norm_samples.append(sample_pixels(norm_img, int(args.pixels_per_image), rng))
            gy, gx = np.gradient(norm_img.astype(np.float32))
            gm = np.sqrt(gx * gx + gy * gy)
            grad_samples.append(sample_pixels(gm, int(args.pixels_per_image), rng))

        if speckle_enabled:
            try:
                filt = apply_speckle_filter(
                    raw,
                    method=speckle_method,
                    kernel_size=speckle_kernel,
                    looks=speckle_looks,
                )
                filt_samples.append(sample_pixels(filt, int(args.pixels_per_image), rng))
                enl_raw = estimate_enl_intensity(raw, rng=rng)
                enl_filt = estimate_enl_intensity(filt, rng=rng)
                enl_rows.append({"name": Path(p).name, "enl_raw_intensity": enl_raw, "enl_filtered_intensity": enl_filt})
            except Exception:
                enl_rows.append({"name": Path(p).name, "enl_raw_intensity": None, "enl_filtered_intensity": None})
        else:
            enl_raw = estimate_enl_intensity(raw, rng=rng)
            enl_rows.append({"name": Path(p).name, "enl_raw_intensity": enl_raw, "enl_filtered_intensity": None})

    def cat(xs):
        if not xs:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0)

    raw_all = cat(raw_samples)
    log_all = cat(log_samples)
    norm_all = cat(norm_samples)
    filt_all = cat(filt_samples)
    grad_all = cat(grad_samples)

    save_hist_plot(raw_all, out_dir / "hist_raw_amplitude.png", "Raw Amplitude (sampled pixels)")
    if log_all.size:
        save_hist_plot(log_all, out_dir / "hist_log.png", "Log-Transformed (sampled pixels)")
    if norm_all.size:
        save_hist_plot(norm_all, out_dir / "hist_normalized.png", "Normalized [0,1] (sampled pixels)")
    if filt_all.size:
        save_hist_plot(filt_all, out_dir / "hist_speckle_filtered.png", "Speckle-Filtered (sampled pixels)")
    if grad_all.size:
        save_hist_plot(grad_all, out_dir / "hist_gradient_magnitude.png", "Gradient Magnitude on Normalized (sampled pixels)", bins=200)

    with open(out_dir / "enl_estimates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "enl_raw_intensity", "enl_filtered_intensity"])
        w.writeheader()
        for r in enl_rows:
            w.writerow(r)

    eligible = [m["eligible_patches_est"] for m in metas if m["eligible_patches_est"] is not None]
    if eligible:
        save_hist_plot(np.array(eligible, dtype=np.float32), out_dir / "eligible_patches_hist.png", "Estimated Patch Positions per Image", bins=60)

    top_patch_images = [
        {"name": m["name"], "eligible_patches_est": m["eligible_patches_est"], "height": m["height"], "width": m["width"]}
        for m in metas_sorted_by_patches[:10]
    ]

    summary = {
        "image_dir": str(cfg.data.image_dir),
        "n_images": int(len(image_paths)),
        "total_size_gb": float(total_bytes / (1024 ** 3)),
        "dtype_counts": dict(dtype_counts),
        "complex_count": int(complex_count),
        "complex_fraction": float(complex_count / max(1, len(metas))),
        "hr_patch_px": int(hr_patch),
        "stride_px": int(stride),
        "top_images_by_patch_positions": top_patch_images,
        "stats_raw_amplitude_samples": basic_stats(raw_all),
        "stats_log_samples": basic_stats(log_all) if log_all.size else None,
        "stats_normalized_samples": basic_stats(norm_all) if norm_all.size else None,
        "stats_speckle_filtered_samples": basic_stats(filt_all) if filt_all.size else None,
        "stats_gradient_magnitude_samples": basic_stats(grad_all) if grad_all.size else None,
        "notes": {
            "complex_handling_in_training": "Training loader converts complex arrays to magnitude via np.abs when detected.",
            "enl_definition": "ENL computed on intensity (amplitude^2) for low-gradient sampled windows; higher ENL indicates smoother homogeneous regions.",
        },
    }

    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote outputs to: {out_dir.resolve()}")
    print(f"Inventory: {str((out_dir / 'dataset_inventory.csv').resolve())}")
    print(f"Summary:    {str((out_dir / 'analysis_summary.json').resolve())}")


if __name__ == "__main__":
    main()