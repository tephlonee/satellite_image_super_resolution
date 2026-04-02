#!/usr/bin/env python3
"""
scripts/run_inference.py
------------------------
Run super-resolution inference on new SAR images.

Supports SRCNN, FSRCNN, SRResNet, RCAN, and GAN models.
Outputs SR images as .tif (if rasterio available) or .png.

Usage:
    python scripts/run_inference.py \
        --config configs/config.yaml \
        --model srcnn \
        --checkpoint checkpoints/srcnn/best_model.pth \
        --input_dir path/to/new/images \
        --output_dir results/sr_output

    python scripts/run_inference.py \
        --config configs/config.yaml \
        --model gan \
        --checkpoint checkpoints/gan/generator_best.pth \
        --input_path single_image.tif \
        --output_dir results/
"""

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.device import get_device, set_seed
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from data.dataset import load_image
from data.preprocessing import SARPreprocessor
from data.augmentation import to_tensor
from evaluation.metrics import compute_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Save output image
# ---------------------------------------------------------------------------

def save_image(array: np.ndarray, path: str) -> None:
    """Save SR result as PNG or TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clip to [0, 1], scale to uint16 for max dynamic range
    array = np.clip(array, 0, 1)

    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            data_u16 = (array * 65535).astype(np.uint16)
            if hasattr(tifffile, "imwrite"):
                tifffile.imwrite(str(path), data_u16)
            else:
                tifffile.imsave(str(path), data_u16)
            logger.info(f"Saved TIFF: {path}")
            return
        except ImportError:
            pass

    # Fallback: save as PNG
    png_path = path.with_suffix(".png")
    from PIL import Image
    Image.fromarray((array * 255).astype(np.uint8)).save(str(png_path))
    logger.info(f"Saved PNG: {png_path}")


# ---------------------------------------------------------------------------
# Tile-based inference for large images
# ---------------------------------------------------------------------------

def infer_image(
    model: torch.nn.Module,
    image: np.ndarray,
    preprocessor: SARPreprocessor,
    device: torch.device,
    scale: int,
    tile_size: int = 256,
    overlap: int = 32,
    stream_path: str | None = None,
) -> np.ndarray | None:
    """
    Run SR inference on a full image using overlapping tile strategy.

    This avoids memory issues with large SAR images and reduces boundary
    artifacts by blending overlapping tile predictions.

    Args:
        model:        SR model (SRCNN or Generator)
        image:        Raw input SAR image (H, W)
        preprocessor: Fitted SARPreprocessor
        device:       Compute device
        scale:        SR scale factor
        tile_size:    LR tile size in pixels
        overlap:      Overlap between adjacent tiles (LR pixels)

    Returns:
        Super-resolved image (H*scale, W*scale), values in [0, 1]
    """
    model.eval()

    # Preprocess full image (preprocessor returns (img, log, norm))
    lr_full, _, _ = preprocessor(image)
    h, w = lr_full.shape
    out_h, out_w = h * scale, w * scale

    # If the output is extremely large, stream tiles directly to disk to avoid RAM OOM
    MAX_PIXELS_RAM = 50_000_000  # ~200MB for float32; adjust as needed
    if stream_path and (out_h * out_w) > MAX_PIXELS_RAM:
        import tifffile
        stream_path = str(Path(stream_path).with_suffix(".tif"))
        Path(stream_path).parent.mkdir(parents=True, exist_ok=True)
        overlap = 0
        stride = tile_size
        with tifffile.TiffWriter(stream_path, bigtiff=True) as tif:
            # Write an empty image first, then update tiles via memmap-like API
            # tifffile supports writing full image at once; we instead assemble per tile to a scratch array
            out = np.memmap(Path(stream_path + ".tmp"), dtype=np.uint16, mode="w+", shape=(out_h, out_w))
            with torch.no_grad():
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y_end = min(y + tile_size, h)
                        x_end = min(x + tile_size, w)
                        tile = lr_full[y:y_end, x:x_end]
                        lr_t = to_tensor(tile).unsqueeze(0).to(device)
                        sr_t = model(lr_t).clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()
                        oy, ox = y * scale, x * scale
                        oh, ow = sr_t.shape
                        out[oy : oy + oh, ox : ox + ow] = (sr_t * 65535).astype(np.uint16)
            tif.write(out, dtype=np.uint16)
            try:
                Path(stream_path + ".tmp").unlink()
            except Exception:
                pass
        logger.info(f"Saved streamed TIFF: {stream_path}")
        return None

    # Output accumulator and weight map (RAM-safe case)
    output = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    stride = tile_size - overlap

    with torch.no_grad():
        for y in range(0, max(h - overlap, 1), stride):
            for x in range(0, max(w - overlap, 1), stride):
                # Crop LR tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = lr_full[y:y_end, x:x_end]

                # To tensor
                lr_t = to_tensor(tile).unsqueeze(0).to(device)

                # Inference
                sr_t = model(lr_t).clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()

                # Map back to output coordinates
                oy, ox = y * scale, x * scale
                oh, ow = sr_t.shape

                output[oy : oy + oh, ox : ox + ow] += sr_t
                weight[oy : oy + oh, ox : ox + ow] += 1.0

    # Normalize by overlap count
    weight = np.maximum(weight, 1.0)
    return output / weight


def _pick_center_window(h: int, w: int, size: int) -> tuple[int, int, int, int]:
    y = max((h - size) // 2, 0)
    x = max((w - size) // 2, 0)
    return y, x, size, size


def _read_tif_window(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    y, x, h, w = window
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            arr = src.read(1, window=rasterio.windows.Window(x, y, w, h))
            if np.iscomplexobj(arr):
                arr = np.abs(arr)
            return np.asarray(np.squeeze(arr), dtype=np.float32)
    except Exception:
        arr = load_image(str(path))
        return np.asarray(arr[y : y + h, x : x + w], dtype=np.float32)


def _get_tif_hw(path: Path) -> tuple[int, int]:
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            return int(src.height), int(src.width)
    except Exception:
        arr = load_image(str(path))
        return int(arr.shape[0]), int(arr.shape[1])


def _scene_score(scene: np.ndarray) -> tuple[float, dict]:
    scene = np.asarray(scene, dtype=np.float32)
    scene = np.abs(scene)
    valid = np.isfinite(scene) & (scene > 0.0)
    frac_valid = float(np.mean(valid.astype(np.float32)))
    if frac_valid < 0.05:
        return -float("inf"), {"frac_valid": frac_valid, "std": 0.0, "p99": 0.0}
    vals = scene[valid]
    std = float(np.std(vals))
    if std < 1e-6:
        return -float("inf"), {"frac_valid": frac_valid, "std": std, "p99": float(np.max(vals))}
    p99 = float(np.percentile(vals, 99))
    p1 = float(np.percentile(vals, 1))
    dr = max(p99 - p1, 0.0)
    score = frac_valid * (np.log1p(dr) + 0.25 * np.log1p(std))
    return float(score), {"frac_valid": frac_valid, "std": std, "p99": p99}


def _select_nonblack_window(
    path: Path,
    scene_size: int,
    tries: int = 30,
) -> tuple[int, int, int, int, dict]:
    h, w = _get_tif_hw(path)
    size = int(scene_size)
    size = max(1, min(size, h, w))

    best_window = _pick_center_window(h, w, size)
    best_scene = _read_tif_window(path, best_window)
    best_score, best_stats = _scene_score(best_scene)

    rng = np.random.default_rng(0)
    if h == size and w == size:
        return best_window[0], best_window[1], best_window[2], best_window[3], best_stats

    for _ in range(max(int(tries), 1)):
        y = int(rng.integers(0, max(h - size + 1, 1)))
        x = int(rng.integers(0, max(w - size + 1, 1)))
        window = (y, x, size, size)
        scene = _read_tif_window(path, window)
        score, stats = _scene_score(scene)
        if score > best_score:
            best_window = window
            best_score = score
            best_stats = stats

    return best_window[0], best_window[1], best_window[2], best_window[3], best_stats


def _downsample_to_lr(hr: np.ndarray, scale: int) -> np.ndarray:
    import cv2

    h, w = hr.shape
    lr_h = max(h // scale, 1)
    lr_w = max(w // scale, 1)
    lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    return np.asarray(lr, dtype=np.float32)


def _resize_to(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    import cv2

    img = np.asarray(image, dtype=np.float32)
    return np.asarray(cv2.resize(img, (int(out_w), int(out_h)), interpolation=cv2.INTER_CUBIC), dtype=np.float32)


def _to_uint8(image01: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(image01, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_viz_panel(
    out_path: str,
    panels: list[tuple[str, np.ndarray]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    label_h = 18

    imgs = []
    for label, arr in panels:
        if arr.ndim == 2:
            img = Image.fromarray(_to_uint8(arr), mode="L")
        else:
            img = Image.fromarray(arr.astype(np.uint8))
        if img.mode != "L":
            img = img.convert("L")
        w, h = img.size
        labeled = Image.new("L", (w, h + label_h), 0)
        draw = ImageDraw.Draw(labeled)
        text = str(label)
        text_w = int(draw.textlength(text, font=font))
        draw.text(((w - text_w) // 2, 2), text, fill=255, font=font)
        labeled.paste(img, (0, label_h))
        imgs.append(labeled)

    widths, heights = zip(*(im.size for im in imgs))
    total_w = int(sum(widths))
    max_h = int(max(heights))
    canvas = Image.new("L", (total_w, max_h))

    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    out_path = str(Path(out_path).with_suffix(".png"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info(f"Saved visualization: {out_path}")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Main inference runner
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAR Super-Resolution Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="srcnn",
        choices=["srcnn", "fsrcnn", "srresnet", "rcan", "gan"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="infer",
        choices=["infer", "self_eval"],
        help="infer: treat inputs as LR and upscale; self_eval: treat inputs as HR, downsample to LR, upscale, and score vs HR",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of SAR images to process")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Single SAR image path")
    parser.add_argument("--output_dir", type=str, default="results/inference",
                        help="Output directory for SR images")
    parser.add_argument(
        "--scene_size",
        type=int,
        default=0,
        help="If >0, run inference on a square crop (scene) of this size instead of the full image. In self_eval mode this is HR pixels; in infer mode this is LR pixels.",
    )
    parser.add_argument(
        "--scene_x",
        type=int,
        default=None,
        help="Optional scene crop X (left) in pixels. Defaults to center crop.",
    )
    parser.add_argument(
        "--scene_y",
        type=int,
        default=None,
        help="Optional scene crop Y (top) in pixels. Defaults to center crop.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="If set, process only this many images from input_dir (sorted order). If omitted and --scene_size>0, defaults to 10.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="If set, do not write SR outputs to disk (still reports timing/metrics).",
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Save a side-by-side visualization for each processed scene/image (PNG).",
    )
    parser.add_argument(
        "--save_best_viz",
        action="store_true",
        help="Save a visualization only for the best result (self_eval mode uses highest PSNR).",
    )
    parser.add_argument("--tile_size", type=int, default=256,
                        help="LR tile size for tiled inference")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Tile overlap (LR pixels)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    scale = cfg.data.get("scale_factor", 4)

    logger.info(f"Loading {args.model} model from: {args.checkpoint}")

    # Build model
    if args.model == "srcnn":
        from models.srcnn import build_srcnn
        model = build_srcnn(cfg).to(device)
    elif args.model == "fsrcnn":
        from models.fsrcnn import build_fsrcnn
        model = build_fsrcnn(cfg).to(device)
    elif args.model == "srresnet":
        from models.srresnet import build_srresnet
        model = build_srresnet(cfg).to(device)
    elif args.model == "rcan":
        from models.rcan import build_rcan
        model = build_rcan(cfg).to(device)
    else:
        from models.gan import build_gan
        model, _ = build_gan(cfg)
        model = model.to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # Collect input images
    extensions = {".tif"}
    if args.input_dir:
        input_paths = sorted(
            p for p in Path(args.input_dir).iterdir()
            if p.suffix.lower() in extensions and "preview" not in p.stem
        )
    elif args.input_path:
        input_paths = [Path(args.input_path)]
    else:
        raise ValueError("Specify --input_dir or --input_path")

    if args.scene_size > 0 and args.num_images is None and args.input_dir:
        args.num_images = 10
    if args.num_images is not None:
        input_paths = input_paths[: max(int(args.num_images), 0)]

    output_root = Path(args.output_dir)
    if output_root.name != args.model:
        output_root = output_root / args.model
    output_root_str = str(output_root)

    logger.info(f"Processing {len(input_paths)} image(s) → {output_root_str}")

    per_image_seconds: list[float] = []
    metric_rows: list[dict] = []
    best_key = -float("inf")
    best_viz: dict | None = None

    for img_path in input_paths:
        logger.info(f"  Processing: {img_path.name}")
        preprocessor = SARPreprocessor(cfg.preprocessing)
        raw = None
        window = None

        if args.scene_size and args.scene_size > 0:
            h, w = _get_tif_hw(img_path)
            scene_size = int(args.scene_size)
            if args.mode == "self_eval":
                scene_size = max((scene_size // scale) * scale, scale)
            if args.scene_x is not None and args.scene_y is not None:
                x = int(args.scene_x)
                y = int(args.scene_y)
                hh = scene_size
                ww = scene_size
                stats = None
            else:
                y, x, hh, ww, stats = _select_nonblack_window(img_path, scene_size=scene_size, tries=30)
                logger.info(
                    f"    picked_scene: y={y} x={x} size={hh} | frac_valid={stats['frac_valid']:.3f} std={stats['std']:.4f} p99={stats['p99']:.4f}"
                )
            x = max(min(x, max(w - ww, 0)), 0)
            y = max(min(y, max(h - hh, 0)), 0)
            window = (y, x, hh, ww)
            raw = _read_tif_window(img_path, window)
        else:
            raw = load_image(str(img_path))

        out_stem = img_path.stem + f"_SR_x{scale}"
        if window is not None:
            y, x, hh, ww = window
            out_stem += f"_scene_y{y}_x{x}_sz{hh}"
        out_name = out_stem + img_path.suffix
        viz_dir = os.path.join(output_root_str, "viz")
        viz_path = os.path.join(viz_dir, out_stem + "_viz.png")

        if args.mode == "self_eval":
            hr_scene = np.asarray(raw, dtype=np.float32)
            lr_scene = _downsample_to_lr(hr_scene, scale=scale)

            hr_proc, _, _ = preprocessor(hr_scene)
            lr_proc, _, _ = preprocessor(lr_scene)

            lr_t = to_tensor(lr_proc).unsqueeze(0).to(device)

            with torch.no_grad():
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                sr_t = model(lr_t).clamp(0, 1)
                _sync_if_cuda(device)
                t1 = time.perf_counter()

            sr = sr_t.squeeze(0).squeeze(0).cpu().numpy()
            hr_target = np.clip(hr_proc, 0, 1)

            h_min = min(sr.shape[0], hr_target.shape[0])
            w_min = min(sr.shape[1], hr_target.shape[1])
            sr = sr[:h_min, :w_min]
            hr_target = hr_target[:h_min, :w_min]
            lr_up = _resize_to(lr_proc, out_h=h_min, out_w=w_min)

            metrics = compute_metrics(sr, hr_target, data_range=1.0)
            metrics["image"] = img_path.name
            metrics["seconds"] = float(t1 - t0)
            metric_rows.append(metrics)
            per_image_seconds.append(float(t1 - t0))

            logger.info(
                f"    self_eval: psnr={metrics['psnr']:.3f} dB | ssim={metrics['ssim']:.4f} | {metrics['seconds']:.3f}s"
            )

            if args.save_viz and not args.no_save:
                _save_viz_panel(
                    viz_path,
                    panels=[
                        ("LR (Upscaled)", lr_up),
                        (f"SR ({args.model.upper()})", sr),
                        ("HR (Original)", hr_target),
                    ],
                )

            key = float(metrics.get("psnr", -float("inf")))
            if key > best_key:
                best_key = key
                best_viz = {
                    "path": os.path.join(viz_dir, "best_viz.png"),
                    "panels": [
                        ("LR (Upscaled)", lr_up),
                        (f"SR ({args.model.upper()})", sr),
                        ("HR (Original)", hr_target),
                    ],
                    "meta": {
                        "image": img_path.name,
                        "psnr": float(metrics["psnr"]),
                        "ssim": float(metrics["ssim"]),
                    },
                }

            if not args.no_save:
                save_image(sr, os.path.join(output_root_str, out_name))
        else:
            if args.scene_size and args.scene_size > 0:
                lr_scene, _, _ = preprocessor(raw)
                lr_t = to_tensor(lr_scene).unsqueeze(0).to(device)

                with torch.no_grad():
                    _sync_if_cuda(device)
                    t0 = time.perf_counter()
                    sr_t = model(lr_t).clamp(0, 1)
                    _sync_if_cuda(device)
                    t1 = time.perf_counter()

                sr = sr_t.squeeze(0).squeeze(0).cpu().numpy()
                per_image_seconds.append(float(t1 - t0))

                logger.info(f"    scene_infer: {t1 - t0:.3f}s")
                if args.save_viz and not args.no_save:
                    lr_up = _resize_to(lr_scene, out_h=sr.shape[0], out_w=sr.shape[1])
                    _save_viz_panel(
                        viz_path,
                        panels=[
                            ("LR (Upscaled)", lr_up),
                            (f"SR ({args.model.upper()})", sr),
                        ],
                    )
                if not args.no_save:
                    save_image(sr, os.path.join(output_root_str, out_name))
            else:
                sr = infer_image(
                    model=model,
                    image=raw,
                    preprocessor=preprocessor,
                    device=device,
                    scale=scale,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    stream_path=None if args.no_save else os.path.join(output_root_str, out_name),
                )

                if sr is not None:
                    if args.save_viz and not args.no_save:
                        lr_full, _, _ = preprocessor(raw)
                        lr_up = _resize_to(lr_full, out_h=sr.shape[0], out_w=sr.shape[1])
                        _save_viz_panel(
                            viz_path,
                            panels=[
                                ("LR (Upscaled)", lr_up),
                                (f"SR ({args.model.upper()})", sr),
                            ],
                        )
                    if not args.no_save:
                        save_image(sr, os.path.join(output_root_str, out_name))

    if per_image_seconds:
        avg_s = float(np.mean(per_image_seconds))
        logger.info(f"Average per-image forward time: {avg_s:.3f}s ({1.0 / max(avg_s, 1e-9):.2f} img/s)")

    if metric_rows:
        avg_psnr = float(np.mean([m["psnr"] for m in metric_rows]))
        avg_ssim = float(np.mean([m["ssim"] for m in metric_rows]))
        logger.info(f"Average self_eval metrics over {len(metric_rows)} images: PSNR={avg_psnr:.3f} dB | SSIM={avg_ssim:.4f}")

    if args.save_best_viz and (best_viz is not None) and (not args.no_save):
        _save_viz_panel(best_viz["path"], best_viz["panels"])
        meta = best_viz["meta"]
        logger.info(f"Best visualization came from: {meta['image']} | psnr={meta['psnr']:.3f} dB | ssim={meta['ssim']:.4f}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
