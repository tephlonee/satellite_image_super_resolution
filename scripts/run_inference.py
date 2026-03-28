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
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.device import get_device, set_seed
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from data.dataset import load_image, generate_lr
from data.preprocessing import SARPreprocessor
from data.augmentation import to_tensor

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
            tifffile.imsave(str(path), (array * 65535).astype(np.uint16))
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
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of SAR images to process")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Single SAR image path")
    parser.add_argument("--output_dir", type=str, default="results/inference",
                        help="Output directory for SR images")
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

    # Build preprocessor (no fitting needed at inference; uses saved stats from training if available)
    preprocessor = SARPreprocessor(cfg.preprocessing)

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

    logger.info(f"Processing {len(input_paths)} image(s) → {args.output_dir}")

    for img_path in input_paths:
        logger.info(f"  Processing: {img_path.name}")
        raw = load_image(str(img_path))

        out_name = img_path.stem + f"_SR_x{scale}" + img_path.suffix
        sr = infer_image(
            model=model,
            image=raw,
            preprocessor=preprocessor,
            device=device,
            scale=scale,
            tile_size=args.tile_size,
            overlap=args.overlap,
            stream_path=os.path.join(args.output_dir, out_name),
        )

        if sr is not None:
            save_image(sr, os.path.join(args.output_dir, out_name))

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
