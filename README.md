# SAR Super-Resolution — Production Codebase

Deep learning super-resolution for **Capella SAR imagery** using PyTorch.  
Supports two model families (SRCNN and GAN), patch-based training, SAR-specific preprocessing, and Optuna hyperparameter tuning — all optimised for small datasets (~40 images).

---

## Project Structure

```
sar_sr/
├── configs/
│   └── config.yaml             ← All settings (data, models, training, tuning)
├── data/
│   ├── preprocessing.py        ← Log transform, normalization, Lee/Frost/Median filters
│   ├── augmentation.py         ← Paired LR-HR augmentation (flip, rotate, crop)
│   └── dataset.py              ← SARDataset, DataLoader factory, train/val/test split
├── models/
│   ├── srcnn.py                ← SRCNNDeep (pixel-shuffle, residual blocks, no BN)
│   └── gan.py                  ← SRGenerator + PatchDiscriminator (LSGAN)
├── train/
│   ├── train_srcnn.py          ← SRCNN trainer (early stopping, cosine LR, checkpointing)
│   └── train_gan.py            ← GAN trainer (warmup → adversarial, separate G/D optimizers)
├── evaluation/
│   ├── metrics.py              ← PSNR, SSIM
│   └── losses.py               ← L1/MSE, TV loss, LSGAN losses
├── utils/
│   ├── config.py               ← YAML/JSON loader with dot-notation access
│   ├── logger.py               ← Structured logging + TrainingLogger
│   ├── checkpoint.py           ← save/load checkpoints, EarlyStopping
│   └── device.py               ← Auto device selection, seed setting
├── tuning/
│   └── tune.py                 ← Optuna study with MedianPruner
├── scripts/
│   ├── run_srcnn.py            ← Train SRCNN end-to-end
│   ├── run_gan.py              ← Train GAN end-to-end
│   ├── run_tuning.py           ← Hyperparameter search
│   └── run_inference.py        ← Inference on new SAR images (tiled)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GeoTIFF support (recommended for Capella images):
```bash
pip install rasterio tifffile
```

### 2. Prepare data

Place your SAR images (`.tif`, `.png`, etc.) in a directory and set `data.image_dir` in `configs/config.yaml`:

```yaml
data:
  image_dir: "path/to/your/sar/images"
  scale_factor: 4
  patch_size: 64
```

The codebase **automatically generates LR/HR pairs** via bicubic downsampling — no pre-processing of images required.

### 3. Train SRCNN

```bash
python scripts/run_srcnn.py --config configs/config.yaml
```

Resume from checkpoint:
```bash
python scripts/run_srcnn.py --config configs/config.yaml --resume checkpoints/srcnn/latest.pth
```

### 4. Train GAN

```bash
python scripts/run_gan.py --config configs/config.yaml
```

### 5. Hyperparameter tuning

```bash
python scripts/run_tuning.py --config configs/config.yaml --model srcnn --n_trials 50
```

Best parameters are saved to `tuning_results_srcnn.json`.

### 6. Inference on new images

```bash
# Single image
python scripts/run_inference.py \
    --config configs/config.yaml \
    --model srcnn \
    --checkpoint checkpoints/srcnn/best_model.pth \
    --input_path my_sar_image.tif \
    --output_dir results/

# Whole directory
python scripts/run_inference.py \
    --config configs/config.yaml \
    --model gan \
    --checkpoint checkpoints/gan/generator_best.pth \
    --input_dir path/to/images/ \
    --output_dir results/sr_output/
```

---

## Key Design Decisions for SAR

| Challenge | Solution |
|-----------|----------|
| **Multiplicative speckle** | Log transform converts speckle to additive noise |
| **Small dataset (~40 images)** | Patch-based training (~50 patches/image), heavy augmentation |
| **Overfitting** | Weight decay (L2), early stopping, cosine LR schedule |
| **GAN hallucination risk** | L1 content loss dominates (weight=1.0); adversarial weight=0.01 |
| **Checkerboard artifacts** | Pixel-shuffle upsampling instead of transposed conv |
| **Batch norm issues** | No BN anywhere — avoids distorting SAR amplitude statistics |
| **Large image inference** | Overlapping tile strategy with blending |

---

## Configuration Reference

All settings live in `configs/config.yaml`. Key sections:

```yaml
data:
  scale_factor: 4          # 2x, 4x, or 8x SR
  patch_size: 64           # HR patch size for training

preprocessing:
  log_transform: true      # Essential for SAR
  speckle_filter:
    enabled: false         # Optional: lee / frost / median
    method: "lee"
    kernel_size: 3

train_srcnn:
  loss: "l1"               # l1 or mse
  tv_loss_weight: 0.0      # > 0 enables TV regularization
  weight_decay: 1.0e-4     # L2 regularization

train_gan:
  pixel_loss_weight: 1.0       # L1 dominates
  adversarial_loss_weight: 0.01  # Keep low for SAR
  warmup_epochs: 10            # G-only warmup before adding D
```

---

## Regularization Strategy

For small SAR datasets the following regularization hierarchy is used:

1. **Data augmentation** (primary): flips, rotations, random crops
2. **Weight decay** (L2): configurable, default 1e-4
3. **Early stopping**: monitors validation PSNR
4. **TV loss** (optional): very light spatial smoothness penalty
5. **No dropout / no batch norm**: inappropriate for SR on amplitude images

---

## Metrics

Both models are evaluated with:
- **PSNR** (dB): higher is better; typical SR gains are 0.5–3 dB over bicubic
- **SSIM**: higher is better; range [0, 1]

Metrics are logged per epoch to console and to `checkpoints/{model}/train.log`.
