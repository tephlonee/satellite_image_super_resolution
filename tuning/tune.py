"""
tuning/tune.py
--------------
Optuna-based hyperparameter tuning for SRCNN and GAN models.

Tunes:
  - Learning rate
  - Weight decay
  - Batch size
  - n_feats (filter count)
  - n_layers (depth)
  - patch_size

Each trial trains for a reduced number of epochs and reports val PSNR.
A pruner (MedianPruner) kills unpromising trials early for efficiency.

Usage:
    python scripts/run_tuning.py --config configs/config.yaml --model srcnn
"""

import copy
import torch
from typing import Optional

from utils.config import load_config, DotDict
from utils.device import get_device, set_seed
from utils.logger import get_logger
from data.dataset import build_dataloaders

logger = get_logger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _srcnn_objective(trial, base_cfg: DotDict, n_epochs: int = 30) -> float:
    """Optuna objective for SRCNN tuning."""
    from models.srcnn import SRCNNDeep
    from evaluation.losses import SRCNNLoss
    from evaluation.metrics import compute_metrics
    from utils.checkpoint import EarlyStopping

    ss = base_cfg.tuning.get("search_space", {})

    # Sample hyperparameters
    lr = trial.suggest_float(
        "learning_rate",
        ss.get("learning_rate", [1e-5, 1e-3])[0],
        ss.get("learning_rate", [1e-5, 1e-3])[1],
        log=True,
    )
    wd = trial.suggest_float(
        "weight_decay",
        ss.get("weight_decay", [1e-5, 1e-3])[0],
        ss.get("weight_decay", [1e-5, 1e-3])[1],
        log=True,
    )
    batch_size = trial.suggest_categorical("batch_size", ss.get("batch_size", [8, 16, 32]))
    n_feats = trial.suggest_categorical("n_feats", ss.get("n_feats", [32, 64, 128]))
    n_layers = trial.suggest_categorical("n_layers", ss.get("n_layers", [4, 6, 8, 10]))
    patch_size = trial.suggest_categorical("patch_size", ss.get("patch_size", [32, 64, 96]))

    # Build modified config for this trial
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["patch_size"] = patch_size
    cfg["train_srcnn"]["batch_size"] = batch_size
    cfg["train_srcnn"]["learning_rate"] = lr
    cfg["train_srcnn"]["weight_decay"] = wd
    cfg["train_srcnn"]["epochs"] = n_epochs
    cfg["train_srcnn"]["early_stopping"] = {"enabled": True, "patience": 10, "min_delta": 1e-5}
    cfg["srcnn"]["n_feats"] = n_feats
    cfg["srcnn"]["n_layers"] = n_layers

    device = get_device(cfg.get("device", "auto"))

    try:
        train_loader, val_loader, _, preprocessor = build_dataloaders(
            cfg,
            batch_size=cfg.train_srcnn.batch_size,
        )

        model = SRCNNDeep(
            n_channels=cfg.srcnn.get("n_channels", 1),
            n_feats=n_feats,
            n_layers=n_layers,
            scale=cfg.data.get("scale_factor", 4),
        ).to(device)

        criterion = SRCNNLoss(loss_type="l1", tv_loss_weight=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        early_stopping = EarlyStopping(patience=10, mode="max")

        best_psnr = 0.0

        for epoch in range(n_epochs):
            # Train
            model.train()
            for lr_img, hr_img in train_loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)
                optimizer.zero_grad()
                sr = model(lr_img)
                loss, _ = criterion(sr, hr_img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            val_psnr = 0.0
            n_val = 0
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img = lr_img.to(device)
                    hr_img = hr_img.to(device)
                    sr = model(lr_img).clamp(0, 1)
                    metrics = compute_metrics(sr, hr_img)
                    val_psnr += metrics["psnr"]
                    n_val += 1
            val_psnr /= max(n_val, 1)

            if val_psnr > best_psnr:
                best_psnr = val_psnr

            # Report to Optuna for pruning
            trial.report(val_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if early_stopping(val_psnr):
                break

        return best_psnr

    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def _gan_objective(trial, base_cfg: DotDict, n_epochs: int = 20) -> float:
    """Optuna objective for GAN generator tuning (warmup only for speed)."""
    from models.gan import SRGenerator
    from evaluation.metrics import compute_metrics
    from utils.checkpoint import EarlyStopping

    ss = base_cfg.tuning.get("search_space", {})

    lr = trial.suggest_float(
        "g_learning_rate",
        ss.get("learning_rate", [1e-5, 1e-3])[0],
        ss.get("learning_rate", [1e-5, 1e-3])[1],
        log=True,
    )
    wd = trial.suggest_float(
        "weight_decay",
        ss.get("weight_decay", [1e-5, 1e-3])[0],
        ss.get("weight_decay", [1e-5, 1e-3])[1],
        log=True,
    )
    batch_size = trial.suggest_categorical("batch_size", ss.get("batch_size", [8, 16]))
    n_feats = trial.suggest_categorical("n_feats", ss.get("n_feats", [32, 64]))
    n_res_blocks = trial.suggest_int("n_res_blocks", 4, 12, step=2)
    patch_size = trial.suggest_categorical("patch_size", ss.get("patch_size", [32, 64]))

    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["patch_size"] = patch_size
    cfg["train_gan"]["batch_size"] = batch_size
    cfg["gan"]["n_feats"] = n_feats
    cfg["gan"]["n_res_blocks"] = n_res_blocks

    device = get_device(cfg.get("device", "auto"))

    try:
        train_loader, val_loader, _, _ = build_dataloaders(
            cfg,
            batch_size=cfg.train_gan.batch_size,
        )

        generator = SRGenerator(
            n_channels=cfg.gan.get("n_channels", 1),
            n_feats=n_feats,
            n_res_blocks=n_res_blocks,
            scale=cfg.data.get("scale_factor", 4),
        ).to(device)

        optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.L1Loss()
        early_stopping = EarlyStopping(patience=8, mode="max")

        best_psnr = 0.0

        for epoch in range(n_epochs):
            generator.train()
            for lr_img, hr_img in train_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                optimizer.zero_grad()
                sr = generator(lr_img)
                loss = criterion(sr, hr_img)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer.step()

            generator.eval()
            val_psnr = 0.0
            n_val = 0
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    sr = generator(lr_img).clamp(0, 1)
                    val_psnr += compute_metrics(sr, hr_img)["psnr"]
                    n_val += 1
            val_psnr /= max(n_val, 1)

            if val_psnr > best_psnr:
                best_psnr = val_psnr

            trial.report(val_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if early_stopping(val_psnr):
                break

        return best_psnr

    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_tuning(
    cfg_path: str,
    model: str = "srcnn",
    n_trials_override: Optional[int] = None,
    n_epochs_per_trial: int = 30,
    seed: int = 42,
) -> dict:
    """
    Launch an Optuna hyperparameter search.

    Args:
        cfg_path:            Path to config YAML
        model:               "srcnn" or "gan"
        n_trials_override:   Override n_trials from config
        n_epochs_per_trial:  Epochs per trial (keep low for speed)
        seed:                Random seed

    Returns:
        Dict with best_params and best_value (PSNR)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Install Optuna: pip install optuna")

    set_seed(seed)
    cfg = load_config(cfg_path)
    tune_cfg = cfg.tuning

    n_trials = n_trials_override or tune_cfg.get("n_trials", 50)
    timeout = tune_cfg.get("timeout", 3600)
    study_name = tune_cfg.get("study_name", f"sar_sr_{model}_tuning")
    storage = tune_cfg.get("storage", None)

    objective_fn = _srcnn_objective if model == "srcnn" else _gan_objective

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize PSNR
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True,
    )

    logger.info(f"Starting Optuna study '{study_name}' | {n_trials} trials | model={model}")

    study.optimize(
        lambda trial: objective_fn(trial, cfg, n_epochs=n_epochs_per_trial),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    logger.info(f"Best trial: PSNR = {study.best_value:.4f} dB")
    logger.info(f"Best params: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
