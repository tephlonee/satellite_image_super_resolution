"""
utils/config.py
---------------
Configuration loading and validation utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Union


class DotDict(dict):
    """Dictionary subclass supporting dot-notation access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, *keys, default=None):
        """Access nested keys safely: cfg.get_nested('train', 'lr', default=1e-4)"""
        d = self
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return DotDict(d) if isinstance(d, dict) else d


def load_config(path: Union[str, Path]) -> DotDict:
    """
    Load a YAML or JSON config file and return a DotDict.

    Args:
        path: Path to config file (.yaml, .yml, or .json)

    Returns:
        DotDict with all config values accessible via dot notation
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(f)
        elif path.suffix == ".json":
            raw = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return DotDict(raw)


def save_config(cfg: dict, path: Union[str, Path]) -> None:
    """Save config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def merge_configs(base: dict, override: dict) -> DotDict:
    """
    Recursively merge override into base config.

    Args:
        base: Base configuration dict
        override: Override values (takes priority)

    Returns:
        Merged DotDict
    """
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return DotDict(merged)
