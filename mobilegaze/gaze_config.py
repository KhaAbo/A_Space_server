"""Configuration loader for Gaze Estimation Pipeline."""

from pathlib import Path
from typing import Any, Dict

import yaml
import torch


class Config:
    """Configuration container with section-based access."""

    def __init__(self, data: Dict[str, Any], root: Path):
        self.data = data
        self.root = root

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value."""
        return self.data.get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        """Get a configuration section as a dictionary."""
        return self.data.get(name, {})

    def path(self, key: str) -> Path:
        """Get a path from the paths section, resolved relative to root."""
        raw = self.data["paths"].get(key)
        if raw is None:
            raise KeyError(f"Path '{key}' not found in config")
        p = Path(raw)
        if p.is_absolute():
            return p
        return self.root / p

    def device(self) -> torch.device:
        """Get the configured device, with auto-detection fallback."""
        device_cfg = self.section("device").get("default")
        return select_device(device_cfg)


def select_device(device: str | None = None) -> torch.device:
    """Select compute device with auto-detection.

    Priority: specified device > CUDA > MPS > CPU
    """
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to config file. If None, uses default config location.

    Returns:
        Config object with loaded settings.
    """
    if path is None:
        path = Path(__file__).parent / "config" / "gaze_config.yml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Root is the mobilegaze folder (where this file lives)
    root = Path(__file__).parent
    return Config(data, root)


# Default config instance (lazy loaded)
_default_config: Config | None = None


def get_config() -> Config:
    """Get the default config instance (singleton pattern)."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config

