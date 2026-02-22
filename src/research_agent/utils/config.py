"""Centralized configuration loading."""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_config_cache: Optional[dict] = None
_CONFIG_FILENAME = "configs/config.yaml"


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing configs/."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety limit
        if (current / "configs").is_dir():
            return current
        current = current.parent
    # Fallback: assume CWD
    return Path.cwd()


def load_config(config_path: Optional[str] = None, *, use_cache: bool = True) -> dict:
    """Load and cache the YAML configuration.

    Args:
        config_path: Override path. If None, auto-discovers configs/config.yaml.
        use_cache: If True (default), returns cached result on subsequent calls.
    """
    global _config_cache
    if use_cache and _config_cache is not None and config_path is None:
        return _config_cache

    if config_path:
        path = Path(config_path)
    else:
        path = _find_project_root() / _CONFIG_FILENAME

    if not path.exists():
        logger.debug(f"Config file not found: {path}")
        result = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            result = {}

    if config_path is None:
        _config_cache = result
    return result


def clear_config_cache():
    """Clear the cached config (useful for testing)."""
    global _config_cache
    _config_cache = None
