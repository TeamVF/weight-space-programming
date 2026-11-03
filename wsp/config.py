"""Load a TOML config and apply WSP_* environment overrides.

Author: wanghaobo
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "model": "mock",
    "effort": "high",
    "runs_dir": "runs",
    "sandbox_timeout_s": 5,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return defaults, then an optional TOML file, then environment overrides.

    Search order for the file: the `path` argument, `WSP_CONFIG`, then
    `config.toml` in the working directory when that file exists.
    """
    config = dict(DEFAULTS)
    source = _resolve_path(path)
    if source is not None:
        loaded = tomllib.loads(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must be a TOML table: {source}")
        config.update(loaded)
    if "WSP_MODEL" in os.environ:
        config["model"] = os.environ["WSP_MODEL"]
    if "WSP_EFFORT" in os.environ:
        config["effort"] = os.environ["WSP_EFFORT"]
    if "WSP_RUNS_DIR" in os.environ:
        config["runs_dir"] = os.environ["WSP_RUNS_DIR"]
    if "WSP_SANDBOX_TIMEOUT" in os.environ:
        config["sandbox_timeout_s"] = float(os.environ["WSP_SANDBOX_TIMEOUT"])
    return config


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    env = os.environ.get("WSP_CONFIG")
    if env:
        return Path(env)
    default = Path("config.toml")
    if default.is_file():
        return default
    return None
