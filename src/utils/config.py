from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top-level YAML, got: {type(data)}")
    return data


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries (overrides win)."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = copy.deepcopy(v)
    return out


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_to_yaml(config: Mapping[str, Any]) -> str:
    """Render a config dict as YAML for saving alongside outputs."""
    return yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)

