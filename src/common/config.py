# src/common/config.py
from __future__ import annotations
import os
import re
import json
import yaml
import time
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import torch

__all__ = [
    "deep_update",
    "load_yaml_with_includes",
    "to_attrdict",
    "cfg_get",
    "set_seed",
    "generate_run_id",
    "parse_overrides",
    "apply_overrides",
    "save_cfg_lock",
    "parse_torch_dtype", 
]

# -----------------------------
# Dict helpers
# -----------------------------
def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict b into dict a (mutates a)."""
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a

def _expand_env(x: Any) -> Any:
    if isinstance(x, str):
        return os.path.expandvars(os.path.expanduser(x))
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_expand_env(v) for v in x)
    return x

def load_yaml_with_includes(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file with support for:
      include:
        - other.yaml
        - sub/another.yaml
    Paths are resolved relative to `path`'s directory.
    Env vars and ~ are expanded.
    """
    path = Path(path).expanduser().resolve()
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("include", []) or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    base = {}
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = (path.parent / inc_path).resolve()
        deep_update(base, load_yaml_with_includes(inc_path))
    deep_update(base, cfg)
    return _expand_env(base)

class AttrDict(dict):
    """Dot-access dict (read/write)."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def to_attrdict(d: Dict[str, Any]) -> Any:
    a = AttrDict()
    for k, v in (d or {}).items():
        a[k] = to_attrdict(v) if isinstance(v, dict) else v
    return a

def cfg_get(obj: Union[dict, AttrDict], path: str, default: Any = None) -> Any:
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur

def parse_torch_dtype(name: str):
    """
    Map strings to torch dtypes.
    Supported: bf16|bfloat16, fp16|float16|half, fp32|float32|float, fp8 variants.
    Falls back to float32 if unknown.
    """
    n = str(name).lower()
    if n in ("bf16", "bfloat16", "bfloat"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32", "float"):
        return torch.float32
    if n in ("fp8", "float8", "e4m3", "e5m2", "fp8_e4m3", "fp8_e5m2"):
        # Prefer e4m3 if present, else e5m2; fallback to fp16 (safe) if neither exists.
        return getattr(torch, "float8_e4m3fn", getattr(torch, "float8_e5m2", torch.float16))
    return torch.float32

# -----------------------------
# Reproducibility & IDs
# -----------------------------
def set_seed(seed: int) -> None:
    import numpy as _np
    import torch as _torch
    random.seed(seed)
    _np.random.seed(seed)  # type: ignore
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

def generate_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    rnd = f"{random.randint(0, 9999):04d}"
    return f"{prefix}-{ts}-{rnd}"

# -----------------------------
# CLI overrides
# -----------------------------
_DOT_RE = re.compile(r"(?<!\\)\.")  # allow escaping \.
def _set_by_dots(root: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = _DOT_RE.split(dotted.replace("\\.", "\0"))
    keys = [k.replace("\0", ".") for k in keys]
    cur = root
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _yaml_parse_scalar(s: str) -> Any:
    # Robust type inference via YAML parser
    try:
        return yaml.safe_load(s)
    except Exception:
        return s

def parse_overrides(args: Iterable[str]) -> Dict[str, Any]:
    """
    Parse a list like:
      ["a.b=3", "training.dtype=bf16", "tags=[x,y]", "flag=true"]
    into a nested dict.
    """
    out: Dict[str, Any] = {}
    for item in args:
        if "=" not in item:
            # treat as boolean True
            _set_by_dots(out, item, True)
            continue
        k, v = item.split("=", 1)
        _set_by_dots(out, k, _yaml_parse_scalar(v))
    return out

def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update(cfg, overrides)

# -----------------------------
# Persist resolved config
# -----------------------------
def save_cfg_lock(cfg: Union[Dict[str, Any], AttrDict], out_dir: Union[str, Path], filename: str = "cfg.lock.yaml") -> str:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with path.open("w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), f, sort_keys=False)
    return str(path)
