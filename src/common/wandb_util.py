# src/common/wandb_util.py
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, Optional

__all__ = [
    "login", "init", "log", "finish", "run_url", "flatten_dict",
    "maybe_init_wandb", "wandb_log", "wandb_finish", "wandb_run_url",
]

try:
    import wandb  # type: ignore
except Exception as e:
    wandb = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# module state
_RUN = None
_WANDB_ENABLED = False

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, nk, sep=sep))
        else:
            items[nk] = v
    return items

# --- low-level API (kept for backward compat) ---
def login(api_key: Optional[str] = None, *, allow_offline: bool = True) -> None:
    if wandb is None:
        if allow_offline and os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled"}:
            print("[wandb] offline/disabled and wandb not installed; proceeding without login.")
            return
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception as e:
            print(f"[wandb] login warning: {e}")

def init(*, project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
         tags: Optional[Iterable[str]] = None, group: Optional[str] = None,
         mode: Optional[str] = None) -> Any:
    if mode:
        os.environ["WANDB_MODE"] = mode
    if wandb is None:
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")
    return wandb.init(project=project, name=name, config=config,
                      tags=list(tags) if tags else None, group=group, reinit=True)

def log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb is None:
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")
    wandb.log(data, step=step) if step is not None else wandb.log(data)

def finish() -> None:
    if wandb is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass

def run_url() -> Optional[str]:
    if wandb is None:
        return None
    try:
        return wandb.run.url if wandb.run is not None else None
    except Exception:
        return None

# --- high-level, safe wrappers used by scripts ---
def maybe_init_wandb(*, project: str, name: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None,
                     api_key: Optional[str] = None, enabled: bool = True,
                     mode: Optional[str] = None, tags: Optional[Iterable[str]] = None,
                     group: Optional[str] = None):
    """Initialize WANDB iff enabled and available. Returns run or None."""
    global _RUN, _WANDB_ENABLED
    _WANDB_ENABLED = bool(enabled)
    if not _WANDB_ENABLED:
        print("[wandb] disabled by flag; not initializing.")
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        print("[wandb] WANDB_MODE=disabled; not initializing.")
        _WANDB_ENABLED = False
        return None
    if wandb is None:
        print(f"[wandb] library not installed ({_IMPORT_ERR}); continuing without logging.")
        _WANDB_ENABLED = False
        return None
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception as e:
            print(f"[wandb] login warning: {e}")
    if mode:
        os.environ["WANDB_MODE"] = mode
    _RUN = wandb.init(project=project, name=name, config=config,
                      tags=list(tags) if tags else None, group=group, reinit=True)
    return _RUN

def wandb_log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if not _WANDB_ENABLED or wandb is None or _RUN is None:
        return
    try:
        wandb.log(data, step=step) if step is not None else wandb.log(data)
    except Exception as e:
        print(f"[wandb] log warning: {e}")

def wandb_finish() -> None:
    global _RUN
    if not _WANDB_ENABLED or wandb is None or _RUN is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass
    _RUN = None

def wandb_run_url() -> Optional[str]:
    if not _WANDB_ENABLED or wandb is None or _RUN is None:
        return None
    try:
        return wandb.run.url
    except Exception:
        return None
