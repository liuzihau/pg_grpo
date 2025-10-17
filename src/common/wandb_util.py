# src/common/wandb_util.py
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, Optional

__all__ = ["login", "init", "log", "finish", "run_url", "flatten_dict"]

try:
    import wandb  # type: ignore
except Exception as e:
    wandb = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def login(api_key: Optional[str] = None, *, allow_offline: bool = True) -> None:
    """
    Log into Weights & Biases. If WANDB_MODE=offline or allow_offline=False is set,
    behavior adapts accordingly.
    """
    if wandb is None:
        if allow_offline and os.environ.get("WANDB_MODE", "").lower() == "offline":
            print("[wandb] offline mode and wandb not installed; proceeding without login.")
            return
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")

    if api_key:
        # Avoid noisy "already logged in" errors
        try:
            wandb.login(key=api_key)
        except Exception as e:
            # If already logged in, wandb raises a benign error; keep going.
            print(f"[wandb] login warning: {e}")

def init(
    *,
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    group: Optional[str] = None,
    mode: Optional[str] = None,  # "online" | "offline" | None
) -> Any:
    """
    Initialize a wandb run. Returns the run object.
    """
    if mode:
        os.environ["WANDB_MODE"] = mode

    if wandb is None:
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")

    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=list(tags) if tags else None,
        group=group,
        reinit=True,
    )
    return run

def log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb is None:
        raise RuntimeError(f"[wandb] Library not available: {_IMPORT_ERR}")
    if step is not None:
        wandb.log(data, step=step)
    else:
        wandb.log(data)

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
