# src/common/io.py
from __future__ import annotations
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "makedirs",
    "atomic_write_bytes",
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_pt",
    "load_pt",
    "list_shards",
    "timestamp",
    "copytree_safe",
    "copy_file_safe",
    "git_info",
]

def makedirs(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    data = json.dumps(obj, indent=indent).encode("utf-8")
    atomic_write_bytes(path, data)

def load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)

def save_yaml(obj: Any, path: str | Path) -> None:
    import yaml
    with Path(path).open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def load_yaml(path: str | Path) -> Any:
    import yaml
    with Path(path).open("r") as f:
        return yaml.safe_load(f)

def save_pt(obj: Any, path: str | Path) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))

def load_pt(path: str | Path) -> Any:
    import torch
    return torch.load(str(path), map_location="cpu")

def list_shards(directory: str | Path, prefix: str = "shard_", suffix: str = ".pt") -> List[str]:
    p = Path(directory)
    return sorted(str(x) for x in p.glob(f"{prefix}*{suffix}"))

def timestamp() -> str:
    import time
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def copytree_safe(src: str | Path, dst: str | Path, symlinks: bool = False, ignore=None) -> None:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore)

def copy_file_safe(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def git_info(root: str | Path | None = None) -> Dict[str, str]:
    """
    Return {'commit': ..., 'branch': ...} if inside a git repo, else {}.
    """
    root = Path(root or ".").resolve()
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {}
