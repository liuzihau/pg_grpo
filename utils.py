
from __future__ import annotations
from typing import Any, Dict
import yaml
import random 
import numpy as np
import torch

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(name: str = "cuda") -> torch.device:
    if name == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        # Convert nested dicts recursively
        return AttrDict({k: to_attrdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_attrdict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_attrdict(x) for x in obj)
    elif isinstance(obj, set):
        return {to_attrdict(x) for x in obj}
    else:
        return obj


import torch

def compute_first_reject_mask(accepted_mask: torch.Tensor) -> torch.Tensor:
    """
    accepted_mask: [T,N] bool/float (1 where accepted)
    Returns first_reject_mask: [T,N] with exactly one 1 at the earliest rejection per sequence (if any)
    """
    T, N = accepted_mask.shape
    acc = accepted_mask > 0.5
    first_reject_mask = torch.zeros_like(accepted_mask, dtype=accepted_mask.dtype)
    for n in range(N):
        acc_n = acc[:, n]
        reject_pos = (~acc_n).nonzero(as_tuple=False)
        if reject_pos.numel() > 0:
            t0 = int(reject_pos[0].item())
            first_reject_mask[t0, n] = 1.0
    return first_reject_mask
