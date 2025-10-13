
from __future__ import annotations
import torch

def accept_prob(p_on_path: torch.Tensor, q_on_path: torch.Tensor, eps: float=1e-8, floor: float=0.0) -> torch.Tensor:
    a = torch.minimum(torch.ones_like(p_on_path), (q_on_path + eps) / (p_on_path + eps))
    if floor > 0.0:
        a = torch.clamp(a, min=floor)
    return a

def survival(a: torch.Tensor) -> torch.Tensor:
    cum = torch.cumprod(a, dim=-1)
    s = torch.cat([torch.ones_like(cum[..., :1]), cum[..., :-1]], dim=-1)
    return s

def eal_reward(a: torch.Tensor) -> torch.Tensor:
    s = survival(a)
    return (s * a).sum(dim=-1)
