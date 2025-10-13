
from __future__ import annotations
import torch

def grpo_loss(logp: torch.Tensor, ref_logp: torch.Tensor, adv: torch.Tensor, survival: torch.Tensor, beta: float=0.05):
    S, K = logp.shape
    adv_tok = adv[:, None].expand(S, K)
    denom = survival.sum().clamp_min(1e-8)
    policy = -((adv_tok * survival) * logp).sum() / denom
    kl = beta * ((logp.exp()) * (logp - ref_logp)).mean()
    loss = policy + kl
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy": float(policy.detach().cpu()),
        "kl": float(kl.detach().cpu())
    }
    return loss, stats
