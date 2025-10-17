# src/reward/boundary.py
from __future__ import annotations
from typing import Dict
import torch

__all__ = ["expected_alpha_and_goodput_from_logps"]

def expected_alpha_and_goodput_from_logps(
    *,
    draft_logp: torch.Tensor,     # [N,T], log p_draft(y_t)
    teacher_logp: torch.Tensor,   # [N,T], log p_teacher(y_t)
    mask: torch.Tensor | None = None,   # [N,T] 1 for valid tokens else 0
    acceptance_cap: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute *expected* speculative-acceptance stats tokenwise, then aggregate.
      alpha_t = min(1, p_draft / (cap * p_teacher)) = clip(exp(logp_d - logp_t - log cap), max=1)
    Returns dict of per-sample scalars: alpha_mean, goodput, reject_rate, accepted_tokens.
    """
    assert draft_logp.shape == teacher_logp.shape, (draft_logp.shape, teacher_logp.shape)
    N, T = draft_logp.shape
    device = draft_logp.device
    if mask is None:
        mask = torch.ones_like(draft_logp, device=device)

    log_cap = float(torch.log(torch.tensor(acceptance_cap, device=device, dtype=draft_logp.dtype)))
    log_ratio = draft_logp - teacher_logp - log_cap
    alpha = torch.exp(log_ratio).clamp(max=1.0) * mask  # [N,T]

    accepted = alpha.sum(dim=1)               # [N]
    valid = mask.sum(dim=1).clamp_min(1.0)    # [N]
    rejects = (valid - accepted)              # [N]

    alpha_mean = accepted / valid
    reject_rate = 1.0 - alpha_mean
    goodput = accepted / (1.0 + rejects).clamp_min(1e-6)  # tokens per teacher call
    return {
        "alpha_mean": alpha_mean,
        "goodput": goodput,
        "reject_rate": reject_rate,
        "accepted_tokens": accepted,
        "valid_tokens": valid,
    }
