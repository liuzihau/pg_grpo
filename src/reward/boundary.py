# src/reward/boundary.py
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn.functional as F

__all__ = [
    "expected_alpha_and_goodput_from_logps",   # (kept) token-wise ratio path
    "alpha_overlap_from_teacher_topk",         # (new) top-K overlap path
]

def expected_alpha_and_goodput_from_logps(
    *,
    draft_logp: torch.Tensor,     # [N,T] log q(y_t) on the sampled tokens
    teacher_logp: torch.Tensor,   # [N,T] log p(y_t) on the same tokens
    mask: Optional[torch.Tensor] = None,  # [N,T] 1=valid, 0=pad/invalid
    acceptance_cap: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    SD-consistent stats.
      alpha_t(y_t) = min(1, p(y_t) / (cap * q(y_t))).
    'accepted_tokens' is the expected accepted span:
        E[L] = sum_t prod_{j<=t} alpha_j
    """
    assert draft_logp.shape == teacher_logp.shape, (draft_logp.shape, teacher_logp.shape)
    device = draft_logp.device
    N, T = draft_logp.shape

    if mask is None:
        mask = torch.ones_like(draft_logp, device=device, dtype=draft_logp.dtype)
    else:
        mask = (mask > 0).to(device=device, dtype=draft_logp.dtype)  # force {0,1} float

    # per-token acceptance on sampled tokens
    log_cap = float(torch.log(torch.tensor(acceptance_cap, device=device, dtype=draft_logp.dtype)))
    log_ratio = teacher_logp - draft_logp - log_cap
    alpha_token = torch.exp(log_ratio).clamp(max=1.0)                # [N,T]

    # --- strictly exclude padded steps from any tokenwise aggregates
    alpha_masked = alpha_token * mask                                # [N,T]

    # tokenwise averages (diagnostic)
    valid = mask.sum(dim=1).clamp_min(1.0)                           # [N]
    alpha_sum = alpha_masked.sum(dim=1)                               # [N]
    alpha_mean = alpha_sum / valid                                    # [N]

    # SD expected span: sum_t prod_{j<=t} alpha_j
    # For padded steps, we set alpha=1 so they don't truncate the product.
    alpha_eff = torch.where(mask > 0, alpha_token, torch.ones_like(alpha_token))
    cp = torch.cumprod(alpha_eff.clamp_min(1e-6), dim=1)             # [N,T]
    expected_span = (cp * mask).sum(dim=1)                            # [N]

    # rejects = (valid - expected_span)
    # goodput = expected_span / (1.0 + rejects).clamp_min(1e-6)
    # reject_rate = 1.0 - (expected_span / valid)

    return {
        "alpha_token": alpha_token,        # [N,T] (raw per-step alpha)
        "alpha_mean": alpha_mean,          # [N]   mean alpha over valid steps only
        "accepted_tokens": expected_span,  # [N]   expected accepted span (use this)
        "valid_tokens": valid,             # [N]
        # "goodput": goodput,                # [N]
        # "reject_rate": reject_rate,        # [N]
        # "alpha_sum": alpha_sum,            # [N]   optional diagnostic
    }


@torch.no_grad()
def alpha_overlap_from_teacher_topk(
    *,
    draft_logits: torch.Tensor,                # [N,T,V] logits on the *T steps of interest*
    teacher_topk_ids: torch.Tensor,            # [N,T,K] token ids from teacher (pregen KD)
    teacher_topk_logprobs: torch.Tensor,       # [N,T,K] log p(x) from teacher on those ids
    mask: Optional[torch.Tensor] = None,       # [N,T] 1=valid, 0=padded
    acceptance_cap: float = 1.0,
    sample_span: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Distribution-overlap acceptance (probabilistic SD):
      alpha_t = sum_x min(q_t(x), p_t(x)/cap).
    We approximate the sum using teacher top-K tokens (and add a conservative tail term).
    Returns per-step alpha, a sampled accepted span (or expected), and goodput-ish metrics.

    This path avoids a teacher forward during GRPO if you use pregen KD teacher top-K.
    """
    N, T, V = draft_logits.shape
    device = draft_logits.device
    if mask is None:
        mask = torch.ones((N, T), dtype=draft_logits.dtype, device=device)

    # Draft probs on the teacher's top-K ids (enough for a good lower bound of overlap)
    logq_full = F.log_softmax(draft_logits, dim=-1)                         # [N,T,V]
    ids = teacher_topk_ids.to(device=device, dtype=torch.long)              # [N,T,K]
    logq_top = torch.gather(logq_full, dim=-1, index=ids)                   # [N,T,K]
    q_top = logq_top.exp()                                                  # [N,T,K]

    # Teacher probs (already log) -> scale by 1/cap for the acceptance rule
    logp_top = teacher_topk_logprobs.to(device=device, dtype=torch.float32) # [N,T,K]
    p_top = (logp_top.exp() / float(acceptance_cap))                        # [N,T,K]

    # Overlap on the covered tokens
    overlap_top = torch.minimum(q_top, p_top).sum(dim=-1)                   # [N,T]

    # Tail masses we didn't enumerate (conservative lower-bound overlap adds min of tails)
    q_tail = (1.0 - q_top.sum(dim=-1)).clamp_min(0.0)                       # [N,T]
    p_tail = (1.0 - p_top.sum(dim=-1)).clamp_min(0.0)                       # [N,T]
    alpha = (overlap_top + torch.minimum(q_tail, p_tail)).clamp(0.0, 1.0)   # [N,T]

    # Respect valid-token mask: treat invalid steps as alpha=1 so they don't stop the span
    alpha_eff = torch.where(mask > 0, alpha, torch.ones_like(alpha))

    # Expected accepted span for a K-step proposal: E[L] = sum_i prod_{j<=i} alpha_j
    cp = torch.cumprod(alpha_eff.clamp_min(1e-6), dim=1)                    # [N,T]
    exp_span = (cp * mask).sum(dim=1)                                       # [N]

    # Monte Carlo sampled span via u ~ U(0,1)
    if sample_span:
        u = torch.rand_like(alpha)
        # accept decisions only on valid steps; invalid => force-accept to not truncate span
        accept = torch.where(mask > 0, (u < alpha).to(alpha.dtype), torch.ones_like(alpha))
        pref = torch.cumprod(accept, dim=1)                                  # 1.. until first 0, then 0..
        span = (pref * mask).sum(dim=1)                                      # [N]
    else:
        span = exp_span

    valid = mask.sum(dim=1).clamp_min(1.0)
    alpha_mean = (alpha * mask).sum(dim=1) / valid
    rejects = (valid - span)
    goodput = span / (1.0 + rejects).clamp_min(1e-6)
    reject_rate = 1.0 - (span / valid)

    return {
        "alpha_token": alpha,        # [N,T]
        "accepted_span": span,       # [N] sampled (or expected if sample_span=False)
        "expected_span": exp_span,   # [N]
        "alpha_mean": alpha_mean,    # [N]
        "goodput": goodput,          # [N]
        "reject_rate": reject_rate,  # [N]
        "accepted_tokens": span,     # [N]
        "valid_tokens": valid,       # [N]
    }
