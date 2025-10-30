from __future__ import annotations
from typing import Dict, Optional, Literal
import torch
import torch.nn.functional as F

__all__ = [
    "expected_alpha_and_goodput_from_logps",   # (kept) sampled-token ratio path
    "expected_span_from_alpha",                # (new) alpha_t -> span
    "alpha_from_full_distributions",           # (new) overlap on full dists
    "kl_from_full_distributions",              # (new) per-step KL (+ mean)
]

@torch.no_grad()
def expected_alpha_and_goodput_from_logps(
    *,
    draft_logp: torch.Tensor,     # [N,T] log q(y_t) on the sampled tokens
    teacher_logp: torch.Tensor,   # [N,T] log p(y_t) on the same tokens
    mask: Optional[torch.Tensor] = None,  # [N,T] 1=valid, 0=pad/invalid
    acceptance_cap: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    SD-consistent stats on the *sampled tokens only*:
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

    draft_logp = draft_logp.float()
    teacher_logp = teacher_logp.float()

    log_cap = float(torch.log(torch.tensor(acceptance_cap, device=device, dtype=draft_logp.dtype)))
    log_ratio = teacher_logp - draft_logp - log_cap
    alpha_token = torch.exp(log_ratio).clamp(max=1.0)                # [N,T]

    return expected_span_from_alpha(alpha_token, mask)


@torch.no_grad()
def expected_span_from_alpha(
    alpha_token: torch.Tensor,   # [N,T]
    mask: torch.Tensor,          # [N,T] float {0,1}
) -> Dict[str, torch.Tensor]:
    """Utility shared by both paths: alpha_t → mean alpha + expected span (masked)."""
    alpha_token = alpha_token.float()
    mask = (mask > 0).to(alpha_token.dtype)

    # tokenwise averages (diagnostic)
    valid = mask.sum(dim=1).clamp_min(1.0)                           # [N]
    alpha_sum = (alpha_token * mask).sum(dim=1)                       # [N]
    alpha_mean = alpha_sum / valid                                    # [N]

    # SD expected span: sum_t prod_{j<=t} alpha_j; for pads, set alpha=1
    alpha_eff = torch.where(mask > 0, alpha_token, torch.ones_like(alpha_token))
    cp = torch.cumprod(alpha_eff.clamp_min(1e-6), dim=1)             # [N,T]
    expected_span = (cp * mask).sum(dim=1)                            # [N]

    return {
        "alpha_token": alpha_token,        # [N,T]
        "alpha_mean": alpha_mean,          # [N]
        "accepted_tokens": expected_span,  # [N]
        "valid_tokens": valid,             # [N]
    }


@torch.no_grad()
def alpha_from_full_distributions(
    *,
    q_logp_full: torch.Tensor,     # [N,T,V] log q_t(v)
    p_logp_full: torch.Tensor,     # [N,T,V] log p_t(v)
    mask: torch.Tensor,            # [N,T] float {0,1}
    acceptance_cap: float = 1.0,
    topk: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Full-distribution overlap:
      alpha_t = sum_v min(q_t(v), p_t(v))  in probability space.
    If topk is provided, we approximate with the union top-K per (N,T), plus
    a single "other" bucket for the remaining mass (lower-bounded overlap).
    """
    assert q_logp_full.shape == p_logp_full.shape
    N, T, V = q_logp_full.shape
    device = q_logp_full.device

    # probs in fp32
    q = torch.exp(q_logp_full.float())
    p = torch.exp(p_logp_full.float())

    if topk is None or topk >= V:
        overlap = torch.minimum(q, p).sum(dim=-1)    # [N,T]
    else:
        # union topK approximation (lower bound)
        k = min(int(topk), V)
        qk_vals, qk_idx = torch.topk(q, k, dim=-1)                  # [N,T,k]
        pk_vals, pk_idx = torch.topk(p, k, dim=-1)                  # [N,T,k]
        # build union indices
        union_idx = torch.concat([qk_idx, pk_idx], dim=-1)          # [N,T,2k]
        union_idx = torch.unique(union_idx, dim=-1)                 # [N,T,<=2k]

        # gather selected masses
        gather_shape = union_idx.shape
        take = union_idx.reshape(N*T, -1)
        q_take = q.reshape(N*T, V).gather(1, take).reshape(gather_shape)  # [N,T,M]
        p_take = p.reshape(N*T, V).gather(1, take).reshape(gather_shape)

        head_overlap = torch.minimum(q_take, p_take).sum(dim=-1)    # [N,T]
        q_head = q_take.sum(dim=-1)
        p_head = p_take.sum(dim=-1)
        # Compress tail into a single bucket (lower bound on true overlap in tail)
        tail_overlap = torch.minimum(1.0 - q_head, 1.0 - p_head)
        overlap = head_overlap + tail_overlap                        # [N,T]

    # optional cap (behaves like sampled-ratio cap)
    if acceptance_cap is not None and acceptance_cap > 0:
        overlap = (overlap / float(acceptance_cap)).clamp(max=1.0)

    return expected_span_from_alpha(overlap, mask)


def kl_from_full_distributions(
    *,
    q_logp_full: torch.Tensor,   # [N,T,V]
    p_logp_full: torch.Tensor,   # [N,T,V]
    mask: torch.Tensor,          # [N,T] float {0,1}
    direction: Literal["q||p", "p||q"] = "q||p",
) -> Dict[str, torch.Tensor]:
    """
    Token-wise KL over the full vocabulary (masked), then mean over valid steps.
    Returns per-sample kl_mean [N] for direct supervised optimisation.
    """
    assert q_logp_full.shape == p_logp_full.shape
    q_logp_full = q_logp_full.float()
    p_logp_full = p_logp_full.float()
    mask = (mask > 0).to(q_logp_full.dtype)

    if direction == "q||p":
        q = torch.exp(q_logp_full)
        kl_t = (q * (q_logp_full - p_logp_full)).sum(dim=-1)  # [N,T]
    elif direction == "p||q":
        p = torch.exp(p_logp_full)
        kl_t = (p * (p_logp_full - q_logp_full)).sum(dim=-1)  # [N,T]
    else:
        raise ValueError(f"Unknown KL direction: {direction}")

    valid = mask.sum(dim=1).clamp_min(1.0)                    # [N]
    kl_mean = (kl_t * mask).sum(dim=1) / valid                # [N]

    return {
        "kl_token": kl_t,     # [N,T]
        "kl_mean": kl_mean,   # [N]
        "valid_tokens": valid # [N]
    }
