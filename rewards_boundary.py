from __future__ import annotations
from typing import Optional, Tuple, Literal
import torch
import torch.nn.functional as F

DivergenceKind = Literal["kl", "ce", "js", "alpha"]


@torch.no_grad()
def _softmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.log_softmax(-1).exp()

def _cross_entropy_from_pqlog(p: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    # CE(p, q) = - sum_v p(v) log q(v)
    return -(p * q_log).sum(dim=-1)

def _forward_kl(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    # KL(p || q) = sum p * (log p - log q)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(dim=-1)

def _js_divergence(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    # JS = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q)
    p = p_log.exp()
    q = q_log.exp()
    m = 0.5 * (p + q) + 1e-8
    m_log = (m + 1e-8).log()
    return 0.5 * (p * (p_log - m_log)).sum(dim=-1) + 0.5 * (q * (q_log - m_log)).sum(dim=-1)

def _alpha_divergence(p_log: torch.Tensor, q_log: torch.Tensor, alpha: float) -> torch.Tensor:
    # D_α(p||q) = (1/(α(1-α))) * (1 - sum p^α q^(1-α))
    p = p_log.exp()
    q = q_log.exp()
    s = (p.clamp_min(1e-12) ** alpha) * (q.clamp_min(1e-12) ** (1 - alpha))
    mix = s.sum(dim=-1).clamp_min(1e-12)
    return (1.0 / (alpha * (1.0 - alpha))) * (1.0 - mix)

def topk_project_logits_to_teacher_support(
    teacher_logits: torch.Tensor, draft_logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N, V = teacher_logits.shape
    k = min(k, V)
    _, topk_idx = teacher_logits.topk(k=k, dim=-1)
    t_log = teacher_logits.gather(-1, topk_idx).log_softmax(-1)  # [T,N,K]
    q_log = draft_logits.gather(-1, topk_idx).log_softmax(-1)    # [T,N,K]
    return t_log, q_log

def compute_token_divergence(
    teacher_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    kind: DivergenceKind = "kl",
    alpha: float = 0.5,
    topk_for_ce: int = 0,
) -> torch.Tensor:
    if topk_for_ce and kind in ("kl", "ce"):
        t_log, q_log = topk_project_logits_to_teacher_support(teacher_logits, draft_logits, topk_for_ce)
        if kind == "ce":
            return _cross_entropy_from_pqlog(t_log.exp(), q_log)
        else:
            return _forward_kl(t_log, q_log)

    t_log = teacher_logits.log_softmax(-1)
    q_log = draft_logits.log_softmax(-1)

    if kind == "ce":
        return _cross_entropy_from_pqlog(t_log.exp(), q_log)
    if kind == "kl":
        return _forward_kl(t_log, q_log)
    if kind == "js":
        return _js_divergence(t_log, q_log)
    if kind == "alpha":
        return _alpha_divergence(t_log, q_log, alpha)
    raise ValueError(f"Unknown divergence kind: {kind}")

def compute_entropy_bonus(draft_logits: torch.Tensor) -> torch.Tensor:
    q_log = draft_logits.log_softmax(-1)
    q = q_log.exp()
    return -(q * q_log).sum(dim=-1)  # [T,N]

def compute_anchor_kl(draft_logits: torch.Tensor, anchor_logits: torch.Tensor) -> torch.Tensor:
    q_log = draft_logits.log_softmax(-1)
    qa_log = anchor_logits.log_softmax(-1)
    q = q_log.exp()
    return (q * (q_log - qa_log)).sum(dim=-1)  # [T,N]

def sequence_reward_sum(
    teacher_logits: torch.Tensor,   # [T,N,V], no grad
    draft_logits: torch.Tensor,     # [T,N,V], no grad fine (reward only)
    accepted_mask: torch.Tensor,    # [T,N]
    first_reject_mask: torch.Tensor,# [T,N]
    *,
    include_first_reject: bool = True,
    divergence: DivergenceKind = "kl",
    alpha: float = 0.5,
    topk_for_ce: int = 0,
    entropy_bonus_coef: float = 0.0,
    anchor_kl_beta: float = 0.0,
    anchor_logits: Optional[torch.Tensor] = None,  # [T,N,V]
    clip_reward: bool = True,
    clip_range: tuple[float, float] = (-20.0, 0.0),
) -> torch.Tensor:
    mask = accepted_mask.to(draft_logits.dtype)
    if include_first_reject:
        mask = (mask + first_reject_mask.to(draft_logits.dtype)).clamp_max(1.0)

    d_t = compute_token_divergence(
        teacher_logits, draft_logits, kind=divergence, alpha=alpha, topk_for_ce=topk_for_ce
    )  # [T,N]

    seq_R = -(d_t * mask).sum(dim=0)  # [N]

    if entropy_bonus_coef != 0.0:
        H_t = compute_entropy_bonus(draft_logits)  # [T,N]
        seq_R = seq_R + entropy_bonus_coef * (H_t * mask).sum(dim=0)

    if anchor_kl_beta != 0.0 and anchor_logits is not None:
        kl_t = compute_anchor_kl(draft_logits, anchor_logits)  # [T,N]
        seq_R = seq_R - anchor_kl_beta * (kl_t * mask).sum(dim=0)

    if clip_reward:
        lo, hi = clip_range
        seq_R = seq_R.clamp(min=lo, max=hi)
    return seq_R
