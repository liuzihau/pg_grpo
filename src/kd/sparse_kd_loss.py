# src/kd/sparse_kd_loss.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

def sparse_kd_kl(
    *,
    d_logits_BT_V: torch.Tensor,        # [B,S,V]
    topk_ids_BTK: torch.Tensor,         # [B,S,K] (int)
    topk_logprobs_BTK: torch.Tensor,    # [B,S,K] (float, teacher log-probs)
    mask_BT: torch.Tensor,              # [B,S]   (0/1 or weights)
    distill_temp: float = 1.0,
    tail_mode: str = "bucket",          # "bucket" | "ignore"
) -> torch.Tensor:
    """
    KL(teacher || student), where teacher is given as sparse top-K + tail mass.
    We compute student log_probs over full vocab, gather top-K, and build a "tail bucket".
    Loss is averaged over (masked) tokens.
    """
    B, S, V = d_logits_BT_V.shape
    K = topk_ids_BTK.shape[-1]
    T = float(distill_temp)

    # Student log-probs (optionally temperature on student)
    log_s = F.log_softmax(d_logits_BT_V / T, dim=-1)  # [B,S,V]
    # Gather top-K student log-probs
    log_s_topk = torch.gather(log_s, dim=-1, index=topk_ids_BTK)  # [B,S,K]
    s_topk = torch.exp(log_s_topk)                                # [B,S,K]
    s_topk_sum = torch.clamp(s_topk.sum(dim=-1), 0.0, 1.0)        # [B,S]
    s_tail = torch.clamp(1.0 - s_topk_sum, 0.0, 1.0)              # [B,S]

    # Teacher probs
    t_topk = torch.exp(topk_logprobs_BTK)                         # [B,S,K]
    t_topk_sum = torch.clamp(t_topk.sum(dim=-1), 0.0, 1.0)        # [B,S]
    t_tail = torch.clamp(1.0 - t_topk_sum, 0.0, 1.0)              # [B,S]

    # Optionally retemper the teacher (approximate by re-normalizing over topk+tail)
    if T != 1.0:
        # raise teacher probs to power (1/T) and renormalize including tail bucket
        t_topk = torch.clamp(t_topk, 1e-40, 1.0)
        t_tail = torch.clamp(t_tail, 1e-40, 1.0)
        t_topk_T = t_topk.pow(1.0 / T)
        t_tail_T = t_tail.pow(1.0 / T)
        Z = t_topk_T.sum(dim=-1) + t_tail_T  # [B,S]
        t_topk = t_topk_T / Z.unsqueeze(-1)
        t_tail = t_tail_T / Z

    # KL = sum_i t_i (log t_i - log s_i) over topk + tail bucket
    kl_topk = (t_topk * (_safe_log(t_topk) - log_s_topk)).sum(dim=-1)  # [B,S]
    if tail_mode == "ignore":
        kl = kl_topk
    else:
        kl_tail = t_tail * (_safe_log(t_tail) - _safe_log(s_tail))      # [B,S]
        kl = kl_topk + kl_tail

    # Weighted mean over tokens
    mask = mask_BT.float()
    denom = torch.clamp(mask.sum(), min=1.0)
    loss = (kl * mask).sum() / denom
    return loss
