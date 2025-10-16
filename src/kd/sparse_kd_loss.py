from __future__ import annotations
import torch
import torch.nn.functional as F

def _log_softmax_all(logits):  # [B,T,V]
    return logits.log_softmax(dim=-1)

def sparse_kd_kl(
    d_logits_BT_V: torch.Tensor,     # [B,T,V] draft logits
    topk_ids_BTK: torch.Tensor,      # [B,T,K] teacher top-K token ids
    topk_logprobs_BTK: torch.Tensor, # [B,T,K] teacher logprobs over full vocab
    mask_BT: torch.Tensor,           # [B,T] 1 for valid cont positions
    *,
    tail_mode: str = "bucket",       # "bucket" | "ignore"
    distill_temp: float = 1.0,
):
    eps = 1e-8
    B,T,V = d_logits_BT_V.shape
    K = topk_ids_BTK.shape[-1]

    if distill_temp != 1.0:
        d_logits_BT_V = d_logits_BT_V / distill_temp
        # teacher probs^ (1/T) re-normalized over K or K+tail
        # Convert teacher logprobs -> probs
        pK = topk_logprobs_BTK.exp().clamp_min(eps)
        pK = (pK ** (1.0/distill_temp))

    else:
        pK = topk_logprobs_BTK.exp().clamp_min(eps)

    # draft log probs on K
    logZ = torch.logsumexp(d_logits_BT_V, dim=-1, keepdim=True)             # [B,T,1]
    q_logits_K = torch.gather(d_logits_BT_V, dim=-1, index=topk_ids_BTK)    # [B,T,K]
    log_qK = q_logits_K - logZ                                              # [B,T,K]
    qK = log_qK.exp().clamp_min(eps)                                        # [B,T,K]

    if tail_mode == "ignore":
        # renormalize both over K
        pK = pK / (pK.sum(dim=-1, keepdim=True).clamp_min(eps))
        qK = qK / (qK.sum(dim=-1, keepdim=True).clamp_min(eps))
        log_pK = (pK + eps).log()
        log_qK = (qK + eps).log()
        kl = (pK * (log_pK - log_qK)).sum(dim=-1)                           # [B,T]
    else:
        # bucket the tail mass
        p_sum = pK.sum(dim=-1).clamp_max(1.0)
        q_sum = qK.sum(dim=-1).clamp_max(1.0)
        p_tail = (1.0 - p_sum).clamp_min(0.0)
        q_tail = (1.0 - q_sum).clamp_min(eps)
        log_pK = (pK + eps).log()
        log_qK = (qK + eps).log()
        log_p_tail = (p_tail + eps).log()
        log_q_tail = (q_tail + eps).log()
        kl_main = (pK * (log_pK - log_qK)).sum(dim=-1)                      # [B,T]
        kl_tail = p_tail * (log_p_tail - log_q_tail)                        # [B,T]
        kl = kl_main + kl_tail

    kl = kl * mask_BT
    denom = mask_BT.sum().clamp_min(1.0)
    return kl.sum() / denom  # scalar
