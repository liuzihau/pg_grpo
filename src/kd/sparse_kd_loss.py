from __future__ import annotations
import torch
import torch.nn.functional as F

def _log_softmax_all(logits):  # [B,T,V]
    return logits.log_softmax(dim=-1)

@torch.compile(dynamic=True, fullgraph=False)  # optional if PyTorch 2.x; remove if troublesome
def sparse_kd_kl(
    *,
    d_logits_BT_V: Tensor,          # [B, S, V] draft logits (REQUIRES GRAD)
    topk_ids_BTK: Tensor,           # [B, S, K] teacher top-K token ids (int64/long)
    topk_logprobs_BTK: Tensor,      # [B, S, K] teacher logprobs for those ids (float32)
    mask_BT: Tensor,                # [B, S] weights/mask (float32), 0 masks token out
    distill_temp: float = 1.0,      # temperature on both sides
    tail_mode: str = "renorm",      # "renorm" | "bucket"
) -> Tensor:
    """
    KL(P_topK || Q) approximated on teacher top-K.
    - P_topK: teacher logprobs restricted to top-K, renormalized so sum=1.
    - Q: draft full softmax; we only gather log Q at those ids.

    Returns a scalar (mean over unmasked tokens).
    """
    assert d_logits_BT_V.requires_grad, "draft logits must require grad"

    B, S, V = d_logits_BT_V.shape
    K = topk_ids_BTK.shape[-1]

    # 1) Teacher top-K -> probabilities with (optional) temperature
    #    log p_i* = log p_i / T  then renorm over K
    lp = topk_logprobs_BTK / max(distill_temp, 1e-6)              # [B,S,K]
    lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)           # log-normalize
    p = lp.exp()                                                  # [B,S,K], sum=1 over K

    # 2) Draft log-softmax over full vocab (with temperature)
    logq_all = torch.log_softmax(d_logits_BT_V / max(distill_temp, 1e-6), dim=-1)  # [B,S,V]

    # 3) Gather log Q at teacher's top-K ids
    logq_topk = torch.gather(logq_all, dim=-1, index=topk_ids_BTK.long())          # [B,S,K]

    # 4) Approx KL on top-K: sum_i p_i (log p_i - log q_i)
    kl_topk = (p * (lp - logq_topk)).sum(dim=-1)                                    # [B,S]

    # Optional "bucket" correction: encourage mass outside top-K
    if tail_mode == "bucket":
        # teacher's leftover mass outside K is small; reward Q allocating some prob to NONE bucket
        # We don't have teacher tail distribution; this simply penalizes Q overly peaky on K.
        # Implement as negative entropy term on Q|topK to spread mass within K
        qK = logq_topk.exp()                                   # [B,S,K]
        ent_qK = -(qK * logq_topk).sum(dim=-1)                 # [B,S]
        kl_topk = kl_topk - 0.01 * ent_qK                      # tiny regularizer

    # 5) Mask + mean
    w = mask_BT.to(kl_topk.dtype)                               # [B,S]
    denom = w.sum().clamp_min(1.0)
    loss = (kl_topk * w).sum() / denom                          # scalar
    return loss