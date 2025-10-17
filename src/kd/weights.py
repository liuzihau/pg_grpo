# src/kd/weights.py
from __future__ import annotations
import torch

@torch.no_grad()
def build_kd_weights(
    *,
    d_logits_BT_V: torch.Tensor,        # [B,S,V]
    topk_ids_BTK: torch.Tensor,         # [B,S,K]
    topk_logprobs_BTK: torch.Tensor,    # [B,S,K]
    cont_mask_BT: torch.Tensor,         # [B,S] 0/1
    margin_gamma: float = 0.5,
    margin_center: float = 1.0,
    w_min: float = 0.2,
    mismatch_lambda: float = 0.3,
) -> torch.Tensor:
    """
    Heuristic per-token weights:
      - margin_w = sigmoid(gamma * ( (t_top1 - t_top2) - center ))
      - upweight tokens where student's argmax != teacher's argmax
      - clip by cont_mask
    """
    B, S, V = d_logits_BT_V.shape
    # Teacher top-1 id at each token
    t_probs = torch.exp(topk_logprobs_BTK)                 # [B,S,K]
    # idx of best topk entry per token
    t_top_idx = torch.argmax(t_probs, dim=-1, keepdim=True)  # [B,S,1]
    t_top1_id = torch.gather(topk_ids_BTK, -1, t_top_idx).squeeze(-1)  # [B,S]

    # Student argmax
    d_argmax = d_logits_BT_V.argmax(dim=-1)  # [B,S]
    mismatch = (d_argmax != t_top1_id).float()

    # Teacher margin top1 - top2
    top2_probs, _ = torch.topk(t_probs, k=min(2, t_probs.shape[-1]), dim=-1)
    if top2_probs.shape[-1] == 1:
        margin = torch.ones_like(cont_mask_BT) * 1e-6
    else:
        margin = top2_probs[..., 0] - top2_probs[..., 1]  # [B,S]

    mg, mc = float(margin_gamma), float(margin_center)
    wmin = float(w_min)
    m = torch.sigmoid(mg * (margin - mc))                 # [B,S]
    margin_w = wmin + (1.0 - wmin) * m

    weights = (margin_w * (1.0 + float(mismatch_lambda) * mismatch)) * cont_mask_BT
    return weights
