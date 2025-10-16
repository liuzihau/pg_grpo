import torch

def kl_topk(draft_logits_TBV, topk_ids_TBK, topk_logprobs_TBK, tail="bucket"):
    # wrap sparse_kd_kl across [T,B,*]
    from src.kd.sparse_kd_loss import sparse_kd_kl
    # reshape to [B,T,V] etc.
    T,B,V = draft_logits_TBV.shape
    K = topk_ids_TBK.shape[-1]
    d = sparse_kd_kl(
        d_logits_BT_V=draft_logits_TBV.transpose(0,1).contiguous(),      # [B,T,V]
        topk_ids_BTK=topk_ids_TBK.transpose(0,1).contiguous(),           # [B,T,K]
        topk_logprobs_BTK=topk_logprobs_TBK.transpose(0,1).contiguous(), # [B,T,K]
        mask_BT=torch.ones(B,T, device=draft_logits_TBV.device),
        tail_mode=tail, distill_temp=1.0,
    )
    return d
