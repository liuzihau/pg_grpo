# kd_trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn

from rewards_boundary import compute_token_divergence

@dataclass
class KDWeights:
    # Margin weighting: floor to keep gradient on low-confidence tokens.
    # weight = w_min + (1 - w_min) * sigmoid(gamma * (margin - center))
    margin_gamma: float = 0.5
    margin_center: float = 1.0
    w_min: float = 0.2
    # Small boost where draft ≠ teacher (don’t over-do this).
    mismatch_lambda: float = 0.3

@dataclass
class KDStepConfig:
    divergence: str = "kl"
    alpha: float = 0.5
    topk_for_ce: int = 0
    # KD: keep both at 0.0 unless you’re experimenting;
    # they can slow down loss reduction.
    entropy_bonus: float = 0.0
    anchor_kl_beta: float = 0.0
    max_grad_norm: float = 1.0
    temperature: float = 0.0  # teacher generation temperature

@torch.no_grad()
def pack_batch(
    tokenizer,
    prompts: List[str],
    target_model: nn.Module,
    device: torch.device,
    max_input_len: int,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_input_len, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    gen = target_model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 1e-6) if temperature > 0.0 else None,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=pad,
    )
    B, Ltot = gen.shape
    cont_mask = torch.zeros_like(gen, dtype=torch.float32)
    prompt_lens = attn_mask.sum(dim=1)
    for i in range(B):
        cont_mask[i, prompt_lens[i]: Ltot] = 1.0
    nonpad = (gen != tokenizer.pad_token_id).to(gen.dtype)
    return gen, nonpad, cont_mask

def kd_step(
    *,
    tokenizer,
    draft: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    device: torch.device,
    kd_cfg: KDStepConfig,
    kd_weights: KDWeights,
    max_input_len: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    full_ids, nonpad_mask, cont_mask = pack_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        target_model=target,
        device=device,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        temperature=kd_cfg.temperature,
    )
    input_ids = full_ids[:, :-1].contiguous()
    attn_mask = nonpad_mask[:, :-1].contiguous()
    cont_region = cont_mask[:, 1:].contiguous()

    with torch.no_grad():
        t_out = target(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        teacher_logits = t_out.logits  # [B,L-1,V]
        # teacher confidence margin
        top2 = torch.topk(teacher_logits, k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1])   # [B,L-1]
        m = torch.sigmoid(kd_weights.margin_gamma * (margin - kd_weights.margin_center))
        margin_w = kd_weights.w_min + (1.0 - kd_weights.w_min) * m  # in [w_min,1]

    draft.train()
    d_out = draft(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    draft_logits = d_out.logits

    with torch.no_grad():
        t_argmax = teacher_logits.argmax(dim=-1)
        d_argmax = draft_logits.argmax(dim=-1)
        mismatch = (t_argmax != d_argmax).float()

    # [T,B,*] layout
    TNV_t = teacher_logits.transpose(0,1).contiguous()
    TNV_d = draft_logits.transpose(0,1).contiguous()
    TN_cont = cont_region.transpose(0,1).contiguous()
    TN_attn = attn_mask.transpose(0,1).contiguous()
    mask_TN = (TN_cont * TN_attn)

    d_t = compute_token_divergence(
        TNV_t, TNV_d, kind=kd_cfg.divergence, alpha=kd_cfg.alpha, topk_for_ce=kd_cfg.topk_for_ce
    )  # [T,B]

    TW = margin_w.transpose(0,1).contiguous()
    MW = (1.0 + kd_weights.mismatch_lambda * mismatch).transpose(0,1).contiguous()
    weights = TW * MW * mask_TN

    main = (d_t * weights).sum() / (weights.sum().clamp_min(1.0))
    total = main  # KD default: no entropy/anchor
    
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(draft.parameters(), kd_cfg.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        avg_cont_len = cont_region.sum(dim=1).mean().item()
        return {
            "train/kd_raw_loss": float(d_t.detach().cpu().mean()),
            "train/kd_total_loss": float(total.detach().cpu()),
            "train/kd_main_div": float(main.detach().cpu()),
            "train/kd_avg_margin": float(margin.mean().detach().cpu()),
            "train/kd_avg_mismatch": float(mismatch.float().mean().detach().cpu()),
            "train/avg_cont_len": avg_cont_len,
        }
