# src/specdec/sim.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _left_pad_to_batch(seqs: List[torch.Tensor], pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """seqs are 1D Long tensors of variable length (no pads). Left-pad to a batch."""
    L = max(int(x.numel()) for x in seqs)
    B = len(seqs)
    out = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        out[i, L - n :] = s.to(device)
        att[i, L - n :] = 1
    return out, att

def _clip_ctx(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if x.numel() <= max_len: return x
    return x[-max_len:]

@torch.no_grad()
def eval_spec_batch_prob(
    *,
    tokenizer,
    draft: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    prompts: List[str],
    K: int,
    max_new_tokens: int,
    temperature: float,
    max_input_len: int,
    acceptance_cap: float = 1.0,      # c in min(1, p_teacher / (c * q_draft))
) -> Dict[str, float]:
    """
    Probabilistic speculative decoding:
    - Draft proposes up to K tokens (using generate(..., output_scores=True)).
    - For t in 1..K: accept each token with prob a_t = min(1, p_teacher(y_t)/(c*q_draft(y_t))).
    - Stop at first rejection; append exactly 1 teacher token there (greedy) to guarantee progress.
    - Count teacher_calls per proposal chunk.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Initial (unpadded) ids for each prompt
    with torch.no_grad():
        enc = tokenizer(prompts, padding=False, truncation=True, max_length=max_input_len, return_tensors=None)
    cur_ids: List[torch.Tensor] = []
    for i in range(len(prompts)):
        ids = torch.tensor(enc["input_ids"][i], dtype=torch.long)
        cur_ids.append(_clip_ctx(ids, max_input_len))

    B = len(cur_ids)
    accepted_total = torch.zeros(B, dtype=torch.long, device=device)
    compared_total = torch.zeros(B, dtype=torch.long, device=device)
    teacher_calls  = torch.zeros(B, dtype=torch.long, device=device)
    done           = torch.zeros(B, dtype=torch.bool, device=device)

    while True:
        if done.all(): break

        # Batch current contexts
        batch_ctx = [_clip_ctx(x.to(device), max_input_len) for x in cur_ids]
        inp, att = _left_pad_to_batch(batch_ctx, pad_id, device)

        # DRAFT propose K tokens (+ logits/scores)
        do_sample = temperature > 0.0
        gen = draft.generate(
            input_ids=inp,
            attention_mask=att,
            max_new_tokens=K,
            do_sample=do_sample,
            temperature=max(temperature, 1e-6) if do_sample else None,
            top_p=None if not do_sample else 1.0,
            use_cache=True,
            pad_token_id=pad_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        seqs = gen.sequences  # [B, Lctx+T]
        scores = gen.scores   # list of T tensors [B, V]
        T_actual = len(scores)
        if T_actual == 0:
            # Nothing was generated; mark any remaining as done to avoid looping
            done[:] = True
            break

        # Extract proposed tokens & draft per-step probs
        proposed: List[torch.Tensor] = []
        q_probs: List[torch.Tensor]  = []
        for i in range(B):
            if done[i]:
                proposed.append(torch.empty(0, dtype=torch.long, device=device))
                q_probs.append(torch.empty(0, dtype=torch.float32, device=device))
                continue
            Lpad = int(att[i].sum().item())
            new_tokens = seqs[i, Lpad : Lpad + T_actual]           # [T_actual]
            # Scores[t]: logits at step t for the *whole batch*; pick row i, then softmax
            step_probs = []
            for t in range(T_actual):
                logits_i = scores[t][i]                            # [V]
                prob_i = F.softmax(logits_i, dim=-1)
                step_probs.append(prob_i[new_tokens[t]].unsqueeze(0))
            q = torch.cat(step_probs, dim=0)                       # [T_actual]
            proposed.append(new_tokens.to(device))
            q_probs.append(q.to(device))

        # Teacher forward on merged inputs
        teacher_inputs: List[torch.Tensor] = []
        for i in range(B):
            if done[i]:
                teacher_inputs.append(batch_ctx[i]); continue
            merged = torch.cat([batch_ctx[i], proposed[i]], dim=0)
            teacher_inputs.append(_clip_ctx(merged, max_input_len))
        t_inp, t_att = _left_pad_to_batch(teacher_inputs, pad_id, device)
        tout = teacher(input_ids=t_inp, attention_mask=t_att, use_cache=False)
        t_logits = tout.logits

        # Slice teacher logits at the K proposed positions (per row)
        t_probs_list: List[torch.Tensor] = []
        t_argmax_list: List[torch.Tensor] = []
        for i in range(B):
            if done[i]:
                t_probs_list.append(torch.empty(0, device=device))
            else:
                L_eff = int(t_att[i].sum().item())
                T_i   = min(T_actual, int(proposed[i].numel()))
                if T_i == 0:
                    t_probs_list.append(torch.empty(0, device=device))
                else:
                    sl = t_logits[i, L_eff - T_i : L_eff, :]       # [T_i, V]
                    probs = F.softmax(sl.float(), dim=-1)          # (float32 for stability)
                    # p_teacher at the proposed token ids
                    idx = proposed[i][:T_i]
                    p = probs.gather(-1, idx.view(-1,1)).squeeze(-1)   # [T_i]
                    t_probs_list.append(p)
                    t_argmax_list.append(probs.argmax(dim=-1))         # [T_i]

        # Per row: accept until first rejection; at rejection, append 1 teacher token
        for i in range(B):
            if done[i]: continue
            T_i = int(proposed[i].numel())
            if T_i == 0:
                done[i] = True
                continue

            acc_in_chunk = 0
            rejected_here = False

            # compare steps
            for t in range(T_i):
                compared_total[i] += 1
                p = t_probs_list[i][t]       # teacher prob at token y_t
                q = q_probs[i][t].clamp_min(1e-12)
                accept_prob = torch.minimum(torch.tensor(1.0, device=device), p / (acceptance_cap * q))
                u = torch.rand((), device=device)
                if u <= accept_prob:
                    acc_in_chunk += 1
                else:
                    rejected_here = True
                    break

            # append accepted tokens (maybe zero)
            if acc_in_chunk > 0:
                accepted_total[i] += acc_in_chunk
                cur_ids[i] = torch.cat([cur_ids[i], proposed[i][:acc_in_chunk].detach().cpu()], dim=0)

            # if rejected, fallback 1 teacher token at that position
            if rejected_here:
                # teacher argmax at the reject position is in t_argmax_list[i][acc_in_chunk]
                t_arg = t_logits[i, int(t_att[i].sum().item()) - T_i + acc_in_chunk, :].argmax().item()
                cur_ids[i] = torch.cat([cur_ids[i], torch.tensor([t_arg], dtype=torch.long)], dim=0)

            # Accounting
            teacher_calls[i] += 1
            if int(accepted_total[i].item()) >= max_new_tokens:
                done[i] = True

        # Force-exit if everyone met quota
        if (accepted_total >= max_new_tokens).all():
            break

    # Stats
    compared_total = compared_total.clamp_min(1)
    teacher_calls  = teacher_calls.clamp_min(1)
    alpha_accept = (accepted_total.float() / compared_total.float()).mean().item()
    goodput      = (accepted_total.float() / teacher_calls.float()).mean().item()
    avg_span     = goodput
    reject_rate  = (1.0 - (accepted_total.float() / compared_total.float())).mean().item()

    return {
        "alpha_accept": alpha_accept,
        "goodput_tokens_per_teacher_call": goodput,
        "avg_accepted_span": avg_span,
        "reject_rate": reject_rate,
        "mean_accepted_tokens": accepted_total.float().mean().item(),
        "mean_teacher_calls": teacher_calls.float().mean().item(),
    }
