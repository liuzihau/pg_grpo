# scripts/train_grpo.py
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from peft import PeftModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides, parse_overrides,
    set_seed, save_cfg_lock, cfg_get, parse_torch_dtype,
)
from src.common.io import save_json, timestamp
from src.common.wandb_util import maybe_init_wandb, wandb_log, wandb_finish
from src.models.load import (
    load_models_for_eval_from_model_dir,
    load_tokenizer_from_training_cfg,
    mark_only_lora_trainable,
    freeze_all_params,
    print_model_layer_report,

)
from src.data.pregen_kd_ds import PreGeneratedTopKDataset   # reuse existing dataset
from src.data.grpo_pregen import collate_grpo_prefixes
from src.data.prompts import load_prompts_for_split, make_manual_splits, truncate_prompt_by_tokens
from src.training.optim import build_adamw
from src.training.schedule import make_warmup_cosine
from src.training.move import move_to_device_non_blocking
from src.reward.boundary import (
    expected_alpha_and_goodput_from_logps,
    expected_span_from_alpha,          # new
    alpha_from_full_distributions,     # new
    kl_from_full_distributions,        # new
)

# ---------------------------- small prompt dataset ----------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return self.prompts[i]


def _prepare_prompts(tokenizer, prompts: List[str], max_input_len: int, S: int) -> List[str]:
    # ensure prompt + max_new <= max_input_len
    cushion = 8
    budget = max(1, int(max_input_len) - int(S) - cushion)
    out: List[str] = []
    for p in prompts:
        out.append(truncate_prompt_by_tokens(tokenizer, p, budget))
    return out

def _left_pad_batch(ids_list: List[List[int]], pad_id: int, max_len: int|None=None) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """left-pad lists of ids => (input_ids[B,L], attn[B,L], true_lengths[B])."""
    Ls = [len(x) for x in ids_list]
    L = max(Ls) if max_len is None else max_len
    B = len(ids_list)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        n = min(len(x), L)
        ids[i, L - n : L] = torch.tensor(x[-n:], dtype=torch.long)
        att[i, L - n : L] = 1
    return ids, att, Ls

def _tokenize_prompts(tokenizer, prompts: List[str], max_input_len: int):
    enc = tokenizer(
        prompts, return_tensors=None, padding=False, truncation=True, max_length=max_input_len
    )
    ids_list = enc["input_ids"]

    # --- FIX: ensure no empty sequences (generate() + rope can NaN on fully-padded rows)
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    fallback = bos if bos is not None else (eos if eos is not None else 0)
    ids_list = [ids if len(ids) > 0 else [fallback] for ids in ids_list]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, attn, lens = _left_pad_batch(ids_list, pad_id=pad_id)
    return input_ids, attn, lens, pad_id


def _hard_refresh_ref_model_(ref_model, draft):
    """Overwrite ref_model weights with the current draft (no grads on ref)."""
    ref_model.load_state_dict(draft.state_dict(), strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_model
# ------------------------------ GRPO core utils -------------------------------
@torch.no_grad()
def _sample_group(
    model, tokenizer, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor,
    group_size: int, max_new: int, temperature: float, eos_id: int|None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Repeat each prompt group_size times and sample max_new tokens.
    Returns sequences [B*G, L_out] and list of prompt lengths (repeated G times).
    """
    B, L = prompt_ids.shape
    # Repeat for group sampling
    rep_ids  = prompt_ids.repeat_interleave(group_size, dim=0).contiguous()
    rep_attn = prompt_attn.repeat_interleave(group_size, dim=0).contiguous()

# --- NEW: move safely only if not sharded
    ids_in  = _on_right_device_for_generate(model, rep_ids)
    att_in  = _on_right_device_for_generate(model, rep_attn)
    if (att_in.sum(dim=1) == 0).any():
        bad = (att_in.sum(dim=1) == 0).nonzero(as_tuple=False).view(-1)
        # pick a safe token to seed generation
        bos = getattr(tokenizer, "bos_token_id", None) or (tokenizer.eos_token_id or 0)
        ids_in[bad, -1] = bos
        att_in[bad, -1] = 1
    try:
        gen = model.generate(
            input_ids=ids_in  ,
            attention_mask=att_in,
            do_sample=True,
            temperature=max(1e-6, float(temperature)),
            max_new_tokens=int(max_new),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,  # we'll recompute logprobs with grads later
        )
    except Exception as e:
        # Fallback: deterministic decode for this batch, or just skip
        print(f"[warn] sampling failed ({e}); falling back to greedy for this batch")
        gen = model.generate(do_sample=False,
            input_ids=ids_in,
            attention_mask=att_in,
            max_new_tokens=int(max_new),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,  # we'll recompute logprobs with grads later
            )
        
    seqs = gen.sequences  # [B*G, L_out] (left-padded)
    # prompt lengths replicate G times
    # Derive from attention (1s for tokens incl. prompt); stable since we padded left
    pr_lens = prompt_attn.sum(dim=1).tolist()
    pr_lens = [x for x in pr_lens for _ in range(group_size)]
    return seqs, pr_lens


def _gather_token_logps_from_forward(
    logits: torch.Tensor,    # [N, L, V]
    seqs: torch.Tensor,      # [N, L]
    pr_lens: List[int],      # len N
    max_new: int,
    pad_id: int,
    logprob_temp: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, Ltot, V = logits.shape
    device = logits.device
    T = int(max_new)
    temp = float(max(1e-6, logprob_temp))

    # log-softmax with gradient preserved
    logp_all = F.log_softmax(logits.float() / temp, dim=-1)  # [N,L,V]

    att = (seqs != pad_id).long()
    seq_lens = att.sum(dim=1)                  # [N]
    pr = torch.tensor(pr_lens, device=device)  # [N]
    gen_lens = (seq_lens - pr).clamp(min=0, max=T)  # [N]

    t = torch.arange(T, device=device).unsqueeze(0).expand(N, T)        # [N,T]
    valid_mask = (t < gen_lens.unsqueeze(1)).float()                    # [N,T]

    # positions for next-token logits (use last prompt token + t)
    pos_logits = (pr - 1).unsqueeze(1) + t                              # [N,T]
    pos_logits_clamped = pos_logits.clamp(min=0, max=Ltot - 1)

    # target token ids at the generated positions
    pos_targets = pr.unsqueeze(1) + t                                   # [N,T]
    pos_targets_clamped = pos_targets.clamp(min=0, max=Ltot - 1)
    tgt_ids = seqs.gather(1, pos_targets_clamped)                       # [N,T]

    # gather next-token log-probs for those target ids
    logp_next = logp_all.gather(1, pos_logits_clamped.unsqueeze(-1).expand(-1, -1, V))  # [N,T,V]
    out_logp = logp_next.gather(2, tgt_ids.unsqueeze(-1)).squeeze(-1)   # [N,T]

    return out_logp, valid_mask

# ---- drop-in helper (uses teacher with KV cache, tiny memory) ----
def _preferred_input_device(model):
    """
    Returns:
      - torch.device('cuda:X') if all params live on the same single CUDA device
      - torch.device('cpu')    if model is sharded across multiple devices (or CPU)
    """
    # (A) Try hf_device_map first (most reliable)
    devs = set()
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for v in model.hf_device_map.values():
            # v can be int, 'cpu', or 'cuda:X'
            if isinstance(v, str):
                devs.add(v)
            elif isinstance(v, int):
                devs.add(f"cuda:{v}")
        # Only one non-CPU device? then choose it
        non_cpu = [d for d in devs if d != "cpu"]
        if len(non_cpu) == 1:
            return torch.device(non_cpu[0])
        # mixed devices (multi-shard) or CPU-only -> keep CPU
        return torch.device("cpu")

    # (B) Fallback: inspect parameters
    p_devs = set()
    for p in model.parameters():
        if p.device is not None:
            p_devs.add(str(p.device))
            if len(p_devs) > 1:
                return torch.device("cpu")  # multiple devices => treat as sharded
    if len(p_devs) == 1:
        one = next(iter(p_devs))
        if one.startswith("cuda"):
            return torch.device(one)
    return torch.device("cpu")

def _on_right_device_for_generate(model, x: torch.Tensor) -> torch.Tensor:
    tgt = _preferred_input_device(model)
    return x if tgt.type == "cpu" else x.to(tgt, non_blocking=True)


def _gather_full_next_logprobs_from_forward(
    logits: torch.Tensor,    # [N, Ltot, V]
    seqs: torch.Tensor,      # [N, Ltot]
    pr_lens: list[int],      # len N
    max_new: int,
    pad_id: int,
    logprob_temp: float | None = None,
    to_cpu: bool = False,    # WARNING: keep False during training that needs grads
) -> tuple[torch.Tensor, torch.Tensor]:
    N, Ltot, V = logits.shape
    device = logits.device
    T = int(max_new)
    temp = float(logprob_temp) if logprob_temp is not None else 1.0

    logits = torch.nan_to_num(logits, neginf=-1e4, posinf=1e4)
    logp_all = F.log_softmax(logits.float() / temp, dim=-1)             # [N,L,V]
    att = (seqs != pad_id).long()
    seq_lens = att.sum(dim=1)
    pr = torch.tensor(pr_lens, device=device)
    gen_lens = (seq_lens - pr).clamp(min=0, max=T)

    t = torch.arange(T, device=device).unsqueeze(0).expand(N, T)        # [N,T]
    valid_mask = (t < gen_lens.unsqueeze(1)).float()                    # [N,T]

    pos_logits = (pr - 1).unsqueeze(1) + t
    pos_logits = pos_logits.clamp(min=0, max=Ltot - 1)                  # [N,T]

    # gather the full next-token log-probs at those positions (preserves grad)
    logp_full = logp_all.gather(
        1, pos_logits.unsqueeze(-1).expand(-1, -1, V)
    )                                                                    # [N,T,V]

    if to_cpu:  # only for evaluation; do NOT use when you need gradients
        logp_full = logp_full.cpu()
        valid_mask = valid_mask.cpu()

    return logp_full, valid_mask


@torch.no_grad()
def teacher_full_next_logprobs_stepwise(
    teacher,
    seqs: torch.Tensor,          # [N, L_out] left-padded: prompt + generated
    pr_lens: list[int],          # len N
    pad_id: int,
    max_new: int,
    device: torch.device | str,
    microbatch: int = 4,
    logprob_temp: float | None = None,
    to_cpu: bool = True,
):
    """
    Build step-wise teacher cache and collect FULL next-token log-probs.
    Returns:
      logp_full [N,T,V], mask [N,T]
    """
    device = torch.device(device)
    teacher = teacher.to(device).eval()

    N, L = seqs.shape
    T = int(max_new)
    temp = float(logprob_temp) if logprob_temp is not None else 1.0

    # pre-allocate (optionally on CPU)
    # we probe V lazily on first step
    out_full = None
    mask = torch.zeros((N, T), dtype=torch.float32, device="cpu" if to_cpu else device)

    # process by small slices of N to bound memory
    for s in range(0, N, microbatch):
        e = min(s + microbatch, N)
        for i in range(s, e):
            seq = seqs[i].to(device, non_blocking=True)
            p = int(pr_lens[i])
            valid_len = int((seq != pad_id).sum().item()) - p
            g = max(0, min(valid_len, T))
            if g == 0:
                continue

            prompt = seq[:p].unsqueeze(0)
            att_pr = (prompt != pad_id).long()
            out_pr = teacher(input_ids=prompt, attention_mask=att_pr, use_cache=True)
            past = out_pr.past_key_values

            prev = seq[p-1].view(1,1)
            # lazy allocate with known V
            if out_full is None:
                with torch.no_grad():
                    probe = teacher(input_ids=prev, use_cache=True, past_key_values=past)
                    V = int(probe.logits.size(-1))
                out_full = torch.empty(
                    (N, T, V), dtype=torch.float32,
                    device="cpu" if to_cpu else device
                )

            for t in range(g):
                out_t = teacher(input_ids=prev, use_cache=True, past_key_values=past)
                logits_last = out_t.logits[:, -1, :]   # [1,V]
                logp = F.log_softmax(logits_last.float() / temp, dim=-1)  # [1,V]
                if to_cpu and logp.device.type != "cpu":
                    logp = logp.cpu()
                out_full[i, t, :] = logp.squeeze(0)
                mask[i, t] = 1.0

                past = out_t.past_key_values
                prev = seq[p + t].view(1,1)

    # if nothing generated at all, allocate degenerate
    if out_full is None:
        # probe V from teacher embeddings if necessary
        V = teacher.get_output_embeddings().weight.size(0)
        out_full = torch.empty((N, T, V), dtype=torch.float32, device="cpu" if to_cpu else device)

    return out_full, mask

@torch.no_grad()
def teacher_logps_for_sampled_tokens_stepwise(
    teacher,
    seqs: torch.Tensor,          # [N, L_out] left-padded: prompt + generated
    pr_lens: list[int],          # len N, #prompt tokens for each seq
    pad_id: int,
    max_new: int,
    device: torch.device | str,
    microbatch: int = 8,         # verify N sequences in small chunks
):
    """
    Returns:
      t_logp_NT: [N,T] log p_teacher(y_t) for the *sampled* tokens of each sequence
      mask_NT:  [N,T] 1 where step is valid (<= generated length), else 0
    Works by:
      1) build teacher KV cache on the prompt once
      2) then step through generated tokens with cache (predict next, gather logp of the sampled token)
    """
    N, L = seqs.shape
    device = torch.device(device)
    teacher = teacher.to(device).eval()

    T = int(max_new)
    t_logp = torch.zeros((N, T), dtype=torch.float32, device=device)
    mask   = torch.zeros((N, T), dtype=torch.float32, device=device)

    # process in small microbatches to save memory
    for s in range(0, N, microbatch):
        e = min(s + microbatch, N)
        for i in range(s, e):
            seq = seqs[i]
            p   = int(pr_lens[i])                         # prompt length for this seq
            # number of generated tokens present in `seq`
            valid_len = int((seq != pad_id).sum().item()) - p
            g = max(0, min(valid_len, T))
            if g == 0:
                continue

            # 1) run teacher on the prompt to build past
            prompt = seq[:p].unsqueeze(0).to(device)      # [1, p]
            att_pr = (prompt != pad_id).long()
            out_pr = teacher(input_ids=prompt,
                             attention_mask=att_pr,
                             use_cache=True)
            past = out_pr.past_key_values

            # 2) step through generated tokens (teacher-forcing)
            # first next-token is predicted from the *last prompt token*
            prev = seq[p-1].view(1,1).to(device)          # x_{p-1}
            for t in range(g):
                out_t = teacher(input_ids=prev,
                                use_cache=True,
                                past_key_values=past)
                logits_last = out_t.logits[:, -1, :]      # [1, V]
                tgt = seq[p+t].view(1).to(device)         # y_t
                logp = F.log_softmax(logits_last.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                t_logp[i, t] = logp
                mask[i, t] = 1.0

                # advance cache and prev token
                past = out_t.past_key_values
                prev = seq[p+t].view(1,1).to(device)

    return t_logp, mask

# -------------------------------- validation -----------------------------------
def _build_val_loader(cfg, tokenizer):
    """Create a validation DataLoader that mirrors the training source."""
    use_pregen_val = bool(cfg_get(cfg, "grpo.valid_use_pregen_prompts",
                           cfg_get(cfg, "grpo.use_pregen_prompts", False)))

    if use_pregen_val:
        ds_root = Path(cfg.grpo.pregen_dir).expanduser().resolve()
        pregen_split = str(cfg_get(cfg, "grpo.valid_pregen_split",
                            cfg_get(cfg, "grpo.pregen_split", "validation")))
        ds = PreGeneratedTopKDataset(ds_root, split=pregen_split)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        def _collate(batch):
            return collate_grpo_prefixes(
                batch=batch,
                tokenizer=tokenizer,
                pad_id=pad_id,
                max_input_len=int(cfg.data.max_input_len),
                max_new_tokens=int(cfg.grpo.max_new),
                offset_strategy=str(cfg_get(cfg, "grpo.valid_offset_strategy", "uniform")),
                offset_stride=int(cfg_get(cfg, "grpo.offset_stride", 8)),
                cushion=int(cfg_get(cfg, "grpo.cushion", 8)),
                seed=None,
            )

        dl = DataLoader(
            ds,
            batch_size=int(cfg_get(cfg, "grpo.valid_batch_size",
                            cfg_get(cfg, "grpo.batch_size", 1))),
            shuffle=False,
            num_workers=int(cfg_get(cfg, "grpo.num_workers", 2)),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=_collate,
            drop_last=False,
        )
        return dl, False  # use_hf_text_prompts_val=False
    else:
        # HF text prompts path
        split = str(cfg_get(cfg, "grpo.valid_hf_split", "validation"))
        prompts = load_prompts_for_split(tokenizer, cfg, split=split)
        if not prompts:
            train_prompts, val_prompts = make_manual_splits(tokenizer, cfg, seed=int(cfg.training.seed))
            prompts = val_prompts
        sample_max = cfg_get(cfg, "grpo.valid_sample_max", None)
        if sample_max:
            prompts = prompts[: int(sample_max)]
        ds = PromptDataset(prompts)
        dl = DataLoader(
            ds,
            batch_size=int(cfg_get(cfg, "grpo.valid_batch_size",
                            cfg_get(cfg, "grpo.batch_size", 1))),
            shuffle=False,
            num_workers=int(cfg_get(cfg, "grpo.num_workers", 2)),
            pin_memory=True,
            drop_last=False,
        )
        return dl, True  # use_hf_text_prompts_val=True


@torch.no_grad()
def _run_validation_once(
    *,
    draft,
    teacher,
    tokenizer,
    val_loader,
    use_hf_text_prompts_val: bool,
    cfg,
    device: torch.device,
    reward_key: str,
    acceptance_source: str,         # <--- NEW
    logprob_temp: float,            # <--- NEW
    full_dist_microbatch: int,      # <--- NEW
    full_dist_cpu_offload: bool,    # <--- NEW
    overlap_topk: int | None,       # <--- NEW
) -> Dict[str, float]:
    """Run validation with group_size=1 and return averaged metrics."""
    draft.eval()
    teacher.eval()

    max_new = int(cfg.grpo.max_new)
    temperature = float(cfg_get(cfg, "grpo.valid_temperature",
                         cfg_get(cfg, "grpo.temperature", 1.0)))
    cap = float(cfg_get(cfg, "grpo.valid_acceptance_cap",
                  cfg_get(cfg, "grpo.acceptance_cap", 1.0)))

    processed = 0
    cap_samples = cfg_get(cfg, "grpo.valid_sample_max", None)
    if cap_samples is not None:
        cap_samples = int(cap_samples)

    reward_sum = 0.0
    alpha_sum  = 0.0
    ats_sum    = 0.0
    valid_tok_sum = 0.0
    n_samples = 0

    for batch in val_loader:
        if use_hf_text_prompts_val:
            prep_prompts = _prepare_prompts(
                tokenizer, batch, max_input_len=int(cfg.data.max_input_len), S=max_new
            )
            prompt_ids, prompt_attn, pr_lens, pad_id = _tokenize_prompts(
                tokenizer, prep_prompts, max_input_len=int(cfg.data.max_input_len)
            )
        else:
            prompt_ids = batch["prompt_ids"]
            prompt_attn = batch["prompt_attn"]
            pr_lens = batch["prompt_attn"].sum(dim=1).tolist()
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        eos_id = tokenizer.eos_token_id
        seqs, rep_pr_lens = _sample_group(
            draft, tokenizer, prompt_ids, prompt_attn,
            group_size=1, max_new=max_new,
            temperature=temperature, eos_id=eos_id
        )

        # common masks
        att = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
        seqs_in = _on_right_device_for_generate(draft, seqs)
        att_in  = _on_right_device_for_generate(draft, att)
        if acceptance_source == "full_overlap":
            # draft full
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = draft(input_ids=seqs_in, attention_mask=att_in, use_cache=False)
            
            q_full, q_mask = _gather_full_next_logprobs_from_forward(
                out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                logprob_temp=logprob_temp, to_cpu=full_dist_cpu_offload
            )
            # teacher full
            p_full, p_mask = teacher_full_next_logprobs_stepwise(
                teacher=teacher,
                seqs=seqs.to(out.logits.device, non_blocking=True),
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
                logprob_temp=logprob_temp,
                to_cpu=full_dist_cpu_offload,
            )
            valid_mask = ((q_mask > 0) & (p_mask > 0)).to(q_full.dtype)
            stats = alpha_from_full_distributions(
                q_logp_full=q_full, p_logp_full=p_full, mask=valid_mask,
                acceptance_cap=cap, topk=overlap_topk
            )
        else:
            # sampled ratio (legacy)
            seqs_in = _on_right_device_for_generate(draft, seqs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = draft(input_ids=seqs_in, attention_mask=att, use_cache=False)

            
            d_logp_NT, gen_mask_NT = _gather_token_logps_from_forward(
                out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                logprob_temp=temperature,
            )
            t_logp_NT, t_mask_NT = teacher_logps_for_sampled_tokens_stepwise(
                teacher=teacher,
                seqs=seqs.to(out.logits.device, non_blocking=True),
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
            )
            valid_mask = ((gen_mask_NT > 0) & (t_mask_NT > 0)).to(d_logp_NT.dtype)
            stats = expected_alpha_and_goodput_from_logps(
                draft_logp=d_logp_NT,
                teacher_logp=t_logp_NT,
                mask=valid_mask,
                acceptance_cap=cap,
            )

        if reward_key == "alpha":
            r = stats["alpha_mean"]
        elif reward_key == "accepted_tokens":
            r = stats["accepted_tokens"]
        else:
            raise NotImplementedError(f"Unknown reward_key: {reward_key}")

        B_now = r.numel()
        reward_sum    += float(r.sum().detach().cpu())
        alpha_sum     += float(stats["alpha_mean"].sum().detach().cpu())
        ats_sum       += float(stats["accepted_tokens"].sum().detach().cpu())
        valid_tok_sum += float(stats["valid_tokens"].sum().detach().cpu())
        n_samples     += B_now
        processed     += B_now

        if cap_samples is not None and processed >= cap_samples:
            break

    if n_samples == 0:
        return {"valid/reward": 0.0, "valid/alpha_mean": 0.0,
                "valid/accepted_tokens": 0.0, "valid/valid_tokens": 0.0}

    return {
        "valid/reward":           reward_sum / n_samples,
        "valid/alpha_mean":       alpha_sum / n_samples,
        "valid/accepted_tokens":  ats_sum / n_samples,
        "valid/valid_tokens":     valid_tok_sum / n_samples,
        "valid/samples":          float(n_samples),
    }


# ----------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_api_key", default=None)
    args = ap.parse_args()

    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(apply_overrides(raw, parse_overrides(args.override)))
    set_seed(int(cfg.training.seed))

    # ----- New training mode & distribution settings -----
    train_mode = str(cfg_get(cfg, "grpo.train_mode", "policy")).lower()             # "policy" | "kl"
    acceptance_source = str(cfg_get(cfg, "grpo.acceptance_source", "sampled_ratio")).lower()  # "sampled_ratio" | "full_overlap"
    kl_direction = str(cfg_get(cfg, "grpo.kl_direction", "q||p")).lower()           # only for train_mode=kl
    kl_loss_coeff = float(cfg_get(cfg, "grpo.kl_loss_coeff", 1.0))
    full_dist_microbatch = int(cfg_get(cfg, "grpo.full_dist_microbatch", 4))
    full_dist_cpu_offload = bool(cfg_get(cfg, "grpo.full_dist_cpu_offload", True))
    if train_mode == "kl" and full_dist_cpu_offload:
        print("[warn] full_dist_cpu_offload=True disables grads; forcing False for KL mode.")
        full_dist_cpu_offload = False
    overlap_topk = cfg_get(cfg, "grpo.overlap_topk", None)
    if overlap_topk is not None:
        overlap_topk = int(overlap_topk)

    reward_key_cfg = str(cfg_get(cfg, "grpo.reward_key", "alpha_mean")).lower()
    checkpoint_every = int(cfg_get(cfg, "grpo.checkpoint_every",
                           cfg_get(cfg, "grpo.kl_ref_update_every", 0)))

    out_dir = Path(cfg.grpo.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cfg_lock(raw, out_dir, filename="cfg.lock.yaml")

    # ------- Load finetuned draft (LoRA) + teacher using model_dir cfg
    kd_model_dir = Path(cfg.grpo.base_model_dir).expanduser().resolve()
    dtype = parse_torch_dtype(cfg_get(cfg, "training.dtype", "bf16"))

    try:
        tokenizer, draft, teacher, train_cfg, stage_kind, used_cfg_path = \
            load_models_for_eval_from_model_dir(
                kd_model_dir, dtype_name=str(dtype).split(".")[-1], device_map="auto"
            )
    except Exception as e:
        # Fallback: load tokenizer from kd cfg, load draft (LoRA) from dir + teacher by name in cfg.grpo.teacher
        # (this path shouldn't trigger if your load.py already handles kd cfg gracefully)
        print(f"[warn] primary model loader failed: {e}")
        # minimal fallback path
        raw_kd = load_yaml_with_includes(kd_model_dir / "cfg.lock.yaml")
        kd_cfg = to_attrdict(raw_kd)
        tokenizer = load_tokenizer_from_training_cfg(kd_cfg)
        draft_base = AutoModelForCausalLM.from_pretrained(
            kd_cfg.models.draft, torch_dtype=dtype, device_map="auto"
        )
        draft = PeftModel.from_pretrained(draft_base, kd_model_dir)
        draft.config.use_cache = False
        draft.train()
        n_train, n_total = mark_only_lora_trainable(draft)
        print(f"[grpo] trainable params (LoRA only): {n_train:,} / {n_total:,}")
        print_model_layer_report(draft, title="GRPO draft (after freezing base)", limit=80, only_lora=True)
        teacher_name = cfg_get(cfg, "grpo.teacher", cfg_get(kd_cfg, "models.target", None)) \
                       or cfg_get(kd_cfg, "models.tokenizer", None)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, torch_dtype=dtype, device_map="auto"
        )
        teacher.config.use_cache = False
        teacher.eval()
        train_cfg, used_cfg_path = kd_cfg, kd_model_dir / "cfg.lock.yaml"
        stage_kind = "kd"

    # ---- ensure draft has trainable params (LoRA) ----
    # Some loaders return a fully-frozen eval model; we must re-enable LoRA grads.
    try:
        n_train, n_total = mark_only_lora_trainable(draft)
    except Exception:
        # fallback: enable any PEFT adapter layers if available
        if hasattr(draft, "enable_adapter_layers"):
            draft.enable_adapter_layers()
        # re-run to count
        n_train, n_total = mark_only_lora_trainable(draft)

    print(f"[trainable] {n_train:,} / {n_total:,} params require_grad on draft")
    assert n_train > 0, (
        "No trainable parameters found on draft. "
        "Likely the loader merged adapters or returned a fully-frozen eval model. "
        "Make sure LoRA adapters are attached and active."
    )

    draft.train()


    # ---- teacher memory policy ----
    td = str(cfg_get(cfg, "grpo.teacher_device", "auto")).lower()
    use_8bit = bool(cfg_get(cfg, "grpo.teacher_8bit", False))
    if td == "cpu":
        teacher.to("cpu")
        torch.cuda.empty_cache()
    else:
        if use_8bit:
            try:
                # reload teacher in 8-bit on GPU
                from transformers import AutoModelForCausalLM
                # resolve teacher name again (already used inside loader)
                try:
                    from src.models.load import _resolve_teacher_name as _resolve_tname
                    run_root = Path(cfg.grpo.base_model_dir).expanduser().resolve()
                    teacher_name = _resolve_tname(train_cfg, run_root)
                except Exception:
                    teacher_name = None
                if teacher_name is None:
                    # fall back to the already-loaded teacher
                    pass
                else:
                    del teacher
                    torch.cuda.empty_cache()
                    teacher = AutoModelForCausalLM.from_pretrained(
                        teacher_name, load_in_8bit=True, device_map="auto"
                    )
            except Exception as e:
                print(f"[warn] 8-bit teacher load failed, using bf16 on GPU: {e}")


    # Ensure teacher is fully frozen (no grads)
    freeze_all_params(teacher)
    teacher.eval()

    # ---- draft: only LoRA trains + checkpointing ----
    if hasattr(draft, "gradient_checkpointing_enable"):
        draft.gradient_checkpointing_enable()
    # PEFT base (robust across PEFT versions)
    for node in (getattr(draft, "base_model", None),
                getattr(getattr(draft, "base_model", None), "model", None)):
        if node is not None and hasattr(node, "gradient_checkpointing_enable"):
            node.gradient_checkpointing_enable()

    # no KV cache during training
    if hasattr(draft, "config"):
        draft.config.use_cache = False
        if hasattr(draft, "enable_input_require_grads"):
            try:
                draft.enable_input_require_grads()
            except Exception:
                pass

    # Ensure only LoRA is trainable on the draft
    if isinstance(draft, PeftModel):
        # Make sure the active adapter is enabled for training
        try:
            # adapter name is usually "default" unless you set a custom name
            active = getattr(draft, "active_adapter", None) or "default"
            if hasattr(draft, "set_adapter"):
                draft.set_adapter(active)
            if hasattr(draft, "enable_adapter_layers"):
                draft.enable_adapter_layers()
        except Exception as _e:
            # Non-fatal: continue and rely on requires_grad filtering below
            pass

    # Freeze base & unfreeze only LoRA/adapter weights
    draft.train()

    # Report layers / LoRA injection
    if bool(cfg_get(cfg, "grpo.print_layers", True)):
        print_model_layer_report(draft,   title="GRPO draft (starting from KD)", limit=80, only_lora=True)
        print_model_layer_report(teacher, title="GRPO teacher (reference)",      limit=40, only_lora=True)

    # Optional reference for KL control: "teacher" | "draft_init" | "none"
    kl_ref = str(cfg_get(cfg, "grpo.kl_ref", "teacher")).lower()
    kl_coeff = float(cfg_get(cfg, "grpo.kl_coeff", 0.01))
    ref_model = None
    if kl_ref == "draft_init":
        # make a frozen copy (weights shared if LoRA, so we load a new base and attach adapters snapshot)
        from copy import deepcopy
        ref_model = deepcopy(draft).eval()
        for p in ref_model.parameters(): p.requires_grad_(False)
    elif kl_ref == "teacher":
        ref_model = teacher
    else:
        ref_model = None

    kl_ref_update_every = int(cfg_get(cfg, "grpo.kl_ref_update_every", 0))
    # ------- Build prompt source -------
    use_pregen = bool(cfg_get(cfg, "grpo.use_pregen_prompts", False))

    if use_pregen:
        # 2a) Use KD pre-generated corpus as prompt source with variable offsets
        ds_root = Path(cfg.grpo.pregen_dir).expanduser().resolve()
        pregen_split = str(cfg_get(cfg, "grpo.pregen_split", cfg.data.split))
        ds = PreGeneratedTopKDataset(ds_root, split=pregen_split)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        def _collate(batch):
            return collate_grpo_prefixes(
                batch=batch,
                tokenizer=tokenizer,
                pad_id=pad_id,
                max_input_len=int(cfg.data.max_input_len),
                max_new_tokens=int(cfg.grpo.max_new),
                offset_strategy=str(cfg_get(cfg, "grpo.offset_strategy", "uniform")),
                offset_stride=int(cfg_get(cfg, "grpo.offset_stride", 8)),
                cushion=int(cfg_get(cfg, "grpo.cushion", 8)),
                seed=None,   # <— no step-based seed; workers seed themselves
            )

        num_workers = int(cfg_get(cfg, "grpo.num_workers", 2))
        dl = DataLoader(
            ds,
            batch_size=int(cfg.grpo.batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
            collate_fn=_collate,
        )
        use_hf_text_prompts = False
    else:
        # 2b) Original: use HF split of raw text prompts
        base_split = cfg.data.split
        prompts = load_prompts_for_split(tokenizer, cfg, split=base_split)
        if not prompts:
            train_prompts, val_prompts = make_manual_splits(tokenizer, cfg, seed=int(cfg.training.seed))
            prompts = train_prompts
        sample_max = cfg_get(cfg, "data.sample_max", None)
        if sample_max:
            prompts = prompts[: int(sample_max)]
        ds = PromptDataset(prompts)
        dl = DataLoader(
            ds,
            batch_size=int(cfg.grpo.batch_size),
            shuffle=True,
            num_workers=int(cfg_get(cfg, "grpo.num_workers", 2)),
            pin_memory=True,
            drop_last=True,
        )
        use_hf_text_prompts = True


    do_valid = int(cfg_get(cfg, "grpo.valid_every", 0)) > 0
    if do_valid:
        val_loader, use_hf_text_prompts_val = _build_val_loader(cfg, tokenizer)
    else:
        val_loader, use_hf_text_prompts_val = None, False


    # ------- Optim & schedule (train only LoRA params) -------
    optim = build_adamw(
        draft,
        lr=float(cfg.grpo.lr),
        weight_decay=float(cfg.grpo.get("weight_decay", 0.0)),
        betas=tuple(cfg.grpo.get("betas", [0.9, 0.95])),
        eps=1e-8,
    )
    sched = make_warmup_cosine(
        optim,
        total_steps=int(cfg.grpo.total_steps),
        warmup_ratio=float(cfg.grpo.get("warmup_ratio", 0.05)),
        min_lr=float(cfg.grpo.get("min_lr", 0.0)),
    )

    # ------- W&B -------
    run = maybe_init_wandb(
        enabled=bool(args.wandb),
        api_key=args.wandb_api_key,
        project=cfg_get(cfg, "logging.project", "grpo-train"),
        name=cfg_get(cfg, "logging.name", "grpo_run"),
        config=json.loads(json.dumps(cfg, default=str)),
        tags=cfg_get(cfg, "logging.tags", None),
        group=cfg_get(cfg, "logging.group", None),
        mode=cfg_get(cfg, "logging.mode", None),
    )

    # ------- Train loop -------
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))
    max_new = int(cfg.grpo.max_new)
    temperature = float(cfg.grpo.temperature)
    # temperature used when converting logits -> log-probs for q/p full distributions
    logprob_temp = cfg_get(cfg, "grpo.logprob_temp", None)
    if logprob_temp is None:
        logprob_temp = temperature   # fall back to sampling temp
    else:
        logprob_temp = float(logprob_temp)

    cap = float(cfg.grpo.acceptance_cap)
    group_size = int(cfg.grpo.group_size)
    grad_clip = float(cfg.training.get("max_grad_norm", 1.0))
    acc_steps = int(cfg.grpo.get("grad_accum_steps", 1))
    print_every = int(cfg.grpo.log_every)

    draft.train()
    pbar = tqdm(range(int(cfg.grpo.total_steps)), desc="GRPO", ncols=100)
    it = iter(dl)


    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = getattr(tokenizer, "bos_token_id", None)

    for m in (draft, teacher):
        if getattr(m.config, "pad_token_id", None) is None:
            m.config.pad_token_id = pad_id
        if getattr(m.config, "eos_token_id", None) is None and eos_id is not None:
            m.config.eos_token_id = eos_id
        if bos_id is not None and getattr(m.config, "bos_token_id", None) is None:
            m.config.bos_token_id = bos_id
        if hasattr(m, "generation_config"):
            gc = m.generation_config
            if getattr(gc, "pad_token_id", None) is None: gc.pad_token_id = m.config.pad_token_id
            if getattr(gc, "eos_token_id", None) is None and eos_id is not None: gc.eos_token_id = eos_id
            if bos_id is not None and getattr(gc, "bos_token_id", None) is None: gc.bos_token_id = bos_id


    for step in pbar:
        try:
            batch_prompts = next(it)
        except StopIteration:
            it = iter(dl)
            batch_prompts = next(it)

        # Tokenize prompts (left-pad) with a safe budget (context)
        # We’ll clip earlier in data_gen/eval the same way
        if use_hf_text_prompts:
            prep_prompts = _prepare_prompts(
                tokenizer, batch_prompts, max_input_len=int(cfg.data.max_input_len), S=max_new
            )
            prompt_ids, prompt_attn, pr_lens, pad_id = _tokenize_prompts(
                tokenizer, prep_prompts, max_input_len=int(cfg.data.max_input_len)
            )
        else:
            # batch_prompts here is a dict from collate_grpo_prefixes
            prompt_ids = batch_prompts["prompt_ids"]      # [B, L]
            prompt_attn = batch_prompts["prompt_attn"]    # [B, L]
            pr_lens = batch_prompts["prompt_attn"].sum(dim=1).tolist()
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # 1) Sample group completions (no grad)
        eos_id = tokenizer.eos_token_id
        with torch.no_grad():
            seqs, rep_pr_lens = _sample_group(
                draft.eval(), tokenizer, prompt_ids, prompt_attn,
                group_size=group_size, max_new=max_new,
                temperature=temperature, eos_id=eos_id
            )
        draft.train()  # back to train for gradient passes

        N, L = seqs.shape  # N = B * group_size

        # 2) Compute draft logprobs on sampled tokens WITH grad
        #    One forward on the full sequences to pull the next-token logits
        att = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long().to(device)
        seqs_in = _on_right_device_for_generate(draft, seqs)
        att_in  = _on_right_device_for_generate(draft, att)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = draft(input_ids=seqs_in, attention_mask=att_in, use_cache=False)

        # Bail out gracefully if logits are non-finite
        if not torch.isfinite(out.logits).all():
            print("[warn] non-finite logits; skipping step & zeroing grads")
            optim.zero_grad(set_to_none=True)
            # (Optional) small LR backoff on instability bursts:
            for pg in optim.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, 1e-6)
            continue
        
        d_logp_NT, gen_mask_NT = _gather_token_logps_from_forward(
            out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
            max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            logprob_temp=temperature,                      # <--- IMPORTANT
        )
        if gen_mask_NT.sum().item() == 0:
            # nothing generated (should be rare); skip to next batch
            optim.zero_grad(set_to_none=True)
            continue

        # 3) Teacher logprobs on the same sampled tokens (no grad, stepwise KV cache)
        t_logp_NT, t_mask_NT = teacher_logps_for_sampled_tokens_stepwise(
            teacher=teacher,
            seqs=seqs.to(out.logits.device, non_blocking=True),          # [N, L_out]
            pr_lens=rep_pr_lens,           # list[int], same as you pass to draft gather
            pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
            max_new=max_new,
            device=device,
            microbatch=full_dist_microbatch,
        )
        
        # 4) Optional ref logprobs for KL
        if kl_ref == "teacher":
            ref_logp_NT = t_logp_NT
        elif kl_ref == "draft_init":
            with torch.no_grad():
                rout = ref_model(input_ids=seqs.to(ref_model.device), attention_mask=att.to(ref_model.device), use_cache=False)
                ref_logp_NT, _ = _gather_token_logps_from_forward(
                    rout.logits.to(d_logp_NT.device), seqs.to(d_logp_NT.device), rep_pr_lens, max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id)
                )
        else:
            ref_logp_NT = None

        # -------------------- REWARD & LOSS (two modes) --------------------
        if train_mode == "policy":
            if acceptance_source == "full_overlap":
                # --- full distribution overlap path ---
                # Build q_full and p_full (log-probs over V for next-token positions)
                q_full, q_mask = _gather_full_next_logprobs_from_forward(
                    out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                    max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                    logprob_temp=logprob_temp, to_cpu=full_dist_cpu_offload
                )
                p_full, p_mask = teacher_full_next_logprobs_stepwise(
                    teacher=teacher,
                    seqs=seqs.to(out.logits.device, non_blocking=True),
                    pr_lens=rep_pr_lens,
                    pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                    max_new=max_new,
                    device=device,
                    microbatch=full_dist_microbatch,
                    logprob_temp=logprob_temp,
                    to_cpu=full_dist_cpu_offload,
                )
                valid_mask_full = ((q_mask > 0) & (p_mask > 0)).to(q_full.dtype)
                stats = alpha_from_full_distributions(
                    q_logp_full=q_full, p_logp_full=p_full, mask=valid_mask_full,
                    acceptance_cap=cap, topk=overlap_topk
                )
            else:
                # --- legacy sampled-token ratio path ---
                valid_mask = ((gen_mask_NT > 0) & (t_mask_NT > 0)).to(d_logp_NT.dtype)
                stats = expected_alpha_and_goodput_from_logps(
                    draft_logp=d_logp_NT.detach(),   # ok to detach for reward
                    teacher_logp=t_logp_NT,
                    mask=valid_mask,
                    acceptance_cap=cap,
                )

            # scalar per-sample reward used by GRPO
            reward_key = str(cfg_get(cfg, "grpo.reward_key", "alpha_mean")).lower()
            if reward_key == "alpha":
                r_N = stats["alpha_mean"]
            elif reward_key == "accepted_tokens":
                r_N = stats["accepted_tokens"]
            else:
                raise NotImplementedError()

            # Group baseline & normalised advantages
            B = prompt_ids.size(0)
            G = group_size
            assert B * G == N, (B, G, N)
            r_BG = r_N.view(B, G)
            base_B = r_BG.mean(dim=1, keepdim=True)
            adv_BG = (r_BG - base_B)
            adv_N = adv_BG.view(N)

            adv_mean = adv_N.mean()
            adv_std  = adv_N.std().clamp_min(1e-3)
            adv_N = (adv_N - adv_mean) / adv_std
            adv_N = adv_N.clamp(-5.0, 5.0)

            # policy loss (mask by valid tokens count)
            valid_t  = (stats["valid_tokens"].to(d_logp_NT.device)).clamp_min(1.0)  # [N]
            sum_logpi = (d_logp_NT * ((gen_mask_NT > 0).to(d_logp_NT.dtype))).sum(dim=1)
            pol_loss = -(adv_N.detach() * (sum_logpi / valid_t)).mean()

            # optional KL regulariser (skip if using teacher as the supervised target elsewhere)
            if ref_logp_NT is not None and kl_coeff > 0:
                kl_per_t = (d_logp_NT - ref_logp_NT) * gen_mask_NT
                kl_term = kl_per_t.sum(dim=1) / valid_t
                kl_loss = kl_coeff * kl_term.mean()
            else:
                kl_loss = torch.zeros((), device=d_logp_NT.device)

            loss = pol_loss + kl_loss

        elif train_mode == "kl":
            # --------- pure supervised KL on on-policy samples (no GRPO) ----------
            # Build full distributions for q/p
            q_full, q_mask = _gather_full_next_logprobs_from_forward(
                out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                logprob_temp=logprob_temp, to_cpu=full_dist_cpu_offload
            )
            p_full, p_mask = teacher_full_next_logprobs_stepwise(
                teacher=teacher,
                seqs=seqs.to(out.logits.device, non_blocking=True),
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
                logprob_temp=logprob_temp,
                to_cpu=full_dist_cpu_offload,
            )
            valid_mask_full = ((q_mask > 0) & (p_mask > 0)).to(q_full.dtype)

            kl_stats = kl_from_full_distributions(
                q_logp_full=q_full, p_logp_full=p_full, mask=valid_mask_full,
                direction=kl_direction
            )
            kl_loss = kl_loss_coeff * kl_stats["kl_mean"].mean()
            loss = kl_loss  # no policy term

            # turn off any old KL-regulariser in this mode to avoid double counting
            ref_logp_NT = None

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        (loss / acc_steps).backward()

        if (step + 1) % acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(draft.parameters(), grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        # ---- logging
        if step % print_every == 0:
            logs = {
                "grpo/loss": float(loss.detach().cpu()),
                "grpo/lr": sched.get_last_lr()[0],
                "grpo/mean_len": float(((gen_mask_NT > 0).sum(dim=1).float().mean()).detach().cpu()),
            }
            if train_mode == "policy":
                logs.update({
                    "grpo/policy_loss": float(pol_loss.detach().cpu()),
                    "grpo/kl_loss": float(kl_loss.detach().cpu()),
                    "grpo/alpha_mean": float(stats["alpha_mean"].mean().detach().cpu()),
                    "grpo/accepted_tokens": float(stats["accepted_tokens"].mean().detach().cpu()),
                })
            else:  # kl
                logs.update({
                    "kl/mean": float(kl_stats["kl_mean"].mean().detach().cpu()),
                })
            wandb_log(logs, step=step)
            if train_mode == "policy":
                pbar.set_postfix(loss=f"{logs['grpo/loss']:.3f}",
                                 ats=f"{logs['grpo/accepted_tokens']:.3f}",
                                 alpha=f"{logs['grpo/alpha_mean']:.3f}")
            else:
                pbar.set_postfix(loss=f"{logs['grpo/loss']:.3f}",
                                 kl=f"{logs['kl/mean']:.3f}")


        # ---- periodic validation
        if do_valid and ((step + 1) % int(cfg.grpo.valid_every) == 0):
            draft.eval()   # make sure no dropout etc.
            with torch.no_grad():
                val_stats = _run_validation_once(
                    draft=draft,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    val_loader=val_loader,
                    use_hf_text_prompts_val=use_hf_text_prompts_val,
                    cfg=cfg,
                    device=device,
                    reward_key=reward_key_cfg,
                    acceptance_source=acceptance_source,
                    logprob_temp=logprob_temp,
                    full_dist_microbatch=full_dist_microbatch,
                    full_dist_cpu_offload=full_dist_cpu_offload,
                    overlap_topk=overlap_topk,
                )
            # log + progress text
            wandb_log(val_stats, step=step)
            pbar.set_postfix_str(
                f"{pbar.postfix} | v/rew={val_stats['valid/reward']:.3f} "
                f"v/α={val_stats['valid/alpha_mean']:.3f} v/ats={val_stats['valid/accepted_tokens']:.2f}"
            )
            draft.train()

        if checkpoint_every > 0 and ((step + 1) % checkpoint_every == 0):
            outdir = out_dir.with_name(out_dir.name + f"-step-{step}") / "lora"
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                draft.save_pretrained(str(outdir))
            except Exception:
                torch.save(draft.state_dict(), outdir / "pytorch_model.bin")

        if train_mode == "policy" and kl_ref == "draft_init" and kl_ref_update_every > 0 and ((step + 1) % kl_ref_update_every == 0):
            _hard_refresh_ref_model_(ref_model, draft)


    # ------- Save adapters + summary -------
    outdir = out_dir / "lora"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        draft.save_pretrained(str(outdir))
    except Exception:
        torch.save(draft.state_dict(), outdir / "pytorch_model.bin")

    save_json(
        {
            "finished_at": timestamp(),
            "steps": int(cfg.grpo.total_steps),
            "base_model_dir": str(kd_model_dir),
            "out_dir": str(outdir),
            "kl_ref": kl_ref,
            "reward_key": reward_key_cfg,
        },
        out_dir / "train_summary.json",
    )
    wandb_finish()
    print(f"[DONE] GRPO LoRA saved at: {outdir}")


if __name__ == "__main__":
    main()






