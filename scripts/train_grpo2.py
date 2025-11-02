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
from src.data.prompts import load_prompts_for_split, make_prompt_batch

# ---------------------------- helpers ----------------------------
def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def _left_pad_batch(ids_list: List[List[int]], pad_id: int):
    L = max(len(x) for x in ids_list)
    out = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        out[i, -len(ids):] = torch.tensor(ids, dtype=torch.long)
        att[i, -len(ids):] = 1
    return out, att, [len(x) for x in ids_list]

def _tokenize_prompts(tokenizer, prompts: List[str], max_input_len: int):
    enc = tokenizer(
        prompts, return_tensors=None, padding=False, truncation=True, max_length=max_input_len
    )
    ids_list = enc["input_ids"]
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

@torch.no_grad()
def _sample_group(model, tokenizer, prompt_ids, prompt_attn,
                  group_size: int, max_new: int, temperature: float, eos_id: int):
    # Repeat prompts G times
    rep_ids  = prompt_ids.repeat_interleave(group_size, dim=0).contiguous()
    rep_attn = prompt_attn.repeat_interleave(group_size, dim=0).contiguous()

    # --- NEW: move safely only if not sharded
    ids_in  = _on_right_device_for_generate(model, rep_ids)
    att_in  = _on_right_device_for_generate(model, rep_attn)

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
    logprob_temp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - logp on the *generated* tokens (next-token targets), shape [N, T]
      - valid mask [N, T] (1 for generated positions)
    """
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
    pos_logits_clamped = pos_logits
    logp_next = logp_all.gather(1, pos_logits_clamped.unsqueeze(-1).expand(-1, -1, V))  # [N,T,V]

    # the targets are the *actual* next tokens at those positions
    pos_targets = pr.unsqueeze(1) + t                                   # [N,T]
    pos_targets_clamped = pos_targets.clamp(min=0, max=Ltot - 1)
    tgt_ids = seqs.gather(1, pos_targets_clamped)                       # [N,T]

    # gather next-token log-probs for those target ids
    logp_next = logp_all.gather(1, pos_logits_clamped.unsqueeze(-1).expand(-1, -1, V))  # [N,T,V]
    out_logp = logp_next.gather(2, tgt_ids.unsqueeze(-1)).squeeze(-1)   # [N,T]

    return out_logp, valid_mask

# ---- device helpers ----
def _preferred_input_device(model):
    """
    Returns:
      - torch.device('cuda:X') if all params live on the same single CUDA device
      - torch.device('cpu')    if model is sharded across multiple devices (or CPU)
    """
    devs = set()
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for v in model.hf_device_map.values():
            if isinstance(v, str):
                devs.add(v)
            elif isinstance(v, int):
                devs.add(f"cuda:{v}")
        non_cpu = [d for d in devs if d != "cpu"]
        if len(non_cpu) == 1:
            return torch.device(non_cpu[0])
        return torch.device("cpu")
    p_devs = set()
    for p in model.parameters():
        if p.device is not None:
            p_devs.add(str(p.device))
            if len(p_devs) > 1:
                return torch.device("cpu")
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
    # the full next-token distribution at those positions
    out_full = logp_all.gather(1, pos_logits.unsqueeze(-1).expand(-1, -1, V))  # [N,T,V]

    return out_full, valid_mask

# ---------------------------- teacher utilities (stepwise w/ KV) ----------------------------
def teacher_logps_for_sampled_tokens_stepwise(
    teacher,
    seqs: torch.Tensor,          # [N, L_out]
    pr_lens: List[int],
    pad_id: int,
    max_new: int,
    device: torch.device,
    microbatch: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return teacher logp on *generated tokens* [N,T] and valid mask [N,T].
    Uses stepwise KV caching to save memory and time.
    """
    T = int(max_new)
    N, L_out = seqs.shape
    # Move to teacher device (single device or CPU for sharded)
    seqs = _on_right_device_for_generate(teacher, seqs)

    # Accumulate in chunks
    out_list = []
    mask_list = []
    with torch.no_grad():
        for i0 in range(0, N, microbatch):
            i1 = min(N, i0 + microbatch)
            chunk = seqs[i0:i1]
            att_chunk = (chunk != pad_id).long()
            # One forward to get logits
            out = teacher(input_ids=chunk, attention_mask=att_chunk, use_cache=True)
            # Gather logp on generated tokens for this chunk
            pr_sub = pr_lens[i0:i1]
            d_logp, valid = _gather_token_logps_from_forward(
                out.logits, chunk, pr_sub, T, pad_id, logprob_temp=None
            )
            out_list.append(d_logp)
            mask_list.append(valid)
    return torch.cat(out_list, dim=0), torch.cat(mask_list, dim=0)

def teacher_full_next_logprobs_stepwise(
    teacher,
    seqs: torch.Tensor,          # [N, L_out]
    pr_lens: List[int],
    pad_id: int,
    max_new: int,
    device: torch.device,
    microbatch: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return teacher full next-token logprobs [N,T,V] and valid mask [N,T].
    Uses stepwise KV caching to save memory and time.
    """
    T = int(max_new)
    N, L_out = seqs.shape
    seqs = _on_right_device_for_generate(teacher, seqs)

    out_list = []
    mask_list = []
    with torch.no_grad():
        for i0 in range(0, N, microbatch):
            i1 = min(N, i0 + microbatch)
            chunk = seqs[i0:i1]
            att_chunk = (chunk != pad_id).long()
            out = teacher(input_ids=chunk, attention_mask=att_chunk, use_cache=True)
            pr_sub = pr_lens[i0:i1]
            full_logp, valid = _gather_full_next_logprobs_from_forward(
                out.logits, chunk, pr_sub, T, pad_id, logprob_temp=None, to_cpu=False
            )
            out_list.append(full_logp)
            mask_list.append(valid)
    return torch.cat(out_list, dim=0), torch.cat(mask_list, dim=0)

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
        p = str(p)
        if len(tokenizer.encode(p)) > budget:
            # naive trunc: keep rightmost tokens (since we left-pad)
            ids = tokenizer.encode(p, truncation=True, max_length=budget)
            p = tokenizer.decode(ids)
        out.append(p)
    return out

# ---------------------------- main training ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--overrides", type=str, default="")
    args = parser.parse_args()

    cfg = to_attrdict(load_yaml_with_includes(args.cfg))
    cfg = apply_overrides(cfg, parse_overrides(args.overrides))
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cfg_lock(cfg, out_dir / "cfg_lock.yaml")

    # seed
    set_seed(int(cfg_get(cfg, "train.seed", 1234)))

    # tokenizer + models
    tokenizer = load_tokenizer_from_training_cfg(cfg)

    draft, teacher = load_models_for_eval_from_model_dir(
        tokenizer, cfg
    )

    # --- Ensure PAD/EOS are defined consistently for generation ---
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    for m in (draft, teacher):
        if getattr(m.config, "pad_token_id", None) is None:
            m.config.pad_token_id = pad_id
        if getattr(m.config, "eos_token_id", None) is None and eos_id is not None:
            m.config.eos_token_id = eos_id
        if hasattr(m, "generation_config"):
            m.generation_config.pad_token_id = m.config.pad_token_id
            m.generation_config.eos_token_id = m.config.eos_token_id

    # report layers and mark only LoRA as trainable
    print_model_layer_report(draft)
    mark_only_lora_trainable(draft)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb
    run = maybe_init_wandb(cfg, out_dir)

    # dataset(s)
    train_prompts = load_prompts_for_split(cfg, split="train")
    val_prompts   = load_prompts_for_split(cfg, split="val")
    train_ds = PromptDataset(train_prompts)
    val_ds   = PromptDataset(val_prompts)

    micro_bsz    = int(cfg_get(cfg, "train.micro_bsz", 2))
    grad_accum   = int(cfg_get(cfg, "train.grad_accum", 8))
    lr           = float(cfg_get(cfg, "train.lr", 5e-5))
    max_steps    = int(cfg_get(cfg, "train.max_steps", 20000))
    log_every    = int(cfg_get(cfg, "train.log_every", 10))
    val_every    = int(cfg_get(cfg, "train.val_every", 200))
    save_every   = int(cfg_get(cfg, "train.save_every", 1000))

    group_size   = int(cfg_get(cfg, "grpo.group_size", 4))
    max_input_len= int(cfg_get(cfg, "data.max_input_len", 1024))
    max_new      = int(cfg_get(cfg, "grpo.max_new", 64))
    temperature  = float(cfg_get(cfg, "grpo.temperature", 0.7))
    cap          = float(cfg_get(cfg, "grpo.acceptance_cap", 1.0))
    kl_ref       = str(cfg_get(cfg, "grpo.kl_ref", "teacher"))  # or 'draft_init' or 'none'
    kl_coeff     = float(cfg_get(cfg, "grpo.kl_coeff", 0.02))
    acceptance_source = str(cfg_get(cfg, "grpo.acceptance_source", "sampled_ratio"))
    logprob_temp = cfg_get(cfg, "grpo.logprob_temp", None)
    full_dist_cpu_offload = bool(cfg_get(cfg, "grpo.full_dist_cpu_offload", False))
    full_dist_microbatch  = int(cfg_get(cfg, "grpo.full_dist_microbatch", 8))

    use_hf_text_prompts = bool(cfg_get(cfg, "data.use_hf_text_prompts", True))
    use_hf_text_prompts_val = bool(cfg_get(cfg, "data.use_hf_text_prompts_val", True))

    # optimizer
    optim = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=lr)

    # ref model if needed
    if kl_ref == "draft_init":
        # clone LoRA-wrapped model as ref (no grad)
        try:
            ref_model = PeftModel.from_pretrained(draft.base_model, draft.peft_config)
        except Exception:
            ref_model = AutoModelForCausalLM.from_config(draft.config)
            ref_model = PeftModel(base_model=ref_model, peft_config=draft.peft_config)
            ref_model.load_state_dict(draft.state_dict(), strict=False)
        freeze_all_params(ref_model)
        ref_model.eval()
    else:
        ref_model = None

    # training
    step = 0
    global_start = timestamp()
    pbar = tqdm(total=max_steps, desc="GRPO")

    train_loader = DataLoader(train_ds, batch_size=micro_bsz, shuffle=True, drop_last=True)

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            # 1) Prepare prompts
            if use_hf_text_prompts:
                prep_prompts = _prepare_prompts(
                    tokenizer, batch, max_input_len=int(cfg.data.max_input_len), S=max_new
                )
                prompt_ids, prompt_attn, pr_lens, pad_id = _tokenize_prompts(
                    tokenizer, prep_prompts, max_input_len=int(cfg.data.max_input_len)
                )
            else:
                raise NotImplementedError("Only text prompts flow shown here")

            # 2) Sample group completions (no grad)
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
            att = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()
            seqs_in = _on_right_device_for_generate(draft, seqs)
            att_in  = _on_right_device_for_generate(draft, att)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = draft(input_ids=seqs_in, attention_mask=att_in, use_cache=False)

            # Bail out gracefully if logits are non-finite
            if not torch.isfinite(out.logits).all():
                print("[warn] non-finite logits; skipping this batch")
                optim.zero_grad(set_to_none=True)
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
                seqs=_on_right_device_for_generate(teacher, seqs),          # [N, L_out]
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
            )
            valid_mask = ((gen_mask_NT > 0) & (t_mask_NT > 0)).to(d_logp_NT.dtype)

            # 4) Optional ref logprobs for KL
            if kl_ref == "teacher":
                ref_logp_NT = t_logp_NT
            elif kl_ref == "draft_init":
                with torch.no_grad():
                    seqs_ref = _on_right_device_for_generate(ref_model, seqs)
                    att_ref  = _on_right_device_for_generate(ref_model, att)
                    rout = ref_model(input_ids=seqs_ref, attention_mask=att_ref, use_cache=False)
                    ref_logp_NT, _ = _gather_token_logps_from_forward(
                        rout.logits, seqs.to(rout.logits.device, non_blocking=True), rep_pr_lens,
                        max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id)
                    )
            else:
                ref_logp_NT = None

            # 5) Compute rewards (alpha etc.) and GRPO loss
            # ... (your reward shaping & GRPO loss code here; unchanged)
            # Suppose we have: loss = loss_grpo(d_logp_NT, t_logp_NT, ref_logp_NT, valid_mask, ...)

            # demo skeleton
            loss = ((d_logp_NT - t_logp_NT).pow(2) * valid_mask).sum() / (valid_mask.sum().clamp_min(1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optim.step()

            # log / step
            if step % log_every == 0:
                wandb_log({"loss": float(loss.item())})
            pbar.update(1)
            step += 1

            # validation
            if step % val_every == 0:
                _run_validation_once(
                    draft, teacher, tokenizer, val_ds, cfg, device, max_new, temperature,
                    acceptance_source, cap, full_dist_cpu_offload, full_dist_microbatch, logprob_temp
                )

            # save
            if step % save_every == 0:
                # (save adapter etc.)
                pass

    # done
    total_dur = timestamp() - global_start
    print(f"[done] total {human_time(total_dur)}")
    wandb_finish()


def _run_validation_once(draft, teacher, tokenizer, val_ds, cfg, device, max_new, temperature,
                         acceptance_source, cap, full_dist_cpu_offload, full_dist_microbatch, logprob_temp):
    draft.eval()
    try:
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=int(cfg_get(cfg, "eval.val_bsz", 8)), shuffle=False)
    except Exception:
        val_loader = [(val_ds[i:i+int(cfg_get(cfg, "eval.val_bsz", 8))]) for i in range(0, len(val_ds), int(cfg_get(cfg, "eval.val_bsz", 8)))]

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
            raise NotImplementedError

        eos_id = tokenizer.eos_token_id
        seqs, rep_pr_lens = _sample_group(
            draft, tokenizer, prompt_ids, prompt_attn,
            group_size=1, max_new=max_new,
            temperature=temperature, eos_id=eos_id
        )

        # common masks
        att = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()

        if acceptance_source == "full_overlap":
            # draft full
            with torch.autocast("cuda", dtype=torch.bfloat16):
                seqs_in = _on_right_device_for_generate(draft, seqs)
                att_in  = _on_right_device_for_generate(draft, att)
                out = draft(input_ids=seqs_in, attention_mask=att_in, use_cache=False)
            
            q_full, q_mask = _gather_full_next_logprobs_from_forward(
                out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                logprob_temp=logprob_temp, to_cpu=full_dist_cpu_offload
            )
            # teacher full
            p_full, p_mask = teacher_full_next_logprobs_stepwise(
                teacher=teacher,
                seqs=_on_right_device_for_generate(teacher, seqs),
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
            )
            valid_mask = ((q_mask > 0) & (p_mask > 0)).to(q_full.dtype)
            # (compute acceptance / metrics from q_full vs p_full)
            # ...
        else:
            # sampled ratio (legacy)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                seqs_in = _on_right_device_for_generate(draft, seqs)
                att_in  = _on_right_device_for_generate(draft, att)
                out = draft(input_ids=seqs_in, attention_mask=att_in, use_cache=False)

            d_logp_NT, gen_mask_NT = _gather_token_logps_from_forward(
                out.logits, seqs.to(out.logits.device, non_blocking=True), rep_pr_lens,
                max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                logprob_temp=logprob_temp,
            )
            t_logp_NT, t_mask_NT = teacher_logps_for_sampled_tokens_stepwise(
                teacher=teacher,
                seqs=_on_right_device_for_generate(teacher, seqs),
                pr_lens=rep_pr_lens,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                max_new=max_new,
                device=device,
                microbatch=full_dist_microbatch,
            )
            valid_mask = ((gen_mask_NT > 0) & (t_mask_NT > 0)).to(d_logp_NT.dtype)
            # (compute acceptance / metrics from d_logp_NT vs t_logp_NT)
            # ...

    draft.train()

# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    main()
