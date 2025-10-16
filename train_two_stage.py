from __future__ import annotations
import argparse, os, math, random
from typing import List, Dict

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import torch.nn as nn
from tqdm.auto import tqdm
import wandb

from utils import load_yaml, set_seed, get_device, to_attrdict
from data import HFChatPromptsDataset, PromptOnlyDataset
from models import load_tokenizer, load_target, load_draft
from lora_setup import attach_lora, enable_lm_head_training
from collector_utils import compute_first_reject_mask
from trainer_hook import BoundaryGRPOHelper
from kd_trainer import kd_step, KDWeights, KDStepConfig, pack_batch
from two_stage_curriculum import Curriculum
from rewards_boundary import compute_token_divergence


# ----------------------------
# small helpers
# ----------------------------
def cfg_get(obj, path, default=None):
    cur = obj
    for key in path.split("."):
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def build_optimizer(model: nn.Module, cfg):
    lr = cfg_get(cfg, "training.lr", 2e-4)
    wd = cfg_get(cfg, "training.weight_decay", 0.01)
    betas = cfg_get(cfg, "training.betas", (0.9, 0.95))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

def build_optimizer_kd(model: nn.Module, cfg):
    lr = cfg_get(cfg, "kd.lr", 5e-4)
    wd = cfg_get(cfg, "kd.weight_decay", 0.0)
    betas = cfg_get(cfg, "kd.betas", (0.9, 0.95))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

def make_warmup_cosine_scheduler(optim, total_steps: int, warmup_ratio: float = 0.05):
    warmup = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

def _is_accelerate_dispatched(m) -> bool:
    return getattr(m, "hf_device_map", None) not in (None, {})

def _move_if_needed(m, device):
    # Respect accelerate hooks / 4bit/8bit placements
    if _is_accelerate_dispatched(m):
        return m
    if getattr(m, "is_loaded_in_8bit", False) or getattr(m, "is_loaded_in_4bit", False):
        return m
    return m.to(device)


# ----------------------------
# data
# ----------------------------
# train_two_stage.py
def build_prompts_list(tokenizer, cfg, split_override: str | None = None) -> List[str]:
    """
    Build prompts from HF (preferred) or local file.
    If HF fails/empty, fall back to local JSONL.
    """
    data_cfg = getattr(cfg, "data", {}) or {}
    source = (getattr(data_cfg, "source", None) or data_cfg.get("source") or "hf").lower()
    split = split_override or getattr(data_cfg, "split", "train")
    errors: List[str] = []

    if source == "hf":
        name  = getattr(data_cfg, "hf_name", None) or data_cfg.get("hf_name")
        if not isinstance(name, str):
            raise ValueError("data.hf_name must be a string like 'allenai/tulu-3-sft-mixture'")
        field = getattr(data_cfg, "messages_field", "messages")
        keep_history = getattr(data_cfg, "keep_history", True)
        sample_max = getattr(data_cfg, "sample_max", None)
        load_kwargs = dict(getattr(data_cfg, "load_kwargs", None) or {})

        # let users pass token/cache/streaming via config:
        # data:
        #   load_kwargs:
        #     token: ${HF_TOKEN}
        #     streaming: false
        #     cache_dir: /some/cache
        tried = []
        for sp in [split, "validation", "test", getattr(data_cfg, "split", "train")]:
            if sp in tried:
                continue
            tried.append(sp)
            try:
                ds = HFChatPromptsDataset(
                    dataset_name=name,
                    split=sp,
                    tokenizer=tokenizer,
                    messages_field=field,
                    sample_max=sample_max,
                    keep_history=keep_history,
                    load_kwargs=load_kwargs,
                )
                items = [rec["prompt"] for rec in ds]
                if len(items) > 0:
                    return items
                else:
                    errors.append(f"HF dataset loaded but 0 prompts after parsing (split={sp}).")
            except Exception as e:
                errors.append(f"HF load failed for split={sp}: {type(e).__name__}: {e}")

        # HF failed or empty: try local file if provided
        local_eval_path = cfg_get(cfg, "eval.prompts_path", None)
        local_train_path = cfg_get(cfg, "data.prompts_path", None)
        for path in [local_train_path, local_eval_path]:
            if path and os.path.exists(path):
                ds = PromptOnlyDataset(path)
                items = [rec["prompt"] for rec in ds]
                if len(items) > 0:
                    print(f"[data] Falling back to local prompts at {path} (n={len(items)})")
                    return items

        # Still nothing -> raise with actionable message
        hint = (
            "No prompts found. Possible fixes:\n"
            "  • Check internet/HF credentials (set `data.load_kwargs.token` or `HF_TOKEN`).\n"
            "  • Verify dataset and split exist (data.hf_name, data.split).\n"
            "  • Provide a local JSONL via `data.prompts_path` (one object per line: {\"prompt\": \"...\"}).\n"
            "  • Or set `data.source: local` and `data.prompts_path: path/to/prompts.jsonl`.\n"
            f"Details:\n- " + "\n- ".join(errors)
        )
        raise RuntimeError(hint)

    # Local JSONL paths
    path = cfg_get(cfg, "data.prompts_path", "data/prompts.jsonl")
    if split_override in ("validation", "test"):
        path = cfg_get(cfg, "eval.prompts_path", None) or cfg_get(cfg, "data.val_prompts_path", path)
    if not os.path.exists(path):
        raise RuntimeError(f"Local prompts file not found: {path}")
    ds = PromptOnlyDataset(path)
    items = [rec["prompt"] for rec in ds]
    if not items:
        raise RuntimeError(f"Local prompts file is empty: {path}")
    return items

# ----------------------------
# rollout for GRPO collection
# ----------------------------
@torch.no_grad()
def _top1_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)

@torch.no_grad()
def rollout_one(
    *,
    prompts: List[str],
    tokenizer,
    draft_model: nn.Module,
    target_model: nn.Module,
    device: torch.device,
    temperature: float,
    spec_window: int,
    max_input_len: int,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    N, _ = input_ids.shape
    T = spec_window

    draft_logits_list, teacher_logits_list = [], []
    accepted_mask = torch.zeros(T, N, device=device)
    old_acts_logp_list = []
    actions = torch.empty(T, N, dtype=torch.long, device=device)
    rejected_seen = torch.zeros(N, dtype=torch.bool, device=device)

    cur_ids, cur_attn = input_ids, attn_mask
    draft_model.eval()
    target_model.eval()

    for t in range(T):
        d_out = draft_model(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
        d_logits = d_out.logits[:, -1, :]
        if temperature <= 0.0:
            sampled = d_logits.argmax(-1)
        else:
            probs = (d_logits / max(temperature, 1e-6)).softmax(-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
        logp_old = d_logits.log_softmax(-1).gather(-1, sampled[:, None]).squeeze(-1)

        t_out = target_model(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
        t_logits = t_out.logits[:, -1, :]

        tgt_top1 = _top1_from_logits(t_logits)
        step_accept = (tgt_top1 == sampled) & (~rejected_seen)
        accepted_mask[t, step_accept] = 1.0
        rejected_seen |= (~step_accept) & (~rejected_seen)

        cur_ids = torch.cat([cur_ids, sampled[:, None]], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones(N, 1, device=device, dtype=cur_attn.dtype)], dim=1)

        draft_logits_list.append(d_logits)
        teacher_logits_list.append(t_logits)
        old_acts_logp_list.append(logp_old)
        actions[t] = sampled

    draft_logits = torch.stack(draft_logits_list, dim=0)
    teacher_logits = torch.stack(teacher_logits_list, dim=0)
    first_reject_mask = compute_first_reject_mask(accepted_mask)

    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "draft_logits": draft_logits,
        "teacher_logits": teacher_logits,
        "accepted_mask": accepted_mask,
        "first_reject_mask": first_reject_mask,
        "actions": actions,
        "old_acts_logp": torch.stack(old_acts_logp_list).reshape(-1),
    }


# ----------------------------
# evaluation routines
# ----------------------------
@torch.no_grad()
def _grab_logits(out) -> torch.Tensor:
    # Works for ModelOutput or tuple/list
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    raise RuntimeError(f"Model forward did not return logits. Got: {type(out)}")

@torch.no_grad()
def eval_kd_alignment(
    *,
    tokenizer,
    draft: nn.Module,
    target: nn.Module,
    device: torch.device,
    prompts: List[str],
    max_input_len: int,
    gen_tokens: int,
) -> Dict[str, float]:
    # teacher-forced pack (same as KD)
    full_ids, nonpad_mask, cont_mask = pack_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        target_model=target,
        device=device,
        max_input_len=max_input_len,
        max_new_tokens=gen_tokens,
        temperature=0.0,
    )

    # Input to forwards: [B, L-1]
    input_ids = full_ids[:, :-1].contiguous()
    attn_mask = nonpad_mask[:, :-1].contiguous()
    cont_region = cont_mask[:, 1:].contiguous()

    # For safety, ensure non-empty
    assert input_ids.numel() > 0, "Empty input_ids in eval_kd_alignment."

    target.eval(); draft.eval()
    t_out = target(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    d_out = draft(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)

    teacher_logits = _grab_logits(t_out)
    draft_logits   = _grab_logits(d_out)

    # Expect [B, L-1, V]
    if teacher_logits.dim() != 3:
        raise RuntimeError(f"teacher_logits has dim {teacher_logits.dim()}, expected 3. Shape={tuple(teacher_logits.shape)}")
    if draft_logits.dim() != 3:
        raise RuntimeError(f"draft_logits has dim {draft_logits.dim()}, expected 3. Shape={tuple(draft_logits.shape)}")

    # [T,B,*] layout
    TNV_t = teacher_logits.transpose(0, 1).contiguous()  # [T,B,V]
    TNV_d = draft_logits.transpose(0, 1).contiguous()    # [T,B,V]
    TN_cont = cont_region.transpose(0, 1).contiguous()   # [T,B]
    TN_attn = attn_mask.transpose(0, 1).contiguous()     # [T,B]
    mask_TN = (TN_cont * TN_attn)

    # divergences
    kl_t   = compute_token_divergence(TNV_t, TNV_d, kind="kl",    alpha=0.5, topk_for_ce=0)  # [T,B]
    a05_t  = compute_token_divergence(TNV_t, TNV_d, kind="alpha", alpha=0.5, topk_for_ce=0)  # [T,B]
    denom  = mask_TN.sum().clamp_min(1.0)

    # top-1 match on continuation region only
    t_argmax = teacher_logits.argmax(dim=-1)  # [B,L-1]
    d_argmax = draft_logits.argmax(dim=-1)    # [B,L-1]
    match = ((t_argmax == d_argmax).float() * cont_region).sum() / cont_region.sum().clamp_min(1.0)

    return {
        "eval/kl_per_tok":        float((kl_t * mask_TN).sum().cpu() / denom),
        "eval/alpha05_per_tok":   float((a05_t * mask_TN).sum().cpu() / denom),
        "eval/top1_match_rate":   float(match.cpu()),
        "eval/avg_cont_len":      float(cont_region.sum(dim=1).mean().cpu()),
    }

@torch.no_grad()
def eval_spec_acceptance(
    *,
    tokenizer,
    draft: nn.Module,
    target: nn.Module,
    device: torch.device,
    prompts: List[str],
    max_input_len: int,
    T: int,
) -> Dict[str, float]:
    out = rollout_one(
        prompts=prompts,
        tokenizer=tokenizer,
        draft_model=draft,
        target_model=target,
        device=device,
        temperature=0.0,
        spec_window=T,
        max_input_len=max_input_len,
    )
    acc = out["accepted_mask"]                 # [T,N]
    per_seq_accepts = acc.sum(dim=0)           # [N]
    full_accept = (per_seq_accepts == T).float()
    return {
        "eval/accept_tokens_mean": float(per_seq_accepts.mean().cpu()),
        "eval/full_accept_rate":   float(full_accept.mean().cpu()),
    }

# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="two_stage_kd_grpo")
    parser.add_argument("--wandb_api_key", type=str, required=True)
    args = parser.parse_args()

    cfg = to_attrdict(load_yaml(args.config))
    set_seed(cfg_get(cfg, "training.seed", 42))
    device = get_device(cfg_get(cfg, "training.device", "cuda"))
    os.makedirs(cfg_get(cfg, "training.output_dir", "./outputs"), exist_ok=True)

    # wandb
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=cfg_get(cfg, "logging.project", "specdec-two-stage"),
        name=cfg_get(cfg, "logging.name", args.run_name),
        config=dict(cfg),
    )

    # tokenizer / models
    tok_name = cfg_get(cfg, "models.tokenizer", "Qwen/Qwen3-0.6B")
    tgt_name = cfg_get(cfg, "models.target",   "Qwen/Qwen3-8B")
    dft_name = cfg_get(cfg, "models.draft",    "Qwen/Qwen3-0.6B")

    tokenizer = load_tokenizer(tok_name)
    target = load_target(tgt_name, dtype=cfg_get(cfg, "training.dtype", "bf16"),
                         device=cfg_get(cfg, "training.device", "cuda"))
    draft  = load_draft(dft_name, dtype=cfg_get(cfg, "training.dtype", "bf16"),
                        device=cfg_get(cfg, "training.device", "cuda"))

    # Ensure padding is defined and consistent
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    for m in (target, draft):
        if getattr(m.config, "pad_token_id", None) is None:
            m.config.pad_token_id = tokenizer.pad_token_id

    try:
        for m in (target, draft):
            if hasattr(m, "config") and hasattr(m.config, "attn_implementation"):
                m.config.attn_implementation = "flash_attention_2"
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

    # LoRA + train LM head
    draft = attach_lora(draft, cfg)
    enable_lm_head_training(draft)

    # placement (respect accelerate)
    target = _move_if_needed(target, device).eval()
    for p in target.parameters(): p.requires_grad_(False)
    draft = _move_if_needed(draft, device)

    # memory knobs
    if hasattr(draft, "gradient_checkpointing_enable"):
        try:
            draft.gradient_checkpointing_enable()
        except TypeError:
            draft.gradient_checkpointing_enable(use_reentrant=False)
    if hasattr(draft, "config"):
        draft.config.use_cache = False
    draft.train()

    # optimizer (KD stage)
    optim = build_optimizer_kd(draft, cfg)
    scheduler = make_warmup_cosine_scheduler(optim, total_steps=cfg_get(cfg, "kd.total_steps", 2000),
                                             warmup_ratio=cfg_get(cfg, "kd.warmup_ratio", 0.05))

    # data
    prompts_all = build_prompts_list(tokenizer, cfg, split_override="train")
    if len(prompts_all) == 0:
        raise RuntimeError("No prompts found for training.")

    # eval prompts
    eval_enabled = bool(cfg_get(cfg, "eval.enabled", True))
    eval_n = int(cfg_get(cfg, "eval.n_prompts", 64))
    eval_gen = int(cfg_get(cfg, "eval.gen_tokens", 128))
    eval_max_in = int(cfg_get(cfg, "eval.max_input_len", cfg_get(cfg, "data.max_input_len", 1024)))
    eval_every_kd = int(cfg_get(cfg, "eval.every_steps_kd", 200))
    eval_every_grpo = int(cfg_get(cfg, "eval.every_steps_grpo", 1000))

    eval_prompts_all = build_prompts_list(tokenizer, cfg, split_override=cfg_get(cfg, "eval.split", "validation"))
    if len(eval_prompts_all) == 0:
        # fallback: sample from train
        eval_prompts_all = prompts_all
    # fix the eval subset deterministically
    rng = random.Random(1234)
    eval_subset = [eval_prompts_all[i] for i in rng.sample(range(len(eval_prompts_all)), k=min(eval_n, len(eval_prompts_all)))]

    # curriculum for KD
    curriculum = Curriculum(
        total_kd_steps=cfg_get(cfg, "kd.total_steps", 2000),
        max_input_len_start=cfg_get(cfg, "kd.curriculum.max_input_len_start", 512),
        max_input_len_end=cfg_get(cfg, "kd.curriculum.max_input_len_end", 2048),
        max_new_tokens_start=cfg_get(cfg, "kd.curriculum.max_new_tokens_start", 128),
        max_new_tokens_end=cfg_get(cfg, "kd.curriculum.max_new_tokens_end", 512),
        hold_ratio=cfg_get(cfg, "kd.curriculum.hold_ratio", 0.25),
    )

    # ---------- Stage 1: KD/SFT ----------
    kd_steps = cfg_get(cfg, "kd.total_steps", 2000)
    kd_batch = cfg_get(cfg, "kd.batch_size", 4)
    kd_log_every = cfg_get(cfg, "kd.log_every", 20)

    kd_weights = KDWeights(
        margin_gamma=cfg_get(cfg, "kd.margin_gamma", 0.5),
        margin_center=cfg_get(cfg, "kd.margin_center", 1.0),
        w_min=cfg_get(cfg, "kd.w_min", 0.2),
        mismatch_lambda=cfg_get(cfg, "kd.mismatch_lambda", 0.3),
    )
    kd_cfg = KDStepConfig(
        divergence=cfg_get(cfg, "reward.divergence", "kl"),
        alpha=float(cfg_get(cfg, "reward.alpha", 0.5)),
        topk_for_ce=int(cfg_get(cfg, "reward.topk_for_ce", 0)),
        entropy_bonus=0.0,
        anchor_kl_beta=0.0,
        max_grad_norm=float(cfg_get(cfg, "training.max_grad_norm", 1.0)),
        temperature=float(cfg_get(cfg, "kd.temperature", 0.0)),
    )

    pbar = tqdm(range(kd_steps), desc="KD/SFT", dynamic_ncols=True)
    for step in pbar:
        max_input_len, max_new_tokens = curriculum.at(step)
        # wrap-around sampling
        start = (step * kd_batch) % len(prompts_all)
        end = start + kd_batch
        batch_prompts = prompts_all[start:end] if end <= len(prompts_all) else (prompts_all[start:] + prompts_all[:(end % len(prompts_all))])

        metrics = kd_step(
            tokenizer=tokenizer,
            draft=draft,
            target=target,
            optimizer=optim,
            prompts=batch_prompts,
            device=device,
            kd_cfg=kd_cfg,
            kd_weights=kd_weights,
            max_input_len=max_input_len,
            max_new_tokens=max_new_tokens,
        )
        scheduler.step()
        if step % kd_log_every == 0:
            wandb.log({**metrics, "train/curr_max_in": max_input_len, "train/curr_max_new": max_new_tokens,
                       "train/lr": scheduler.get_last_lr()[0]}, step=step)
            pbar.set_postfix(loss=f'{metrics["train/kd_total_loss"]:.3f}', cont=f'{metrics["train/avg_cont_len"]:.1f}')

        # ---- KD evaluation ----
        if eval_enabled and (step % max(1, eval_every_kd) == 0):
            draft.eval()
            try:
                kd_eval = eval_kd_alignment(
                    tokenizer=tokenizer,
                    draft=draft,
                    target=target,
                    device=device,
                    prompts=eval_subset,
                    max_input_len=eval_max_in,
                    gen_tokens=eval_gen,
                )
                wandb.log(kd_eval, step=step)
            finally:
                draft.train()

    # save after KD
    outdir = cfg_get(cfg, "training.output_dir", "./outputs")
    os.makedirs(outdir, exist_ok=True)
    torch.save({"model": draft.state_dict(), "opt": optim.state_dict()}, os.path.join(outdir, "draft_after_kd.pt"))

    # ---------- Stage 2: GRPO ----------
    # new optimizer for GRPO
    optim = build_optimizer(draft, cfg)

    grpo_steps = cfg_get(cfg, "grpo.total_steps", 30000)
    group_size = cfg_get(cfg, "training.group_size", 4)
    spec_window = cfg_get(cfg, "training.S", 2)
    grpo_batch_prompts = cfg_get(cfg, "training.batch_prompts", 2)
    grpo_temp = cfg_get(cfg, "training.temperature", 0.8)
    max_input_len_grpo = cfg_get(cfg, "data.max_input_len", 1024)
    save_every = cfg_get(cfg, "training.save_every", 2000)

    helper = BoundaryGRPOHelper(cfg, draft_module=draft, optim=optim)

    pbar = tqdm(range(grpo_steps), desc="GRPO", dynamic_ncols=True)
    for step in pbar:
        start = (step * grpo_batch_prompts) % len(prompts_all)
        end = start + grpo_batch_prompts
        batch_prompts = prompts_all[start:end] if end <= len(prompts_all) else (prompts_all[start:] + prompts_all[:(end % len(prompts_all))])

        group = [
            rollout_one(
                prompts=batch_prompts,
                tokenizer=tokenizer,
                draft_model=draft,
                target_model=target,
                device=device,
                temperature=grpo_temp,
                spec_window=spec_window,
                max_input_len=max_input_len_grpo,
            )
            for _ in range(group_size)
        ]
        metrics = helper.step_on_group(group=group, step=step, log_fn=wandb.log)
        pbar.set_postfix(R=f'{metrics["train/seq_reward_mean"]:.3f}',
                         KL=f'{metrics["train/ppo_approx_kl"]:.4f}',
                         clip=f'{metrics["train/ppo_clipfrac"]:.2f}')

        # ---- GRPO evaluation ----
        if eval_enabled and (step % max(1, eval_every_grpo) == 0):
            draft.eval()
            try:
                # Teacher-forced alignment
                kd_eval = eval_kd_alignment(
                    tokenizer=tokenizer,
                    draft=draft,
                    target=target,
                    device=device,
                    prompts=eval_subset,
                    max_input_len=eval_max_in,
                    gen_tokens=eval_gen,
                )
                # Spec acceptance sim (greedy)
                spec_eval = eval_spec_acceptance(
                    tokenizer=tokenizer,
                    draft=draft,
                    target=target,
                    device=device,
                    prompts=eval_subset[: min(16, len(eval_subset))],  # keep this small; it’s stepwise
                    max_input_len=eval_max_in,
                    T=min(64, eval_gen),
                )
                wandb.log({**{f"eval_grpo/{k.split('/',1)[1]}": v for k,v in kd_eval.items()},
                           **{f"eval_grpo/{k.split('/',1)[1]}": v for k,v in spec_eval.items()}},
                          step=step)
            finally:
                draft.train()

        if step > 0 and (step % save_every == 0):
            torch.save({"model": draft.state_dict(), "opt": optim.state_dict()}, os.path.join(outdir, f"draft_grpo_step{step}.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()
