from __future__ import annotations
import argparse, os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb

from utils import load_yaml, set_seed, get_device, to_attrdict
from models import load_tokenizer, load_target, load_draft
from lora_setup import attach_lora
from trainer_hook import BoundaryGRPOHelper
from collector_utils import compute_first_reject_mask


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


def _is_accelerate_dispatched(m) -> bool:
    # Transformers sets this when using device_map / accelerate hooks.
    return getattr(m, "hf_device_map", None) not in (None, {})

def _move_if_needed(m, device):
    # If model is already sharded/dispatched by accelerate (device_map), do NOT .to(device)
    if _is_accelerate_dispatched(m):
        return m
    # BitsAndBytes 4/8-bit models are also typically device-mapped
    if getattr(m, "is_loaded_in_8bit", False) or getattr(m, "is_loaded_in_4bit", False):
        return m
    return m.to(device)

@torch.no_grad()
def _top1_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)

def _sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return _top1_from_logits(logits)
    probs = (logits / max(temperature, 1e-6)).softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def _gather_logp(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(dim=-1)
    return logp.gather(-1, ids.view(-1, 1)).squeeze(-1)

def _append_tokens(input_ids: torch.Tensor, new_ids: torch.Tensor) -> torch.Tensor:
    return torch.cat([input_ids, new_ids.view(-1, 1)], dim=1)

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
    # ---- COLLECT UNDER no_grad (no graphs kept) ----
    with torch.no_grad():
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
        draft_model.eval()   # eval for collection; we’ll train during recompute
        target_model.eval()

        # TODO sample and accept method should be alpha related 
        for t in range(T):
            d_out = draft_model(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
            d_logits = d_out.logits[:, -1, :]                 # [N,V]
            probs = (d_logits / max(temperature, 1e-6)).softmax(-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1) # [N]
            logp_old = d_logits.log_softmax(-1).gather(-1, sampled[:, None]).squeeze(-1)

            t_out = target_model(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
            t_logits = t_out.logits[:, -1, :]

            tgt_top1 = t_logits.argmax(-1)
            step_accept = (tgt_top1 == sampled) & (~rejected_seen)
            accepted_mask[t, step_accept] = 1.0
            rejected_seen |= (~step_accept) & (~rejected_seen)

            # extend prefix
            cur_ids = torch.cat([cur_ids, sampled[:, None]], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones(N, 1, device=device, dtype=cur_attn.dtype)], dim=1)

            # store detached things for reward + replay
            draft_logits_list.append(d_logits)   # detached by no_grad
            teacher_logits_list.append(t_logits)
            old_acts_logp_list.append(logp_old)
            actions[t] = sampled

        draft_logits   = torch.stack(draft_logits_list, dim=0)     # [T,N,V]
        teacher_logits = torch.stack(teacher_logits_list, dim=0)   # [T,N,V]
        first_reject_mask = compute_first_reject_mask(accepted_mask)

        return {
            "input_ids": enc["input_ids"].to(device),        # starting prefix (replay)
            "attention_mask": enc["attention_mask"].to(device),
            "draft_logits": draft_logits,                    # detached
            "teacher_logits": teacher_logits,                # detached
            "accepted_mask": accepted_mask,
            "first_reject_mask": first_reject_mask,
            "actions": actions,                               # [T,N] ints
            "old_acts_logp": torch.stack(old_acts_logp_list).reshape(-1),  # [T*N], detached
        }


def build_dataset_from_cfg(tokenizer, cfg) -> List[str]:
    """
    Returns a list[str] prompts.
    Fixes the HFValidationError by ensuring we pass a *string repo id* to datasets.load_dataset,
    never the full config dict.
    """
    from data import PromptOnlyDataset, HFChatPromptsDataset

    data_cfg = getattr(cfg, "data", {}) or {}
    source = getattr(data_cfg, "source", None) or data_cfg.get("source")

    if source == "hf":
        name  = getattr(data_cfg, "hf_name", None) or data_cfg.get("hf_name")
        if not isinstance(name, str):
            raise ValueError(f"data.hf_name must be a string repo id like 'allenai/tulu-3-sft-mixture', got: {name!r}")
        split = getattr(data_cfg, "split", "train")
        field = getattr(data_cfg, "messages_field", "messages")
        sample_max = getattr(data_cfg, "sample_max", None)
        keep_history = getattr(data_cfg, "keep_history", True)
        load_kwargs = getattr(data_cfg, "load_kwargs", None) or {}
        ds = HFChatPromptsDataset(
            dataset_name=name,
            split=split,
            tokenizer=tokenizer,
            messages_field=field,
            sample_max=sample_max,
            keep_history=keep_history,
            load_kwargs=load_kwargs,
        )
        return [rec["prompt"] for rec in ds]

    # Fallback: local prompts.jsonl
    prompts_path = cfg_get(cfg, "data.prompts_path", "data/prompts.jsonl")
    ds = PromptOnlyDataset(prompts_path)
    return [rec["prompt"] for rec in ds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--run_name", type=str, default="grpo_boundary")
    parser.add_argument("--wandb_api_key", type=str, required=True)
    args = parser.parse_args()

    cfg = to_attrdict(load_yaml(args.config))
    set_seed(cfg_get(cfg, "training.seed", 42))
    device = get_device(cfg_get(cfg, "training.device", "cuda"))
    os.makedirs(cfg_get(cfg, "training.output_dir", "./outputs"), exist_ok=True)

    # W&B login/init
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=cfg_get(cfg, "logging.project", "grpo-specdec"),
        name=cfg_get(cfg, "logging.name", args.run_name),
        config=dict(cfg),
    )

    # Tokenizer / model names
    tok_name = cfg_get(cfg, "models.tokenizer", "Qwen/Qwen3-0.6B")
    tgt_name = cfg_get(cfg, "models.target",   "Qwen/Qwen3-0.6B")
    dft_name = cfg_get(cfg, "models.draft",    "Qwen/Qwen3-0.6B")

    # Load
    tokenizer = load_tokenizer(tok_name)
    target = load_target(tgt_name, dtype=cfg_get(cfg, "training.dtype", "bf16"),
                        device=cfg_get(cfg, "training.device", "cuda"))
    draft  = load_draft(dft_name,  dtype=cfg_get(cfg, "training.dtype", "bf16"),
                        device=cfg_get(cfg, "training.device", "cuda"))

    # IMPORTANT: attach LoRA BEFORE doing any manual .to(device) (and avoid .to if device_map is set)
    draft = attach_lora(draft, cfg)

    # Now only move if not already accelerate-dispatched
    device = get_device(cfg_get(cfg, "training.device", "cuda"))
    target = _move_if_needed(target, device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    draft = _move_if_needed(draft, device)

    optim = build_optimizer(draft, cfg)

    # Prompts (HF or local)
    prompts_all = build_dataset_from_cfg(tokenizer, cfg)

    total_steps  = cfg_get(cfg, "training.max_steps", 1000)
    batch_prompts= cfg_get(cfg, "training.batch_prompts", 2)
    group_size   = cfg_get(cfg, "training.group_size", 4)
    max_input_len= cfg_get(cfg, "data.max_input_len", 1024)
    spec_window  = cfg_get(cfg, "training.S", 2)
    temperature  = cfg_get(cfg, "training.temperature", 0.8)

    helper = BoundaryGRPOHelper(cfg, draft_module=draft, optim=optim)

    pbar = tqdm(range(total_steps), desc="Training", dynamic_ncols=True)
    for step in pbar:
        # sample prompts
        if len(prompts_all) < batch_prompts:
            raise RuntimeError(f"Not enough prompts ({len(prompts_all)}) for batch_prompts={batch_prompts}")
        batch = prompts_all[step * batch_prompts % len(prompts_all) : (step + 1) * batch_prompts % len(prompts_all)]
        if len(batch) < batch_prompts:
            # wrap-around
            batch = (prompts_all[(step * batch_prompts) % len(prompts_all):] +
                     prompts_all[:batch_prompts - len(batch)])

        # collect GRPO group
        group = [
            rollout_one(
                prompts=batch,
                tokenizer=tokenizer,
                draft_model=draft,
                target_model=target,
                device=device,
                temperature=temperature,
                spec_window=spec_window,
                max_input_len=max_input_len,
            )
            for _ in range(group_size)
        ]

        metrics = helper.step_on_group(group=group, step=step, log_fn=wandb.log)
        pbar.set_postfix(
            R=f'{metrics["train/seq_reward_mean"]:.3f}',
            KL=f'{metrics["train/ppo_approx_kl"]:.4f}',
            clip=f'{metrics["train/ppo_clipfrac"]:.2f}',
        )

        if step > 0 and step % cfg_get(cfg, "training.save_every", 1000) == 0:
            outdir = cfg_get(cfg, "training.output_dir", "./outputs")
            ckpt = os.path.join(outdir, f"draft_step{step}.pt")
            torch.save({"step": step, "model": draft.state_dict(), "opt": optim.state_dict()}, ckpt)

    wandb.finish()


if __name__ == "__main__":
    main()
