# scripts/eval_specdec.py
from __future__ import annotations
import argparse, os, json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides, parse_overrides,
    set_seed, save_cfg_lock, cfg_get, parse_torch_dtype,
)
from src.common.io import save_json, makedirs, timestamp
from src.common.wandb_util import maybe_init_wandb, wandb_log, wandb_finish
from src.models.load import load_models_for_eval_from_model_dir
from src.data.prompts import load_prompts_for_split, make_manual_splits, truncate_prompt_by_tokens
from src.specdec.sim import eval_spec_batch_prob

def _prepare_prompts(tokenizer, prompts: List[str], max_input_len: int, S: int) -> List[str]:
    out: List[str] = []
    budget = max(1, max_input_len - 1)  # conservative cushion
    for p in prompts:
        out.append(truncate_prompt_by_tokens(tokenizer, p, budget))
    return out

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

    # --- Output dir & lock
    out_dir = Path(cfg.eval.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cfg_lock(raw, out_dir, filename="eval.lock.yaml")

    # --- Load tokenizer + models by *inspecting model_dir config*
    model_dir = Path(cfg.eval.model_dir).expanduser().resolve()
    dtype = parse_torch_dtype(cfg_get(cfg, "eval.dtype", cfg_get(cfg, "training.dtype", "bf16")))
    tokenizer, draft, teacher, train_cfg, stage_kind, used_cfg_path = \
        load_models_for_eval_from_model_dir(model_dir, dtype_name=str(dtype).split(".")[-1], device_map="auto")

    # --- Dataset split (only data.* in eval.yaml is required; no model names)
    base_split = cfg.data.split
    prompts = load_prompts_for_split(tokenizer, cfg, split=base_split)
    if not prompts:
        # fall back to manual splits using eval config (same helper as data_gen)
        train_prompts, val_prompts = make_manual_splits(tokenizer, cfg, seed=int(cfg.training.seed))
        prompts = val_prompts if base_split != "train" else train_prompts

    # Sample a subset if requested
    n = int(cfg.eval.num_samples)
    if n and n < len(prompts):
        prompts = prompts[:n]

    # Truncate to fit model context (≤ max_input_len - max_new safety)
    S = int(cfg.eval.K)
    prepped_prompts = _prepare_prompts(
        tokenizer, prompts, max_input_len=int(cfg.data.max_input_len), S=S
    )

    # --- W&B
    run = maybe_init_wandb(
        enabled=bool(args.wandb),
        api_key=args.wandb_api_key,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config={
            "stage_kind": stage_kind,
            "eval_model_dir": str(model_dir),
            "used_training_cfg": str(used_cfg_path),
            "eval_cfg": raw,
        },
    )

    # --- Eval (batched)
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))
    B = int(cfg.eval.batch_size)
    K = int(cfg.eval.K)
    T = int(cfg.eval.max_new)
    temp = float(cfg.eval.temperature)
    cap = float(cfg.eval.get("acceptance_cap", 1.0))

    agg = {
        "alpha_accept": 0.0,
        "goodput_tokens_per_teacher_call": 0.0,
        "avg_accepted_span": 0.0,
        "reject_rate": 0.0,
        "mean_accepted_tokens": 0.0,
        "mean_teacher_calls": 0.0,
        "count": 0,
    }

    pbar = tqdm(range(0, len(prepped_prompts), B), desc="SpecDec Eval", ncols=100)
    for off in pbar:
        batch_prompts = prepped_prompts[off : off + B]
        stats = eval_spec_batch_prob(
            tokenizer=tokenizer,
            draft=draft,
            teacher=teacher,
            device=device,
            prompts=batch_prompts,
            K=K,
            max_new_tokens=T,
            temperature=temp,
            max_input_len=int(cfg.data.max_input_len),
            acceptance_cap=cap,
        )
        # Online average across batches
        agg["count"] += 1
        for k in stats:
            if k in agg:
                agg[k] += (stats[k] - agg[k]) / agg["count"]
            else:
                agg[k] = stats[k]

        if args.wandb:
            wandb_log({
                "eval/alpha_accept": stats["alpha_accept"],
                "eval/goodput": stats["goodput_tokens_per_teacher_call"],
                "eval/reject_rate": stats["reject_rate"],
                "eval/avg_span": stats["avg_accepted_span"],
            })

        pbar.set_postfix(alpha=f"{agg['alpha_accept']:.3f}",
                         goodput=f"{agg['goodput_tokens_per_teacher_call']:.3f}")

    # --- Save metrics locally
    out_json = {
        "created_at": timestamp(),
        "model_dir": str(model_dir),
        "used_training_cfg": str(used_cfg_path),
        "stage_kind": stage_kind,
        "n_samples": len(prepped_prompts),
        "K": K,
        "max_new": T,
        "temperature": temp,
        "acceptance_cap": cap,
        "metrics": {k: v for k, v in agg.items() if k != "count"},
    }
    save_json(out_json, out_dir / "metrics.json")
    print("[DONE] Saved metrics ->", out_dir / "metrics.json")

    wandb_finish()

if __name__ == "__main__":
    main()
