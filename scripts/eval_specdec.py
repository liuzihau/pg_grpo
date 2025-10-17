from __future__ import annotations
import argparse, os, json, gc
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
from src.models.load import load_models_for_eval_from_model_dir, load_base_draft_from_training_cfg
from src.data.prompts import load_prompts_for_split, make_manual_splits, truncate_prompt_by_tokens
from src.specdec.sim import eval_spec_batch_prob


def _prepare_prompts(tokenizer, prompts: List[str], max_input_len: int, S: int) -> List[str]:
    # Left-truncate to fit context; S is unused but kept for symmetry with data_gen.
    out: List[str] = []
    budget = max(1, max_input_len - 1)
    for p in prompts:
        out.append(truncate_prompt_by_tokens(tokenizer, p, budget))
    return out


def _read_json(p: Path):
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _discover_pregen_dir(train_cfg, model_dir: Path) -> Path | None:
    pregen_dir = cfg_get(train_cfg, "kd.pregen_dir", None)
    if pregen_dir:
        return Path(pregen_dir).expanduser().resolve()
    for c in [model_dir / "train_summary.json",
              model_dir.parent / "train_summary.json",
              model_dir.parent.parent / "train_summary.json"]:
        j = _read_json(c)
        if j and j.get("pregen_dir"):
            return Path(j["pregen_dir"]).expanduser().resolve()
    return None


def _load_pregen_data_cfg(pregen_dir: Path):
    cfg_path = pregen_dir / "cfg.lock.yaml"
    if not cfg_path.is_file():
        return None
    raw = load_yaml_with_includes(cfg_path)
    return to_attrdict(raw)


def _eval_one_variant(
    *,
    label: str,
    tokenizer,
    draft,
    teacher,
    device,
    prompts: List[str],
    B: int,
    K: int,
    T: int,
    temp: float,
    max_input_len: int,
    acceptance_cap: float,
    run_to_wandb: bool,
) -> Dict[str, float]:
    agg = {
        "alpha_accept": 0.0,
        "goodput_tokens_per_teacher_call": 0.0,
        "avg_accepted_span": 0.0,
        "reject_rate": 0.0,
        "mean_accepted_tokens": 0.0,
        "mean_teacher_calls": 0.0,
        "count": 0,
    }
    pbar = tqdm(range(0, len(prompts), B), desc=f"SpecDec Eval ({label})", ncols=100)
    for off in pbar:
        batch_prompts = prompts[off : off + B]
        stats = eval_spec_batch_prob(
            tokenizer=tokenizer,
            draft=draft,
            teacher=teacher,
            device=device,
            prompts=batch_prompts,
            K=K,
            max_new_tokens=T,
            temperature=temp,
            max_input_len=max_input_len,
            acceptance_cap=acceptance_cap,
        )
        agg["count"] += 1
        for k in stats:
            if k in agg:
                agg[k] += (stats[k] - agg[k]) / agg["count"]
            else:
                agg[k] = stats[k]

        if run_to_wandb:
            wandb_log({
                f"eval/{label}/alpha_accept": stats["alpha_accept"],
                f"eval/{label}/goodput": stats["goodput_tokens_per_teacher_call"],
                f"eval/{label}/reject_rate": stats["reject_rate"],
                f"eval/{label}/avg_span": stats["avg_accepted_span"],
            })

        pbar.set_postfix(
            alpha=f"{agg['alpha_accept']:.3f}",
            goodput=f"{agg['goodput_tokens_per_teacher_call']:.3f}"
        )

    # strip the running counter
    agg.pop("count", None)
    return agg


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

    # --- Load tokenizer + models from the training cfg in model_dir
    model_dir = Path(cfg.eval.model_dir).expanduser().resolve()
    dtype = parse_torch_dtype(cfg_get(cfg, "training.dtype", "bf16"))
    tokenizer, ft_draft, teacher, train_cfg, stage_kind, used_cfg_path = \
        load_models_for_eval_from_model_dir(
            model_dir, dtype_name=str(dtype).split(".")[-1], device_map="auto"
        )

    # --- Dataset: prefer KD corpus config's validation
    pregen_dir = _discover_pregen_dir(train_cfg, model_dir)
    if pregen_dir is None:
        print("[eval] Could not discover KD corpus; falling back to eval.yaml:data.*")
        data_cfg_for_eval = cfg
    else:
        pregen_cfg = _load_pregen_data_cfg(pregen_dir)
        if pregen_cfg is None or getattr(pregen_cfg, "data", None) is None:
            print(f"[eval] Found KD corpus at {pregen_dir}, but no cfg.lock.yaml; using eval.yaml:data.*")
            data_cfg_for_eval = cfg
        else:
            merged = dict(pregen_cfg)
            merged["data"] = dict(pregen_cfg.data)
            for k, v in dict(getattr(cfg, "data", {})).items():
                merged["data"][k] = v
            data_cfg_for_eval = to_attrdict(merged)

    base_split = cfg_get(cfg, "data.split", None) or "validation"
    prompts = load_prompts_for_split(tokenizer, data_cfg_for_eval, split=base_split)
    if not prompts:
        train_prompts, val_prompts = make_manual_splits(
            tokenizer, data_cfg_for_eval, seed=int(cfg.training.seed)
        )
        prompts = val_prompts if base_split != "train" else train_prompts

    # Optional sampling
    n = int(cfg.eval.num_samples)
    if n and n < len(prompts):
        prompts = prompts[:n]

    prepped_prompts = _prepare_prompts(
        tokenizer, prompts,
        max_input_len=int(cfg_get(data_cfg_for_eval, "data.max_input_len", cfg.data.max_input_len)),
        S=int(cfg.eval.K),
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

    # --- Shared knobs
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))
    B = int(cfg.eval.batch_size)
    K = int(cfg.eval.K)
    T = int(cfg.eval.max_new)
    temp = float(cfg.eval.temperature)
    cap = float(cfg.eval.get("acceptance_cap", 1.0))
    max_input_len = int(cfg_get(data_cfg_for_eval, "data.max_input_len", cfg.data.max_input_len))

    # -------- Pass 1: FINETUNED (LoRA) --------
    metrics_ft = _eval_one_variant(
        label="finetuned",
        tokenizer=tokenizer, draft=ft_draft, teacher=teacher, device=device,
        prompts=prepped_prompts, B=B, K=K, T=T, temp=temp,
        max_input_len=max_input_len, acceptance_cap=cap,
        run_to_wandb=bool(args.wandb),
    )
    (out_dir / "finetuned").mkdir(parents=True, exist_ok=True)
    save_json({
        "created_at": timestamp(),
        "variant": "finetuned",
        "model_dir": str(model_dir),
        "used_training_cfg": str(used_cfg_path),
        "stage_kind": stage_kind,
        "n_samples": len(prepped_prompts),
        "K": K, "max_new": T, "temperature": temp, "acceptance_cap": cap,
        "metrics": metrics_ft,
    }, out_dir / "finetuned" / "metrics.json")
    print("[DONE] Finetuned metrics ->", out_dir / "finetuned" / "metrics.json")

    # Free the LoRA draft to save memory
    del ft_draft
    torch.cuda.empty_cache()
    gc.collect()

    # -------- Pass 2: BASE draft (pre-finetune), if requested --------
    compare = bool(cfg_get(cfg, "eval.compare_with_base", True))
    if compare:
        base_draft = load_base_draft_from_training_cfg(
            train_cfg, dtype_name=str(dtype).split(".")[-1], device_map="auto"
        )
        metrics_base = _eval_one_variant(
            label="base",
            tokenizer=tokenizer, draft=base_draft, teacher=teacher, device=device,
            prompts=prepped_prompts, B=B, K=K, T=T, temp=temp,
            max_input_len=max_input_len, acceptance_cap=cap,
            run_to_wandb=bool(args.wandb),
        )
        (out_dir / "base").mkdir(parents=True, exist_ok=True)
        save_json({
            "created_at": timestamp(),
            "variant": "base",
            "model_dir": str(model_dir),
            "used_training_cfg": str(used_cfg_path),
            "stage_kind": stage_kind,
            "n_samples": len(prepped_prompts),
            "K": K, "max_new": T, "temperature": temp, "acceptance_cap": cap,
            "metrics": metrics_base,
        }, out_dir / "base" / "metrics.json")
        print("[DONE] Base metrics ->", out_dir / "base" / "metrics.json")

        # Optional: quick delta print
        try:
            delta_goodput = metrics_ft["goodput_tokens_per_teacher_call"] - metrics_base["goodput_tokens_per_teacher_call"]
            print(f"[delta] finetuned - base goodput: {delta_goodput:+.3f}")
        except Exception:
            pass

    wandb_finish()


if __name__ == "__main__":
    main()
