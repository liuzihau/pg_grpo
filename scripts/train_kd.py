# scripts/train_kd.py
from __future__ import annotations
import argparse, os, math, json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides,
    parse_overrides, set_seed, save_cfg_lock, cfg_get, parse_torch_dtype,
)
from src.common.io import save_json, timestamp, makedirs
from src.common.wandb_util import maybe_init_wandb, wandb_log, wandb_finish

from src.models.tokenizer import load_tokenizer_leftpad
from src.models.load import load_draft_with_lora
from src.data.pregen_kd_ds import PreGeneratedTopKDataset, collate_topk
from src.kd.sparse_kd_loss import sparse_kd_kl
from src.kd.weights import build_kd_weights
from src.training.optim import build_adamw
from src.training.schedule import make_warmup_cosine
from src.training.move import move_to_device_non_blocking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dot.notation=VALUE overrides")
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_api_key", default=None)
    args = ap.parse_args()

    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(apply_overrides(raw, parse_overrides(args.override)))
    set_seed(int(cfg.training.seed))

    out_dir = Path(cfg.kd.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = save_cfg_lock(raw, out_dir, filename="cfg.lock.yaml")
    print(f"[cfg] saved resolved config -> {lock_path}")

    # ---- Tokenizer (must match teacher/draft vocab mapping) ----
    tok_name = cfg_get(cfg, "models.tokenizer", cfg.models.draft)
    tokenizer = load_tokenizer_leftpad(tok_name)

    # ---- Model (draft + LoRA) ----
    dtype = parse_torch_dtype(cfg_get(cfg, "training.dtype", "bf16"))
    model = load_draft_with_lora(
        base_name=cfg.models.draft,
        tokenizer=tokenizer,
        lora_cfg=cfg.lora,
        dtype=dtype,
        device_map="auto",       # single A100: uses that device
        gradient_checkpointing=bool(cfg.training.get("grad_ckpt", True)),
        use_cache=False,
    )
    model.train()

    # ---- Data ----
    # We re-tokenize prompt_text with *this* tokenizer to ensure vocab consistency.
    ds_root = Path(cfg.kd.pregen_dir)
    ds = PreGeneratedTopKDataset(ds_root, split="train")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _collate(batch):
        # Keep collate on CPU (safe for multiple workers)
        return collate_topk(
            batch=batch,
            tokenizer=tokenizer,
            pad_id=pad_id,
            max_input_len=int(cfg.data.max_input_len),
        )

    num_workers = int(cfg_get(cfg, "kd.num_workers", 4))
    dl = DataLoader(
        ds,
        batch_size=int(cfg.kd.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate,
        drop_last=True,
    )

    # ---- Optim & schedule ----
    total_steps = int(cfg.kd.total_steps)
    optim = build_adamw(
        model,
        lr=float(cfg.kd.lr),
        weight_decay=float(cfg.kd.get("weight_decay", 0.0)),
        betas=tuple(cfg.kd.get("betas", [0.9, 0.95])),
        eps=1e-8,
    )
    sched = make_warmup_cosine(
        optim,
        total_steps=total_steps,
        warmup_ratio=float(cfg.kd.get("warmup_ratio", 0.05)),
        min_lr=float(cfg.kd.get("min_lr", 0.0)),
    )

    # ---- Logging ----
    wb_project = cfg_get(cfg, "logging.project", "kd-train")
    wb_name    = cfg_get(cfg, "logging.name",    "kd_run")
    wb_mode    = cfg_get(cfg, "logging.mode",    None)      # e.g. "online" | "offline"
    wb_tags    = cfg_get(cfg, "logging.tags",    None)
    wb_group   = cfg_get(cfg, "logging.group",   None)

    wb_cfg = json.loads(json.dumps(cfg, default=str))  # make cfg wandb-serializable
    run = maybe_init_wandb(
        enabled=bool(args.wandb),
        api_key=args.wandb_api_key,
        project=wb_project,
        name=wb_name,
        config=wb_cfg,
        mode=wb_mode,
        tags=wb_tags,
        group=wb_group,
    )

    # ---- Train loop ----
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))
    scaler = None  # BF16 on A100: no need for GradScaler

    log_every = int(cfg.kd.log_every)
    grad_clip = float(cfg.training.get("max_grad_norm", 1.0))
    acc_steps = int(cfg.kd.get("grad_accum_steps", 1))
    kd_temp = float(cfg.kd.get("distill_temp", 1.0))
    tail_mode = str(cfg.kd.get("topk_tail", "bucket"))

    it = iter(dl)
    pbar = tqdm(range(total_steps), desc="KD (top-K)", ncols=100)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        # Move to GPU in main process only
        batch = move_to_device_non_blocking(batch, device)

        # Forward
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = out.logits  # [B, L, V]

        # last S positions predict the S continuation tokens
        S = int(batch["cont_mask"].shape[1])
        d_logits_BT_V = logits[:, -S:, :]  # [B, S, V]

        # Optional per-token weights (margin + mismatch); stays on device
        weights = build_kd_weights(
            d_logits_BT_V=d_logits_BT_V,
            topk_ids_BTK=batch["topk_ids"],
            topk_logprobs_BTK=batch["topk_logprobs"],
            cont_mask_BT=batch["cont_mask"].float(),
            margin_gamma=float(cfg.kd.get("margin_gamma", 0.5)),
            margin_center=float(cfg.kd.get("margin_center", 1.0)),
            w_min=float(cfg.kd.get("w_min", 0.2)),
            mismatch_lambda=float(cfg.kd.get("mismatch_lambda", 0.3)),
        )

        loss = sparse_kd_kl(
            d_logits_BT_V=d_logits_BT_V,
            topk_ids_BTK=batch["topk_ids"],
            topk_logprobs_BTK=batch["topk_logprobs"],
            mask_BT=weights,                # incorporate weights
            distill_temp=kd_temp,
            tail_mode=tail_mode,            # "bucket" (default) or "ignore"
        ) / acc_steps

        loss.backward()

        if (step + 1) % acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            logs = {
                "train/loss": float(loss.detach().cpu()) * acc_steps,
                "train/lr": sched.get_last_lr()[0],
                "train/avg_cont_len": float(batch["cont_len"].float().mean().cpu()),
            }
            wandb_log(logs, step=step)
            pbar.set_postfix(loss=f"{logs['train/loss']:.3f}",
                             lr=f"{logs['train/lr']:.2e}")

    # ---- Save adapters + training summary ----
    outdir = out_dir / "lora"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(outdir))
    except Exception:
        torch.save(model.state_dict(), outdir / "pytorch_model.bin")

    save_json(
        {
            "finished_at": timestamp(),
            "steps": total_steps,
            "model": cfg.models.draft,
            "pregen_dir": str(ds_root),
            "out_dir": str(outdir),
        },
        out_dir / "train_summary.json",
    )
    wandb_finish()
    print(f"[DONE] KD LoRA saved at: {outdir}")


if __name__ == "__main__":
    main()
