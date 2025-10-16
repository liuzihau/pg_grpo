from __future__ import annotations
import os, argparse, math,yaml
from typing import Any, Dict
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from src.data.pregen_kd_ds import PreGeneratedTopKDataset, collate_topk
from src.kd.sparse_kd_loss import sparse_kd_kl

def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _deep_update(a[k], v)
        else:
            a[k] = v
    return a

def load_yaml_with_includes(path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for rel in includes:
        inc_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        sub = load_yaml_with_includes(inc_path)
        _deep_update(merged, sub)
    _deep_update(merged, cfg)
    return merged

class Attr(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def to_attrdict(d: Dict[str, Any]) -> Any:
    a = Attr()
    for k, v in d.items():
        a[k] = to_attrdict(v) if isinstance(v, dict) else v
    return a

def cfg_get(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur

def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_attr(d): return Attr(d)

def set_seed(seed: int):
    import random
    random.seed(seed); torch.manual_seed(seed)

def build_optimizer(model, cfg):
    lr = float(cfg_get(cfg, "kd.lr", 5e-4))
    wd = float(cfg_get(cfg, "kd.weight_decay", 0.0))
    betas = tuple(cfg.kd.get("betas", [0.9, 0.95]))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

def make_sched(optim, total_steps, warmup_ratio=0.05):
    warm = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(s):
        if s < warm: return s / warm
        prog = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wandb_api_key", required=True)
    args = ap.parse_args()

    # new:
    raw = load_yaml_with_includes(args.config)
    cfg  = to_attrdict(raw)
    set_seed(int(cfg.training.seed))
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))

    # wandb
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=cfg.logging.project, name=cfg.logging.name, config=cfg)

    # tokenizer
    tok_name = cfg.models.get("tokenizer", cfg.draft.name)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # draft model (Unsloth optional; fallback to PEFT)
    try:
        from unsloth import FastLanguageModel
        quant_4bit = bool(cfg.lora.get("qlora", False))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.draft.name,
            max_seq_length=cfg.data.max_input_len,
            dtype=cfg.training.dtype,
            load_in_4bit=quant_4bit,
            device_map="auto",
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(cfg.lora.r),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            target_modules=list(cfg.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.draft.name,
            torch_dtype=getattr(torch, cfg.training.dtype),
            device_map="auto",
        )
        peft = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(cfg.lora.r),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
            target_modules=list(cfg.lora.target_modules),
            bias="none",
        )
        model = get_peft_model(model, peft)

    # memory knobs
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable(use_reentrant=False)
    if hasattr(model, "config"):
        model.config.use_cache = False
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    # data
    ds = PreGeneratedTopKDataset(cfg_get(cfg, "kd.pregen_dir", "data/kd_corpus/qwen8b_S64_topk64"))
    pad_id = tokenizer.pad_token_id

    def _collate(batch):
        return collate_topk(batch, pad_id=pad_id, device=device, draft_dtype=getattr(torch, cfg.training.dtype))

    dl = DataLoader(ds, batch_size=int(cfg.kd.batch_size), shuffle=True,
                    num_workers=2, pin_memory=True, collate_fn=_collate)

    optim = build_optimizer(model, cfg)
    sched = make_sched(optim, int(cfg.kd.total_steps), cfg.kd.get("warmup_ratio", 0.05))

    total_steps = int(cfg_get(cfg, "kd.total_steps", 2000))
    log_every = int(cfg.kd.log_every)

    pbar = tqdm(range(total_steps), desc="KD (top-K)", ncols=100)
    it = iter(dl)
    for step in pbar:
        try: batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        logits = out.logits  # [B,L,V]
        # last S positions predict the S continuation tokens
        S = int(batch["cont_mask"].shape[1])
        d_logits_BT_V = logits[:, -S:, :]  # [B,S,V]

        # weights
        with torch.no_grad():
            # teacher margin from top-K: top1 - top2 (per token)
            topk = batch["topk_logprobs"]  # [B,S,K] logprobs
            top2,_ = torch.topk(topk, k=2, dim=-1)
            margin = top2[..., 0] - top2[..., 1]  # [B,S]
            mg = cfg.kd.get("margin_gamma", 0.5)
            mc = cfg.kd.get("margin_center", 1.0)
            wmin = cfg.kd.get("w_min", 0.2)
            m = torch.sigmoid(mg * (margin - mc))
            margin_w = wmin + (1.0 - wmin) * m

            # mismatch vs draft argmax over full vocab
            d_argmax = d_logits_BT_V.argmax(dim=-1)  # [B,S]
            t_argmax = torch.gather(batch["topk_ids"], -1, torch.zeros_like(batch["topk_ids"][..., :1]))
            t_argmax = t_argmax.squeeze(-1)  # top-1 id per step
            mismatch = (d_argmax != t_argmax).float()

            weights = (margin_w * (1.0 + cfg.kd.get("mismatch_lambda", 0.3) * mismatch)) * batch["cont_mask"]

        loss = sparse_kd_kl(
            d_logits_BT_V=d_logits_BT_V,
            topk_ids_BTK=batch["topk_ids"],
            topk_logprobs_BTK=batch["topk_logprobs"],
            mask_BT=weights,  # incorporate weights in mask (scaled later)
            tail_mode=str(cfg.kd.get("topk_tail", "bucket")),
            distill_temp=float(cfg.kd.get("distill_temp", 1.0)),
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.max_grad_norm))
        optim.step(); sched.step()

        if step % log_every == 0:
            wandb.log({
                "train/kd_total_loss": float(loss.detach().cpu()),
                "train/lr": sched.get_last_lr()[0],
                "train/avg_cont_len": float(batch["cont_len"].float().mean().cpu()),
            }, step=step)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    # save LoRA/adapters
    outdir = os.path.join("outputs", "kd_lora")
    os.makedirs(outdir, exist_ok=True)
    try:
        model.save_pretrained(outdir)
    except Exception:
        torch.save(model.state_dict(), os.path.join(outdir, "pytorch_model.bin"))
    wandb.finish()

if __name__ == "__main__":
    main()
