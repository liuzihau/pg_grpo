# scripts/train_grpo_trl.py
from __future__ import annotations
import argparse
from dataclasses import asdict
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from trl import GRPOConfig, GRPOTrainer

from src.grpo.reward_sparse_spec import SpecDivergenceReward, SpecRewardConfig


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    import random
    random.seed(seed); torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wandb_api_key", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    # --- tokenizer ---
    tok_name = cfg["models"].get("tokenizer", cfg["draft"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # --- draft policy (LoRA/Unsloth can be plugged; here plain HF for clarity) ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["draft"]["name"],
        torch_dtype=getattr(torch, cfg["training"]["dtype"]),
        device_map="auto",
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # --- dataset of prompts ---
    # Replace with your HFChatPromptsDataset or local prompts loader
    ds = load_dataset(cfg["data"]["hf_name"], split=cfg["data"]["split"])
    field = cfg["data"].get("messages_field", "messages")
    prompts: List[str] = []
    for rec in ds:
        msgs = rec.get(field, rec.get("conversations", []))
        if isinstance(msgs, list) and len(msgs) > 0:
            try:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                # fallback
                parts = []
                for m in msgs:
                    role = (m.get("role") or m.get("from") or "user").capitalize()
                    parts.append(f"{role}: {m.get('content') or m.get('value','')}")
                parts.append("Assistant:")
                prompt = "\n".join(parts)
            prompts.append(prompt)
    train_ds = {"prompt": prompts}  # TRL accepts simple dict datasets too

    # --- reward fn (vLLM-backed) ---
    rcfg = SpecRewardConfig(
        topk_for_ce=cfg["reward"].get("topk_for_ce", 64),
        divergence=cfg["reward"].get("divergence", "kl"),
        alpha=float(cfg["reward"].get("alpha", 0.5)),
        tail_mode=cfg["reward"].get("tail_mode", "bucket"),
        distill_temp=float(cfg["reward"].get("distill_temp", 1.0)),
        include_first_reject=bool(cfg["reward"].get("include_first_reject", True)),
        entropy_bonus=float(cfg["reward"].get("entropy_bonus", 0.0)),
        anchor_kl_beta=float(cfg["reward"].get("anchor_kl_beta", 0.0)),
        reward_clip=bool(cfg["reward"].get("clip_reward", True)),
        reward_clip_range=tuple(cfg["reward"].get("clip_range", [-20.0, 0.0])),

        teacher_model=cfg["models"]["target"],
        tensor_parallel_size=int(cfg["vllm"]["tensor_parallel_size"]),
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
        teacher_temperature=float(cfg["grpo"].get("teacher_temperature", 0.0)),
        teacher_top_p=float(cfg["grpo"].get("teacher_top_p", 1.0)),
        teacher_top_k=int(cfg["grpo"].get("teacher_top_k", 0)),
    )
    reward_fn = SpecDivergenceReward(tokenizer, rcfg)

    # --- TRL GRPO config ---
    gcfg = GRPOConfig(
        model_name=cfg["draft"]["name"],
        learning_rate=float(cfg["training"]["lr"]),
        per_device_train_batch_size=int(cfg["grpo"].get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg["grpo"].get("grad_accum", 1)),
        max_prompt_length=int(cfg["data"]["max_input_len"]),
        max_completion_length=int(cfg["grpo"]["S"]),
        group_size=int(cfg["grpo"]["group_size"]),
        kl_coef=0.0,                    # we do our own anchor/divergence in reward
        log_with="wandb",
        tracker_project_name=cfg["logging"]["project"],
        output_dir=cfg.get("training", {}).get("output_dir", "outputs/grpo"),
        bf16=(cfg["training"]["dtype"] == "bf16"),
        seed=int(cfg["training"]["seed"]),
        save_steps=int(cfg["grpo"].get("save_every", 2000)),
    )

    # --- WandB ---
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["name"], config=cfg)

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        reward_function=reward_fn,  # our callable
        tokenizer=tokenizer,
        args=gcfg,
        train_dataset=train_ds,     # {"prompt": [...]}
    )
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
