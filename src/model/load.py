# src/model/load.py
from __future__ import annotations
from typing import Any
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from .lora import LoraCfg

def load_draft_with_lora(
    base_name: str,
    tokenizer,
    lora_cfg: Any,
    dtype: torch.dtype,
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
    use_cache: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    # Ensure pad id set on config for left padding
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = bool(use_cache)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable(use_reentrant=False)

    # LoRA
    lc = LoraCfg(
        r=int(lora_cfg.get("r", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [
            # Common Qwen/Llama linear names
            "q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"
        ])),
        bias=str(lora_cfg.get("bias", "none")),
    )
    peft_conf = LoraConfig(
        task_type="CAUSAL_LM",
        r=lc.r,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=lc.target_modules,
        bias=lc.bias,
    )
    model = get_peft_model(model, peft_conf)
    return model
