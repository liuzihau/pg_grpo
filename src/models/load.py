# src/models/load.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from src.common.config import load_yaml_with_includes, to_attrdict, cfg_get, parse_torch_dtype

# ---------- existing helpers (keep yours) ----------
def _find_training_cfg_path(model_dir: Union[str, Path]) -> Path:
    d = Path(model_dir).expanduser().resolve()
    cands = [
        d / "cfg.lock.yaml",
        d / "kd_train.yaml",
        d / "grpo_train.yaml",
        d.parent / "cfg.lock.yaml",
    ]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find a training config in or near: {d}\n"
        "Expected one of: cfg.lock.yaml, kd_train.yaml, grpo_train.yaml"
    )

def _infer_stage_kind(cfg: Any) -> str:
    if getattr(cfg, "kd", None) is not None:
        return "kd"
    if getattr(cfg, "grpo", None) is not None:
        return "grpo"
    return "kd"

def load_tokenizer_from_training_cfg(train_cfg: Any):
    tok_name = cfg_get(train_cfg, "models.tokenizer", None) \
            or cfg_get(train_cfg, "models.draft", None) \
            or cfg_get(train_cfg, "models.target", None)
    if tok_name is None:
        raise ValueError("Cannot resolve tokenizer name from training cfg (models.tokenizer/draft/target).")
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

# ---------- NEW: draft loader with fresh LoRA for KD training ----------
def _default_lora_targets(model: nn.Module) -> list[str]:
    """
    Reasonable defaults for LLaMA/Mistral/Qwen families.
    If you passed explicit target_modules in cfg.lora.target_modules we will use that instead.
    """
    common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Fall back to leaf linear names if none of those exists (other archs).
    exists = set()
    for n, m in model.named_modules():
        leaf = n.split(".")[-1]
        if isinstance(m, nn.Linear):
            exists.add(leaf)
    if any(t in exists for t in common):
        return common
    # Fallback: all linear leaves except lm_head
    return sorted(list({n.split(".")[-1] for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and "lm_head" not in n}))

def load_draft_with_lora(
    *,
    base_name: str,
    tokenizer,                       # not used here but kept for signature parity / future use
    lora_cfg: Any,                   # expects fields: r, alpha, dropout, bias, target_modules?
    dtype: torch.dtype,
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
    use_cache: bool = False,
):
    """
    Load the draft base model and attach a *fresh* LoRA adapter for KD training.
    Returns a PEFT-wrapped model ready for .train().
    """
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if hasattr(base, "config"):
        base.config.use_cache = use_cache
    if gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass

    # Resolve target modules
    explicit_targets = None
    if hasattr(lora_cfg, "target_modules"):
        explicit_targets = list(lora_cfg.target_modules) if lora_cfg.target_modules is not None else None
    targets = explicit_targets or _default_lora_targets(base)

    # Build LoRA config
    r = int(getattr(lora_cfg, "r", 16))
    alpha = int(getattr(lora_cfg, "alpha", 32))
    dropout = float(getattr(lora_cfg, "dropout", 0.05))
    bias = getattr(lora_cfg, "bias", "none")

    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias=bias,
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

    model = get_peft_model(base, peft_cfg)

    # Small quality-of-life: ensure lm_head has grads (PEFT keeps base frozen except adapters).
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        for p in model.lm_head.parameters():
            p.requires_grad_(True)

    model.train()
    return model

# ---------- existing eval loader (keep yours, unchanged) ----------
def load_models_for_eval_from_model_dir(
    model_dir: Union[str, Path],
    dtype_name: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[Any, Any, Any, Any, str, Path]:
    model_dir = Path(model_dir).expanduser().resolve()
    cfg_path = _find_training_cfg_path(model_dir)
    raw = load_yaml_with_includes(cfg_path)
    train_cfg = to_attrdict(raw)
    stage_kind = _infer_stage_kind(train_cfg)

    draft_name  = cfg_get(train_cfg, "models.draft", None)
    target_name = cfg_get(train_cfg, "models.target", None)
    if draft_name is None or target_name is None:
        raise ValueError(f"models.draft/models.target missing in training cfg: {cfg_path}")

    dtype = parse_torch_dtype(dtype_name or cfg_get(train_cfg, "training.dtype", "bf16"))
    tok = load_tokenizer_from_training_cfg(train_cfg)

    base = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=dtype, device_map=device_map
    )
    try:
        draft = PeftModel.from_pretrained(base, model_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to attach LoRA from {model_dir}. "
            f"Make sure this folder contains adapter_config.json + adapter weights. Error: {e}"
        )
    if hasattr(draft, "config"):
        draft.config.use_cache = False
    draft.eval()

    teacher = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map=device_map
    )
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    teacher.eval()

    return tok, draft, teacher, train_cfg, stage_kind, cfg_path
