# src/models/load.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.common.config import load_yaml_with_includes, to_attrdict, cfg_get, parse_torch_dtype

def _find_training_cfg_path(model_dir: Union[str, Path]) -> Path:
    """
    Look for a config snapshot *inside* the model_dir or its parent.
    Priority:
      1) model_dir/cfg.lock.yaml
      2) model_dir/kd_train.yaml or model_dir/grpo_train.yaml
      3) parent/cfg.lock.yaml
    """
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
    # If kd.* block exists we call it KD; if grpo.* exists, GRPO; else KD as default
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

def load_models_for_eval_from_model_dir(
    model_dir: Union[str, Path],
    dtype_name: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[Any, Any, Any, Any, str, Path]:
    """
    Returns: (tokenizer, draft_lora_model, teacher_model, train_cfg, stage_kind, cfg_path)
      - tokenizer: HF tokenizer (left-pad)
      - draft_lora_model: base draft + LoRA adapter loaded from model_dir
      - teacher_model: target teacher in eval mode
      - train_cfg: AttrDict of the saved training cfg
      - stage_kind: "kd" or "grpo"
      - cfg_path: resolved config path used
    """
    model_dir = Path(model_dir).expanduser().resolve()
    cfg_path = _find_training_cfg_path(model_dir)
    raw = load_yaml_with_includes(cfg_path)
    train_cfg = to_attrdict(raw)
    stage_kind = _infer_stage_kind(train_cfg)

    # Resolve names from training cfg (no need to pass via eval cfg)
    draft_name  = cfg_get(train_cfg, "models.draft", None)
    target_name = cfg_get(train_cfg, "models.target", None)
    if draft_name is None or target_name is None:
        raise ValueError(f"models.draft/models.target missing in training cfg: {cfg_path}")

    # dtype
    dtype = parse_torch_dtype(dtype_name or cfg_get(train_cfg, "training.dtype", "bf16"))

    # Tokenizer
    tok = load_tokenizer_from_training_cfg(train_cfg)

    # Load draft base + attach LoRA from model_dir
    base = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=dtype, device_map=device_map
    )
    # PeftModel.from_pretrained attaches adapters saved via save_pretrained(adapter_dir)
    try:
        draft = PeftModel.from_pretrained(base, model_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to attach LoRA from {model_dir}. "
            f"Make sure this folder contains adapter_config.json + adapter weights. Error: {e}"
        )
    if hasattr(draft, "config"):
        draft.config.use_cache = False
    draft.eval()  # eval graph for speed; we won’t backprop during eval

    # Load teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map=device_map
    )
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    teacher.eval()

    return tok, draft, teacher, train_cfg, stage_kind, cfg_path
