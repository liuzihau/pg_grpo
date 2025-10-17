# src/models/load.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PeftModel

from src.common.config import (
    load_yaml_with_includes, to_attrdict, cfg_get, parse_torch_dtype
)

# ---------------------------
# KD TRAINING: base + LoRA
# ---------------------------

def load_draft_with_lora(
    *,
    base_name: str,
    tokenizer,                          # unused here but kept for API compat
    lora_cfg: Dict[str, Any],
    dtype: torch.dtype,
    device_map: Union[str, Dict[str, int]] = "auto",
    gradient_checkpointing: bool = True,
    use_cache: bool = False,
):
    """
    Load the draft base model and wrap it with a fresh LoRA adapter for training.
    Signature kept 100% compatible with scripts/train_kd.py.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Gradient checkpointing & cache flags
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = bool(use_cache)

    # Build LoRA config (sensible defaults if some keys are absent)
    lora_args = {
        "r":             int(lora_cfg.get("r", 16)),
        "lora_alpha":    int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 32))),
        "lora_dropout":  float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
        "bias":          str(lora_cfg.get("bias", "none")),
        "task_type":     "CAUSAL_LM",
        # if target_modules not specified, default to common QKV/Proj + MLP
        "target_modules": lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        # optional: rank pattern / modules_to_save etc.
        "modules_to_save": lora_cfg.get("modules_to_save", None),
        "rank_pattern":    lora_cfg.get("rank_pattern", None),
        "use_rslora":      bool(lora_cfg.get("use_rslora", False)),
        "use_dora":        bool(lora_cfg.get("use_dora", False)),
    }
    # Filter None values that Peft may not like
    lora_args = {k: v for k, v in lora_args.items() if v is not None}
    peft_cfg = LoraConfig(**lora_args)

    model = get_peft_model(model, peft_cfg)
    # Keep train mode; caller sets .train()
    return model


# ---------------------------
# EVAL HELPERS
# ---------------------------

def _resolve_adapter_dir(model_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Return (run_root, adapter_dir).
    Accept either:
      - <run_root>/lora/{adapter files}
      - <adapter_dir> directly (contains adapter_config.json)
    """
    d = Path(model_dir).expanduser().resolve()
    # Direct adapter folder?
    if (d / "adapter_config.json").is_file() or \
       (d / "adapter_model.bin").is_file() or \
       (d / "adapter_model.safetensors").is_file():
        return d.parent, d
    # Run root with lora/
    if (d / "lora" / "adapter_config.json").is_file() or \
       (d / "lora" / "adapter_model.bin").is_file() or \
       (d / "lora" / "adapter_model.safetensors").is_file():
        return d, d / "lora"
    # Fallback: any lora/ dir
    if (d / "lora").is_dir():
        return d, d / "lora"
    raise FileNotFoundError(
        f"No LoRA adapter found in {d}. Pass the run root that contains lora/, "
        "or the adapter folder itself."
    )

def _find_training_cfg_path(run_root: Path) -> Path:
    cands = [
        run_root / "cfg.lock.yaml",
        run_root / "kd_train.yaml",
        run_root / "grpo_train.yaml",
        run_root.parent / "cfg.lock.yaml",
    ]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Training config not found near {run_root}. "
        "Expected cfg.lock.yaml, kd_train.yaml, or grpo_train.yaml."
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

def _read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def _resolve_teacher_name(train_cfg: Any, run_root: Path) -> Optional[str]:
    # 1) explicit in training cfg
    t = cfg_get(train_cfg, "models.target", None)
    if t:
        return t

    # 2) from kd.pregen_dir in training cfg
    pregen_dir = cfg_get(train_cfg, "kd.pregen_dir", None)

    # 3) or from train_summary.json written by train_kd
    if pregen_dir is None:
        summary = _read_json(run_root / "train_summary.json") or {}
        pregen_dir = summary.get("pregen_dir")

    if pregen_dir:
        man = Path(pregen_dir).expanduser().resolve() / "manifest.json"
        m = _read_json(man) or {}
        # data_gen writes {"model": <teacher_name>, ...}
        teacher = m.get("model") or m.get("teacher")
        if teacher:
            return teacher

    return None

def mark_only_lora_trainable(model) -> tuple[int, int]:
    """
    Freeze all non-LoRA weights; unfreeze only lora_* params.
    Returns (num_trainable, num_total).
    """
    num_trainable = 0
    num_total = 0
    for n, p in model.named_parameters():
        num_total += p.numel()
        is_lora = ("lora_" in n) or ("loraA" in n) or ("lora_B" in n) or ("lora_A" in n)
        p.requires_grad_(is_lora)
        if is_lora:
            num_trainable += p.numel()
    return num_trainable, num_total


def load_models_for_eval_from_model_dir(
    model_dir: Union[str, Path],
    dtype_name: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[Any, Any, Any, Any, str, Path]:
    """
    Returns: (tokenizer, draft_lora_model, teacher_model, train_cfg, stage_kind, cfg_path)
    - tokenizer: left-padded tokenizer inferred from training cfg
    - draft_lora_model: base draft + attached LoRA from model_dir (or model_dir/lora)
    - teacher_model: target model (from training cfg or KD manifest)
    - train_cfg: AttrDict of saved training cfg
    - stage_kind: "kd" or "grpo"
    - cfg_path: path to the cfg used
    """
    run_root, adapter_dir = _resolve_adapter_dir(model_dir)
    cfg_path = _find_training_cfg_path(run_root)

    raw = load_yaml_with_includes(cfg_path)
    train_cfg = to_attrdict(raw)
    stage_kind = _infer_stage_kind(train_cfg)

    draft_name = cfg_get(train_cfg, "models.draft", None)
    if draft_name is None:
        raise ValueError(f"models.draft missing in training cfg: {cfg_path}")

    target_name = _resolve_teacher_name(train_cfg, run_root)
    if target_name is None:
        raise ValueError(
            "Could not resolve teacher model name.\n"
            f"- Looked in {cfg_path} (models.target) and in KD corpus manifest via kd.pregen_dir/train_summary.json.\n"
            "Fix by either:\n"
            "  a) adding models.target to your training cfg, or\n"
            "  b) ensuring train_summary.json contains 'pregen_dir' with manifest.json {'model': ...}."
        )

    dtype = parse_torch_dtype(dtype_name or cfg_get(train_cfg, "training.dtype", "bf16"))

    # Tokenizer
    tok = load_tokenizer_from_training_cfg(train_cfg)

    # Draft + LoRA
    base = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=dtype, device_map=device_map
    )
    try:
        draft = PeftModel.from_pretrained(base, adapter_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to attach LoRA from {adapter_dir}. "
            f"Expected adapter_config.json + weights. Error: {e}"
        )
    if hasattr(draft, "config"):
        draft.config.use_cache = False
    draft.eval()

    # Teacher
    teacher = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map=device_map
    )
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    teacher.eval()

    return tok, draft, teacher, train_cfg, stage_kind, cfg_path


def load_base_draft_from_training_cfg(
    train_cfg: Any,
    dtype_name: Optional[str] = None,
    device_map: str = "auto",
):
    """
    Load ONLY the base 'models.draft' (no LoRA) using the dtype in training cfg
    (or an override). Returns a HF CausalLM in eval mode with use_cache=False.
    """
    draft_name = cfg_get(train_cfg, "models.draft", None)
    if draft_name is None:
        raise ValueError("models.draft missing in training cfg; cannot load base draft.")

    dtype = parse_torch_dtype(dtype_name or cfg_get(train_cfg, "training.dtype", "bf16"))
    model = AutoModelForCausalLM.from_pretrained(
        draft_name, torch_dtype=dtype, device_map=device_map
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model


# --- LoRA / parameter reporting helpers --------------------------------------
def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def _collect_lora_parents_from_state_dict(model):
    """
    Collect parent module names that have LoRA params.
    Works for typical PEFT keys: *.lora_A.*, *.lora_B.*, *.lora_embedding_*.
    """
    parents = set()
    for k in model.state_dict().keys():
        if "lora_A" in k or "lora_B" in k or "lora_embedding_" in k or ".lora_" in k:
            # normalize: strip the last 2 segments for lora_[AB].weight etc.
            parts = k.split(".")
            # find the 'lora_*' segment and drop everything from there
            try:
                idx = next(i for i, p in enumerate(parts) if p.startswith("lora"))
                parent = ".".join(parts[:idx])
            except StopIteration:
                # fallback: drop the last segment
                parent = ".".join(parts[:-1])
            if parent:
                parents.add(parent)
    return sorted(parents)

def print_model_layer_report(model, title="Model", limit=40, only_lora=True):
    """
    Print a compact report:
      - trainable vs total params
      - LoRA-injected parent modules (first/last N=limit//2 to keep it readable)
    If only_lora=False, it also prints all leaf module names (capped by limit).
    """
    print(f"\n[report] {title}")
    trainable, total = _count_params(model)
    print(f"  params: trainable={trainable:,}  total={total:,}")

    # Prefer a PEFT-aware message if available
    try:
        from peft import PeftModel  # type: ignore
        is_peft = isinstance(model, PeftModel)
    except Exception:
        is_peft = False

    lora_parents = _collect_lora_parents_from_state_dict(model)
    if lora_parents:
        print(f"  LoRA-injected modules: {len(lora_parents)}")
        if len(lora_parents) <= limit:
            for name in lora_parents:
                print(f"    - {name}")
        else:
            head = lora_parents[: limit // 2]
            tail = lora_parents[-(limit // 2) :]
            for name in head:
                print(f"    - {name}")
            print(f"    ... ({len(lora_parents) - len(head) - len(tail)} more) ...")
            for name in tail:
                print(f"    - {name}")
    else:
        if is_peft:
            print("  [note] PEFT model detected but no LoRA keys in state_dict (unusual).")
        else:
            print("  No LoRA adapters detected.")

    if not only_lora:
        # dump a capped list of all leaf modules
        leaf_names = []
        for name, module in model.named_modules():
            # leaf: no submodules
            if not any(True for _ in module.children()):
                leaf_names.append(name)
        leaf_names = sorted(set(leaf_names))
        print(f"  Leaf modules: {len(leaf_names)}")
        if len(leaf_names) <= limit:
            for name in leaf_names:
                print(f"    * {name}")
        else:
            head = leaf_names[: limit // 2]
            tail = leaf_names[-(limit // 2) :]
            for name in head:
                print(f"    * {name}")
            print(f"    ... ({len(leaf_names) - len(head) - len(tail)} more) ...")
            for name in tail:
                print(f"    * {name}")
    print("")