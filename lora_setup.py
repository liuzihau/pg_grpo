from __future__ import annotations
from typing import Iterable, List
from peft import LoraConfig, get_peft_model, TaskType

def _unique_leaf_names(model) -> List[str]:
    leaves = set()
    for name, _ in model.named_modules():
        if not name: continue
        leaves.add(name.rsplit(".", 1)[-1])
    return sorted(leaves)

def _filter_present_targets(model, requested: Iterable[str]) -> List[str]:
    leaves = set(_unique_leaf_names(model))
    return [t for t in requested if t in leaves]

def attach_lora(model, cfg):
    lora_cfg = getattr(cfg, "lora", None) or (cfg.get("lora") if isinstance(cfg, dict) else None)
    if lora_cfg is None:
        raise ValueError("No 'lora' section in cfg")

    target_modules = list(lora_cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]))
    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.05))
    bias = lora_cfg.get("bias", "none")
    task_type = lora_cfg.get("task_type", "CAUSAL_LM")

    present = _filter_present_targets(model, target_modules)
    if not present:
        available = _unique_leaf_names(model)
        raise ValueError(f"None of requested LoRA target_modules found. requested={target_modules}, available-sample={available[:50]}")

    peft_config = LoraConfig(
        task_type=getattr(TaskType, task_type),
        target_modules=present,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
    )
    model = get_peft_model(model, peft_config)
    try: model.print_trainable_parameters()
    except Exception: pass
    return model

def enable_lm_head_training(model):
    """Make the LM head trainable (helps calibration / alignment)."""
    lm = getattr(model, "lm_head", None)
    if lm is None:
        return
    for p in lm.parameters():
        p.requires_grad_(True)
