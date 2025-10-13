
from __future__ import annotations
from typing import List
from peft import LoraConfig, get_peft_model

def attach_lora(model, target_modules: List[str], r: int = 16, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model
