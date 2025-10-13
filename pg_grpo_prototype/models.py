
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_target(model_name: str, dtype: str = "bf16", device: str = "cuda"):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype_map.get(dtype, torch.bfloat16), device_map=device
    )
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def load_draft(model_name: str, dtype: str = "bf16", device: str = "cuda"):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype_map.get(dtype, torch.bfloat16), device_map=device
    )
    return model
