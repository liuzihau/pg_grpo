# src/common/hardware.py
from __future__ import annotations
import os
from typing import Dict, Any, Tuple

import torch

def gpu_info() -> Tuple[int, list[int]]:
    """Return (n_gpus, total_mem_bytes_per_gpu)."""
    if not torch.cuda.is_available():
        return 0, []
    n = torch.cuda.device_count()
    mems = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mems.append(props.total_memory)
    return n, mems

def is_big_gpu(mems: list[int]) -> bool:
    """Heuristic: treat >=75GB as 'big' (A100-80GB or similar)."""
    return any(m >= 75 * 1024**3 for m in mems)

def autoscale_vllm(cfg) -> Dict[str, Any]:
    """
    Produce vLLM overrides that are safe for the current machine.
    - On A100-80GB: allow compile/cudagraphs, higher utilization, larger max_model_len.
    - On 24GB 3090: force eager mode, lower utilization, fp16 KV cache, smaller max_model_len.
    """
    n, mems = gpu_info()
    big = is_big_gpu(mems)
    S = int(getattr(getattr(cfg, "gen", {}), "S", 64))
    req_max_in = int(getattr(getattr(cfg, "data", {}), "max_input_len", 2048))

    # Leave some safety headroom for KV/graphs/etc.
    safety = 192
    # Cap max_model_len to something safer on consumer cards.
    if big:
        max_model_len = req_max_in + S + safety
    else:
        # 24GB: keep contexts modest
        max_model_len = min(req_max_in + S + safety, 3328)

    overrides = {
        "tensor_parallel_size": n if (n > 1) else 1,
        "gpu_memory_utilization": 0.90 if big else 0.80,
        "max_model_len": max_model_len,
        "kv_cache_dtype": "bf16" if big else "fp16",
        "enforce_eager": False if big else True,
        # a hint you can use for batching upstream
        "batch_size_hint": 32 if big else 8,
    }
    return overrides

def apply_vllm_overrides(vllm_cfg, overrides: Dict[str, Any]) -> None:
    """Only set fields that aren't already set in config."""
    for k, v in overrides.items():
        if k == "batch_size_hint":
            continue
        if getattr(vllm_cfg, k, None) is None:
            setattr(vllm_cfg, k, v)

def hf_max_memory(percent: float = 0.90) -> Dict[int, int] | None:
    """
    Build a `max_memory` map for HF `from_pretrained(..., device_map="auto", max_memory=...)`.
    """
    n, mems = gpu_info()
    if n == 0:
        return None
    return {i: int(mems[i] * percent) for i in range(n)}

def autoscale_batch_size(base_on_big: int, base_on_small: int) -> int:
    """Pick a batch size based on hardware."""
    n, mems = gpu_info()
    return base_on_big if is_big_gpu(mems) else base_on_small
