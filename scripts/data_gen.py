# data_gen.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import random
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides,
    parse_overrides, set_seed, save_cfg_lock, cfg_get
)
from src.common.io import makedirs, save_json, save_pt, list_shards, timestamp
from src.data.prompts import load_prompts_for_split, make_manual_splits, truncate_prompt_by_tokens
from src.data.vllm_extract import pairs_from_logprob_step


def _clamp_k_for_vllm(k: int) -> int:
    # vLLM 0.11 allows up to 20 top-k logprobs
    return int(max(0, min(20, k)))


def _build_llm(cfg):
    kwargs = dict(
        model=cfg.models.target,
        tensor_parallel_size=int(cfg.vllm.tensor_parallel_size),
        dtype=cfg.vllm.dtype,
        gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
    )
    # vLLM 0.11: max_model_len is valid; guard in case of older builds
    if getattr(cfg.vllm, "max_model_len", None) is not None:
        kwargs["max_model_len"] = int(cfg.vllm.max_model_len)
    return LLM(**kwargs)


def _build_sampling_params(cfg, S: int, K: int, seed: int) -> SamplingParams:
    return SamplingParams(
        max_tokens=S,
        temperature=float(cfg.gen.temperature),
        top_p=float(cfg.gen.top_p),
        top_k=int(cfg.gen.top_k),
        n=1,
        seed=seed,
        logprobs=K,             # top-K logprobs for GENERATED tokens
        prompt_logprobs=None,   # do not request prompt logprobs
    )


def _prepare_prompts(tokenizer, prompts: List[str], max_input_len: int, S: int) -> List[str]:
    """Ensure decoder prompt length <= max_model_len - S via left truncation."""
    out: List[str] = []
    budget = max(1, max_input_len - 1)  # conservative cushion
    for p in prompts:
        out.append(truncate_prompt_by_tokens(tokenizer, p, budget))
    return out


def _gen_one_split(
    *,
    split_name: str,
    prompts: List[str],
    tokenizer,
    llm,
    cfg,
    out_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    S = int(cfg.gen.S)
    K_req = int(cfg.gen.K)
    K = _clamp_k_for_vllm(K_req)
    if K != K_req:
        print(f"[warn] gen.K={K_req} > 20, clamped to {K} for vLLM 0.11")

    # Truncate prompts to fit (max_model_len == data.max_input_len here)
    prepped = _prepare_prompts(
        tokenizer, prompts, max_input_len=int(cfg.data.max_input_len), S=S
    )

    bs = int(cfg.gen.batch_size)
    shard_size = int(cfg.gen.shard_size)

    sp = _build_sampling_params(cfg, S=S, K=K, seed=seed)
    shard_idx = 0
    buf: List[Dict[str, Any]] = []
    written = 0

    pbar = tqdm(range(0, len(prepped), bs), desc=f"[{split_name}] vLLM gen", ncols=100)
    for off in pbar:
        batch_prompts = prepped[off : off + bs]
        outs = llm.generate(batch_prompts, sp)

        for out in outs:
            prompt_text = out.prompt
            gen = out.outputs[0]
            token_ids = gen.token_ids               # list[int], length S' <= S
            token_logprobs = gen.logprobs           # list[*], len S', see pairs_from_logprob_step()

            # Collect top-K per step
            step_topk_ids: List[List[int]] = []
            step_topk_lps: List[List[float]] = []
            for step in range(len(token_ids)):
                pairs = pairs_from_logprob_step(token_logprobs[step], tokenizer)
                if not pairs:
                    continue
                # already sorted in many backends, sort anyway
                pairs.sort(key=lambda x: x[1], reverse=True)
                ids = [int(p[0]) for p in pairs[:K]]
                lps = [float(p[1]) for p in pairs[:K]]
                step_topk_ids.append(ids)
                step_topk_lps.append(lps)

            cont_len = len(step_topk_ids)
            if cont_len == 0:
                continue

            # pad to S
            pad_id = tokenizer.eos_token_id
            cont_ids = (token_ids + [pad_id] * (S - len(token_ids)))[:S]

            ids_arr = np.zeros((S, K), dtype="int32")
            lps_arr = np.full((S, K), -1e30, dtype="float32")
            for t in range(min(S, cont_len)):
                ids_t = step_topk_ids[t]
                lps_t = step_topk_lps[t]
                n = min(K, len(ids_t))
                ids_arr[t, :n] = ids_t[:n]
                lps_arr[t, :n] = lps_t[:n]

            enc_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
            rec = {
                "prompt_text": prompt_text,
                "prompt_ids": enc_ids,
                "cont_ids": cont_ids,
                "cont_len": min(S, cont_len),
                "topk_ids": torch.tensor(ids_arr, dtype=torch.int32),
                "topk_logprobs": torch.tensor(lps_arr, dtype=torch.float32),
            }
            buf.append(rec)
            written += 1

            if len(buf) >= shard_size:
                shard_path = out_dir / split_name / f"shard_{shard_idx:05d}.pt"
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                save_pt(buf, shard_path)
                buf = []
                shard_idx += 1

    if buf:
        shard_path = out_dir / split_name / f"shard_{shard_idx:05d}.pt"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        save_pt(buf, shard_path)

    manifest = {
        "split": split_name,
        "S": S,
        "K": K,
        "num_records": written,
        "shard_size": shard_size,
        "shards": [{"path": Path(p).name, "size": None} for p in list_shards(out_dir / split_name)],
        "seed": seed,
        "temperature": float(cfg.gen.temperature),
        "top_p": float(cfg.gen.top_p),
        "top_k": int(cfg.gen.top_k),
    }
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="dot.notation=VALUE overrides")
    args = ap.parse_args()

    # Load + overrides
    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(apply_overrides(raw, parse_overrides(args.override)))
    set_seed(int(cfg.training.seed))

    out_dir = Path(cfg.gen.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = save_cfg_lock(raw, out_dir, filename="cfg.lock.yaml")
    print(f"[cfg] saved resolved config -> {lock_path}")

    # Tokenizer (left-padding for decoder-only)
    tok_name = cfg_get(cfg, "models.tokenizer", cfg.models.target)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # vLLM teacher
    llm = _build_llm(cfg)

    # Build splits (HF or manual)
    # 1) Try to read target split directly; if empty, fall back to manual splits
    base_split = cfg.data.split
    train_prompts = load_prompts_for_split(tokenizer, cfg, split=base_split)
    if not train_prompts:
        print("[info] HF split empty or missing -> using manual_splits over the whole dataset")
        train_prompts, val_prompts = make_manual_splits(tokenizer, cfg)
    else:
        # If validation split exists, try to load; otherwise manual sample from train pool
        val_prompts = load_prompts_for_split(tokenizer, cfg, split="validation")
        if not val_prompts and cfg_get(cfg, "data.manual_splits", None):
            train_prompts, val_prompts = make_manual_splits(tokenizer, cfg, seed=int(cfg.training.seed))

    # Optional dedupe (exact string match)
    if bool(cfg_get(cfg, "data.dedupe", True)):
        def _dedupe(lst: List[str]) -> List[str]:
            seen, out = set(), []
            for s in lst:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out
        train_prompts = _dedupe(train_prompts)
        if val_prompts:
            val_prompts = _dedupe(val_prompts)

    save_json({"train": len(train_prompts), "validation": len(val_prompts or [])},
              out_dir / "split_sizes.json")

    # Generate each split
    manifests = {"created_at": timestamp(), "model": cfg.models.target, "root_out": str(out_dir)}
    manifests["train"] = _gen_one_split(
        split_name="train", prompts=train_prompts, tokenizer=tokenizer, llm=llm, cfg=cfg, out_dir=out_dir, seed=int(cfg.training.seed)
    )
    if val_prompts:
        manifests["validation"] = _gen_one_split(
            split_name="validation", prompts=val_prompts, tokenizer=tokenizer, llm=llm, cfg=cfg, out_dir=out_dir, seed=int(cfg.training.seed) + 1
        )

    save_json(manifests, out_dir / "manifest.json")
    print(f"[DONE] Wrote KD corpus into: {out_dir}")


if __name__ == "__main__":
    main()
