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


def _prepare_prompts(tokenizer, prompts: List[str],
                     max_input_len: int, S: int,
                     model_max_len: int | None = None,
                     cushion: int = 8) -> List[str]:
    """
    Left-truncate so that: prompt_len <= min(max_input_len, model_max_len) - S - cushion
    """
    cap = int(max_input_len)
    if model_max_len is not None:
        cap = min(cap, int(model_max_len))
    budget = max(1, cap - int(S) - int(cushion))

    out: List[str] = []
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

    prepped = _prepare_prompts(
        tokenizer,
        prompts,
        max_input_len=int(cfg.data.max_input_len),
        S=S,
        model_max_len=int(getattr(cfg.vllm, "max_model_len", 10**9)),
        cushion=8,
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
    # This handles both: (a) true HF splits or (b) "manual" re-split of a base split.
    kd_train_prompts, grpo_train_prompts, val_prompts = make_manual_splits(
        tokenizer, cfg, seed=int(cfg.training.seed)
    )

    # Optional dedupe (exact string match)
    if bool(cfg_get(cfg, "data.dedupe", True)):
        def _dedupe(lst: List[str]) -> List[str]:
            seen, out = set(), []
            for s in lst:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out
        kd_train_prompts = _dedupe(kd_train_prompts)
        grpo_train_prompts = _dedupe(grpo_train_prompts)
        if val_prompts:
            val_prompts = _dedupe(val_prompts)

    split_sizes = {
        "kd_train": len(kd_train_prompts),
        "grpo_train": len(grpo_train_prompts),
        "validation": len(val_prompts or []),
    }
    save_json(split_sizes, out_dir / "split_sizes.json")

    # ----- Generate each split -----
    manifests = {
        "created_at": timestamp(),
        "model": cfg.models.target,
        "root_out": str(out_dir),
        "splits": split_sizes,
    }

    # KD data goes under split name "train" (keeps train_kd.py unchanged)
    manifests["train"] = _gen_one_split(
        split_name="train",
        prompts=kd_train_prompts,
        tokenizer=tokenizer,
        llm=llm,
        cfg=cfg,
        out_dir=out_dir,
        seed=int(cfg.training.seed)
    )

    # GRPO prompt pool (with teacher top-K) goes under split name "grpo"
    manifests["grpo"] = _gen_one_split(
        split_name="grpo",
        prompts=grpo_train_prompts,
        tokenizer=tokenizer,
        llm=llm,
        cfg=cfg,
        out_dir=out_dir,
        seed=int(cfg.training.seed) + 1
    )

    # Optional validation (same format as KD)
    if val_prompts:
        manifests["validation"] = _gen_one_split(
            split_name="validation",
            prompts=val_prompts,
            tokenizer=tokenizer,
            llm=llm,
            cfg=cfg,
            out_dir=out_dir,
            seed=int(cfg.training.seed) + 2
        )

    save_json(manifests, out_dir / "manifest.json")
    print(f"[DONE] Wrote KD/GRPO corpus into: {out_dir}")


if __name__ == "__main__":
    main()
