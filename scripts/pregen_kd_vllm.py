# scripts/pregen_kd_vllm.py
from __future__ import annotations
import os, json, argparse, random
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams


# -----------------------------
# Helper
# -----------------------------
def _pairs_from_cand(cand, tokenizer):
    """
    Normalize a single step's candidate logprobs into a list[(token_id:int, logprob:float)].
    Supports vLLM 0.11 shapes:
      - dict: { token(str|int) -> logprob(float) }
      - list: [TokenLogprob(token_id=int, logprob=float), ...]
    """
    pairs = []
    if isinstance(cand, dict):
        for tk, lp in cand.items():
            # tk may be int (already an ID) or str (a token string)
            if isinstance(tk, int):
                tid = tk
            else:
                try:
                    tid = tokenizer.convert_tokens_to_ids(tk)
                except Exception:
                    # If something odd slips through, skip it safely
                    continue
            if tid is None or tid < 0:
                continue
            pairs.append((int(tid), float(lp)))
    else:
        # Assume iterable of objects with .token_id and .logprob
        for c in cand:
            tid = getattr(c, "token_id", None)
            lp  = getattr(c, "logprob", None)
            if tid is None or lp is None:
                continue
            pairs.append((int(tid), float(lp)))
    return pairs

# -----------------------------
# YAML loader with `include:` support
# -----------------------------
def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _deep_update(a[k], v)
        else:
            a[k] = v
    return a

def load_yaml_with_includes(path: str) -> Dict[str, Any]:
    import yaml
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for rel in includes:
        inc_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        sub = load_yaml_with_includes(inc_path)
        _deep_update(merged, sub)
    _deep_update(merged, cfg)
    return merged

class Attr(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
def to_attrdict(d: Dict[str, Any]) -> Any:
    a = Attr()
    for k, v in d.items():
        if isinstance(v, dict):
            a[k] = to_attrdict(v)
        else:
            a[k] = v
    return a

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------
# Prompt extraction (allenai/tulu-3-sft-mixture)
# -----------------------------
def hf_prompts(tokenizer, cfg, split: str) -> List[str]:
    name = cfg.data.hf_name
    field = cfg.data.messages_field
    keep_history = bool(cfg.data.keep_history)
    lw = cfg.data.get("load_kwargs", {}) or {}
    # try desired split, then validation, test, then cfg.data.split
    for sp in [split, "validation", "test", cfg.data.split]:
        try:
            ds = load_dataset(name, split=sp, **lw)
        except Exception:
            continue
        items = []
        for rec in ds:
            msgs = rec.get(field, rec.get("conversations", rec.get("conversation", [])))
            if not isinstance(msgs, list):
                continue
            norm = []
            for m in msgs:
                if isinstance(m, dict):
                    role = str(m.get("role", m.get("from", ""))).lower()
                    content = m.get("content", m.get("value", ""))
                    if content:
                        norm.append({"role": role, "content": content})
            # keep up to and including the last user turn
            last_user = None
            for i in range(len(norm) - 1, -1, -1):
                if norm[i]["role"] == "user":
                    last_user = i
                    break
            if last_user is None:
                continue
            kept = norm[: last_user + 1] if keep_history else [norm[last_user]]
            try:
                prompt = tokenizer.apply_chat_template(
                    kept, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = [f"{m['role'].capitalize()}: {m['content']}" for m in kept]
                parts.append("Assistant:")
                prompt = "\n".join(parts)
            items.append(prompt)
        if items:
            return items
    return []

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # Load config with include support
    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(raw)

    # Sanity: kd_pregen must exist
    if not hasattr(cfg, "kd_pregen") or cfg.kd_pregen is None:
        raise RuntimeError(
            "Config missing `kd_pregen` block. Ensure defaults.yaml includes configs/kd_pregen.yaml "
            "and that file defines:\n"
            "kd_pregen:\n  out_dir: ...\n  num_samples: ...\n  shard_size: ...\n  S: ...\n  K: ..."
        )

    kdpg = cfg.kd_pregen
    out_dir      = kdpg.out_dir
    num_samples  = int(kdpg.num_samples)
    shard_size   = int(kdpg.shard_size)
    S            = int(kdpg.S)
    K            = int(kdpg.K)
    split        = kdpg.get("split", cfg.data.split)
    seed         = int(kdpg.get("seed", cfg.training.seed))
    temperature  = float(kdpg.get("temperature", 0.0))
    top_p        = float(kdpg.get("top_p", 1.0))
    top_k        = int(kdpg.get("top_k", 0))

    if not out_dir:
        raise RuntimeError("`kd_pregen.out_dir` must be set in the config.")

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer
    tok_name = cfg.models.get("tokenizer", cfg.models.target)
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # vLLM teacher
    llm = LLM(
        model=cfg.models.target,
        tensor_parallel_size=int(cfg.vllm.tensor_parallel_size),
        dtype=cfg.vllm.dtype,
        gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
    )

    # Prompts
    prompts = hf_prompts(tokenizer, cfg, split)
    if not prompts:
        raise RuntimeError("No prompts from HF (allenai/tulu-3-sft-mixture). Check `data.*` in configs/data.yaml.")

    idxs = random.sample(range(len(prompts)), k=min(len(prompts), num_samples))

    # vLLM sampling params (continuation logprobs only)
    sp = SamplingParams(
        max_tokens=S,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1,
        seed=seed,           # optional but handy for reproducibility
        logprobs=K,          # top-K logprobs for GENERATED tokens
        prompt_logprobs=None # don't request prompt logprobs
    )

    manifest = {
        "S": S,
        "K": K,
        "shards": [],
        "num_samples": len(idxs),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    buf = []
    shard = 0

    pbar = tqdm(range(0, len(idxs), 32), desc="vLLM KD pregen", ncols=100)
    for off in pbar:
        batch_idx = idxs[off : off + 32]
        batch_prompts = [prompts[i] for i in batch_idx]
        outs = llm.generate(batch_prompts, sp)

        for out in outs:
            prompt_text = out.prompt
            gen = out.outputs[0]
            token_ids = gen.token_ids               # [S'] (<= S)
            token_logprobs = gen.logprobs           # list length S'

            # Build top-K arrays per step
            step_topk_ids = []
            step_topk_lps = []
            for step in range(len(token_ids)):
                cand = token_logprobs[step]
                pairs = _pairs_from_cand(cand, tokenizer)
                if not pairs:
                    continue
                pairs.sort(key=lambda x: x[1], reverse=True)
                pairs = pairs[:K]
                ids = [p[0] for p in pairs]
                lps = [p[1] for p in pairs]
                step_topk_ids.append(ids)
                step_topk_lps.append(lps)


            cont_len = len(step_topk_ids)
            if cont_len == 0:
                continue

            # pad to S
            pad_id = tokenizer.eos_token_id
            cont_ids = (token_ids + [pad_id] * (S - len(token_ids)))[:S]

            import numpy as np
            ids_arr = np.zeros((S, K), dtype="int32")
            lps_arr = np.full((S, K), -1e30, dtype="float32")
            for t in range(min(S, cont_len)):
                ids_t = step_topk_ids[t]
                lps_t = step_topk_lps[t]
                n = min(K, len(ids_t))
                ids_arr[t, :n] = ids_t[:n]
                lps_arr[t, :n] = lps_t[:n]

            rec = {
                "prompt_text": prompt_text,
                "prompt_ids": tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist(),
                "cont_ids": cont_ids,
                "cont_len": min(S, cont_len),
                "topk_ids": torch.tensor(ids_arr, dtype=torch.int32),
                "topk_logprobs": torch.tensor(lps_arr, dtype=torch.float32),
            }
            buf.append(rec)

            if len(buf) >= shard_size:
                path = os.path.join(out_dir, f"shard_{shard:05d}.pt")
                torch.save(buf, path)
                manifest["shards"].append({"path": os.path.basename(path), "size": len(buf)})
                buf = []
                shard += 1

    if buf:
        path = os.path.join(out_dir, f"shard_{shard:05d}.pt")
        torch.save(buf, path)
        manifest["shards"].append({"path": os.path.basename(path), "size": len(buf)})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] KD corpus in {out_dir}")

if __name__ == "__main__":
    main()
