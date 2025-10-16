from __future__ import annotations
import os, json, argparse, random
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ---- utils
def load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def to_attrdict(d):
    class A(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
    return A(d)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

def hf_prompts(tokenizer, cfg, split: str) -> List[str]:
    name = cfg.data.hf_name
    field = cfg.data.messages_field
    keep_history = bool(cfg.data.keep_history)
    lw = cfg.data.get("load_kwargs", {}) or {}
    for sp in [split, "validation", "test", cfg.data.split]:
        try:
            ds = load_dataset(name, split=sp, **lw)
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
                # keep up to last user
                last_user = None
                for i in range(len(norm)-1, -1, -1):
                    if norm[i]["role"] == "user":
                        last_user = i; break
                if last_user is None:
                    continue
                kept = norm[:last_user+1] if keep_history else [norm[last_user]]
                try:
                    prompt = tokenizer.apply_chat_template(kept, tokenize=False, add_generation_prompt=True)
                except Exception:
                    parts = []
                    for m in kept:
                        parts.append(f"{m['role'].capitalize()}: {m['content']}")
                    parts.append("Assistant:")
                    prompt = "\n".join(parts)
                items.append(prompt)
            if items: return items
        except Exception:
            continue
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = to_attrdict(load_yaml(args.config))
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

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # tokenizer
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

    prompts = hf_prompts(tokenizer, cfg, split)
    if not prompts:
        raise RuntimeError("No prompts from HF")

    idxs = random.sample(range(len(prompts)), k=min(len(prompts), num_samples))

    # vLLM sampling params
    sp = SamplingParams(
        max_tokens=S, temperature=temperature, top_p=top_p, top_k=top_k,
        n=1, use_beam_search=False, logprobs=K, prompt_logprobs=False  # only need continuation logprobs
    )

    manifest = {"S": S, "K": K, "shards": [], "num_samples": len(idxs),
                "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed}
    buf = []; shard = 0

    pbar = tqdm(range(0, len(idxs), 32), desc="vLLM KD pregen", ncols=100)
    for off in pbar:
        batch_idx = idxs[off: off+32]
        batch_prompts = [prompts[i] for i in batch_idx]
        outs = llm.generate(batch_prompts, sp)

        for out in outs:
            prompt_text = out.prompt
            gen = out.outputs[0]
            token_ids = gen.token_ids               # [S'] (<= S)
            token_logprobs = gen.logprobs           # List[List[TokenLogprob]] length S'

            # Build top-K arrays per step
            step_topk_ids = []
            step_topk_lps = []
            for step in range(len(token_ids)):
                top = token_logprobs[step]  # dict or list of candidate logprobs
                # vLLM returns a dict {token: logprob} or list of CandidateLogprob; normalize to top-K arrays
                pairs = []
                if isinstance(top, dict):
                    for k_, v_ in top.items():
                        try:
                            tid = tokenizer.convert_tokens_to_ids(k_)
                        except Exception:
                            continue
                        if tid is not None and tid >= 0:
                            pairs.append((tid, float(v_)))
                else:
                    # list of CandidateLogprob
                    for cand in top:
                        pairs.append((cand.token_id, float(cand.logprob)))
                # keep best K
                pairs.sort(key=lambda x: x[1], reverse=True)
                pairs = pairs[:K]
                if not pairs:
                    continue
                ids = [p[0] for p in pairs]
                lps = [p[1] for p in pairs]
                step_topk_ids.append(ids)
                step_topk_lps.append(lps)

            cont_len = len(step_topk_ids)
            if cont_len == 0:
                continue

            # pad to S
            pad_id = tokenizer.eos_token_id
            cont_ids = (token_ids + [pad_id]*(S - len(token_ids)))[:S]
            # pad top-K arrays
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
                buf = []; shard += 1

    if buf:
        path = os.path.join(out_dir, f"shard_{shard:05d}.pt")
        torch.save(buf, path)
        manifest["shards"].append({"path": os.path.basename(path), "size": len(buf)})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] KD corpus in {out_dir}")

if __name__ == "__main__":
    main()
