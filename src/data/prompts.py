# src/data/prompts.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import random
from datasets import load_dataset

def _normalize_messages(rec, field: str) -> List[Dict[str, str]]:
    msgs = rec.get(field, rec.get("conversations", rec.get("conversation", [])))
    if not isinstance(msgs, list):
        return []
    norm = []
    for m in msgs:
        if isinstance(m, dict):
            role = str(m.get("role", m.get("from", ""))).lower()
            content = m.get("content", m.get("value", ""))
            if content:
                norm.append({"role": role, "content": content})
    return norm

def _build_prompt_from_msgs(tokenizer, msgs: List[Dict[str, str]], keep_history: bool) -> str | None:
    # keep up to and including the last user turn
    last_user = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            last_user = i
            break
    if last_user is None:
        return None
    kept = msgs[: last_user + 1] if keep_history else [msgs[last_user]]
    try:
        prompt = tokenizer.apply_chat_template(kept, tokenize=False, add_generation_prompt=True)
        return prompt
    except Exception:
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in kept]
        parts.append("Assistant:")
        return "\n".join(parts)

def load_prompts_for_split(tokenizer, cfg, split: str) -> List[str]:
    """
    Try to load prompts for a specific HF split. Returns [] if unavailable.
    """
    name = cfg.data.hf_name
    field = cfg.data.messages_field
    keep_history = bool(cfg.data.keep_history)
    lw = cfg.data.get("load_kwargs", {}) or {}
    sample_max = cfg.data.get("sample_max", None)

    try:
        ds = load_dataset(name, split=split, **lw)
    except Exception:
        return []

    out: List[str] = []
    for i, rec in enumerate(ds):
        msgs = _normalize_messages(rec, field)
        if not msgs:
            continue
        p = _build_prompt_from_msgs(tokenizer, msgs, keep_history)
        if p:
            out.append(p)
        if sample_max is not None and len(out) >= int(sample_max):
            break
    return out

def make_manual_splits(tokenizer, cfg, seed: int | None = None) -> Tuple[List[str], List[str]]:
    """
    If HF split is not usable, load the base split (cfg.data.split) and derive train/val sizes
    from cfg.data.manual_splits.
    """
    base_split = cfg.data.split
    prompts = load_prompts_for_split(tokenizer, cfg, split=base_split)
    if not prompts:
        # try validation or test as a source pool
        for sp in ["validation", "test", "train"]:
            prompts = load_prompts_for_split(tokenizer, cfg, split=sp)
            if prompts:
                break

    ms = cfg.data.get("manual_splits", {})
    n_train = int(ms.get("train", max(1, int(len(prompts) * 0.9))))
    n_val   = int(ms.get("validation", len(prompts) - n_train))

    idx = list(range(len(prompts)))
    rnd = random.Random(seed or 1234)
    rnd.shuffle(idx)

    train_idx = idx[: min(n_train, len(idx))]
    val_idx   = idx[min(n_train, len(idx)) : min(n_train + n_val, len(idx))]

    train_prompts = [prompts[i] for i in train_idx]
    val_prompts   = [prompts[i] for i in val_idx]
    return train_prompts, val_prompts

def truncate_prompt_by_tokens(tokenizer, prompt: str, max_tokens: int) -> str:
    """
    Left-truncate the prompt to at most `max_tokens` to avoid exceeding model length
    when adding continuation S tokens. Decoder-only models prefer left padding.
    """
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    if len(ids) <= max_tokens:
        return prompt
    trimmed = ids[-max_tokens:]
    # keep special tokens as-is if any; we don't skip special tokens in decoding
    return tokenizer.decode(trimmed, skip_special_tokens=False)
