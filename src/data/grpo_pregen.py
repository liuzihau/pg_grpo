# src/data/grpo_pregen.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random
import torch

def _left_pad(seqs: List[List[int]], pad_id: int, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    Ls = [len(x) for x in seqs]
    L = max(Ls) if max_len is None else max_len
    B = len(seqs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(seqs):
        n = min(len(x), L)
        ids[i, L - n : L] = torch.tensor(x[-n:], dtype=torch.long)
        att[i, L - n : L] = 1
    return ids, att

def _choose_offset(cont_len: int, *, strategy: str, stride: int, rng: random.Random) -> int:
    if cont_len <= 0:
        return 0
    strategy = (strategy or "uniform").lower()
    stride = max(1, int(stride))
    if strategy == "stride":
        # choose uniformly from {0, stride, 2*stride, ...}
        n = max(1, (cont_len + stride - 1) // stride)
        k = rng.randrange(n)
        return min(k * stride, max(0, cont_len - 1))
    # default: uniform
    return rng.randrange(cont_len)

def collate_grpo_prefixes(
    batch: Sequence[Dict[str, Any]],
    *,
    tokenizer,
    pad_id: int,
    max_input_len: int,
    max_new_tokens: int,
    offset_strategy: str = "uniform",
    offset_stride: int = 8,
    cushion: int = 8,
    seed: Optional[int] = None,      # <-- now optional
) -> Dict[str, torch.Tensor]:
    """
    Build prefix = prompt_ids + cont_ids[:offset] per record, respecting a budget:
    len(prefix) <= max_input_len - cushion - max_new_tokens.
    """
    rng = random.Random() if seed is None else random.Random(seed)  # <-- no step dependency
    input_lists: List[List[int]] = []
    chosen_offsets: List[int] = []
    cont_lens: List[int] = []

    budget = max(1, int(max_input_len) - int(max_new_tokens) - int(cushion))
    for rec in batch:
        prompt_ids: List[int] = list(rec["prompt_ids"])
        cont_ids:   List[int] = list(rec["cont_ids"])
        cont_len:   int       = int(rec["cont_len"])
        cont_lens.append(cont_len)
        if offset_strategy == "fix":
            o = 32
        else:
            o = _choose_offset(cont_len, strategy=offset_strategy, stride=offset_stride, rng=rng)
        prefix = prompt_ids + cont_ids[:o]
        if len(prefix) > budget:
            prefix = prefix[-budget:]

        input_lists.append(prefix)
        chosen_offsets.append(o)

    ids, att = _left_pad(input_lists, pad_id=pad_id)
    return {
        "prompt_ids": ids,
        "prompt_attn": att,
        "offsets": torch.tensor(chosen_offsets, dtype=torch.long),
        "cont_len": torch.tensor(cont_lens, dtype=torch.long),
    }
