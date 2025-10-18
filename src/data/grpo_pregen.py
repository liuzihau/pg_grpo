# src/data/grpo_pregen.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random
import torch

# We will read from PreGeneratedTopKDataset, but we don't need its K tables here.

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
        # choose uniformly from the grid {0, stride, 2*stride, ...}
        n = max(1, (cont_len + stride - 1) // stride)
        k = rng.randrange(n)
        off = min(k * stride, max(0, cont_len - 1))
        return off
    # default: uniform over [0, cont_len-1]
    return rng.randrange(cont_len)

def collate_grpo_prefixes(
    batch: Sequence[Dict[str, Any]],
    *,
    tokenizer,
    pad_id: int,
    max_input_len: int,
    max_new_tokens: int,
    offset_strategy: str = "uniform",   # "uniform" | "stride"
    offset_stride: int = 8,
    cushion: int = 8,
    seed: int = 1234,
) -> Dict[str, torch.Tensor]:
    """
    For each record {prompt_ids, cont_ids, cont_len}, choose an offset 'o' and build:
        prefix_ids = prompt_ids + cont_ids[:o]
    Then left-pad to a batch and return tensors ready for sampling:
        - prompt_ids [B, L]
        - prompt_attn [B, L]
        - offsets [B]
        - cont_len [B]
    We enforce: len(prefix_ids) <= max_input_len - cushion - max_new_tokens
    """
    rng = random.Random(seed)
    input_lists: List[List[int]] = []
    chosen_offsets: List[int] = []
    cont_lens: List[int] = []

    budget = max(1, int(max_input_len) - int(max_new_tokens) - int(cushion))
    for rec in batch:
        prompt_ids: List[int] = list(rec["prompt_ids"])
        cont_ids:   List[int] = list(rec["cont_ids"])
        cont_len:   int       = int(rec["cont_len"])
        cont_lens.append(cont_len)

        # pick offset
        o = _choose_offset(cont_len, strategy=offset_strategy, stride=offset_stride, rng=rng)
        # build prefix
        prefix = prompt_ids + cont_ids[:o]

        # enforce budget (keep most recent tokens)
        if len(prefix) > budget:
            prefix = prefix[-budget:]

        input_lists.append(prefix)
        chosen_offsets.append(o)

    ids, att = _left_pad(input_lists, pad_id=pad_id)
    return {
        "prompt_ids": ids,                 # [B, L]
        "prompt_attn": att,                # [B, L]
        "offsets": torch.tensor(chosen_offsets, dtype=torch.long),  # [B]
        "cont_len": torch.tensor(cont_lens, dtype=torch.long),      # [B]
    }
