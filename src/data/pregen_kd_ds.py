# src/data/pregen_kd_ds.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import glob
import torch

# ... your PreGeneratedTopKDataset stays the same ...

def _left_pad(seqs: List[List[int]], pad_id: int, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-pad a list of token id sequences to a common length.
    Returns (input_ids [B, L], attention_mask [B, L]).
    """
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


def collate_topk(
    batch: Sequence[Dict[str, Any]],
    *,
    tokenizer,                 # not re-tokenizing prompt_text; we use stored prompt_ids
    pad_id: int,
    max_input_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Build teacher-forcing inputs:
      input_ids = prompt_ids + cont_ids[:cont_len-1] (clipped left to max_input_len)
    Align KD targets to the last S positions of logits:
      - bottom-align the S×K top-k arrays so rows (S-cont_len: S-1) are valid
      - cont_mask has 1s only on those bottom 'cont_len' rows (else 0)
    """
    B = len(batch)

    # Per-sample staging
    input_lists: List[List[int]] = []
    cont_lens: List[int] = []
    S_list: List[int] = []     # S is fixed by corpus, but we read per rec to be safe

    # We’ll build these then stack: [B, S, K]
    topk_ids_list: List[torch.Tensor] = []
    topk_lps_list: List[torch.Tensor] = []
    cont_mask_list: List[torch.Tensor] = []

    for rec in batch:
        prompt_ids: List[int] = list(rec["prompt_ids"])
        cont_ids:   List[int] = list(rec["cont_ids"])        # length S with EOS pad
        cont_len:   int       = int(rec["cont_len"])
        tk_ids:     torch.Tensor = rec["topk_ids"]           # [S, K], int32
        tk_lps:     torch.Tensor = rec["topk_logprobs"]      # [S, K], float32

        S = int(len(cont_ids))
        K = int(tk_ids.shape[1])
        S_list.append(S)
        cont_lens.append(cont_len)

        # ---- Build teacher-forcing inputs: prompt + first (cont_len-1) ----
        prefix_len = max(cont_len - 1, 0)
        inp = prompt_ids + cont_ids[:prefix_len]

        # clip left to respect max_input_len (keep the most recent tokens)
        if len(inp) > max_input_len:
            inp = inp[-max_input_len:]
        input_lists.append(inp)

        # ---- Bottom-align KD tables to match logits[:, -S:, :] ----
        # rows [0 : S - cont_len) => invalid (masked), rows [S-cont_len : S) => valid continuation steps
        tk_ids = tk_ids.to(torch.long)            # [S, K]
        tk_lps = tk_lps.to(torch.float32)         # [S, K]

        # Create empty bottom-aligned buffers
        ids_btm = torch.zeros((S, K), dtype=torch.long)
        lps_btm = torch.full((S, K), -1e30, dtype=torch.float32)
        if cont_len > 0:
            ids_btm[S - cont_len : S, :] = tk_ids[:cont_len, :]
            lps_btm[S - cont_len : S, :] = tk_lps[:cont_len, :]

        mask = torch.zeros((S,), dtype=torch.float32)
        if cont_len > 0:
            mask[S - cont_len : S] = 1.0

        topk_ids_list.append(ids_btm)
        topk_lps_list.append(lps_btm)
        cont_mask_list.append(mask)

    # Sanity: ensure S is constant across the batch (it should be your corpus S)
    S_unique = set(S_list)
    if len(S_unique) != 1:
        raise ValueError(f"Inconsistent S in batch: {S_unique}")
    S = S_list[0]
    K = topk_ids_list[0].shape[1]

    # Left-pad inputs to batch
    input_ids, attention_mask = _left_pad(input_lists, pad_id=pad_id)
    # outputs on CPU; main process moves to GPU with non_blocking=True
    batch_out = {
        "input_ids": input_ids,                              # [B, L]
        "attention_mask": attention_mask,                    # [B, L]
        "cont_len": torch.tensor(cont_lens, dtype=torch.long),
        "cont_mask": torch.stack(cont_mask_list, dim=0),     # [B, S] (bottom 1s)
        "topk_ids": torch.stack(topk_ids_list, dim=0),       # [B, S, K] bottom-aligned
        "topk_logprobs": torch.stack(topk_lps_list, dim=0),  # [B, S, K] bottom-aligned
    }
    return batch_out
