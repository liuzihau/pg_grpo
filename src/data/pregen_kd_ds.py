# src/data/pregen_kd_ds.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import glob
import torch
from torch.utils.data import Dataset

__all__ = ["PreGeneratedTopKDataset", "collate_topk"]

class PreGeneratedTopKDataset(Dataset):
    """
    KD corpus produced by scripts/data_gen.py.

    Folder layout:
      <root>/
        manifest.json
        train/
          shard_00000.pt
          shard_00001.pt
          ...
        validation/ (optional)

    Each shard_*.pt is a list[dict] with keys:
      - "prompt_text": str
      - "prompt_ids": list[int]
      - "cont_ids": list[int]             # length S, EOS-padded
      - "cont_len": int                   # 0..S
      - "topk_ids": torch.Tensor [S, K]   # int32
      - "topk_logprobs": torch.Tensor[S,K]# float32
    """

    def __init__(self, root: Union[str, Path], split: str = "train"):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        # List shards (sorted for stable indexing)
        self.shards: List[Path] = [
            Path(p) for p in sorted(glob.glob(str(self.split_dir / "shard_*.pt")))
        ]
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.split_dir}")

        # Pre-scan shard sizes once to build a flat index
        self._shard_sizes: List[int] = []
        for p in self.shards:
            data = torch.load(p, map_location="cpu")
            if not isinstance(data, (list, tuple)):
                raise ValueError(f"Shard {p} is not a list; got type={type(data)}")
            self._shard_sizes.append(len(data))

        # Prefix sums for O(log N) shard lookup
        self._cum_sizes: List[int] = []
        total = 0
        for sz in self._shard_sizes:
            total += sz
            self._cum_sizes.append(total)
        self._total = total

        # Lazy shard cache
        self._cache_path: Optional[Path] = None
        self._cache_data: Optional[List[Dict[str, Any]]] = None

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += self._total
        if not (0 <= idx < self._total):
            raise IndexError(idx)
        # binary search over cum sizes
        lo, hi = 0, len(self._cum_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo
        prev_cum = self._cum_sizes[shard_idx - 1] if shard_idx > 0 else 0
        off = idx - prev_cum
        return shard_idx, off

    def _ensure_loaded(self, shard_idx: int):
        path = self.shards[shard_idx]
        if self._cache_path == path and self._cache_data is not None:
            return
        data = torch.load(path, map_location="cpu")
        # basic sanity
        if not isinstance(data, list):
            raise ValueError(f"Expected list of records in shard {path}, got {type(data)}")
        self._cache_path = path
        self._cache_data = data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sidx, off = self._locate(idx)
        self._ensure_loaded(sidx)
        rec = self._cache_data[off]  # type: ignore[index]

        # Normalize types/shapes for the collate:
        # prompt_ids, cont_ids -> list[int]
        prompt_ids = list(rec["prompt_ids"])
        cont_ids   = list(rec["cont_ids"])
        cont_len   = int(rec["cont_len"])

        # top-k tensors on CPU with proper dtypes
        topk_ids: torch.Tensor = rec["topk_ids"]
        topk_lp:  torch.Tensor = rec["topk_logprobs"]
        if topk_ids.dtype != torch.long:
            topk_ids = topk_ids.long()
        else:
            topk_ids = topk_ids.cpu()
        if topk_lp.dtype != torch.float32:
            topk_lp = topk_lp.to(torch.float32)
        else:
            topk_lp = topk_lp.cpu()

        return dict(
            prompt_ids=prompt_ids,
            cont_ids=cont_ids,
            cont_len=cont_len,
            topk_ids=topk_ids,
            topk_logprobs=topk_lp,
        )


# ---------------- Collate (kept compatible with data_gen output) ----------------

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
        tk_ids:     torch.Tensor = rec["topk_ids"]           # [S, K], int64/long
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
    # S = S_list[0]  # not used below directly

    # Left-pad inputs to batch
    input_ids, attention_mask = _left_pad(input_lists, pad_id=pad_id)

    batch_out = {
        "input_ids": input_ids,                              # [B, L]
        "attention_mask": attention_mask,                    # [B, L]
        "cont_len": torch.tensor(cont_lens, dtype=torch.long),
        "cont_mask": torch.stack(cont_mask_list, dim=0),     # [B, S] (bottom 1s)
        "topk_ids": torch.stack(topk_ids_list, dim=0),       # [B, S, K] bottom-aligned
        "topk_logprobs": torch.stack(topk_lps_list, dim=0),  # [B, S, K] bottom-aligned
    }
    return batch_out
