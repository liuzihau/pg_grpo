# src/data/pregen_kd_ds.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _list_pt_shards(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.glob("*.pt")])

class PreGeneratedTopKDataset(Dataset):
    """
    Reads shards produced by scripts/data_gen.py.
    Each record contains:
      - prompt_text: str
      - cont_len: int
      - topk_ids:  [S, K] int32
      - topk_logprobs: [S, K] float32   (teacher log-probs)
    """
    def __init__(self, root: Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.manifest = _load_json(self.root / "manifest.json")
        self.S = int(self.manifest.get("S", 64))
        self.K = int(self.manifest.get("K", 20))
        self.shards = _list_pt_shards(self.root / split)
        assert self.shards, f"No shards found in {self.root / split}"
        # Build global index -> (shard_idx, local_idx)
        self._index: List[Tuple[int, int]] = []
        for si, sp in enumerate(self.shards):
            data = torch.load(sp, map_location="cpu")
            for j in range(len(data)):
                self._index.append((si, j))
        self._shard_cache = {}  # small lazy cache

    def __len__(self):
        return len(self._index)

    def _get_shard(self, si: int):
        if si not in self._shard_cache:
            self._shard_cache.clear()
            self._shard_cache[si] = torch.load(self.shards[si], map_location="cpu")
        return self._shard_cache[si]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, li = self._index[idx]
        shard = self._get_shard(si)
        rec = shard[li]
        return {
            "prompt_text": rec["prompt_text"],
            "cont_len": int(rec["cont_len"]),
            "topk_ids": rec["topk_ids"],               # [S,K] int32 CPU tensor
            "topk_logprobs": rec["topk_logprobs"],     # [S,K] float32 CPU tensor
        }

def collate_topk(
    *,
    batch: List[Dict[str, Any]],
    tokenizer,
    pad_id: int,
    max_input_len: int,
) -> Dict[str, torch.Tensor]:
    # Tokenize prompts on CPU, left-padded to the longest (and truncated to max_input_len)
    texts = [b["prompt_text"] for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_input_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]             # [B, L]
    attention_mask = enc["attention_mask"]   # [B, L]

    S = batch[0]["topk_ids"].shape[0]
    K = batch[0]["topk_ids"].shape[1]

    topk_ids = torch.stack([b["topk_ids"] for b in batch], dim=0).long()            # [B,S,K]
    topk_logprobs = torch.stack([b["topk_logprobs"] for b in batch], dim=0).float() # [B,S,K]
    cont_len = torch.tensor([b["cont_len"] for b in batch], dtype=torch.long)       # [B]
    # Build mask [B,S]
    cont_mask = torch.zeros((len(batch), S), dtype=torch.float32)
    for i, Lc in enumerate(cont_len.tolist()):
        cont_mask[i, :Lc] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "cont_len": cont_len,
        "cont_mask": cont_mask,
    }
