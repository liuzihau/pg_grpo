from __future__ import annotations
import os, json
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class PreGeneratedTopKDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        man = os.path.join(root, "manifest.json")
        if not os.path.exists(man):
            raise FileNotFoundError(f"manifest.json not found in {root}")
        with open(man, "r") as f:
            self.manifest = json.load(f)
        self.cum = []
        c = 0
        for s in self.manifest["shards"]:
            self.cum.append((c, s["path"], s["size"]))
            c += s["size"]
        self.total = c
        self._cache = {"path": None, "items": None}

    def __len__(self): return self.total

    def _load(self, path: str):
        items = torch.load(os.path.join(self.root, path), map_location="cpu")
        self._cache = {"path": path, "items": items}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for start, path, size in self.cum:
            if start <= idx < start+size:
                if self._cache["path"] != path:
                    self._load(path)
                return self._cache["items"][idx - start]
        raise IndexError(idx)

def left_pad(ids_list: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(ids_list)
    L = max(x.numel() for x in ids_list)
    out = torch.full((B,L), pad_id, dtype=torch.long)
    mask = torch.zeros((B,L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        l = x.numel()
        out[i, -l:] = x
        mask[i, -l:] = 1
    return out, mask

def collate_topk(batch: List[Dict[str,Any]], pad_id: int, device, draft_dtype):
    prompts = [torch.tensor(b["prompt_ids"], dtype=torch.long) for b in batch]
    cont_ids = [torch.tensor(b["cont_ids"], dtype=torch.long) for b in batch]
    cont_len = torch.tensor([int(b["cont_len"]) for b in batch], dtype=torch.long)
    # input_ids = prompt || cont[:cont_len-1], predict cont tokens
    seqs = []
    for p, c, Lc in zip(prompts, cont_ids, cont_len):
        seqs.append(torch.cat([p, c[:max(int(Lc-1), 0)]], dim=0))
    input_ids, attn_mask = left_pad(seqs, pad_id=pad_id)
    B, L = input_ids.shape
    S = int(max(int(b["cont_len"]) for b in batch))
    K = int(batch[0]["topk_ids"].shape[1]) if S>0 else 0

    topk_ids = torch.stack([b["topk_ids"] for b in batch], dim=0)[:, :S, :]       # [B,S,K]
    topk_lps = torch.stack([b["topk_logprobs"] for b in batch], dim=0)[:, :S, :]  # [B,S,K]
    cont_mask = torch.zeros((B,S), dtype=torch.float32)
    for i, Lc in enumerate(cont_len):
        cont_mask[i, :int(Lc)] = 1.0

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "cont_ids": torch.stack([c for c in cont_ids], 0)[:, :S].to(device),
        "cont_len": cont_len.to(device),
        "cont_mask": cont_mask.to(device),
        "topk_ids": topk_ids.to(device),
        "topk_logprobs": topk_lps.to(device),
    }
