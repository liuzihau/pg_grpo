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

def collate_topk(batch, pad_id: int, draft_dtype: torch.dtype = torch.float32):
    B = len(batch)
    S = max(int(b["cont_len"]) for b in batch)
    K = int(batch[0]["topk_ids"].shape[1])

    max_prompt = max(len(b["prompt_ids"]) for b in batch)

    input_ids      = torch.full((B, max_prompt + S), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    cont_mask      = torch.zeros((B, S), dtype=torch.float32)
    topk_ids       = torch.zeros((B, S, K), dtype=torch.long)
    topk_logprobs  = torch.full((B, S, K), -1e30, dtype=torch.float32)
    cont_len       = torch.zeros((B,), dtype=torch.long)

    for i, b in enumerate(batch):
        pids = torch.tensor(b["prompt_ids"], dtype=torch.long)
        Lp = pids.numel()
        input_ids[i, :Lp] = pids
        attention_mask[i, :Lp] = 1

        Si = min(S, int(b["cont_len"]))
        cont_len[i] = Si
        cont_mask[i, :Si] = 1

        ids = b["topk_ids"][:Si].to(torch.long)
        lps = b["topk_logprobs"][:Si].to(torch.float32)
        topk_ids[i, :Si] = ids
        topk_logprobs[i, :Si] = lps

    return {
        "input_ids": input_ids,              # CPU tensors
        "attention_mask": attention_mask,
        "cont_mask": cont_mask,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "cont_len": cont_len,
    }