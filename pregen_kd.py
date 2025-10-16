# pregen_kd.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from kd_trainer import KDStepConfig, KDWeights
from rewards_boundary import compute_token_divergence

class TeacherHead(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, dtype=None, device=None):
        super().__init__()
        self.register_buffer("weight", weight.to(dtype=dtype, device=device), persistent=False)  # [V,H]
        self.register_buffer("bias",   None if bias is None else bias.to(dtype=dtype, device=device), persistent=False)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,T,H] -> logits [B,T,V]
        B, T, H = h.shape
        v = torch.matmul(h.reshape(B*T, H), self.weight.t())
        if self.bias is not None:
            v = v + self.bias
        return v.reshape(B, T, -1)

class PreGeneratedKDDataset(Dataset):
    """
    Random-access view over shard files (torch.save lists).
    """
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        man_path = os.path.join(root, "manifest.json")
        if not os.path.exists(man_path):
            # fallback: infer shard list
            shards = [fn for fn in os.listdir(root) if fn.startswith("shard_") and fn.endswith(".pt")]
            shards.sort()
            self.manifest = {"S": None, "hidden_dtype": "fp16", "shards": []}
            for fn in shards:
                path = os.path.join(root, fn)
                data = torch.load(path, map_location="cpu")
                self.manifest["shards"].append({"path": fn, "size": len(data)})
                if self.manifest["S"] is None and len(data) > 0:
                    self.manifest["S"] = len(data[0]["cont_ids"])
                del data
        else:
            with open(man_path, "r") as f:
                self.manifest = json.load(f)
        self.cum = []
        c = 0
        for s in self.manifest["shards"]:
            self.cum.append((c, s["path"], s["size"]))
            c += s["size"]
        self.total = c
        self._cache = {"path": None, "items": None}

    def __len__(self): return self.total

    def _load_shard(self, path: str):
        items = torch.load(os.path.join(self.root, path), map_location="cpu")
        self._cache = {"path": path, "items": items}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # map idx -> (path, local_idx)
        for start, path, size in self.cum:
            if start <= idx < start + size:
                local = idx - start
                if self._cache["path"] != path:
                    self._load_shard(path)
                rec = self._cache["items"][local]
                return rec
        raise IndexError(idx)

def left_pad_1d(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns (padded_ids [B,L], attn_mask [B,L])
    B = len(seqs)
    L = max(int(x.numel()) for x in seqs)
    out = torch.full((B, L), pad_id, dtype=seqs[0].dtype)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        l = int(s.numel())
        out[i, -l:] = s
        mask[i, -l:] = 1
    return out, mask

def collate_pregen_kd(batch: List[Dict[str, Any]], pad_id: int, device, target_dtype) -> Dict[str, torch.Tensor]:
    """
    Makes a batch for KD:
      - input_ids  = [prompt || cont[:cont_len]]  (left-padded)
      - attention_mask
      - cont_mask  = mask over the last T_max positions where t < cont_len
      - teacher_hidden: [B, T_max, H]
      - cont_len: [B]
    """
    # tensors
    prompts = [torch.tensor(b["prompt_ids"], dtype=torch.long) for b in batch]
    cont_ids_all = [torch.tensor(b["cont_ids"], dtype=torch.long) for b in batch]
    cont_lens = torch.tensor([int(b["cont_len"]) for b in batch], dtype=torch.long)
    # build input sequences: prompt || cont[:cont_len]
    seqs = []
    for p, c, Lc in zip(prompts, cont_ids_all, cont_lens):
        Lc = int(Lc.item())
        seqs.append(torch.cat([p, c[:max(Lc, 0)]], dim=0))
    input_ids, attn_mask = left_pad_1d(seqs, pad_id=pad_id)  # [B,L]
    B, L = input_ids.shape
    T_max = int(cont_lens.max().item())
    # teacher hidden
    hidden = []
    for b in batch:
        t = b["teacher_normed_hidden"]  # [S,H] fp16
        h = torch.tensor(t, dtype=torch.float16)  # robust if saved as np
        Lc = int(b["cont_len"])
        h = h[:Lc, :]  # only real continuation
        hidden.append(h)
    H = hidden[0].shape[-1] if T_max > 0 else int(batch[0]["teacher_normed_hidden"].shape[-1])
    teacher_hidden = torch.zeros((B, T_max, H), dtype=torch.float16)
    cont_mask = torch.zeros((B, T_max), dtype=torch.float32)
    for i, h in enumerate(hidden):
        l = h.shape[0]
        if l > 0:
            teacher_hidden[i, T_max - l:, :] = h
            cont_mask[i, T_max - l:] = 1.0

    # to device / dtype
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "teacher_hidden": teacher_hidden.to(device=device, dtype=target_dtype),
        "cont_mask": cont_mask.to(device),
        "cont_len": cont_lens.to(device),
    }

def kd_step_pregen(
    *,
    draft: nn.Module,
    teacher_head: TeacherHead,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    kd_cfg: KDStepConfig,
    kd_weights: KDWeights,
) -> Dict[str, float]:
    """
    KD update using pre-generated teacher hidden states.
    """
    draft.train()
    out = draft(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    d_logits_full = out.logits  # [B,L,V]

    # Slice last T_max positions (these include the continuation predictions)
    T_max = int(batch["teacher_hidden"].shape[1])
    if T_max == 0:
        return {"train/kd_total_loss": 0.0, "train/kd_main_div": 0.0, "train/kd_avg_margin": 0.0, "train/kd_avg_mismatch": 0.0, "train/avg_cont_len": 0.0}

    d_logits_BT = d_logits_full[:, -T_max:, :]       # [B,T,V]
    t_logits_BT = teacher_head(batch["teacher_hidden"])  # [B,T,V]
    B, T, V = d_logits_BT.shape

    # masks
    mask_BT = batch["cont_mask"]  # [B,T] (1 for real continuation positions)
    denom = mask_BT.sum().clamp_min(1.0)

    # margins & mismatch
    with torch.no_grad():
        top2 = torch.topk(t_logits_BT, k=2, dim=-1).values  # [B,T,2]
        margin = top2[..., 0] - top2[..., 1]                # [B,T]
        m = torch.sigmoid(kd_weights.margin_gamma * (margin - getattr(kd_weights, "margin_center", 1.0)))
        w_min = getattr(kd_weights, "w_min", 0.2)
        margin_w = w_min + (1.0 - w_min) * m                # [B,T] in [w_min,1]

        t_argmax = t_logits_BT.argmax(dim=-1)               # [B,T]
        d_argmax = d_logits_BT.argmax(dim=-1)
        mismatch = (t_argmax != d_argmax).float()

    TW = margin_w
    MW = 1.0 + kd_weights.mismatch_lambda * mismatch
    weights_BT = (TW * MW * mask_BT).to(d_logits_BT.dtype)  # [B,T]

    # compute divergence on [T,N,V] layout
    TNV_t = t_logits_BT.transpose(0,1).contiguous()  # [T,B,V]
    TNV_d = d_logits_BT.transpose(0,1).contiguous()  # [T,B,V]
    d_t = compute_token_divergence(
        TNV_t, TNV_d, kind=kd_cfg.divergence, alpha=kd_cfg.alpha, topk_for_ce=kd_cfg.topk_for_ce
    )  # [T,B]

    loss_main = (d_t * weights_BT.transpose(0,1)).sum() / denom
    total = loss_main  # KD: no anchor/entropy

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(draft.parameters(), kd_cfg.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        return {
            "train/kd_total_loss": float(total.detach().cpu()),
            "train/kd_main_div":   float(loss_main.detach().cpu()),
            "train/kd_avg_margin": float(margin.mean().detach().cpu()),
            "train/kd_avg_mismatch": float(mismatch.mean().detach().cpu()),
            "train/avg_cont_len":  float(batch["cont_len"].float().mean().detach().cpu()),
        }
