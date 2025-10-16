
from __future__ import annotations
import os, json, random, argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils import load_yaml, set_seed, to_attrdict
from models import load_tokenizer, load_target
from data import HFChatPromptsDataset, PromptOnlyDataset


@torch.no_grad()
def _get_final_norm_module(model: nn.Module):
    for name in ["model.norm", "transformer.norm", "transformer.final_layernorm", "norm"]:
        cur = model
        ok = True
        for part in name.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.Module):
            return cur
    return nn.Identity()


def _get_lm_head(model: nn.Module):
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        head = model.get_output_embeddings()
    elif hasattr(model, "lm_head"):
        head = model.lm_head
    else:
        raise RuntimeError("Could not locate LM head on teacher model.")
    weight = head.weight.detach().cpu()
    bias = None
    if hasattr(head, "bias") and head.bias is not None:
        bias = head.bias.detach().cpu()
    return weight, bias


def _build_prompts_list(tokenizer, cfg, split: str) -> List[str]:
    data_cfg = getattr(cfg, "data", {}) or {}
    source = (getattr(data_cfg, "source", None) or data_cfg.get("source") or "hf").lower()
    if source == "hf":
        name  = getattr(data_cfg, "hf_name", None) or data_cfg.get("hf_name")
        if not isinstance(name, str):
            raise ValueError("data.hf_name must be a string like 'allenai/tulu-3-sft-mixture'")
        field = getattr(data_cfg, "messages_field", "messages")
        keep_history = getattr(data_cfg, "keep_history", True)
        sample_max = getattr(data_cfg, "sample_max", None)
        load_kwargs = dict(getattr(data_cfg, "load_kwargs", None) or {})
        for sp in [split, "validation", "test", getattr(data_cfg, "split", "train")]:
            try:
                ds = HFChatPromptsDataset(
                    dataset_name=name,
                    split=sp,
                    tokenizer=tokenizer,
                    messages_field=field,
                    sample_max=sample_max,
                    keep_history=keep_history,
                    load_kwargs=load_kwargs,
                )
                items = [rec["prompt"] for rec in ds]
                if items:
                    return items
            except Exception:
                continue
        raise RuntimeError("HF dataset could not be loaded; consider `data.prompts_path`.")
    # local
    path = getattr(data_cfg, "prompts_path", None) or "data/prompts.jsonl"
    ds = PromptOnlyDataset(path)
    items = [rec["prompt"] for rec in ds]
    if not items:
        raise RuntimeError(f"Local prompts file is empty or missing: {path}")
    return items


def _pick_indices(n_total: int, n_want: int, seed: int) -> List[int]:
    n = min(n_total, n_want)
    rng = random.Random(seed)
    return rng.sample(range(n_total), k=n)


@torch.no_grad()
def _sample_next_token_from_logits(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int
) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    scores = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        topk_vals, topk_idx = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1)
        probs = topk_vals.softmax(dim=-1)
        choice = torch.multinomial(probs, 1).squeeze(-1)
        return topk_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        mask = cdf <= top_p
        mask[..., 0] = True
        masked_logits = torch.where(mask, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        probs = masked_logits.softmax(dim=-1)
        choice = torch.multinomial(probs, 1).squeeze(-1)
        return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    probs = scores.softmax(dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.no_grad()
def _generate_one_record(
    *, tokenizer, teacher: nn.Module, prompt_text: str, device: torch.device,
    S: int, temperature: float, top_p: float, top_k: int
) -> Dict[str, Any]:
    enc = tokenizer(
        [prompt_text],
        padding=False,
        truncation=True,
        max_length=getattr(teacher.config, "max_position_embeddings", 4096),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    eos = tokenizer.eos_token_id

    final_norm = _get_final_norm_module(teacher)

    cont_ids: List[int] = []
    normed_hidden_steps: List[torch.Tensor] = []

    cur_ids, cur_attn = input_ids, attn_mask
    teacher.eval()

    for _ in range(S):
        out = teacher(input_ids=cur_ids, attention_mask=cur_attn,
                      use_cache=False, output_hidden_states=True, return_dict=True)
        last_h = out.hidden_states[-1][:, -1, :]          # [1,H] pre-final-norm
        normed = final_norm(last_h)                       # [1,H]
        normed_hidden_steps.append(normed.squeeze(0).to(torch.float16).cpu())

        # Prefer model-provided logits if available
        if hasattr(out, "logits") and out.logits is not None:
            last_logits = out.logits[:, -1, :]            # [1,V]
        else:
            # This path should rarely trigger; head will be applied in training anyway
            raise RuntimeError("Teacher did not return logits; please update model/transformers.")

        next_tok = _sample_next_token_from_logits(last_logits, temperature, top_p, top_k)
        cont_ids.append(int(next_tok.item()))

        cur_ids = torch.cat([cur_ids, next_tok.view(1, 1)], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(next_tok.view(1,1))], dim=1)

        if eos is not None and int(next_tok.item()) == int(eos):
            break

    cont_len = len(cont_ids)
    if cont_len < S:
        cont_ids = cont_ids + [int(eos)] * (S - cont_len)

    return {
        "prompt_ids": enc["input_ids"].squeeze(0).tolist(),
        "cont_ids": cont_ids,             # length S
        "cont_len": cont_len,
        "teacher_normed_hidden": torch.stack(
            normed_hidden_steps + ([torch.zeros_like(normed_hidden_steps[0])]*(S - cont_len))
        )[:S, :].contiguous(),           # [S,H] fp16
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = to_attrdict(load_yaml(args.config))
    pg = getattr(cfg, "kd_pregen", None)
    if not pg:
        raise RuntimeError("Config is missing `kd_pregen` section.")

    # Read params from config
    out_dir      = getattr(pg, "out_dir", None)
    num_samples  = int(getattr(pg, "num_samples", 10000))
    shard_size   = int(getattr(pg, "shard_size", 2000))
    S            = int(getattr(pg, "S", 64))
    split        = str(getattr(pg, "split", getattr(getattr(cfg, "data", {}), "split", "train")))
    seed         = int(getattr(pg, "seed", getattr(getattr(cfg, "training", {}), "seed", 1234)))
    temperature  = float(getattr(pg, "temperature", getattr(getattr(cfg, "kd", {}), "temperature", 0.0)))
    top_p        = float(getattr(pg, "top_p", getattr(getattr(cfg, "kd", {}), "top_p", 1.0)))
    top_k        = int(getattr(pg, "top_k", getattr(getattr(cfg, "kd", {}), "top_k", 0)))

    if not out_dir:
        raise RuntimeError("`kd_pregen.out_dir` must be set in the config.")

    set_seed(seed)

    # device & dtype
    device_str = getattr(getattr(cfg, "training", {}), "device", "cuda")
    dtype_str  = getattr(getattr(cfg, "training", {}), "dtype", "bf16")
    device = torch.device(device_str)

    # tokenizer / teacher
    tok_name = getattr(getattr(cfg, "models", {}), "tokenizer", getattr(getattr(cfg, "models", {}), "target", None))
    tgt_name = getattr(getattr(cfg, "models", {}), "target", None)
    assert tgt_name, "models.target must be set"

    tokenizer = load_tokenizer(tok_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    teacher = load_target(tgt_name, dtype=dtype_str, device=device_str)
    teacher.eval()

    # save teacher head once
    os.makedirs(out_dir, exist_ok=True)
    head_path = os.path.join(out_dir, "teacher_head.pt")
    if not os.path.exists(head_path):
        w, b = _get_lm_head(teacher)
        torch.save({"weight": w.half(), "bias": None if b is None else b.half()}, head_path)

    # prompts
    prompts_all = _build_prompts_list(tokenizer, cfg, split=split)
    if len(prompts_all) == 0:
        raise RuntimeError("No prompts loaded for generation.")
    indices = _pick_indices(len(prompts_all), num_samples, seed)

    # shard loop
    shard, buf = 0, []
    manifest = {"S": S, "hidden_dtype": "fp16", "shards": [], "num_samples": num_samples,
                "split": split, "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed}
    pbar = tqdm(indices, desc="Generate KD corpus", dynamic_ncols=True)
    for idx in pbar:
        rec = _generate_one_record(
            tokenizer=tokenizer,
            teacher=teacher,
            prompt_text=prompts_all[idx],
            device=device,
            S=S,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        buf.append(rec)
        if len(buf) >= shard_size:
            path = os.path.join(out_dir, f"shard_{shard:05d}.pt")
            torch.save(buf, path)
            manifest["shards"].append({"path": os.path.basename(path), "size": len(buf)})
            buf = []
            shard += 1

    if buf:
        path = os.path.join(out_dir, f"shard_{shard:05d}.pt")
        torch.save(buf, path)
        manifest["shards"].append({"path": os.path.basename(path), "size": len(buf)})

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Saved KD corpus to {out_dir} with {len(manifest['shards'])} shards.")

if __name__ == "__main__":
    main()
