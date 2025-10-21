#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time, gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ---- local utils you already have ----
from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides, parse_overrides,
    set_seed, cfg_get, parse_torch_dtype,
)
from src.common.io import save_json
from src.models.load import print_model_layer_report

# ------------------------------
# Data loading (pregen or HF)
# ------------------------------

def _load_prompts_from_pregen(pregen_dir: Union[str, Path], split: str, sample_max: Optional[int]) -> List[str]:
    """
    Read prompt_text from PT shards under: <pregen_dir>/<split>/shard_*.pt
    """
    pregen_dir = Path(pregen_dir)
    split_dir = pregen_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No pregen split dir: {split_dir}")

    prompts: List[str] = []
    for pt in sorted(split_dir.glob("shard_*.pt")):
        buf = torch.load(pt, map_location="cpu")
        # buf is a list[dict] with "prompt_text"
        for rec in buf:
            p = rec.get("prompt_text", None)
            if p:
                prompts.append(p)
                if sample_max is not None and len(prompts) >= int(sample_max):
                    return prompts
    return prompts

def _normalize_messages(rec, field: str):
    msgs = rec.get(field, rec.get("conversations", rec.get("conversation", [])))
    if not isinstance(msgs, list): return []
    norm = []
    for m in msgs:
        if isinstance(m, dict):
            role = str(m.get("role", m.get("from", ""))).lower()
            content = m.get("content", m.get("value", ""))
            if content: norm.append({"role": role, "content": content})
    return norm

def _prompt_from_msgs(tokenizer, msgs, keep_history=True) -> Optional[str]:
    last_user = None
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i]["role"] == "user": last_user = i; break
    if last_user is None: return None
    kept = msgs[:last_user+1] if keep_history else [msgs[last_user]]
    try:
        return tokenizer.apply_chat_template(kept, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in kept]
        parts.append("Assistant:")
        return "\n".join(parts)

def _load_prompts_from_hf(tokenizer, name: str, split: str, sample_max: Optional[int], keep_history: bool=True, load_kwargs: dict|None=None) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset(name, split=split, **(load_kwargs or {}))
    out: List[str] = []
    for rec in ds:
        msgs = _normalize_messages(rec, "messages")
        if not msgs: continue
        p = _prompt_from_msgs(tokenizer, msgs, keep_history)
        if p:
            out.append(p)
            if sample_max is not None and len(out) >= int(sample_max):
                break
    return out

def _left_truncate_to_budget(tok, text, budget):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() <= budget:
        return text
    return tok.decode(ids[-budget:], skip_special_tokens=False)

# ------------------------------
# Model loading (HF or PEFT)
# ------------------------------

def _is_sharded(model) -> bool:
    dm = getattr(model, "hf_device_map", None)
    return bool(dm) and len(set(dm.values())) > 1

def _first_device(model):
    # safe for single-device models
    return next(model.parameters()).device

def _place_for(model, t):
        return t.to(next(model.parameters()).device)

def _token_for(model, tok_id: int):
    dev = next(model.parameters()).device
    return torch.tensor([[tok_id]], dtype=torch.long, device=dev)

def _is_adapter_folder(p: Union[str, Path]) -> bool:
    d = Path(p)
    return (d / "adapter_config.json").is_file() or \
           (d / "adapter_model.bin").is_file() or \
           (d / "adapter_model.safetensors").is_file()

def _try_read_adapter_base(adapter_dir: Union[str, Path]) -> Optional[str]:
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.is_file(): return None
    try:
        j = json.loads(cfg.read_text())
        return j.get("base_model_name_or_path") or j.get("base_model_name") or None
    except Exception:
        return None

def _load_causal_lm(spec: Optional[str],
                    *,
                    dtype: torch.dtype,
                    device_map: str,
                    use_cache: bool,
                    fallback_base: Optional[str] = None,
                    load_in_8bit: bool = False):
    """
    spec is None => return None
    spec is HF model id or local path => return AutoModelForCausalLM
    spec is a LoRA adapter dir => attach to base (from adapter_config or fallback_base)
    """
    if spec in (None, "", "null", "None"):
        return None
    # LoRA adapter?
    if _is_adapter_folder(spec):
        base_name = _try_read_adapter_base(spec) or fallback_base
        if base_name is None:
            raise ValueError(f"Adapter at {spec} has no base in adapter_config.json and no fallback base provided.")
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit if hasattr(AutoModelForCausalLM, "from_pretrained") else False
        )
        model = PeftModel.from_pretrained(base, spec)
    else:
        # full HF model id / folder
        kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        model = AutoModelForCausalLM.from_pretrained(spec, **kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = bool(use_cache)
    model.eval()
    return model

# --- helper: replay a short list of tokens on top of a given past
def _replay_tokens(model, past, token_ids: list[int]):
    cur_prev = None
    for tok in token_ids:
        cur_prev = _token_for(model, tok)             # [1,1] on model's device
        out = model(input_ids=cur_prev, use_cache=True, past_key_values=past)
        past = out.past_key_values                    # advance cache by one
    return past, cur_prev

# ------------------------------
# SD simulator (greedy or sampled)
# ------------------------------
@torch.no_grad()
def _sd_simulate_one(
    *,
    tokenizer,
    draft,
    teacher,
    prompt_text: str,
    K: int,
    max_new: int,
    temperature: float,
    acceptance_cap: float,
    device: torch.device,
    max_input_len: int,
) -> Dict[str, Any]:
    """
    Standard SD loop with alpha_t computed as distribution overlap:
      alpha_t = sum_x min(q_t(x), p_t(x)/cap)
    (independent of greedy/sampled accept decision).
    """

    # ---- helpers (assume you have these; keep same signatures)
    

    # ---- tokenize & truncate prompt
    budget = max(1, max_input_len - 1)   # tiny cushion
    prompt_text = _left_truncate_to_budget(tokenizer, prompt_text, budget)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids_cpu = enc["input_ids"]  # [1, P]

    # make sure models on device / in eval
    draft = draft.eval()
    teacher = teacher.eval()

    # knobs
    use_greedy = (float(temperature) <= 0.0)
    tau = float(temperature)
    cap = float(max(acceptance_cap, 1e-6))

    total_gen = 0
    accepted_total = 0
    teacher_calls = 0
    alphas: List[float] = []

    # ---- build prompt caches (teacher & draft)
    d_out = draft(input_ids=_place_for(draft, input_ids_cpu[:, :-1]), use_cache=True)
    draft_past = d_out.past_key_values
    cur_prev = _place_for(draft, input_ids_cpu[:, -1:])  # last prompt token

    t_out = teacher(input_ids=_place_for(teacher, input_ids_cpu[:, :-1]), use_cache=True)
    teacher_past = t_out.past_key_values
    teacher_prev = _place_for(teacher, input_ids_cpu[:, -1:])

    start_t = time.perf_counter()

    while total_gen < max_new:
        # teacher_calls += 1           # REMOVE: this was counting blocks, not calls

        # --------- 1) DRAFT proposes up to K tokens
        prop_tokens: List[int] = []
        prop_q_probs: List[torch.Tensor] = []
        draft_past_confirmed = draft_past                 # NEW: snapshot start-of-block cache

        for _ in range(K):
            d = draft(input_ids=cur_prev, use_cache=True, past_key_values=draft_past)
            draft_past = d.past_key_values
            d_logits = d.logits[:, -1, :]
            q_logits = d_logits / tau if tau > 0 else d_logits
            q_probs = F.softmax(q_logits, dim=-1)[0]
            next_id = int(d_logits.argmax(-1).item()) if use_greedy else int(torch.multinomial(q_probs, 1).item())
            prop_tokens.append(next_id)
            prop_q_probs.append(q_probs.detach())
            cur_prev = _token_for(draft, next_id)
            if len(prop_tokens) + total_gen >= max_new:
                break

        if len(prop_tokens) == 0:
            break

        # --------- 2) TEACHER verifies sequentially
        accepted_in_block = 0
        rejected = False
        teacher_correction_id = None
        with open('./test.txt', 'a', encoding='utf-8') as f:
            f.write("========teacher verify start========\n")
        teacher_calls += 1                             # NEW: count actual teacher forwards
        for t, tok in enumerate(prop_tokens):
            tt = teacher(input_ids=teacher_prev, use_cache=True, past_key_values=teacher_past)
            
            teacher_past = tt.past_key_values
            t_logits = tt.logits[:, -1, :]

            # alpha_t overlap (single step, vector [V])
            q_probs = prop_q_probs[t].to(t_logits.device)
            p_probs = F.softmax(t_logits, dim=-1)[0]
            alpha_t = torch.minimum(q_probs, p_probs / cap).sum().clamp_max(1.0)  # <= 1 if cap>=1
            alphas.append(float(alpha_t.item()))

            k = 3
            q_vals, q_ids = torch.topk(q_probs, k=k, dim=-1, largest=True, sorted=True)
            p_vals, p_ids = torch.topk(p_probs, k=k, dim=-1, largest=True, sorted=True)
            s = "q: "
            for r, (tid, prob) in enumerate(zip(q_ids.tolist(), q_vals.tolist())):
                s += f"#{r}: id={tid}, prob={prob:.4f} "
            s += "\n p: "
            for r, (tid, prob) in enumerate(zip(p_ids.tolist(), p_vals.tolist())):
                s += f"#{r}: id={tid}, prob={prob:.4f} "
            s += '\n'
            with open('./test.txt', 'a', encoding='utf-8') as f:
                f.write(f"{alpha_t.item():.4f}\n {s}")

            if use_greedy:
                top_id = int(t_logits.argmax(-1).item())
                if top_id == tok:
                    accepted_in_block += 1
                    teacher_prev = _token_for(teacher, tok)
                else:
                    teacher_prev = _token_for(teacher, top_id)
                    teacher_correction_id = top_id
                    rejected = True
                    break
            else:
                p_logp_tok = F.log_softmax(t_logits, dim=-1)[0, tok]
                q_logp_tok = torch.log(q_probs[tok].clamp_min(1e-12))
                accept_prob = torch.exp(p_logp_tok - q_logp_tok - math.log(cap)).clamp(max=1.0)
                if torch.rand((), device=t_logits.device) < accept_prob:
                    accepted_in_block += 1
                    teacher_prev = _token_for(teacher, tok)
                else:
                    top_id = int(t_logits.argmax(-1).item())  # can also sample; argmax is standard
                    teacher_prev = _token_for(teacher, top_id)
                    teacher_correction_id = top_id
                    rejected = True
                    break

            if total_gen + accepted_in_block >= max_new:
                break

        # --------- 3) COMMIT (fix the cache sync)
        if rejected:
            # roll back to start-of-block cache, then replay accepted + teacher correction
            draft_past = draft_past_confirmed             # NEW
            replay = prop_tokens[:accepted_in_block] + [teacher_correction_id]
            draft_past, cur_prev = _replay_tokens(draft, draft_past, replay)  # NEW
            accepted_total += accepted_in_block
            total_gen += accepted_in_block + 1            # +1 for teacher token emitted at rejection
        else:
            # all proposed tokens were accepted; our current draft_past already matches that state
            accepted_total += accepted_in_block           # == len(prop_tokens)
            total_gen += accepted_in_block
            cur_prev = teacher_prev                       # keep tokens/devices aligned

        if total_gen >= max_new:
            break


    wall_time = time.perf_counter() - start_t
    return {
        "accepted_tokens": accepted_total,  # total accepted (draft) tokens
        "alphas": alphas,                   # per-proposed-token alpha via overlap
        "teacher_calls": teacher_calls,
        "wall_time": wall_time,
    }

# ------------------------------
# Driver
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(apply_overrides(raw, parse_overrides(args.override)))
    set_seed(int(cfg.runtime.seed))

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer: prefer explicit; else teacher
    tok_name = cfg_get(cfg, "models.tokenizer", None) or cfg.models.teacher_model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ---- models
    dtype = parse_torch_dtype(cfg_get(cfg, "runtime.dtype", "bf16"))
    device = torch.device(cfg.runtime.device)
    device_map = str(cfg_get(cfg, "runtime.device_map", "auto"))
    t8 = bool(cfg_get(cfg, "runtime.teacher_8bit", False))

    teacher = _load_causal_lm(
        cfg.models.teacher_model,
        dtype=dtype, device_map=device_map,
        use_cache=bool(cfg.runtime.teacher_use_cache),
        load_in_8bit=t8
    )
    if teacher is None:
        raise ValueError("teacher_model must be provided for speculative decoding eval.")

    # we’ll load drafts on demand to keep memory lower
    model_specs = [
        ("base", cfg.models.base_model),
        ("kd",   cfg.models.kd_model),
        ("grpo", cfg.models.grpo_model),
    ]

    # ---- data
    if bool(cfg.data.use_pregen):
        prompts = _load_prompts_from_pregen(
            cfg.data.pregen_folder, str(cfg.data.split), cfg_get(cfg, "data.sample_max", None)
        )
    else:
        prompts = _load_prompts_from_hf(
            tokenizer,
            name=str(cfg.data.dataset_name),
            split=str(cfg.data.split),
            sample_max=cfg_get(cfg, "data.sample_max", None),
            keep_history=True,
            load_kwargs=cfg_get(cfg, "data.load_kwargs", {}),
        )
    if not prompts:
        raise RuntimeError("No prompts loaded for evaluation.")

    if cfg.eval.num_prompts is not None:
        prompts = prompts[: int(cfg.eval.num_prompts)]

    # ---- SD knobs
    K = int(cfg.specdec.K)
    max_new = int(cfg.specdec.max_new)
    temp = float(cfg.specdec.temperature)
    cap = float(cfg.specdec.acceptance_cap)
    max_input_len = int(cfg.data.max_input_len)
    log_every = int(cfg.eval.log_every)

    # ---- loop variants
    for label, spec in model_specs:
        if spec in (None, "", "null", "None"):
            print(f"[eval] Skipping {label}: no model provided.")
            continue

        # If adapter, try to infer base from adapter_config; else fall back to cfg.models.base_model
        fallback_base = cfg.models.base_model if _is_adapter_folder(spec) else None
        draft = _load_causal_lm(
            spec, dtype=dtype, device_map=device_map,
            use_cache=bool(cfg.runtime.draft_use_cache),
            fallback_base=fallback_base
        )
        print_model_layer_report(draft, title=f"Eval draft ({label})", limit=60, only_lora=False)

        totals = {
            "accepted_tokens": 0,
            "alphas_sum": 0.0,
            "alphas_count": 0,
            "teacher_calls": 0,
            "wall_time": 0.0,
            "samples": 0,
        }

        pbar = tqdm(range(len(prompts)), desc=f"[SD eval] {label}", ncols=100)
        for i in pbar:
            with open('./test.txt', 'a', encoding='utf-8') as f:
                f.write(f"{i}th inference start:\n")
            stats = _sd_simulate_one(
                tokenizer=tokenizer,
                draft=draft,
                teacher=teacher,
                prompt_text=prompts[i],
                K=K,
                max_new=max_new,
                temperature=temp,
                acceptance_cap=cap,
                device=device,
                max_input_len=max_input_len,
            )

            totals["accepted_tokens"] += int(stats["accepted_tokens"])
            totals["alphas_sum"]      += float(sum(stats["alphas"]))
            totals["alphas_count"]    += int(len(stats["alphas"]))
            totals["teacher_calls"]   += int(stats["teacher_calls"])
            totals["wall_time"]       += float(stats["wall_time"])
            totals["samples"]         += 1

            if (i+1) % max(1, log_every) == 0:
                mean_alpha = (totals["alphas_sum"] / max(1, totals["alphas_count"]))
                goodput    = (totals["accepted_tokens"] + totals["teacher_calls"]) / max(1.0, totals["teacher_calls"])
                pbar.set_postfix(alpha=f"{mean_alpha:.3f}", gp=f"{goodput:.3f}")

        # aggregate
        mean_alpha = (totals["alphas_count"] and totals["alphas_sum"] / totals["alphas_count"]) or 0.0
        goodput    = (totals["accepted_tokens"] + totals["teacher_calls"])/ max(1.0, totals["teacher_calls"])
        mean_accept= totals["accepted_tokens"] / max(1, totals["samples"])

        report = {
            "model_label": label,
            "model_spec": str(spec),
            "K": K,
            "max_new": max_new,
            "temperature": temp,
            "acceptance_cap": cap,
            "n_prompts": totals["samples"],
            "mean_alpha": mean_alpha,                      # avg acceptance prob per proposed token
            "mean_accepted_tokens": mean_accept,          # per prompt
            "goodput_tokens_per_teacher_call": goodput,   # accepted / teacher_calls
            "total_wall_time_sec": totals["wall_time"],
            "teacher_calls": totals["teacher_calls"],
        }

        out_path = out_dir / f"{label}_report.json"
        save_json(report, out_path)
        print(f"[DONE] {label} report -> {out_path}")

        # free draft before next variant
        del draft
        torch.cuda.empty_cache()
        gc.collect()

    print("[ALL DONE] SD eval complete.")

if __name__ == "__main__":
    main()
