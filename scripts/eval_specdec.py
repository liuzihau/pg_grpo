# scripts/eval_specdec.py
from __future__ import annotations
import os, argparse, random, json, csv, datetime
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# small config helpers
# -----------------------------
def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = _deep_update(a[k], v)
        else:
            a[k] = v
    return a

def load_yaml_with_includes(path: str) -> Dict[str, Any]:
    import yaml
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for rel in includes:
        inc_path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        sub = load_yaml_with_includes(inc_path)
        _deep_update(merged, sub)
    _deep_update(merged, cfg)
    return merged

class Attr(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

def to_attrdict(d: Dict[str, Any]) -> Any:
    a = Attr()
    for k, v in d.items():
        a[k] = to_attrdict(v) if isinstance(v, dict) else v
    return a

def cfg_get(obj, path, default=None):
    cur = obj
    for key in path.split("."):
        cur = (cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None))
        if cur is None:
            return default
    return cur

def parse_torch_dtype(name: str):
    n = str(name).lower()
    if n in ("bf16","bfloat16","bfloat"): return torch.bfloat16
    if n in ("fp16","float16","half"):    return torch.float16
    if n in ("fp32","float32","float"):   return torch.float32
    return torch.float32

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)


def first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return None

def resolve_model_names(cfg, args=None):
    # Prefer CLI overrides if provided
    draft_cli    = getattr(args, "draft_name", None) if args else None
    teacher_cli  = getattr(args, "teacher_name", None) if args else None
    tok_cli      = getattr(args, "tokenizer_name", None) if args else None

    draft_name = first_nonempty(
        draft_cli,
        cfg_get(cfg, "models.draft"),
        cfg_get(cfg, "draft.name"),
        cfg_get(cfg, "model_draft.name"),
        cfg_get(cfg, "models.tokenizer"),   # last-resort fallback
    )
    teacher_name = first_nonempty(
        teacher_cli,
        cfg_get(cfg, "models.target"),
        cfg_get(cfg, "target.name"),
        cfg_get(cfg, "model_target.name"),
    )
    tok_name = first_nonempty(
        tok_cli,
        cfg_get(cfg, "models.tokenizer"),
        cfg_get(cfg, "tokenizer.name"),
        cfg_get(cfg, "draft.name"),
        cfg_get(cfg, "models.draft"),
        teacher_name,  # absolute last fallback
    )

    missing = []
    if not draft_name:   missing.append("draft")
    if not teacher_name: missing.append("teacher")
    if not tok_name:     missing.append("tokenizer")
    if missing:
        raise RuntimeError(
            "Missing model names in config ({}). Checked multiple paths.\n"
            "You can also pass --draft_name / --teacher_name / --tokenizer_name.\n"
            "Loaded config snippets:\n"
            f"  models: {dict(cfg.get('models', {})) if isinstance(cfg.get('models', {}), dict) else cfg.get('models', {})}\n"
            f"  draft:  {dict(cfg.get('draft', {}))  if isinstance(cfg.get('draft', {}),  dict) else cfg.get('draft', {})}\n"
            f"  target: {dict(cfg.get('target', {})) if isinstance(cfg.get('target', {}), dict) else cfg.get('target', {})}\n"
        )
    return tok_name, draft_name, teacher_name

# -----------------------------
# data: prompts from HF chat set
# -----------------------------
def hf_prompts(tokenizer, cfg, split: str) -> List[str]:
    name = cfg.data.hf_name
    field = cfg.data.messages_field
    keep_history = bool(cfg.data.keep_history)
    lw = cfg.data.get("load_kwargs", {}) or {}
    for sp in [split, "validation", "test", cfg.data.split]:
        try:
            ds = load_dataset(name, split=sp, **lw)
        except Exception:
            continue
        items = []
        print(f"Start loading prompt... {len(ds)} in total")
        for i, rec in enumerate(ds[:500]):
            if (i + 1) % (len(ds[:500])//10) == 0:
                print(f"loading {i}th prompt...")
            msgs = rec.get(field, rec.get("conversations", rec.get("conversation", [])))
            if not isinstance(msgs, list):
                continue
            norm = []
            for m in msgs:
                if isinstance(m, dict):
                    role = str(m.get("role", m.get("from", ""))).lower()
                    content = m.get("content", m.get("value", ""))
                    if content:
                        norm.append({"role": role, "content": content})
            # keep up to and including last user turn
            last_user = None
            for i in range(len(norm)-1, -1, -1):
                if norm[i]["role"] == "user":
                    last_user = i; break
            if last_user is None:
                continue
            kept = norm[: last_user+1] if keep_history else [norm[last_user]]
            try:
                prompt = tokenizer.apply_chat_template(
                    kept, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = [f"{m['role'].capitalize()}: {m['content']}" for m in kept]
                parts.append("Assistant:")
                prompt = "\n".join(parts)
            items.append(prompt)
        if items:
            return items
    return []

# -----------------------------
# padding helpers
# -----------------------------
def left_pad_to_batch(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of 1D Long tensors into [B, Lmax] + attention mask."""
    Lmax = max(x.numel() for x in seqs) if seqs else 1
    B = len(seqs)
    input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.numel()
        input_ids[i, Lmax-L:] = s
        attn[i, Lmax-L:] = 1
    return input_ids, attn

def clip_ctx(ids: torch.Tensor, max_len: int) -> torch.Tensor:
    if ids.numel() <= max_len: return ids
    return ids[-max_len:]

# -----------------------------
# speculative decoding (chunked K)
# -----------------------------
@torch.no_grad()
def eval_spec_batch(
    *,
    tokenizer,
    draft: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    prompts: List[str],
    K: int,
    max_new_tokens: int,
    temperature: float,
    max_input_len: int,
) -> Dict[str, float]:
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_input_len, return_tensors="pt"
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Extract non-pad slices per row
    ids = enc["input_ids"]
    att = enc["attention_mask"]
    cur_ids_list = []
    for i in range(ids.size(0)):
        last_idx = (att[i] == 1).nonzero()
        if last_idx.numel() == 0:
            cur_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
        else:
            cur_ids_list.append(ids[i, : last_idx[-1].item()+1])

    B = len(cur_ids_list)
    accepted_total = torch.zeros(B, dtype=torch.long, device=device)
    compared_total = torch.zeros(B, dtype=torch.long, device=device)
    teacher_calls  = torch.zeros(B, dtype=torch.long, device=device)
    done           = torch.zeros(B, dtype=torch.bool, device=device)

    while True:
        if done.all(): break

        cur_ids_list = [clip_ctx(x.to(device), max_input_len) for x in cur_ids_list]
        inp, attn = left_pad_to_batch(cur_ids_list, pad_id)
        inp = inp.to(device); attn = attn.to(device)

        do_sample = temperature > 0.0
        gen_out = draft.generate(
            input_ids=inp,
            attention_mask=attn,
            max_new_tokens=K,
            do_sample=do_sample,
            temperature=max(temperature, 1e-6) if do_sample else None,
            top_p=None if not do_sample else 1.0,
            use_cache=True,
            pad_token_id=pad_id,
        )
        proposed_list = []
        for i in range(B):
            Lpad = (attn[i] == 1).sum().item()
            full = gen_out[i]
            proposed = full[Lpad: Lpad + K]
            proposed_list.append(proposed.to(device))

        teacher_inp_list = []
        for i in range(B):
            if done[i]:
                teacher_inp_list.append(cur_ids_list[i]); continue
            merged = torch.cat([cur_ids_list[i], proposed_list[i]], dim=0)
            teacher_inp_list.append(clip_ctx(merged, max_input_len))
        t_inp, t_attn = left_pad_to_batch(teacher_inp_list, pad_id)
        t_inp = t_inp.to(device); t_attn = t_attn.to(device)

        tout = teacher(input_ids=t_inp, attention_mask=t_attn, use_cache=False)
        t_logits = tout.logits

        t_top1_list = []
        for i in range(B):
            if done[i]:
                t_top1_list.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            L_eff = (t_attn[i] == 1).sum().item()
            T_i = min(K, proposed_list[i].numel())
            if T_i == 0:
                t_top1_list.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            slice_logits = t_logits[i, L_eff - T_i : L_eff, :]
            t_top1 = slice_logits.argmax(dim=-1)
            t_top1_list.append(t_top1)

        for i in range(B):
            if done[i]: continue
            prop = proposed_list[i]; t_top1 = t_top1_list[i]
            T_i = t_top1.numel()
            if T_i == 0:
                done[i] = True; continue

            acc_in_chunk, mismatch_seen = 0, False
            for t in range(T_i):
                compared_total[i] += 1
                if t_top1[t].item() == prop[t].item() and not mismatch_seen:
                    acc_in_chunk += 1
                else:
                    mismatch_seen = True
                    break

            if acc_in_chunk > 0:
                accepted_total[i] += acc_in_chunk
                cur_ids_list[i] = torch.cat([cur_ids_list[i], prop[:acc_in_chunk].detach()], dim=0)

            teacher_calls[i] += 1
            if accepted_total[i].item() >= max_new_tokens:
                done[i] = True

        if (accepted_total >= max_new_tokens).all():
            break

    compared_total = compared_total.clamp_min(1)
    alpha_accept = (accepted_total.float() / compared_total.float()).mean().item()
    goodput      = (accepted_total.float() / teacher_calls.clamp_min(1).float()).mean().item()
    avg_span     = (accepted_total.float() / teacher_calls.clamp_min(1).float()).mean().item()
    reject_rate  = (1.0 - (accepted_total.float() / compared_total.float())).mean().item()

    return {
        "alpha_accept": alpha_accept,
        "goodput_tokens_per_teacher_call": goodput,
        "avg_accepted_span": avg_span,
        "reject_rate": reject_rate,
        "mean_accepted_tokens": accepted_total.float().mean().item(),
        "mean_teacher_calls": teacher_calls.float().mean().item(),
    }

# -----------------------------
# load models
# -----------------------------
def load_draft_with_optional_lora(base_name: str, lora_path: str | None, dtype, device_map="auto"):
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map=device_map)
    if lora_path:
        try:
            base = PeftModel.from_pretrained(base, lora_path)
        except Exception:
            sd_path = os.path.join(lora_path, "pytorch_model.bin")
            if os.path.exists(sd_path):
                sd = torch.load(sd_path, map_location="cpu")
                base.load_state_dict(sd, strict=False)
    if hasattr(base, "config"):
        base.config.use_cache = True
        if getattr(base.config, "pad_token_id", None) is None and hasattr(base.config, "eos_token_id"):
            base.config.pad_token_id = base.config.eos_token_id
    base.eval()
    return base

# -----------------------------
# wandb + local IO helpers
# -----------------------------
def maybe_wandb_init(enable: bool, api_key: str | None, project: str, run_name: str, config: dict):
    if not enable: return None
    import wandb
    if api_key:
        wandb.login(key=api_key)
    return wandb.init(project=project, name=run_name, config=config)

def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_csv(path: str, row: dict, header_order: List[str]):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if new_file:
            w.writeheader()
        w.writerow(row)

def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")

# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapters (for draft)")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)

    # logging
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    ap.add_argument("--wandb_api_key", type=str, default=None)
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="Local output dir for JSON/CSV")

    ap.add_argument("--draft_name", type=str, default=None, help="Override draft model id")
    ap.add_argument("--teacher_name", type=str, default=None, help="Override teacher model id")
    ap.add_argument("--tokenizer_name", type=str, default=None, help="Override tokenizer id")

    args = ap.parse_args()

    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(raw)
    set_seed(int(cfg_get(cfg, "training.seed", 1234)))

    device = torch.device(cfg_get(cfg, "training.device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype  = parse_torch_dtype(cfg_get(cfg, "training.dtype", "bf16"))
    tok_name, draft_name, teacher_name = resolve_model_names(cfg, args)
    
    print(f"[eval_specdec] tokenizer={tok_name} | draft={draft_name} | teacher={teacher_name}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # teacher
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto")
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    teacher.eval()

    # drafts
    draft_base = load_draft_with_optional_lora(draft_name, None, dtype=dtype, device_map="auto")
    draft_lora = load_draft_with_optional_lora(draft_name, args.lora_path, dtype=dtype, device_map="auto") if args.lora_path else None

    # prompts
    prompts_all = hf_prompts(tokenizer, cfg, args.split)
    print(f"Total prompts: {prompts_all}")
    if not prompts_all:
        raise RuntimeError(f"No prompts from HF split='{args.split}'. Check configs/data.yaml.")
    if args.num_samples > 0:
        idxs = random.sample(range(len(prompts_all)), k=min(args.num_samples, len(prompts_all)))
        prompts = [prompts_all[i] for i in idxs]
    else:
        prompts = prompts_all

    # eval settings
    max_input_len = int(cfg_get(cfg, "data.max_input_len", 1024))
    B = int(cfg_get(cfg, "eval.batch_size", 8))
    K = int(args.K); max_new = int(args.max_new); temp = float(args.temperature)

    # local out
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("outputs", "eval_spec", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "eval_manifest.json"), {
        "config": raw, "split": args.split, "num_samples": len(prompts),
        "K": K, "max_new": max_new, "temperature": temp, "out_dir": out_dir,
        "lora_path": args.lora_path,
    })

    # wandb
    print("Start initializing wandb...")
    w_project = args.wandb_project or cfg_get(cfg, "logging.project", "specdec-eval")
    w_runname = args.wandb_run_name or f"spec_eval_K{K}_N{len(prompts)}_{timestamp}"
    wb = maybe_wandb_init(args.wandb, args.wandb_api_key, w_project, w_runname, {
        "K": K, "max_new": max_new, "temperature": temp, "num_samples": len(prompts),
        "draft": draft_name, "teacher": teacher_name, "lora_path": args.lora_path,
        "split": args.split
    })
    print("wandb ok")

    headers = ["model_tag", "alpha_accept", "goodput_tokens_per_teacher_call",
               "avg_accepted_span", "reject_rate", "mean_accepted_tokens", "mean_teacher_calls"]

    def run_model(draft_model, tag: str):
        metrics_accum = {k: 0.0 for k in headers if k != "model_tag"}
        count = 0
        csv_path = os.path.join(out_dir, f"{slugify(tag)}_batches.csv")

        pbar = tqdm(range(0, len(prompts), B), desc=f"SpecEval [{tag}]", ncols=100)
        for off in pbar:
            batch_prompts = prompts[off: off+B]
            m = eval_spec_batch(
                tokenizer=tokenizer,
                draft=draft_model,
                teacher=teacher,
                device=device,
                prompts=batch_prompts,
                K=K,
                max_new_tokens=max_new,
                temperature=temp,
                max_input_len=max_input_len,
            )
            for k in metrics_accum:
                metrics_accum[k] += m[k] * len(batch_prompts)
            count += len(batch_prompts)
            pbar.set_postfix(alpha=f"{m['alpha_accept']:.3f}", gp=f"{m['goodput_tokens_per_teacher_call']:.2f}")

            # per-batch logging
            row = {"model_tag": tag, **m}
            append_csv(csv_path, row, headers)
            if wb is not None:
                import wandb
                wandb.log({f"{tag}/alpha_accept": m["alpha_accept"],
                           f"{tag}/goodput": m["goodput_tokens_per_teacher_call"],
                           f"{tag}/avg_span": m["avg_accepted_span"],
                           f"{tag}/reject_rate": m["reject_rate"],
                           f"{tag}/mean_accepted_tokens": m["mean_accepted_tokens"],
                           f"{tag}/mean_teacher_calls": m["mean_teacher_calls"]})

        for k in metrics_accum:
            metrics_accum[k] /= max(1, count)

        summary = {"model_tag": tag, **metrics_accum}
        # save local summary
        save_json(os.path.join(out_dir, f"{slugify(tag)}_summary.json"), summary)

        # wandb summary
        if wb is not None:
            import wandb
            wandb.log({f"{tag}/alpha_accept_mean": summary["alpha_accept"],
                       f"{tag}/goodput_mean": summary["goodput_tokens_per_teacher_call"],
                       f"{tag}/avg_span_mean": summary["avg_accepted_span"],
                       f"{tag}/reject_rate_mean": summary["reject_rate"]})
        print(f"\n=== {tag} RESULTS ===")
        for k, v in summary.items():
            if k == "model_tag": continue
            print(f"{k:>32s}: {v:.6f}")
        print("")
        return summary

    # Run base + LoRA (if provided)
    
    _ = run_model(draft_base, "BASE_DRAFT")
    if args.lora_path:
        _ = run_model(draft_lora, "LORA_DRAFT")

    if wb is not None:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
