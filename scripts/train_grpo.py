# scripts/train_grpo.py
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from src.common.config import (
    load_yaml_with_includes, to_attrdict, apply_overrides, parse_overrides,
    set_seed, save_cfg_lock, cfg_get, parse_torch_dtype,
)
from src.common.io import save_json, timestamp
from src.common.wandb_util import maybe_init_wandb, wandb_log, wandb_finish
from src.models.load import (
    load_models_for_eval_from_model_dir,
    load_tokenizer_from_training_cfg,
    mark_only_lora_trainable,
    print_model_layer_report,

)
from src.data.prompts import load_prompts_for_split, make_manual_splits, truncate_prompt_by_tokens
from src.training.optim import build_adamw
from src.training.schedule import make_warmup_cosine
from src.training.move import move_to_device_non_blocking
from src.reward.boundary import expected_alpha_and_goodput_from_logps


# ---------------------------- small prompt dataset ----------------------------

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return self.prompts[i]

def _prepare_prompts(tokenizer, prompts: List[str], max_input_len: int, S: int) -> List[str]:
    out: List[str] = []
    budget = max(1, max_input_len - 1)  # conservative cushion
    for p in prompts:
        out.append(truncate_prompt_by_tokens(tokenizer, p, budget))
    return out

def _left_pad_batch(ids_list: List[List[int]], pad_id: int, max_len: int|None=None) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """left-pad lists of ids => (input_ids[B,L], attn[B,L], true_lengths[B])."""
    Ls = [len(x) for x in ids_list]
    L = max(Ls) if max_len is None else max_len
    B = len(ids_list)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        n = min(len(x), L)
        ids[i, L - n : L] = torch.tensor(x[-n:], dtype=torch.long)
        att[i, L - n : L] = 1
    return ids, att, Ls

def _tokenize_prompts(tokenizer, prompts: List[str], max_input_len: int):
    enc = tokenizer(
        prompts, return_tensors=None, padding=False, truncation=True, max_length=max_input_len
    )
    ids_list = enc["input_ids"]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids, attn, lens = _left_pad_batch(ids_list, pad_id=pad_id)
    return input_ids, attn, lens, pad_id


# ------------------------------ GRPO core utils -------------------------------

@torch.no_grad()
def _sample_group(
    model, tokenizer, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor,
    group_size: int, max_new: int, temperature: float, eos_id: int|None
) -> Tuple[torch.Tensor, List[int]]:
    """
    Repeat each prompt group_size times and sample max_new tokens.
    Returns sequences [B*G, L_out] and list of prompt lengths (repeated G times).
    """
    B, L = prompt_ids.shape
    # Repeat for group sampling
    rep_ids  = prompt_ids.repeat_interleave(group_size, dim=0).contiguous()
    rep_attn = prompt_attn.repeat_interleave(group_size, dim=0).contiguous()

    gen = model.generate(
        inputs=rep_ids.to(model.device),
        attention_mask=rep_attn.to(model.device),
        do_sample=True,
        temperature=max(1e-6, float(temperature)),
        max_new_tokens=int(max_new),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=eos_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False,  # we'll recompute logprobs with grads later
    )
    seqs = gen.sequences  # [B*G, L_out] (left-padded)
    # prompt lengths replicate G times
    # Derive from attention (1s for tokens incl. prompt); stable since we padded left
    pr_lens = prompt_attn.sum(dim=1).tolist()
    pr_lens = [x for x in pr_lens for _ in range(group_size)]
    return seqs, pr_lens


def _gather_token_logps_from_forward(
    logits: torch.Tensor,    # [N, L, V]
    seqs: torch.Tensor,      # [N, L]
    pr_lens: List[int],      # len N, prompt token counts
    max_new: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract log-probs for the generated tokens y_{1:T} given the model's logits.
    We use the teacher-forcing alignment: logits at pos (p-1+t) predict token at (p+t).
    Returns (logp [N,T], mask [N,T]).
    """
    N, L, V = logits.shape
    device = logits.device
    T = int(max_new)

    # log softmax once
    logp_full = F.log_softmax(logits, dim=-1)  # [N,L,V]

    out_logp = torch.zeros((N, T), dtype=logits.dtype, device=device)
    mask = torch.zeros((N, T), dtype=logits.dtype, device=device)

    # true sequence lengths from padding
    att = (seqs != pad_id).long()             # [N,L]
    seq_lens = att.sum(dim=1)                 # [N]
    pr_lens_t = torch.tensor(pr_lens, device=device, dtype=torch.long)  # [N]
    gen_lens = (seq_lens - pr_lens_t).clamp(min=0, max=T)               # [N]

    for i in range(N):
        g = int(gen_lens[i])
        if g <= 0:
            continue
        p = int(pr_lens[i])
        pos = torch.arange(g, device=device)
        logits_idx = p - 1 + pos              # predicts token at p+pos
        tgt_ids = seqs[i, p + pos]
        out_logp[i, :g] = logp_full[i, logits_idx, tgt_ids]
        mask[i, :g] = 1.0

    return out_logp, mask

# ----------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_api_key", default=None)
    args = ap.parse_args()

    raw = load_yaml_with_includes(args.config)
    cfg = to_attrdict(apply_overrides(raw, parse_overrides(args.override)))
    set_seed(int(cfg.training.seed))

    out_dir = Path(cfg.grpo.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cfg_lock(raw, out_dir, filename="cfg.lock.yaml")

    # ------- Load finetuned draft (LoRA) + teacher using model_dir cfg
    kd_model_dir = Path(cfg.grpo.base_model_dir).expanduser().resolve()
    dtype = parse_torch_dtype(cfg_get(cfg, "training.dtype", "bf16"))

    try:
        tokenizer, draft, teacher, train_cfg, stage_kind, used_cfg_path = \
            load_models_for_eval_from_model_dir(
                kd_model_dir, dtype_name=str(dtype).split(".")[-1], device_map="auto"
            )
    except Exception as e:
        # Fallback: load tokenizer from kd cfg, load draft (LoRA) from dir + teacher by name in cfg.grpo.teacher
        # (this path shouldn't trigger if your load.py already handles kd cfg gracefully)
        print(f"[warn] primary model loader failed: {e}")
        # minimal fallback path
        from peft import PeftModel
        raw_kd = load_yaml_with_includes(kd_model_dir / "cfg.lock.yaml")
        kd_cfg = to_attrdict(raw_kd)
        tokenizer = load_tokenizer_from_training_cfg(kd_cfg)
        draft_base = AutoModelForCausalLM.from_pretrained(
            kd_cfg.models.draft, torch_dtype=dtype, device_map="auto"
        )
        draft = PeftModel.from_pretrained(draft_base, kd_model_dir)
        draft.config.use_cache = False
        draft.train()
        n_train, n_total = mark_only_lora_trainable(draft)
        print(f"[grpo] trainable params (LoRA only): {n_train:,} / {n_total:,}")
        print_model_layer_report(draft, title="GRPO draft (after freezing base)", limit=80, only_lora=True)
        teacher_name = cfg_get(cfg, "grpo.teacher", cfg_get(kd_cfg, "models.target", None)) \
                       or cfg_get(kd_cfg, "models.tokenizer", None)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, torch_dtype=dtype, device_map="auto"
        )
        teacher.config.use_cache = False
        teacher.eval()
        train_cfg, used_cfg_path = kd_cfg, kd_model_dir / "cfg.lock.yaml"
        stage_kind = "kd"

    # Report layers / LoRA injection
    if bool(cfg_get(cfg, "grpo.print_layers", True)):
        print_model_layer_report(draft,   title="GRPO draft (starting from KD)", limit=80, only_lora=True)
        print_model_layer_report(teacher, title="GRPO teacher (reference)",      limit=40, only_lora=True)

    # Optional reference for KL control: "teacher" | "draft_init" | "none"
    kl_ref = str(cfg_get(cfg, "grpo.kl_ref", "teacher")).lower()
    kl_coeff = float(cfg_get(cfg, "grpo.kl_coeff", 0.01))
    ref_model = None
    if kl_ref == "draft_init":
        # make a frozen copy (weights shared if LoRA, so we load a new base and attach adapters snapshot)
        from copy import deepcopy
        ref_model = deepcopy(draft).eval()
        for p in ref_model.parameters(): p.requires_grad_(False)
    elif kl_ref == "teacher":
        ref_model = teacher
    else:
        ref_model = None

    # ------- Build prompt pool (HF split) -------
    base_split = cfg.data.split
    prompts = load_prompts_for_split(tokenizer, cfg, split=base_split)
    if not prompts:
        train_prompts, val_prompts = make_manual_splits(tokenizer, cfg, seed=int(cfg.training.seed))
        prompts = train_prompts
    # Subsample if needed
    sample_max = cfg_get(cfg, "data.sample_max", None)
    if sample_max:
        prompts = prompts[: int(sample_max)]

    # A tiny dataset that yields prompts; dataloader shuffles each epoch
    ds = PromptDataset(prompts)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.grpo.batch_size),
        shuffle=True,
        num_workers=int(cfg_get(cfg, "grpo.num_workers", 2)),
        pin_memory=True,
        drop_last=True,
    )

    # ------- Optim & schedule (train only LoRA params) -------
    optim = build_adamw(
        draft,
        lr=float(cfg.grpo.lr),
        weight_decay=float(cfg.grpo.get("weight_decay", 0.0)),
        betas=tuple(cfg.grpo.get("betas", [0.9, 0.95])),
        eps=1e-8,
    )
    sched = make_warmup_cosine(
        optim,
        total_steps=int(cfg.grpo.total_steps),
        warmup_ratio=float(cfg.grpo.get("warmup_ratio", 0.05)),
        min_lr=float(cfg.grpo.get("min_lr", 0.0)),
    )

    # ------- W&B -------
    run = maybe_init_wandb(
        enabled=bool(args.wandb),
        api_key=args.wandb_api_key,
        project=cfg_get(cfg, "logging.project", "grpo-train"),
        name=cfg_get(cfg, "logging.name", "grpo_run"),
        config=json.loads(json.dumps(cfg, default=str)),
        tags=cfg_get(cfg, "logging.tags", None),
        group=cfg_get(cfg, "logging.group", None),
        mode=cfg_get(cfg, "logging.mode", None),
    )

    # ------- Train loop -------
    device = torch.device(cfg_get(cfg, "training.device", "cuda"))
    max_new = int(cfg.grpo.max_new)
    temperature = float(cfg.grpo.temperature)
    cap = float(cfg.grpo.acceptance_cap)
    group_size = int(cfg.grpo.group_size)
    grad_clip = float(cfg.training.get("max_grad_norm", 1.0))
    acc_steps = int(cfg.grpo.get("grad_accum_steps", 1))
    print_every = int(cfg.grpo.log_every)

    draft.train()
    pbar = tqdm(range(int(cfg.grpo.total_steps)), desc="GRPO", ncols=100)
    it = iter(dl)

    for step in pbar:
        try:
            batch_prompts = [next(it) for _ in range(1)]
            # DataLoader returns strings already
            batch_prompts = batch_prompts[0]
        except StopIteration:
            it = iter(dl)
            batch_prompts = next(it)

        # Tokenize prompts (left-pad) with a safe budget (context)
        # We’ll clip earlier in data_gen/eval the same way
        prep_prompts = _prepare_prompts(
            tokenizer, batch_prompts, max_input_len=int(cfg.data.max_input_len), S=max_new
        )
        prompt_ids, prompt_attn, pr_lens, pad_id = _tokenize_prompts(
            tokenizer, prep_prompts, max_input_len=int(cfg.data.max_input_len)
        )

        # 1) Sample group completions (no grad)
        eos_id = tokenizer.eos_token_id
        with torch.no_grad():
            seqs, rep_pr_lens = _sample_group(
                draft.eval(), tokenizer, prompt_ids, prompt_attn,
                group_size=group_size, max_new=max_new,
                temperature=temperature, eos_id=eos_id
            )
        draft.train()  # back to train for gradient passes

        N, L = seqs.shape  # N = B * group_size

        # 2) Compute draft logprobs on sampled tokens WITH grad
        #    One forward on the full sequences to pull the next-token logits
        att = (seqs != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long().to(device)
        out = draft(input_ids=seqs.to(device), attention_mask=att, use_cache=False)
        d_logp_NT, gen_mask_NT = _gather_token_logps_from_forward(
            out.logits, seqs.to(device), rep_pr_lens, max_new=max_new, pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id)
        )
        if gen_mask_NT.sum().item() == 0:
            # nothing generated (should be rare); skip to next batch
            optim.zero_grad(set_to_none=True)
            continue

        # 3) Teacher logprobs on same tokens (no grad)
        with torch.no_grad():
            tout = teacher(input_ids=seqs.to(teacher.device), attention_mask=att.to(teacher.device), use_cache=False)
            t_logp_NT, _ = _gather_token_logps_from_forward(
                tout.logits.to(d_logp_NT.device), seqs.to(d_logp_NT.device), rep_pr_lens, max_new=max_new,
                pad_id=(tokenizer.pad_token_id or tokenizer.eos_token_id)
            )
        # 4) Optional ref logprobs for KL
        if kl_ref == "teacher":
            ref_logp_NT = t_logp_NT
        elif kl_ref == "draft_init":
            with torch.no_grad():
                rout = ref_model(input_ids=seqs.to(ref_model.device), attention_mask=att.to(ref_model.device), use_cache=False)
                ref_logp_NT, _ = _gather_token_logps_from_forward(
                    rout.logits.to(d_logp_NT.device), seqs.to(d_logp_NT.device), rep_pr_lens, max_new=max_new
                )
        else:
            ref_logp_NT = None

        # 5) Rewards from expected acceptance / goodput
        rewards = expected_alpha_and_goodput_from_logps(
            draft_logp=d_logp_NT.detach(), teacher_logp=t_logp_NT, mask=gen_mask_NT, acceptance_cap=cap
        )
        # scalar per-sample reward used by GRPO
        reward_key = str(cfg_get(cfg, "grpo.reward_key", "goodput")).lower()
        if reward_key == "alpha":
            r_N = rewards["alpha_mean"]
        elif reward_key == "accepted_tokens":
            r_N = rewards["accepted_tokens"]
        else:
            r_N = rewards["goodput"]

        # 6) Group Relative baseline (mean within each prompt-group)
        B = prompt_ids.size(0)
        G = group_size
        assert B * G == N, (B, G, N)
        r_BG = r_N.view(B, G)
        base_B = r_BG.mean(dim=1, keepdim=True)           # [B,1]
        adv_BG = (r_BG - base_B)                           # [B,G]
        adv_N = adv_BG.view(N)

        # Policy loss = - E[adv * sum_t logpi_t / valid_t]
        valid_t = gen_mask_NT.sum(dim=1).clamp_min(1.0)    # [N]
        sum_logpi = (d_logp_NT * gen_mask_NT).sum(dim=1)   # [N]
        pol_loss = -(adv_N.detach() * (sum_logpi / valid_t)).mean()

        # KL control (optional)
        if ref_logp_NT is not None and kl_coeff > 0:
            kl_per_t = (d_logp_NT - ref_logp_NT) * gen_mask_NT  # E_{pi} log pi/ref
            kl_term = kl_per_t.sum(dim=1) / valid_t
            kl_loss = kl_coeff * kl_term.mean()
        else:
            kl_loss = torch.zeros((), device=d_logp_NT.device)

        loss = pol_loss + kl_loss
        (loss / acc_steps).backward()

        if (step + 1) % acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(draft.parameters(), grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        # ---- logging
        if step % print_every == 0:
            logs = {
                "grpo/loss": float(loss.detach().cpu()),
                "grpo/policy_loss": float(pol_loss.detach().cpu()),
                "grpo/kl_loss": float(kl_loss.detach().cpu()),
                "grpo/lr": sched.get_last_lr()[0],
                "grpo/alpha_mean": float(rewards["alpha_mean"].mean().detach().cpu()),
                "grpo/goodput": float(rewards["goodput"].mean().detach().cpu()),
                "grpo/reject_rate": float(rewards["reject_rate"].mean().detach().cpu()),
                "grpo/mean_len": float(valid_t.mean().detach().cpu()),
            }
            wandb_log(logs, step=step)
            pbar.set_postfix(loss=f"{logs['grpo/loss']:.3f}",
                             gp=f"{logs['grpo/goodput']:.3f}",
                             alpha=f"{logs['grpo/alpha_mean']:.3f}")

    # ------- Save adapters + summary -------
    outdir = out_dir / "lora"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        draft.save_pretrained(str(outdir))
    except Exception:
        torch.save(draft.state_dict(), outdir / "pytorch_model.bin")

    save_json(
        {
            "finished_at": timestamp(),
            "steps": int(cfg.grpo.total_steps),
            "base_model_dir": str(kd_model_dir),
            "out_dir": str(outdir),
            "kl_ref": kl_ref,
            "reward_key": reward_key,
        },
        out_dir / "train_summary.json",
    )
    wandb_finish()
    print(f"[DONE] GRPO LoRA saved at: {outdir}")


if __name__ == "__main__":
    main()
