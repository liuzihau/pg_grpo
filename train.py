
from __future__ import annotations
import argparse, math, random, statistics, os
import torch, torch.nn.functional as F

from utils import load_yaml, set_seed, get_device, to_attrdict
from data import PromptOnlyDataset
from models import load_tokenizer, load_target, load_draft
from lora_setup import attach_lora
from rewards import accept_prob, survival as survival_fn
from grpo import grpo_loss

# --------- Optional tqdm ---------
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# --------- Optional W&B ---------
class _NoopWB:
    def __init__(self): self.run=None
    def init(self, **kw): return self
    def log(self, *a, **k): pass
    def finish(self): pass
    class Table:
        def __init__(self, columns): self.columns=columns; self._rows=[]
        def add_data(self, *row): self._rows.append(row)
    class Histogram:
        def __init__(self, arr): self.arr=list(arr)
try:
    import wandb
except Exception:
    wandb = _NoopWB()

def cfg_get(obj, path, default):
    """Nested getattr for AttrDict/dict with default, using dotted path."""
    cur = obj
    for key in path.split("."):
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def wb_init(cfg, cfg_raw):
    use = cfg_get(cfg, "logging.use_wandb", False)
    mode = cfg_get(cfg, "logging.mode", "disabled")
    if (not use) or mode == "disabled":
        return _NoopWB()
    run = wandb.init(
        project=cfg_get(cfg, "logging.project", "grpo-prefix-growth"),
        entity=cfg_get(cfg, "logging.entity", None),
        name=cfg_get(cfg, "logging.run_name", None),
        mode=mode,
        config=cfg_raw
    )
    return wandb

def wb_log(d, step=None):
    try:
        if step is None: wandb.log(d)
        else: wandb.log(d, step=step)
    except Exception:
        pass

# --------- KV helpers ---------
@torch.no_grad()
def init_kv(model, input_ids, attention_mask=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return {"past": out.past_key_values, "attn": attention_mask, "last_tok": input_ids[:, -1:]}

@torch.no_grad()
def extend_kv(model, state, new_tokens):
    out = model(input_ids=new_tokens, attention_mask=state["attn"], past_key_values=state["past"], use_cache=True)
    state["past"] = out.past_key_values
    state["last_tok"] = new_tokens[:, -1:]
    return state

@torch.no_grad()
def ref_logp_on_path(ref_model, path_tokens, kv_state):
    S, K = path_tokens.size()
    step_kv = [kv_state["past"] for _ in range(S)]
    attn = kv_state["attn"]
    out_logp = torch.zeros((S, K), device=path_tokens.device)
    for s in range(S):
        for t in range(K):
            tok = path_tokens[s:s+1, t:t+1]
            out = ref_model(input_ids=tok, attention_mask=attn, past_key_values=step_kv[s], use_cache=True)
            step_kv[s] = out.past_key_values
            logp = F.log_softmax(out.logits[:, -1, :], dim=-1)
            out_logp[s, t] = logp.gather(1, tok).squeeze(1)
    return out_logp

def nucleus_sample_probs(logits, top_p, temperature):
    logits = logits / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumprobs > top_p).float().argmax(dim=-1)
    vocab = probs.size(-1)
    ar = torch.arange(vocab, device=probs.device).unsqueeze(0).expand_as(probs)
    mask_sorted = ar > cutoff.unsqueeze(1)
    mask_vocab = torch.zeros_like(mask_sorted, dtype=torch.bool)
    mask_vocab.scatter_(1, sorted_idx, mask_sorted)
    masked_probs = torch.where(mask_vocab, torch.zeros_like(probs), probs)
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
    return masked_probs

def sample_S_chunks(draft, kv_state, S, K_chunk, temperature, top_p):
    device = kv_state["last_tok"].device
    step_kv = [kv_state["past"] for _ in range(S)]
    attn = kv_state["attn"]
    tokens = torch.empty((S, K_chunk), dtype=torch.long, device=device)
    logps  = torch.empty((S, K_chunk), dtype=torch.float32, device=device)
    last_inputs = [kv_state["last_tok"].clone() for _ in range(S)]

    for t in range(K_chunk):
        logits_rows = []
        for s in range(S):
            out = draft(input_ids=last_inputs[s], attention_mask=attn, past_key_values=step_kv[s], use_cache=True)
            step_kv[s] = out.past_key_values
            logits_rows.append(out.logits[:, -1, :])  # [1,V]
        logits = torch.cat(logits_rows, dim=0)  # [S,V]
        masked_probs = nucleus_sample_probs(logits, top_p, temperature)
        next_tokens = torch.multinomial(masked_probs, num_samples=1)  # [S,1]
        next_logp = torch.log(torch.gather(F.softmax(logits, dim=-1), 1, next_tokens).clamp_min(1e-12))  # [S,1]
        for s in range(S):
            tok = next_tokens[s:s+1, :]
            out = draft(input_ids=tok, attention_mask=attn, past_key_values=step_kv[s], use_cache=True)
            step_kv[s] = out.past_key_values
            last_inputs[s] = tok
        tokens[:, t:t+1] = next_tokens
        logps[:, t:t+1] = next_logp

    step_state = {"past": step_kv, "attn": attn, "last_tok": torch.stack([li.squeeze(0) for li in last_inputs], dim=0)}
    return tokens, logps, step_state

@torch.no_grad()
def target_score_chunks(target, kv_state, prop_tokens):
    S, K = prop_tokens.size()
    step_kv = [kv_state["past"] for _ in range(S)]
    attn = kv_state["attn"]
    q_on_path = torch.empty((S, K), dtype=torch.float32, device=prop_tokens.device)
    for s in range(S):
        for t in range(K):
            tok = prop_tokens[s:s+1, t:t+1]
            out = target(input_ids=tok, attention_mask=attn, past_key_values=step_kv[s], use_cache=True)
            step_kv[s] = out.past_key_values
            logp = F.log_softmax(out.logits[:, -1, :], dim=-1)
            q_on_path[s, t] = logp.gather(1, tok).exp().squeeze(1)
    return q_on_path, {"past": step_kv, "attn": attn, "last_tok": prop_tokens[:, -1:].contiguous()}

def train_on_prompt(prompt_ids, models, cfg, optim, tokenizer, global_step, pbar=None):
    target, draft, ref_draft = models["target"], models["draft"], models["ref"]
    S, K_chunk, M = cfg.training.S, cfg.training.K_chunk, cfg.training.M
    beta = cfg.training.kl_beta
    alpha_floor = cfg.reward.alpha_floor

    attn = (prompt_ids != tokenizer.pad_token_id).long()
    tgt_kv = init_kv(target, prompt_ids, attn)
    drf_kv = init_kv(draft, prompt_ids, attn)
    ref_kv = init_kv(ref_draft, prompt_ids, attn)

    accepted = 0
    steps = 0
    inner_total = max(1, math.ceil(M / max(1, K_chunk)))
    # inner tqdm
    inner_bar = None
    try:
        if cfg_get(cfg, "progress.use_tqdm", True) and tqdm is not None and pbar is None:
            inner_bar = tqdm(total=inner_total, leave=False, desc="prompt")
    except Exception:
        inner_bar = None

    while accepted < M and steps < cfg.training.max_steps:
        steps += 1; global_step += 1
        # 1) Propose S chunks
        prop_tokens, logp_draft, drf_step = sample_S_chunks(
            draft, drf_kv, S, K_chunk, cfg.training.temperature, cfg.training.top_p
        )

        # 2) Target scoring
        q_on_path, tgt_step = target_score_chunks(target, tgt_kv, prop_tokens)
        p_on_path = logp_draft.exp().clamp_min(1e-12)

        # 3) EAL reward + per-step acceptance lengths
        a = accept_prob(p_on_path, q_on_path, floor=alpha_floor)  # [S,K]
        surv = survival_fn(a)                                     # [S,K]
        R = (surv * a).sum(dim=1)                                 # [S] = EAL per sequence
        eal_mean = float(R.mean().detach().cpu())
        eal_std  = float(R.std(unbiased=False).detach().cpu())
        adv = (R - R.mean()).detach()

        # 4) GRPO loss (+ KL)
        ref_logp = ref_logp_on_path(ref_draft, prop_tokens, ref_kv)
        loss, stats = grpo_loss(logp_draft, ref_logp, adv, surv, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), cfg.training.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        # 5) Grow prefix (best-of-S) and compute observed acceptance len
        s_best = int(torch.argmax(R))
        a_best = a[s_best]             # [K]
        tok_best = prop_tokens[s_best] # [K]
        bern = torch.rand_like(a_best)
        keep = (bern < a_best).float()
        keep_prefix = torch.cumprod(keep + 1e-9, dim=0)
        accepted_len = int((keep_prefix > 0).sum().item())

        if accepted_len > 0:
            ext = tok_best[:accepted_len].unsqueeze(0)  # [1,T]
            tgt_kv = extend_kv(target, tgt_kv, ext)
            drf_kv = extend_kv(draft, drf_kv, ext)
            ref_kv = extend_kv(ref_draft, ref_kv, ext)
            accepted += accepted_len

        # ---- Logging ----
        if (global_step % cfg_get(cfg, "logging.log_every", 10)) == 0:
            wb_log({
                "loss/total": float(stats["loss"]),
                "loss/policy": float(stats["policy"]),
                "loss/kl": float(stats["kl"]),
                "reward/mean": eal_mean,
                "reward/std":  eal_std,
                # "average acceptance length within GRPO sequences":
                "accept/eal_mean": eal_mean,
                "accept/observed_len": accepted_len,
                "accept/observed_total": accepted
            }, step=global_step)

        # tqdm progress
        try:
            if inner_bar is not None:
                inner_bar.set_postfix({"acc_len": accepted_len, "acc_total": f"{accepted}/{M}", "loss": f"{float(stats['loss']):.3f}", "R": f"{eal_mean:.2f}"})
                if accepted_len > 0:
                    inner_bar.update(1)
            if pbar is not None:
                pbar.set_postfix_str(f"acc_total {accepted}/{M}")
        except Exception:
            pass

    try:
        if inner_bar is not None:
            inner_bar.close()
    except Exception:
        pass
    return global_step

@torch.no_grad()
def eval_specdec(prompts, tok, models, cfg):
    """Real speculative decoding: 1 token at a time; record real accepted length and per-position alpha."""
    target, draft = models["target"], models["draft"]
    device = next(draft.parameters()).device
    rnd = random.Random(cfg_get(cfg, "eval.seed", 12345))

    idxs = list(range(len(prompts)))
    rnd.shuffle(idxs)
    num_prompts = cfg_get(cfg, "eval.num_prompts", 16)
    if num_prompts and num_prompts > 0:
        idxs = idxs[:num_prompts]

    K = cfg_get(cfg, "eval.K_eval_max", 9)
    accept_lens = []
    alpha_pos_sum = torch.zeros(K, dtype=torch.float64, device=device)
    alpha_pos_cnt = torch.zeros(K, dtype=torch.float64, device=device)
    alpha_means_per_prompt = []

    table = None
    if cfg_get(cfg, "eval.log_alphas_table", True) and hasattr(wandb, "Table"):
        table = wandb.Table(columns=["prompt_id", "pos", "alpha_t"])

    for j, idx in enumerate(idxs):
        enc = tok(prompts[idx], return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = (input_ids != tok.pad_token_id).long()

        tgt_kv = init_kv(target, input_ids, attn)
        drf_kv = init_kv(draft,  input_ids, attn)

        alphas = torch.full((K,), float('nan'), device=device, dtype=torch.float32)
        real_len = 0
        rejected = False

        for t in range(K):
            # Draft one token
            out = draft(input_ids=drf_kv["last_tok"], attention_mask=drf_kv["attn"], past_key_values=drf_kv["past"], use_cache=True)
            drf_kv["past"] = out.past_key_values
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            masked = nucleus_sample_probs(out.logits[:, -1, :], cfg_get(cfg, "eval.top_p", 0.9), cfg_get(cfg, "eval.temperature", 0.8))
            tok_next = torch.multinomial(masked, num_samples=1)  # [1,1]

            # p_t and q_t for that sampled token
            p_t = probs.gather(1, tok_next).clamp_min(1e-12)  # [1,1]
            out_t = target(input_ids=tok_next, attention_mask=tgt_kv["attn"], past_key_values=tgt_kv["past"], use_cache=True)
            tgt_kv["past"] = out_t.past_key_values
            q_t = F.softmax(out_t.logits[:, -1, :], dim=-1).gather(1, tok_next).clamp_min(1e-12)

            a_t = torch.minimum(torch.ones_like(p_t), (q_t / p_t)).squeeze().float()
            alphas[t] = a_t

            # real accept decision
            if not rejected:
                u = torch.rand((), device=device)
                if u < a_t:
                    drf_kv = extend_kv(draft, drf_kv, tok_next)
                    tgt_kv = extend_kv(target, tgt_kv, tok_next)
                    real_len += 1
                else:
                    rejected = True
                    if not cfg_get(cfg, "eval.continue_after_reject", True):
                        break
            # EOS handling
            if cfg_get(cfg, "eval.stop_on_eos", True) and tok_next.item() == tok.eos_token_id:
                break

        accept_lens.append(real_len)
        # aggregate per position
        for t in range(K):
            v = alphas[t].item()
            if not (v != v):  # not NaN
                alpha_pos_sum[t] += v
                alpha_pos_cnt[t] += 1
                if table is not None:
                    table.add_data(idx, t+1, v)

        finite_vals = alphas[~torch.isnan(alphas)]
        if finite_vals.numel() > 0:
            alpha_means_per_prompt.append(float(finite_vals.double().mean().cpu()))

    # Aggregations
    accept_len_mean = float(sum(accept_lens) / max(1, len(accept_lens)))
    accept_len_median = float(statistics.median(accept_lens)) if accept_lens else 0.0
    accept_len_std = float(statistics.pstdev(accept_lens)) if len(accept_lens) > 1 else 0.0
    alpha_pos_mean = (alpha_pos_sum / torch.clamp(alpha_pos_cnt, min=1)).tolist()
    alpha_mean_macro = float(sum(alpha_means_per_prompt) / max(1, len(alpha_means_per_prompt)))
    total_alpha_sum = float(alpha_pos_sum.sum().cpu())
    total_alpha_cnt = float(alpha_pos_cnt.sum().cpu())
    alpha_mean_micro = (total_alpha_sum / max(1e-9, total_alpha_cnt)) if total_alpha_cnt > 0 else 0.0

    metrics = {
        "eval/accept_len_mean": accept_len_mean,
        "eval/accept_len_median": accept_len_median,
        "eval/accept_len_std": accept_len_std,
        "eval/alpha_mean_macro": alpha_mean_macro,
        "eval/alpha_mean_micro": alpha_mean_micro,
    }
    for i, m in enumerate(alpha_pos_mean, start=1):
        metrics[f"eval/alpha_pos_mean/{i}"] = float(m)

    if hasattr(wandb, "Histogram") and cfg_get(cfg, "logging.enable_hist", True):
        try:
            metrics["eval/accept_len_hist"] = wandb.Histogram(accept_lens)
        except Exception:
            pass

    return metrics, table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    args = parser.parse_args()

    cfg_raw = load_yaml(args.config)
    cfg = to_attrdict(cfg_raw)
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)

    # W&B
    wb = wb_init(cfg, cfg_raw)

    # Load tokenizer, models
    tok = load_tokenizer(cfg.models.tokenizer)
    target = load_target(cfg.models.target, dtype=cfg.training.dtype, device=cfg.training.device)
    draft  = load_draft(cfg.models.draft,  dtype=cfg.training.dtype, device=cfg.training.device)
    draft = attach_lora(draft, cfg.lora.target_modules, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout)
    draft.train()

    # Reference policy: frozen copy
    import copy as _copy
    ref_draft = _copy.deepcopy(draft).eval()
    for p in ref_draft.parameters(): p.requires_grad_(False)

    # Optimizer
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, draft.parameters()),
                            lr=cfg.training.lr, weight_decay=cfg.training.weight_decay, betas=(0.9, 0.95))

    # Data
    ds = PromptOnlyDataset(args.prompts)
    if len(ds) == 0:
        raise RuntimeError("Empty dataset. Put prompts in data/prompts.jsonl")

    # Epoch loop
    global_step = 0
    epochs = max(1, cfg_get(cfg, "training.num_epochs", 1))
    outer_bar = None
    try:
        if cfg_get(cfg, "progress.use_tqdm", True) and tqdm is not None:
            outer_bar = tqdm(total=epochs, desc="epochs")
    except Exception:
        outer_bar = None

    for epoch in range(1, epochs+1):
        prompt_indices = list(range(len(ds)))
        limit = cfg_get(cfg, "training.batch_prompts", 0)
        if limit and limit > 0:
            prompt_indices = prompt_indices[:limit]

        inner_epoch_bar = None
        try:
            if cfg_get(cfg, "progress.use_tqdm", True) and tqdm is not None:
                inner_epoch_bar = tqdm(total=len(prompt_indices), leave=False, desc=f"epoch {epoch}")
        except Exception:
            inner_epoch_bar = None

        for i in prompt_indices:
            prompt = ds[i]["prompt"]
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            global_step = train_on_prompt(input_ids, {"target": target, "draft": draft, "ref": ref_draft},
                                          cfg, opt, tok, global_step, pbar=inner_epoch_bar)
            try:
                if inner_epoch_bar is not None:
                    inner_epoch_bar.update(1)
            except Exception:
                pass

        try:
            if inner_epoch_bar is not None:
                inner_epoch_bar.close()
        except Exception:
            pass

        # ----- Evaluation phase -----
        if cfg_get(cfg, "eval.enabled", True) and (epoch % max(1, cfg_get(cfg, "eval.every_epochs", 1)) == 0):
            raw_prompts = [ds[i]["prompt"] for i in range(len(ds))]
            metrics, table = eval_specdec(raw_prompts, tok, {"target": target, "draft": draft}, cfg)
            wb_log(metrics)
            if table is not None:
                wb_log({"eval/alphas_table": table})

        try:
            if outer_bar is not None:
                outer_bar.update(1)
        except Exception:
            pass

    try:
        if outer_bar is not None:
            outer_bar.close()
    except Exception:
        pass
    try:
        wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    main()
