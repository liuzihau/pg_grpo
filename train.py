
from __future__ import annotations
import copy
import argparse, torch, torch.nn.functional as F
from utils import load_yaml, set_seed, get_device, to_attrdict
from data import PromptOnlyDataset
from models import load_tokenizer, load_target, load_draft
from lora_setup import attach_lora
from rewards import accept_prob, survival as survival_fn
from grpo import grpo_loss

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
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)

        # top-p nucleus sampling per row
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumprobs > top_p).float().argmax(dim=-1)
        vocab = probs.size(-1)
        ar = torch.arange(vocab, device=device).unsqueeze(0).expand(S, -1)
        cut = cutoff.unsqueeze(1).expand(-1, vocab)
        mask_sorted = ar > cut
        mask_vocab = torch.zeros_like(mask_sorted, dtype=torch.bool)
        mask_vocab.scatter_(1, sorted_idx, mask_sorted)
        masked_probs = torch.where(mask_vocab, torch.zeros_like(probs), probs)
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)

        next_tokens = torch.multinomial(masked_probs, num_samples=1)  # [S,1]
        next_logp = torch.log(torch.gather(probs, 1, next_tokens).clamp_min(1e-12))  # [S,1]

        # Advance each sample's KV with its chosen token
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

def train_on_prompt(prompt_ids, models, cfg, optim, tokenizer):
    device = prompt_ids.device
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
    while accepted < M and steps < cfg.training.max_steps:
        steps += 1
        # 1) Propose S chunks
        prop_tokens, logp_draft, drf_step = sample_S_chunks(
            draft, drf_kv, S, K_chunk, cfg.training.temperature, cfg.training.top_p
        )

        # 2) Target scoring
        q_on_path, tgt_step = target_score_chunks(target, tgt_kv, prop_tokens)
        p_on_path = logp_draft.exp().clamp_min(1e-12)

        # 3) EAL reward
        a = accept_prob(p_on_path, q_on_path, floor=alpha_floor)
        surv = survival_fn(a)
        R = (surv * a).sum(dim=1)   # [S]
        adv = (R - R.mean()).detach()

        # 4) GRPO loss (+ KL)
        ref_logp = ref_logp_on_path(ref_draft, prop_tokens, ref_kv)
        loss, stats = grpo_loss(logp_draft, ref_logp, adv, surv, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), cfg.training.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        # 5) Grow prefix (best-of-S)
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

        if steps % cfg.training.log_every == 0:
            print(f"[step {steps}] acc_len={accepted_len} R_mean={float(R.mean().cpu()):.3f} "
                  f"loss={stats['loss']:.4f} policy={stats['policy']:.4f} kl={stats['kl']:.4f} "
                  f"accepted_total={accepted}/{M}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--prompts", type=str, default="data/prompts.jsonl")
    args = parser.parse_args()

    cfg_d = load_yaml(args.config)
    cfg = to_attrdict(cfg_d)
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)

    tok = load_tokenizer(cfg.models.tokenizer)
    target = load_target(cfg.models.target, dtype=cfg.training.dtype, device=cfg.training.device)
    draft  = load_draft(cfg.models.draft,  dtype=cfg.training.dtype, device=cfg.training.device)
    draft = attach_lora(draft, cfg.lora.target_modules, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout)
    draft.train()

    ref_draft = copy.deepcopy(draft).eval()
    for p in ref_draft.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, draft.parameters()),
                            lr=cfg.training.lr, weight_decay=cfg.training.weight_decay, betas=(0.9, 0.95))

    ds = PromptOnlyDataset(args.prompts)
    if len(ds) == 0:
        raise RuntimeError("Empty dataset. Put prompts in data/prompts.jsonl")

    # Iterate a few prompts (mini-batch over prompts not implemented for simplicity)
    for i, item in enumerate(ds):
        prompt = item["prompt"]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        train_on_prompt(input_ids, {"target": target, "draft": draft, "ref": ref_draft}, cfg, opt, tok)
        if (i+1) >= cfg.training.batch_prompts:
            break

if __name__ == "__main__":
    main()
