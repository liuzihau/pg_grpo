from __future__ import annotations
import os, argparse, math
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from vllm import LLM, SamplingParams

from src.specdec.acceptance import first_reject_mask
from src.kd.sparse_kd_loss import sparse_kd_kl

def load_yaml(path):
    import yaml
    with open(path,"r") as f: return yaml.safe_load(f)

def set_seed(s): import random; random.seed(s); torch.manual_seed(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wandb_api_key", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device","cuda"))

    wandb.login(key=args.wandb_api_key)
    wandb.init(project=cfg["logging"]["project"], name=cfg["logging"]["name"], config=cfg)

    tok_name = cfg["models"].get("tokenizer", cfg["draft"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Draft (LoRA or Unsloth path if you prefer; reuse from train_kd.py)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["draft"]["name"], device_map="auto", torch_dtype=getattr(torch, cfg["training"]["dtype"])
    )
    peft = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft)
    if hasattr(model,"gradient_checkpointing_enable"):
        try: model.gradient_checkpointing_enable()
        except TypeError: model.gradient_checkpointing_enable(use_reentrant=False)
    if hasattr(model,"config"): model.config.use_cache = False
    model.train()

    # vLLM teacher for verification (fast)
    llm = LLM(
        model=cfg["models"]["target"],
        tensor_parallel_size=int(cfg["vllm"]["tensor_parallel_size"]),
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
    )

    group_size = int(cfg["grpo"]["group_size"])
    S = int(cfg["grpo"]["S"])
    temperature = float(cfg["grpo"]["temperature"])
    clip_eps = float(cfg["ppo"]["clip_eps"])

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    steps = int(cfg["grpo"]["total_steps"])

    pbar = tqdm(range(steps), desc="GRPO", ncols=100)
    for step in pbar:
        # TODO: sample batch prompts from your dataset util (reuse HF loader from KD)
        # For brevity, assume `batch_prompts: List[str]`
        batch_prompts = ["Explain overfitting in simple terms."] * 2

        # Rollout draft S steps (sampled)
        enc = tokenizer(batch_prompts, padding=True, truncation=True, max_length=cfg["data"]["max_input_len"], return_tensors="pt").to(device)
        B = enc["input_ids"].shape[0]
        cur_ids = enc["input_ids"]; cur_attn = enc["attention_mask"]
        actions, old_logps, draft_logits_steps = [], [], []
        for t in range(S):
            out = model(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
            last = out.logits[:, -1, :]
            if temperature <= 0.0:
                act = last.argmax(dim=-1)
            else:
                probs = (last / max(temperature, 1e-6)).softmax(dim=-1)
                act = torch.multinomial(probs, 1).squeeze(-1)
            logp = last.log_softmax(dim=-1).gather(-1, act[:,None]).squeeze(-1)
            actions.append(act); old_logps.append(logp)
            draft_logits_steps.append(last)
            cur_ids = torch.cat([cur_ids, act[:,None]], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones(B,1, device=device, dtype=cur_attn.dtype)], dim=1)

        # vLLM probe: teacher top-1 accept + top-K logprobs on generated tokens
        # Ask vLLM to return logprobs for generated continuation tokens
        sp = SamplingParams(max_tokens=0, logprobs=cfg["reward"]["topk_for_ce"], prompt_logprobs=True)
        texts = [tokenizer.decode(ids) for ids in cur_ids]  # cheap re-decode
        outs = llm.generate(texts, sp)

        # Build acceptance mask from teacher top-1
        accepted = torch.zeros(S, B, device=device)
        topk_ids = torch.zeros(B, S, cfg["reward"]["topk_for_ce"], dtype=torch.int64, device=device)
        topk_lps = torch.full_like(topk_ids, -1e30, dtype=torch.float32)

        for b, o in enumerate(outs):
            prompt_token_logprobs = o.prompt_logprobs  # per input token
            # last S entries correspond to generated tokens
            lastS = prompt_token_logprobs[-S:]
            for t in range(S):
                cand = lastS[t]
                # find top-1 id
                # normalize to list of (id, lp)
                pairs = []
                if isinstance(cand, dict):
                    for tk, lp in cand.items():
                        tid = tokenizer.convert_tokens_to_ids(tk)
                        if tid is not None and tid >= 0: pairs.append((tid, float(lp)))
                else:
                    for c in cand:
                        pairs.append((c.token_id, float(c.logprob)))
                pairs.sort(key=lambda x: x[1], reverse=True)
                # fill arrays
                for k in range(min(len(pairs), topk_ids.shape[-1])):
                    topk_ids[b, t, k] = pairs[k][0]
                    topk_lps[b, t, k] = pairs[k][1]
                # accept if teacher top-1 equals action
                if len(pairs) > 0:
                    accepted[t, b] = 1.0 if pairs[0][0] == int(actions[t][b].item()) else 0.0

        first_rej = first_reject_mask(accepted)  # [T,B]
        mask_TB = torch.clamp(accepted + first_rej, max=1.0)

        # draft logits at each step as [T,B,V]
        T = S
        d_logits_TBV = torch.stack(draft_logits_steps, dim=0)  # [T,B,V]
        # compute sparse KL on masked positions
        loss_div = sparse_kd_kl(
            d_logits_BT_V=d_logits_TBV.transpose(0,1).contiguous(),        # [B,T,V]
            topk_ids_BTK=topk_ids.transpose(0,1).contiguous(),             # [T,B,K]→[B,T,K]
            topk_logprobs_BTK=topk_lps.transpose(0,1).contiguous(),
            mask_BT=mask_TB.transpose(0,1).contiguous(),                   # [B,T]
            tail_mode="bucket",
            distill_temp=1.0,
        )

        # PPO-style ratio clip on sampled actions
        old_acts_logp = torch.stack(old_logps, dim=0)  # [T,B]
        new_acts_logp = []
        for t in range(S):
            logits = draft_logits_steps[t]
            act = actions[t]
            new_acts_logp.append(logits.log_softmax(-1).gather(-1, act[:,None]).squeeze(-1))
        new_acts_logp = torch.stack(new_acts_logp, dim=0)
        adv = -loss_div.detach()  # cheap scalar baseline (you can replace with per-step adv if you like)
        ratio = (new_acts_logp - old_acts_logp).exp()
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        # weight by mask
        ppo_obj = -torch.min(ratio*adv, clipped*adv) * mask_TB
        loss_ppo = ppo_obj.sum() / (mask_TB.sum().clamp_min(1.0))

        total = loss_ppo + loss_div  # divergence + policy improvement

        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
        optim.step()

        wandb.log({"train/grpo_total": float(total.detach().cpu()),
                   "train/grpo_div": float(loss_div.detach().cpu()),
                   "train/grpo_ppo": float(loss_ppo.detach().cpu()),
                  }, step=step)
        pbar.set_postfix(loss=f"{float(total.detach().cpu()):.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()
