# src/grpo/reward_sparse_spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams


@dataclass
class SpecRewardConfig:
    topk_for_ce: int = 64             # teacher K
    divergence: str = "kl"            # "kl" | "alpha" (Tsallis) | "ce"
    alpha: float = 0.5                # for alpha-div
    tail_mode: str = "bucket"         # "bucket" | "ignore" (sparse KD tail mass)
    distill_temp: float = 1.0         # temperature for teacher/draft distill
    include_first_reject: bool = True
    entropy_bonus: float = 0.0
    anchor_kl_beta: float = 0.0
    reward_clip: bool = True
    reward_clip_range: Tuple[float, float] = (-20.0, 0.0)

    # vLLM probe options (teacher)
    teacher_model: str = "Qwen/Qwen3-8B"
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.92

    # generation (probe) knobs
    teacher_temperature: float = 0.0
    teacher_top_p: float = 1.0
    teacher_top_k: int = 0


class SpecDivergenceReward:
    """
    Reward(p_i, q_i) =  - sum_t mask_t * D_t(teacher || draft)
                        + entropy_bonus * sum_t mask_t * H_t(draft)
                        - anchor_kl_beta * sum_t mask_t * KL(q || q_anchor_ema)

    mask = accepted ∪ (first reject if enabled), where acceptance is teacher top-1 == generated token id
    """
    def __init__(self, tokenizer, cfg: SpecRewardConfig):
        self.tok = tokenizer
        self.cfg = cfg
        # teacher via vLLM
        self.llm = LLM(
            model=cfg.teacher_model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
        self._anchor_buffers: dict[int, torch.Tensor] = {}  # keyed by T_max: [T_max, V], EMA logits

    # ---------- helpers ----------
    @torch.no_grad()
    def _probe_teacher_topk(self, full_texts: List[str], T_per_sample: List[int], K: int):
        """
        Returns:
          topk_ids:      List[Tensor [T,K]] per sample
          topk_logprobs: List[Tensor [T,K]] per sample
          top1_ids:      List[Tensor [T]]   per sample
        """
        sp = SamplingParams(
            max_tokens=0,
            logprobs=K,
            prompt_logprobs=True,
            temperature=self.cfg.teacher_temperature,
            top_p=self.cfg.teacher_top_p,
            top_k=self.cfg.teacher_top_k,
        )
        outs = self.llm.generate(full_texts, sp)
        topk_ids, topk_lps, top1_ids = [], [], []
        for sample_out, T in zip(outs, T_per_sample):
            # vLLM returns per-token prompt_logprobs; take last T positions
            lastS = sample_out.prompt_logprobs[-T:] if T > 0 else []
            ids = torch.zeros(T, K, dtype=torch.long)
            lps = torch.full((T, K), -1e30, dtype=torch.float32)
            t1  = torch.full((T,), -1, dtype=torch.long)
            for t, cand in enumerate(lastS):
                pairs = []
                if isinstance(cand, dict):
                    for tk, lp in cand.items():
                        tid = self.tok.convert_tokens_to_ids(tk)
                        if tid is not None and tid >= 0:
                            pairs.append((tid, float(lp)))
                else:
                    for c in cand:
                        pairs.append((c.token_id, float(c.logprob)))
                pairs.sort(key=lambda x: x[1], reverse=True)
                if pairs:
                    t1[t] = pairs[0][0]
                for k in range(min(K, len(pairs))):
                    ids[t, k] = pairs[k][0]
                    lps[t, k] = pairs[k][1]
            topk_ids.append(ids)
            topk_lps.append(lps)
            top1_ids.append(t1)
        return topk_ids, topk_lps, top1_ids

    @staticmethod
    @torch.no_grad()
    def _first_reject_mask(accepted_T: torch.Tensor) -> torch.Tensor:
        # accepted_T: [T]
        T = accepted_T.size(0)
        out = torch.zeros_like(accepted_T)
        seen = False
        for t in range(T):
            if seen:
                continue
            if accepted_T[t] == 0:
                out[t] = 1
                seen = True
        return out

    @staticmethod
    def _entropy_per_token(logits_BTV: torch.Tensor) -> torch.Tensor:
        # [B,T,V] -> [B,T]
        logp = logits_BTV.log_softmax(-1)
        p = logp.exp()
        H = -(p * logp).sum(dim=-1)
        return H

    @staticmethod
    def _sparse_kl_per_token(
        d_logits_BTV: torch.Tensor,  # [B,T,V]
        topk_ids_BTK: torch.Tensor,  # [B,T,K]
        topk_logprobs_BTK: torch.Tensor,  # [B,T,K]
        tail_mode: str = "bucket",
        distill_temp: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns per-token KL: [B,T]
        """
        eps = 1e-8
        B,T,V = d_logits_BTV.shape
        K = topk_ids_BTK.shape[-1]
        # draft side
        d_logits = d_logits_BTV / distill_temp if distill_temp != 1.0 else d_logits_BTV
        logZ = torch.logsumexp(d_logits, dim=-1, keepdim=True)             # [B,T,1]
        q_logits_K = torch.gather(d_logits, dim=-1, index=topk_ids_BTK)    # [B,T,K]
        log_qK = q_logits_K - logZ
        qK = log_qK.exp().clamp_min(eps)

        # teacher probs over K (temperature-reweighted)
        pK = topk_logprobs_BTK.exp().clamp_min(eps)
        if distill_temp != 1.0:
            pK = (pK ** (1.0/distill_temp))

        if tail_mode == "ignore":
            pK = pK / (pK.sum(dim=-1, keepdim=True).clamp_min(eps))
            qK = qK / (qK.sum(dim=-1, keepdim=True).clamp_min(eps))
            log_pK = (pK + eps).log()
            log_qK = (qK + eps).log()
            kl = (pK * (log_pK - log_qK)).sum(dim=-1)                      # [B,T]
        else:
            p_sum = pK.sum(dim=-1).clamp_max(1.0)
            q_sum = qK.sum(dim=-1).clamp_max(1.0)
            p_tail = (1.0 - p_sum).clamp_min(0.0)
            q_tail = (1.0 - q_sum).clamp_min(eps)
            log_pK = (pK + eps).log()
            log_qK = (qK + eps).log()
            log_p_tail = (p_tail + eps).log()
            log_q_tail = (q_tail + eps).log()
            kl_main = (pK * (log_pK - log_qK)).sum(dim=-1)
            kl_tail = p_tail * (log_p_tail - log_q_tail)
            kl = kl_main + kl_tail
        return kl  # [B,T]

    @torch.no_grad()
    def __call__(self, prompts: List[str], samples: List[str], *, model, tokenizer, **kwargs) -> List[float]:
        """
        TRL calls reward_fn(prompts, samples, model=model, tokenizer=tokenizer, **extras).
        We compute one scalar per sample.
        """
        device = next(model.parameters()).device
        B = len(prompts)
        assert B == len(samples)

        # ----- tokenize + build full sequences -----
        enc_p = tokenizer(prompts, padding=True, truncation=True,
                          max_length=tokenizer.model_max_length, return_tensors="pt")
        enc_s = tokenizer(samples, padding=False, truncation=True,
                          max_length=tokenizer.model_max_length, return_tensors="pt")
        # Build per-sample continuation length T
        T_list = [int(x.size(0)) for x in enc_s["input_ids"]]
        T_max = max(T_list) if T_list else 0

        # Concatenate prompt || sample
        full_ids_list = []
        for i in range(B):
            full_ids_list.append(torch.cat([enc_p["input_ids"][i], enc_s["input_ids"][i]], dim=0))
        # left pad to common length
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        L = max(int(x.numel()) for x in full_ids_list)
        full_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        for i, ids in enumerate(full_ids_list):
            l = int(ids.numel()); full_ids[i, -l:] = ids; attn[i, -l:] = 1
        full_ids = full_ids.to(device); attn = attn.to(device)

        # ----- one draft forward to get logits over all positions -----
        with torch.no_grad():
            out = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B,L,V]
        V = logits.size(-1)

        # Slice last T_i tokens for each sample -> pack to [B,T_max,V], mask
        d_logits = torch.zeros(B, T_max, V, device=device, dtype=logits.dtype)
        cont_mask = torch.zeros(B, T_max, device=device, dtype=torch.float32)
        cont_ids = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)
        for i in range(B):
            Ti = T_list[i]
            if Ti == 0: continue
            d_logits[i, :Ti, :] = logits[i, -Ti-1:-1, :]  # predict each cont token
            cont_mask[i, :Ti] = 1.0
            cont_ids[i, :Ti] = enc_s["input_ids"][i].to(device)

        # ----- vLLM teacher probe (top-K logprobs for last T tokens) -----
        # Decode exact full texts from token IDs to keep same tokenizer path
        full_texts = [tokenizer.decode(full_ids_list[i], skip_special_tokens=False) for i in range(B)]
        topk_ids_list, topk_lps_list, top1_ids_list = self._probe_teacher_topk(full_texts, T_list, self.cfg.topk_for_ce)

        # stack/pad to [B,T_max,K]
        K = self.cfg.topk_for_ce
        topk_ids = torch.zeros(B, T_max, K, dtype=torch.long, device=device)
        topk_lps = torch.full((B, T_max, K), -1e30, dtype=torch.float32, device=device)
        teacher_top1 = torch.full((B, T_max), -1, dtype=torch.long, device=device)
        for i in range(B):
            Ti = T_list[i]
            if Ti == 0: continue
            topk_ids[i, :Ti] = topk_ids_list[i].to(device)
            topk_lps[i, :Ti] = topk_lps_list[i].to(device)
            teacher_top1[i, :Ti] = top1_ids_list[i].to(device)

        # ----- acceptance + first reject masks -----
        accepted = (teacher_top1[:, :T_max] == cont_ids[:, :T_max]).float() * cont_mask
        if self.cfg.include_first_reject:
            first_rej = torch.stack([self._first_reject_mask(accepted[i, :T_list[i]]) for i in range(B)], dim=0)
            # pad first_rej to T_max
            padded_fr = torch.zeros_like(accepted)
            for i in range(B):
                Ti = T_list[i]
                padded_fr[i, :Ti] = first_rej[i, :Ti]
            mask = torch.clamp(accepted + padded_fr, max=1.0)
        else:
            mask = accepted

        # ----- divergence per token (sparse top-K KL by default) -----
        kl_per_tok = self._sparse_kl_per_token(
            d_logits_BTV=d_logits,
            topk_ids_BTK=topk_ids,
            topk_logprobs_BTK=topk_lps,
            tail_mode=self.cfg.tail_mode,
            distill_temp=self.cfg.distill_temp,
        )  # [B,T_max]

        # ----- optional entropy bonus -----
        bonus = 0.0
        if self.cfg.entropy_bonus != 0.0:
            H = self._entropy_per_token(d_logits)        # [B,T_max]
            bonus = (H * mask).sum(dim=1) * self.cfg.entropy_bonus  # [B]

        # ----- optional anchor-KL (EMA over logits, keyed by T_max) -----
        anch = 0.0
        if self.cfg.anchor_kl_beta > 0.0 and T_max > 0:
            # EMA buffer shape [T_max, V]
            buf = self._anchor_buffers.get(T_max)
            cur = d_logits.mean(dim=0)  # [T_max, V] (batch-avg)
            if buf is None or buf.shape != cur.shape:
                buf = cur.detach().clone()
            else:
                buf = buf.lerp(cur.detach(), 0.01)
            self._anchor_buffers[T_max] = buf

            # KL(q || q_anchor)
            q_log = d_logits.log_softmax(-1)             # [B,T,V]
            qa_log = buf.unsqueeze(0).log_softmax(-1)    # [1,T,V]
            q = q_log.exp()
            kl_anchor = (q * (q_log - qa_log)).sum(dim=-1)  # [B,T]
            anch = (kl_anchor * mask).sum(dim=1) * self.cfg.anchor_kl_beta  # [B]

        # ----- sequence rewards -----
        seq_R = -(kl_per_tok * mask).sum(dim=1)          # [B]
        if self.cfg.entropy_bonus != 0.0:
            seq_R = seq_R + bonus
        if self.cfg.anchor_kl_beta > 0.0:
            seq_R = seq_R - anch

        if self.cfg.reward_clip:
            lo, hi = self.cfg.reward_clip_range
            seq_R = seq_R.clamp(min=lo, max=hi)

        return [float(x) for x in seq_R.detach().cpu()]
