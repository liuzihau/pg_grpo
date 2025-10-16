from __future__ import annotations
from typing import Dict, List, Tuple, Literal, Optional
import torch
import torch.nn as nn

from rewards_boundary import sequence_reward_sum
from grpo import grpo_loss, normalize_advantages


def _cfg_get(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def _gather_logp(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(-1)
    return logp.gather(-1, ids.view(-1,1)).squeeze(-1)

class _RewardCfg:
    def __init__(self, cfg):
        self.divergence: Literal["kl","ce","js","alpha"] = _cfg_get(cfg, "reward.divergence", "kl")
        self.alpha: float = float(_cfg_get(cfg, "reward.alpha", 0.5))
        self.topk_for_ce: int = int(_cfg_get(cfg, "reward.topk_for_ce", 0))
        self.include_first_reject: bool = bool(_cfg_get(cfg, "reward.include_first_reject", True))
        self.entropy_bonus: float = float(_cfg_get(cfg, "reward.entropy_bonus", 0.0))
        self.anchor_kl_beta: float = float(_cfg_get(cfg, "reward.anchor_kl_beta", 0.0))
        self.clip_reward: bool = bool(_cfg_get(cfg, "reward.clip_reward", True))
        clip_range = _cfg_get(cfg, "reward.clip_range", [-20.0, 0.0])
        self.clip_range: Tuple[float,float] = (
            float(clip_range[0]), float(clip_range[1])
        ) if isinstance(clip_range, (list,tuple)) and len(clip_range)==2 else (-20.0, 0.0)
        self.advantage_norm: Literal["std","none"] = _cfg_get(cfg, "reward.advantage_norm", "std")

class BoundaryGRPOHelper:
    """
    Streaming PPO/GRPO:
      * compute rewards from detached logits
      * compute advantages
      * recompute current logp per trajectory with grad and immediately backward(loss/K)
    """

    def __init__(self, cfg, draft_module: nn.Module, optim: torch.optim.Optimizer):
        self.cfg = cfg
        self.draft = draft_module
        self.optim = optim
        self.r = _RewardCfg(cfg)
        self.anchor_logit_ema: Optional[torch.Tensor] = None

        self.clip_eps: float = float(_cfg_get(cfg, "training.ppo_clip_eps", 0.2))
        self.max_grad_norm: float = float(_cfg_get(cfg, "training.max_grad_norm", 1.0))

    @torch.no_grad()
    def _update_anchor_ema(self, curr_logits: torch.Tensor, momentum: float = 0.99):
        if self.anchor_logit_ema is None:
            self.anchor_logit_ema = curr_logits.detach()
        else:
            self.anchor_logit_ema = momentum * self.anchor_logit_ema + (1 - momentum) * curr_logits.detach()

    def _recompute_logp_stream(
        self,
        input_ids: torch.Tensor,           # [N,L0]
        attention_mask: torch.Tensor,      # [N,L0]
        actions: torch.Tensor,             # [T,N]
    ) -> torch.Tensor:
        """
        Replays the same trajectory and returns current-policy logp at the sampled actions.
        Runs with grad, but frees each step once used.
        """
        self.draft.train()
        N = input_ids.size(0)
        T = actions.size(0)
        cur_ids = input_ids
        cur_attn = attention_mask
        logps = []
        for t in range(T):
            out = self.draft(input_ids=cur_ids, attention_mask=cur_attn, use_cache=False)
            logits = out.logits[:, -1, :]
            logps.append(_gather_logp(logits, actions[t]))
            # extend
            cur_ids = torch.cat([cur_ids, actions[t].view(N,1)], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones(N,1, device=cur_attn.device, dtype=cur_attn.dtype)], dim=1)
        return torch.stack(logps, dim=0).reshape(-1)  # [T*N]

    def step_on_group(
        self,
        group: List[Dict[str, torch.Tensor]],
        step: int,
        log_fn=None,
    ) -> Dict[str, float]:
        # 1) sequence rewards (detached tensors)
        seq_rewards = []
        for out in group:
            if self.r.anchor_kl_beta > 0.0:
                self._update_anchor_ema(out["draft_logits"], momentum=0.99)

            R = sequence_reward_sum(
                teacher_logits=out["teacher_logits"],
                draft_logits=out["draft_logits"],
                accepted_mask=out["accepted_mask"],
                first_reject_mask=out["first_reject_mask"],
                include_first_reject=self.r.include_first_reject,
                divergence=self.r.divergence,
                alpha=self.r.alpha,
                topk_for_ce=self.r.topk_for_ce,
                entropy_bonus_coef=self.r.entropy_bonus,
                anchor_kl_beta=self.r.anchor_kl_beta,
                anchor_logits=self.anchor_logit_ema if self.r.anchor_kl_beta > 0.0 else None,
                clip_reward=self.r.clip_reward,
                clip_range=self.r.clip_range,
            )  # [N]
            seq_rewards.append(R.mean())

        seq_rewards = torch.stack(seq_rewards, dim=0)  # [K]
        # 2) group baseline / advantages
        if self.r.advantage_norm == "std":
            adv_group = normalize_advantages(seq_rewards, mode="std")
        else:
            adv_group = seq_rewards - seq_rewards.mean()

        # 3) streaming PPO: recompute logp with grad per trajectory and backward immediately
        self.optim.zero_grad(set_to_none=True)
        clipfracs = []
        kls = []
        K = len(group)

        for k, out in enumerate(group):
            cur_logp = self._recompute_logp_stream(
                input_ids=out["input_ids"],
                attention_mask=out["attention_mask"],
                actions=out["actions"],
            )  # requires_grad=True
            old_logp = out["old_acts_logp"].detach()   # [T*N], no grad
            adv = adv_group[k].detach().expand_as(cur_logp)

            ppo = grpo_loss(
                logp_actions=cur_logp,
                old_logp_actions=old_logp,
                advantages=adv,
                clip_eps=self.clip_eps,
            )
            # Scale and backward now to free graph
            (ppo["loss"] / K).backward()
            clipfracs.append(ppo["clipfrac"].detach())
            kls.append(ppo["approx_kl"].detach())

        torch.nn.utils.clip_grad_norm_(self.draft.parameters(), self.max_grad_norm)
        self.optim.step()

        metrics = {
            "train/seq_reward_mean": float(seq_rewards.mean().detach().cpu()),
            "train/seq_reward_std": float((seq_rewards.std().detach().cpu()).item() + 1e-9),
            "train/ppo_clipfrac": float(torch.stack(clipfracs).mean().cpu()),
            "train/ppo_approx_kl": float(torch.stack(kls).mean().cpu()),
        }
        if log_fn is not None:
            try: log_fn(metrics, step=step)
            except TypeError: log_fn(metrics)
        return metrics
