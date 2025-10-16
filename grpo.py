from typing import Dict, Literal
import torch

def normalize_advantages(
    R: torch.Tensor,
    mode: Literal["std", "none"] = "std",
) -> torch.Tensor:
    if mode == "none":
        return R
    mean = R.mean()
    std = R.std().clamp_min(1e-6)
    return (R - mean) / std

def grpo_loss(
    logp_actions: torch.Tensor,
    old_logp_actions: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> Dict[str, torch.Tensor]:
    ratio = torch.exp(logp_actions - old_logp_actions)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.mean(torch.min(unclipped, clipped))
    approx_kl = 0.5 * torch.mean((logp_actions - old_logp_actions) ** 2)
    clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float())
    return {"loss": loss, "approx_kl": approx_kl, "clipfrac": clipfrac}
