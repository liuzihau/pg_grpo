import torch

@torch.no_grad()
def first_reject_mask(accepted_mask: torch.Tensor) -> torch.Tensor:
    # accepted_mask: [T,B] (1 if accepted)
    T,B = accepted_mask.shape
    out = torch.zeros_like(accepted_mask)
    rejected_seen = torch.zeros(B, dtype=torch.bool, device=accepted_mask.device)
    for t in range(T):
        acc = accepted_mask[t].bool()
        rej = (~acc) & (~rejected_seen)
        out[t, rej] = 1.0
        rejected_seen |= rej
    return out
