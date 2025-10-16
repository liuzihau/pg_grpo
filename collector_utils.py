import torch

def compute_first_reject_mask(accepted_mask: torch.Tensor) -> torch.Tensor:
    """
    accepted_mask: [T,N] 1 where step accepted
    returns first_reject_mask: [T,N] with one 1 at earliest rejection (if any)
    """
    T, N = accepted_mask.shape
    acc = accepted_mask > 0.5
    first_reject_mask = torch.zeros_like(accepted_mask, dtype=accepted_mask.dtype)
    for n in range(N):
        acc_n = acc[:, n]
        reject_pos = (~acc_n).nonzero(as_tuple=False)
        if reject_pos.numel() > 0:
            t0 = int(reject_pos[0].item())
            first_reject_mask[t0, n] = 1.0
    return first_reject_mask
