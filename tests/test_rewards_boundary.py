# tests/test_rewards_boundary.py
import math
import torch
import pytest

from rewards_boundary import sequence_reward_sum


def _logits_from_probs(p: torch.Tensor) -> torch.Tensor:
    """
    Helper: if p is already a proper probability vector per last dim (sums to 1),
    then softmax(log(p)) == p. This lets us craft exact distributions easily.
    """
    # clamp to avoid log(0)
    return (p.clamp_min(1e-12)).log()


def test_identical_logits_zero_reward_kl():
    """
    If teacher and draft logits produce identical distributions and we use KL,
    the per-token divergence is 0, so the sequence reward should be 0 (before clipping).
    """
    torch.manual_seed(0)
    T, N, V = 4, 3, 5

    # Create arbitrary logits; use same tensor for teacher & draft
    teacher_logits = torch.randn(T, N, V)
    draft_logits = teacher_logits.clone()

    # Accept everything, no first reject
    accepted_mask = torch.ones(T, N)
    first_reject_mask = torch.zeros(T, N)

    R = sequence_reward_sum(
        teacher_logits=teacher_logits,
        draft_logits=draft_logits,
        accepted_mask=accepted_mask,
        first_reject_mask=first_reject_mask,
        include_first_reject=True,
        divergence="kl",       # KL(p||q)
        clip_reward=False,     # disable clipping to check exact 0
    )  # [N]

    assert torch.allclose(R, torch.zeros_like(R), atol=1e-6), f"Got {R}"


def test_handmade_single_step_kl_matches():
    """
    Tiny analytic case (T=1, N=1, V=2) with exact distributions, check the reward equals -KL(p||q).
    """
    T, N, V = 1, 1, 2

    # Choose simple probabilities with no zeros
    p = torch.tensor([[[0.7, 0.3]]])  # [T,N,V]
    q = torch.tensor([[[0.4, 0.6]]])  # [T,N,V]

    teacher_logits = _logits_from_probs(p)
    draft_logits = _logits_from_probs(q)

    accepted_mask = torch.tensor([[1.0]])  # accept that single position
    first_reject_mask = torch.tensor([[0.0]])

    # Expected KL(p||q) = sum_i p_i * (log p_i - log q_i)
    KL = (p * (p.log() - q.log())).sum(dim=-1)  # [T,N] -> [[value]]
    expected_reward = -KL.sum(dim=0)   # [N] = [-KL]

    R = sequence_reward_sum(
        teacher_logits=teacher_logits,
        draft_logits=draft_logits,
        accepted_mask=accepted_mask,
        first_reject_mask=first_reject_mask,
        include_first_reject=True,
        divergence="kl",
        clip_reward=False,
    )  # [N]

    torch.testing.assert_close(R, expected_reward, atol=1e-6, rtol=1e-6)


def test_masks_include_first_reject_toggle():
    """
    Construct T=3, N=1, V=2 with different p,q per time step.
    - accepted_mask selects t=0 and t=2
    - first_reject_mask selects t=1
    We check:
      a) include_first_reject=True => all 3 steps contribute
      b) include_first_reject=False => only t=0 and t=2 contribute
    """
    T, N, V = 3, 1, 2

    # Per-step distributions (teacher p_t, draft q_t)
    p = torch.tensor([
        [[0.8, 0.2]],   # t=0
        [[0.6, 0.4]],   # t=1
        [[0.3, 0.7]],   # t=2
    ])  # [T,N,V]
    q = torch.tensor([
        [[0.5, 0.5]],   # t=0
        [[0.7, 0.3]],   # t=1
        [[0.2, 0.8]],   # t=2
    ])  # [T,N,V]

    teacher_logits = _logits_from_probs(p)
    draft_logits = _logits_from_probs(q)

    # Masks:
    accepted_mask = torch.tensor([[1.0],   # t=0 accepted
                                  [0.0],   # t=1 not accepted
                                  [1.0]])  # t=2 accepted
    first_reject_mask = torch.tensor([[0.0],  # t=0
                                      [1.0],  # t=1 is the first reject
                                      [0.0]]) # t=2

    # Manual KL per t (scalar each)
    KL_t = (p * (p.log() - q.log())).sum(dim=-1).squeeze(-1)  # [T,N] -> [T,1] -> [T]
    # a) include_first_reject=True => all three positions counted
    expected_all = -(KL_t[0] + KL_t[1] + KL_t[2]).unsqueeze(0)  # [N]=[1]
    # b) include_first_reject=False => only accepted positions t=0 and t=2
    expected_accept_only = -(KL_t[0] + KL_t[2]).unsqueeze(0)    # [N]=[1]

    R_all = sequence_reward_sum(
        teacher_logits=teacher_logits,
        draft_logits=draft_logits,
        accepted_mask=accepted_mask,
        first_reject_mask=first_reject_mask,
        include_first_reject=True,
        divergence="kl",
        clip_reward=False,
    )
    R_accept = sequence_reward_sum(
        teacher_logits=teacher_logits,
        draft_logits=draft_logits,
        accepted_mask=accepted_mask,
        first_reject_mask=first_reject_mask,
        include_first_reject=False,
        divergence="kl",
        clip_reward=False,
    )

    torch.testing.assert_close(R_all, expected_all, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(R_accept, expected_accept_only, atol=1e-6, rtol=1e-6)
