"""
soft_dtw.py — Soft-DTW (Cuturi & Blondel, ICML 2017) as a PyTorch autograd op.

Phase 2a deliverable (see rmd/RESEARCH_PLAN.md §7.1). Replaces the hard
min in the DTW recurrence with softmin_γ, making the alignment landscape
fully differentiable:

    softmin_γ(a, b, c) = −γ · log( e^{−a/γ} + e^{−b/γ} + e^{−c/γ} )

As γ → 0 this recovers hard DTW; larger γ averages over all alignments.

DESIGN NOTES:

1. LOCAL COST = EUCLIDEAN (not squared).
   The friend's hard-DTW losses (train_wlasl.py:dtw_path + path-sum) use
   torch.norm per frame pair. We keep the same local cost so soft-DTW is a
   drop-in replacement at the same scale — the learned `threshold` parameter
   and the alpha pull-term stay meaningful.

2. CUSTOM BACKWARD (Cuturi-Blondel Algorithm 2).
   Forward stores the accumulated-cost matrix R; backward computes the
   expected-alignment matrix E in O(NM) instead of letting autograd unroll
   the whole DP graph (which would be slow and memory-hungry). Gradients
   w.r.t. the embeddings then flow through the torch-built cost matrix.

3. ANTI-DIAGONAL VECTORIZATION.
   Both passes iterate over anti-diagonals (N+M steps of vectorized ops)
   rather than N·M Python iterations. Single-pair API; loop over prototypes
   at the call site (sequences here are short, 30–60 frames).

4. STABILITY.
   softmin uses logsumexp. In backward, the exp arguments are ≤ 0 by the
   softmin inequality, so no overflow; underflow to 0 is the correct limit.

USAGE:
    from soft_dtw import soft_dtw

    d = soft_dtw(x, y, gamma=1.0)   # x: (N, E), y: (M, E) torch tensors
    d.backward()                     # gradients flow into x and y
"""

import math

import torch


class _SoftDTWFn(torch.autograd.Function):
    """Soft-DTW value of a precomputed local-cost matrix D (N, M)."""

    @staticmethod
    def forward(ctx, D, gamma):
        N, M = D.shape
        dev, dtype = D.device, D.dtype
        inf = math.inf

        R = D.new_full((N + 2, M + 2), inf)
        R[0, 0] = 0.0

        # Anti-diagonal DP: cells (i, j), 1-based into R, i+j = d.
        for d in range(2, N + M + 1):
            i = torch.arange(max(1, d - M), min(N, d - 1) + 1, device=dev)
            j = d - i
            r_diag = R[i - 1, j - 1]
            r_up = R[i - 1, j]
            r_left = R[i, j - 1]
            stacked = torch.stack([r_diag, r_up, r_left])
            softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
            R[i, j] = D[i - 1, j - 1] + softmin

        ctx.save_for_backward(D, R)
        ctx.gamma = gamma
        return R[N, M].clone()

    @staticmethod
    def backward(ctx, grad_output):
        D, R = ctx.saved_tensors
        gamma = ctx.gamma
        N, M = D.shape
        dev = D.device

        D_pad = D.new_zeros((N + 2, M + 2))
        D_pad[1:N + 1, 1:M + 1] = D

        R = R.clone()
        R[:, M + 1] = -math.inf
        R[N + 1, :] = -math.inf
        R[N + 1, M + 1] = R[N, M]

        E = D.new_zeros((N + 2, M + 2))
        E[N + 1, M + 1] = 1.0

        for d in range(N + M, 1, -1):
            i = torch.arange(max(1, d - M), min(N, d - 1) + 1, device=dev)
            j = d - i
            r_ij = R[i, j]
            a = torch.exp((R[i + 1, j] - r_ij - D_pad[i + 1, j]) / gamma)
            b = torch.exp((R[i, j + 1] - r_ij - D_pad[i, j + 1]) / gamma)
            c = torch.exp((R[i + 1, j + 1] - r_ij - D_pad[i + 1, j + 1]) / gamma)
            E[i, j] = a * E[i + 1, j] + b * E[i, j + 1] + c * E[i + 1, j + 1]

        grad_D = E[1:N + 1, 1:M + 1] * grad_output
        return grad_D, None


def soft_dtw_from_cost(D: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Soft-DTW value of an explicit (N, M) local-cost matrix."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0 (use hard DTW for gamma = 0)")
    return _SoftDTWFn.apply(D, float(gamma))


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Soft-DTW between two sequences with Euclidean local cost.

    Args:
        x: (N, E) tensor.
        y: (M, E) tensor.
        gamma: softmin temperature (> 0).

    Returns:
        0-dim tensor. Differentiable w.r.t. x and y.
    """
    D = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)
    return soft_dtw_from_cost(D, gamma)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _hard_dtw_reference(x: torch.Tensor, y: torch.Tensor) -> float:
    """Plain hard DTW with Euclidean local cost (independent reference)."""
    import numpy as np
    xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    N, M = len(xn), len(yn)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = float(np.linalg.norm(xn[i - 1] - yn[j - 1]))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[N, M])


def _test_gamma_to_zero_recovers_hard_dtw():
    torch.manual_seed(0)
    x = torch.randn(9, 4, dtype=torch.float64)
    y = torch.randn(6, 4, dtype=torch.float64)
    hard = _hard_dtw_reference(x, y)
    soft = soft_dtw(x, y, gamma=1e-3).item()
    assert abs(soft - hard) < 1e-2, f"gamma->0 mismatch: soft {soft} vs hard {hard}"
    print(f"test_gamma_to_zero_recovers_hard_dtw PASSED ({soft:.6f} ~= {hard:.6f})")


def _test_soft_below_hard_and_monotone_in_gamma():
    torch.manual_seed(1)
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(7, 4, dtype=torch.float64)
    hard = _hard_dtw_reference(x, y)
    prev = hard + 1e-9
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        v = soft_dtw(x, y, gamma=gamma).item()
        assert v <= prev + 1e-9, f"not monotone at gamma={gamma}: {v} > {prev}"
        prev = v
    print("test_soft_below_hard_and_monotone_in_gamma PASSED")


def _test_gradcheck_cost_matrix():
    torch.manual_seed(2)
    D = torch.rand(5, 4, dtype=torch.float64, requires_grad=True) + 0.1
    ok = torch.autograd.gradcheck(
        lambda d: _SoftDTWFn.apply(d, 1.0), (D,), eps=1e-6, atol=1e-6
    )
    assert ok
    print("test_gradcheck_cost_matrix PASSED")


def _test_gradient_flows_to_embeddings():
    torch.manual_seed(3)
    x = torch.randn(10, 8, requires_grad=True)
    y = torch.randn(12, 8, requires_grad=True)
    v = soft_dtw(x, y, gamma=0.5)
    v.backward()
    for name, t in [("x", x), ("y", y)]:
        g = t.grad
        assert g is not None and torch.isfinite(g).all(), f"bad grad for {name}"
        assert g.abs().sum() > 0, f"zero grad for {name}"
    print("test_gradient_flows_to_embeddings PASSED")


def _test_identical_sequences_near_zero():
    torch.manual_seed(4)
    x = torch.randn(7, 3, dtype=torch.float64)
    v = soft_dtw(x, x.clone(), gamma=1e-3).item()
    assert abs(v) < 1e-2, f"self-distance not ~0: {v}"
    print(f"test_identical_sequences_near_zero PASSED ({v:.2e})")


def _test_cuda_matches_cpu_if_available():
    if not torch.cuda.is_available():
        print("test_cuda_matches_cpu_if_available SKIPPED (no CUDA)")
        return
    torch.manual_seed(5)
    x = torch.randn(20, 16)
    y = torch.randn(25, 16)
    v_cpu = soft_dtw(x, y, gamma=1.0).item()
    v_gpu = soft_dtw(x.cuda(), y.cuda(), gamma=1.0).item()
    assert abs(v_cpu - v_gpu) < 1e-3, f"CPU/CUDA mismatch: {v_cpu} vs {v_gpu}"
    print(f"test_cuda_matches_cpu_if_available PASSED ({v_cpu:.4f} ~= {v_gpu:.4f})")


if __name__ == "__main__":
    _test_gamma_to_zero_recovers_hard_dtw()
    _test_soft_below_hard_and_monotone_in_gamma()
    _test_gradcheck_cost_matrix()
    _test_gradient_flows_to_embeddings()
    _test_identical_sequences_near_zero()
    _test_cuda_matches_cpu_if_available()
    print("\nAll tests passed.")
