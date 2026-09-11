"""
soft_dtw.py: Soft-DTW (Cuturi & Blondel, ICML 2017) as a PyTorch autograd op.

Replaces the hard min in the DTW recurrence with softmin_γ, making the
alignment landscape fully differentiable:

    softmin_γ(a, b, c) = −γ · log( e^{−a/γ} + e^{−b/γ} + e^{−c/γ} )

As γ → 0 this recovers hard DTW; larger γ averages over all alignments.

DESIGN NOTES:

1. LOCAL COST = EUCLIDEAN (not squared).
   The original hard-DTW losses (train_wlasl.py:dtw_path + path-sum) use
   torch.norm per frame pair. We keep the same local cost so soft-DTW is a
   drop-in replacement at the same scale, so the learned `threshold` parameter
   and the alpha pull-term stay meaningful.

2. CUSTOM BACKWARD (Cuturi-Blondel Algorithm 2).
   Forward stores the accumulated-cost matrix R; backward computes the
   expected-alignment matrix E in O(NM) instead of letting autograd unroll
   the whole DP graph (which would be slow and memory-hungry). Gradients
   w.r.t. the embeddings then flow through the torch-built cost matrix.

3. ANTI-DIAGONAL VECTORIZATION.
   Both passes iterate over anti-diagonals (N+M steps of vectorized ops)
   rather than N·M Python iterations. Single-pair API (soft_dtw) plus a
   batched variant (soft_dtw_many / soft_dtw_pairs) that runs the same DP
   for P variable-length pairs simultaneously: one Python loop of
   Nmax+Mmax steps total instead of one loop per pair. Batched and
   single-pair results are equivalence-tested against each other below.

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
# Batched variant: P variable-length pairs in one DP
# ---------------------------------------------------------------------------

class _SoftDTWBatchFn(torch.autograd.Function):
    """Soft-DTW of P padded cost matrices D (P, Nmax, Mmax) with true
    per-pair lengths n, m (P,). Values/gradients match _SoftDTWFn per pair.

    Padding safety: in the forward DP a cell (i, j) only reads cells with
    indices <= (i, j), so garbage in the padded region never reaches the
    valid region; the answer is read at exactly (n_b, m_b). In the backward
    pass, cells outside each pair's valid region are forced to R = -inf
    (making their softmin weight exp(...) = 0) and their E is pinned to 0,
    except the per-pair virtual terminal (n_b+1, m_b+1) which seeds E = 1
    exactly as in Cuturi-Blondel Algorithm 2.
    """

    @staticmethod
    def forward(ctx, D, n, m, gamma):
        P, N, M = D.shape
        dev = D.device
        inf = math.inf

        R = D.new_full((P, N + 2, M + 2), inf)
        R[:, 0, 0] = 0.0
        for d in range(2, N + M + 1):
            i = torch.arange(max(1, d - M), min(N, d - 1) + 1, device=dev)
            j = d - i
            r_diag = R[:, i - 1, j - 1]              # (P, L)
            r_up = R[:, i - 1, j]
            r_left = R[:, i, j - 1]
            stacked = torch.stack([r_diag, r_up, r_left])
            softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=0)
            R[:, i, j] = D[:, i - 1, j - 1] + softmin

        ctx.save_for_backward(D, R, n, m)
        ctx.gamma = gamma
        idx = torch.arange(P, device=dev)
        return R[idx, n, m].clone()

    @staticmethod
    def backward(ctx, grad_output):
        D, R, n, m = ctx.saved_tensors
        gamma = ctx.gamma
        P, N, M = D.shape
        dev = D.device
        idx = torch.arange(P, device=dev)

        D_pad = D.new_zeros((P, N + 2, M + 2))
        D_pad[:, 1:N + 1, 1:M + 1] = D
        # per-pair virtual terminal must carry zero local cost
        D_pad[idx, n + 1, m + 1] = 0.0

        # invalidate everything beyond each pair's (n_b, m_b) rectangle...
        ii = torch.arange(N + 2, device=dev).view(1, -1, 1)
        jj = torch.arange(M + 2, device=dev).view(1, 1, -1)
        invalid = (ii > n.view(-1, 1, 1)) | (jj > m.view(-1, 1, 1))
        R = R.clone()
        R[invalid] = -math.inf
        # ...except the virtual terminal, which anchors the recursion
        R[idx, n + 1, m + 1] = R[idx, n, m]

        E = D.new_zeros((P, N + 2, M + 2))
        E[idx, n + 1, m + 1] = 1.0

        n_col = n.view(-1, 1)
        m_col = m.view(-1, 1)
        for d in range(N + M, 1, -1):
            i = torch.arange(max(1, d - M), min(N, d - 1) + 1, device=dev)
            j = d - i
            r_ij = R[:, i, j]
            a = torch.exp((R[:, i + 1, j] - r_ij - D_pad[:, i + 1, j]) / gamma)
            b = torch.exp((R[:, i, j + 1] - r_ij - D_pad[:, i, j + 1]) / gamma)
            c = torch.exp((R[:, i + 1, j + 1] - r_ij - D_pad[:, i + 1, j + 1]) / gamma)
            acc = (a * E[:, i + 1, j] + b * E[:, i, j + 1]
                   + c * E[:, i + 1, j + 1])
            # valid cells get the recursion; invalid cells keep their current
            # value (0 everywhere except each pair's terminal, which stays 1
            # until every cell that reads it has been processed)
            valid = (i.view(1, -1) <= n_col) & (j.view(1, -1) <= m_col)
            E[:, i, j] = torch.where(valid, acc, E[:, i, j])

        # the terminal's seed E=1 sits at a *padded* D position for pairs
        # shorter than the grid, so zero it and no gradient leaks into padding
        E[idx, n + 1, m + 1] = 0.0
        grad_D = E[:, 1:N + 1, 1:M + 1] * grad_output.view(-1, 1, 1)
        return grad_D, None, None, None


def _pad_stack(seqs, length, dtype, device):
    """Stack variable-length (T_i, E) tensors into (B, length, E) with zeros."""
    B = len(seqs)
    E = seqs[0].shape[1]
    out = torch.zeros(B, length, E, dtype=dtype, device=device)
    for b, s in enumerate(seqs):
        out[b, :s.shape[0]] = s
    return out


def soft_dtw_many(x: torch.Tensor, ys, gamma: float = 1.0) -> torch.Tensor:
    """Soft-DTW of one query x (N, E) against a list of targets [(M_b, E)].

    Returns a (len(ys),) tensor; differentiable w.r.t. x and every y.
    """
    return soft_dtw_pairs([x], ys, gamma).squeeze(0)


def soft_dtw_pairs(xs, ys, gamma: float = 1.0) -> torch.Tensor:
    """All-pairs soft-DTW between lists xs [(N_e, E)] and ys [(M_b, E)].

    Returns (len(xs), len(ys)) tensor, one batched DP over all pairs.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 (use hard DTW for gamma = 0)")
    Ex, B = len(xs), len(ys)
    dtype, device = xs[0].dtype, xs[0].device
    n_list = [s.shape[0] for s in xs]
    m_list = [s.shape[0] for s in ys]
    Nmax, Mmax = max(n_list), max(m_list)

    X = _pad_stack(xs, Nmax, dtype, device)   # (Ex, Nmax, E)
    Y = _pad_stack(ys, Mmax, dtype, device)   # (B, Mmax, E)
    Xp = X.unsqueeze(1).expand(Ex, B, Nmax, X.shape[2]).reshape(Ex * B, Nmax, -1)
    Yp = Y.unsqueeze(0).expand(Ex, B, Mmax, Y.shape[2]).reshape(Ex * B, Mmax, -1)
    D = torch.cdist(Xp, Yp, p=2)              # (Ex*B, Nmax, Mmax)

    n = torch.tensor(n_list, device=device).repeat_interleave(B)
    m = torch.tensor(m_list, device=device).repeat(Ex)
    vals = _SoftDTWBatchFn.apply(D, n, m, float(gamma))
    return vals.view(Ex, B)


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


def _test_batched_matches_single_values():
    torch.manual_seed(6)
    for gamma in [0.1, 1.0, 5.0]:
        xs = [torch.randn(n, 6, dtype=torch.float64) for n in (9, 4, 12)]
        ys = [torch.randn(m, 6, dtype=torch.float64) for m in (7, 12, 3, 10)]
        batched = soft_dtw_pairs(xs, ys, gamma=gamma)
        for e, x in enumerate(xs):
            for b, y in enumerate(ys):
                single = soft_dtw(x, y, gamma=gamma).item()
                got = batched[e, b].item()
                assert abs(single - got) < 1e-8, \
                    f"value mismatch pair({e},{b}) gamma={gamma}: {got} vs {single}"
    print("test_batched_matches_single_values PASSED")


def _test_batched_matches_single_grads():
    torch.manual_seed(7)
    xs = [torch.randn(n, 5, dtype=torch.float64, requires_grad=True) for n in (8, 5)]
    ys = [torch.randn(m, 5, dtype=torch.float64, requires_grad=True) for m in (6, 11, 4)]
    w = torch.randn(2, 3, dtype=torch.float64)  # mix pairs unevenly

    (soft_dtw_pairs(xs, ys, gamma=0.7) * w).sum().backward()
    g_batched = [t.grad.clone() for t in xs + ys]
    for t in xs + ys:
        t.grad = None

    loss = sum(w[e, b] * soft_dtw(xs[e], ys[b], gamma=0.7)
               for e in range(2) for b in range(3))
    loss.backward()
    g_single = [t.grad.clone() for t in xs + ys]

    for gb, gs in zip(g_batched, g_single):
        assert torch.allclose(gb, gs, atol=1e-8), \
            f"grad mismatch: max diff {(gb - gs).abs().max().item()}"
    print("test_batched_matches_single_grads PASSED")


def _test_batched_gradcheck_padded():
    torch.manual_seed(8)
    D = torch.rand(3, 6, 5, dtype=torch.float64, requires_grad=True) + 0.1
    n = torch.tensor([6, 4, 5])
    m = torch.tensor([5, 3, 5])
    ok = torch.autograd.gradcheck(
        lambda d: _SoftDTWBatchFn.apply(d, n, m, 0.8), (D,), eps=1e-6, atol=1e-6
    )
    assert ok
    print("test_batched_gradcheck_padded PASSED")


def _test_batched_cuda_matches_cpu_if_available():
    if not torch.cuda.is_available():
        print("test_batched_cuda_matches_cpu_if_available SKIPPED (no CUDA)")
        return
    torch.manual_seed(9)
    xs = [torch.randn(n, 16) for n in (30, 45)]
    ys = [torch.randn(m, 16) for m in (60, 25, 40)]
    v_cpu = soft_dtw_pairs(xs, ys, gamma=1.0)
    v_gpu = soft_dtw_pairs([x.cuda() for x in xs], [y.cuda() for y in ys], gamma=1.0)
    assert torch.allclose(v_cpu, v_gpu.cpu(), atol=1e-2), \
        f"CPU/CUDA mismatch: {(v_cpu - v_gpu.cpu()).abs().max().item()}"
    print("test_batched_cuda_matches_cpu_if_available PASSED")


def _bench_batched_vs_single():
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(10)
    xs = [torch.randn(60, 256, device=dev) for _ in range(10)]
    ys = [torch.randn(90, 256, device=dev) for _ in range(21)]

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    for _ in range(2):  # warm-up
        soft_dtw_pairs(xs[:2], ys[:2]).sum()
    sync()
    t0 = time.time()
    soft_dtw_pairs(xs, ys).sum()
    sync()
    t_batched = time.time() - t0

    t0 = time.time()
    for x in xs:
        for y in ys:
            soft_dtw(x, y)
    sync()
    t_single = time.time() - t0
    print(f"bench ({dev}, 10x21 pairs, 60x90x256): "
          f"single {t_single:.2f}s vs batched {t_batched:.3f}s "
          f"({t_single / max(t_batched, 1e-9):.1f}x)")


if __name__ == "__main__":
    _test_gamma_to_zero_recovers_hard_dtw()
    _test_soft_below_hard_and_monotone_in_gamma()
    _test_gradcheck_cost_matrix()
    _test_gradient_flows_to_embeddings()
    _test_identical_sequences_near_zero()
    _test_cuda_matches_cpu_if_available()
    _test_batched_matches_single_values()
    _test_batched_matches_single_grads()
    _test_batched_gradcheck_padded()
    _test_batched_cuda_matches_cpu_if_available()
    _bench_batched_vs_single()
    print("\nAll tests passed.")
