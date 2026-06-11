"""
dba.py — DTW Barycenter Averaging (Petitjean 2011) for embedding sequences.

Implements Phase 1a deliverables 5.1.1–5.1.3 (see rmd/RESEARCH_PLAN.md):

  - full_dtw_with_path: standard DTW with alignment path reconstruction.
  - medoid: the "central" sequence under DTW distance, used as init.
  - dba: iterative barycenter averaging.

REFERENCE:
  Petitjean, Ketterlin, Gançarski.
  "A global averaging method for dynamic time warping, with applications to
   clustering." Pattern Recognition, 44(3), 2011.

DESIGN NOTES:

1. WHY NOT REUSE partial_DTW FROM server/model/classify.py?
   partial_DTW returns COSTS only, not the alignment path, and uses the
   subsequence initialization (free start anywhere in the query). DBA averages
   WHOLE sequences, so we use standard DTW with both-endpoints-pinned
   alignment and reconstruct the path.

2. PERFORMANCE.
   The DP fill is numba-jitted (mirrors classify.py / train_wlasl.py style)
   because eval_harness.py recomputes DBA inside every episode. The O(N+M)
   backtrack stays in plain Python.

3. TIE-BREAKING.
   Backtrack preference on equal costs is (up, left, diagonal) — identical to
   NB2's `dtw_path` (train_wlasl.py) argmin order, so paths are reproducible
   against the friend's reference implementation.

4. CONVERGENCE.
   DBA monotonically decreases the sum of within-cluster DTW costs (Petitjean
   §3). We track that sum and stop when the relative decrease < tol.

USAGE:
    from dba import dba, full_dtw_with_path

    # For each WLASL class:
    embeddings = [embed_sequence(model, ld) for ld in class_recordings]
    barycenter = dba(embeddings, max_iters=20, tol=1e-4, init='medoid')
"""

import random
from typing import List, Tuple

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# DTW with alignment path
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dtw_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fill the (N, M) accumulated-cost matrix for standard DTW."""
    N, _ = x.shape
    M, _ = y.shape
    D = np.empty((N, M))

    D[0, 0] = np.linalg.norm(x[0] - y[0])
    for i in range(1, N):
        D[i, 0] = np.linalg.norm(x[i] - y[0]) + D[i - 1, 0]
    for j in range(1, M):
        D[0, j] = np.linalg.norm(x[0] - y[j]) + D[0, j - 1]

    for i in range(1, N):
        for j in range(1, M):
            dist = np.linalg.norm(x[i] - y[j])
            cost_diag = D[i - 1, j - 1]
            cost_up = D[i - 1, j]
            cost_left = D[i, j - 1]
            best = cost_diag
            if cost_up < best:
                best = cost_up
            if cost_left < best:
                best = cost_left
            D[i, j] = dist + best
    return D


def full_dtw_with_path(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, List[Tuple[int, int]]]:
    """Standard DTW (not subsequence) with path reconstruction.

    Args:
        x: (N, E) numpy array.
        y: (M, E) numpy array.

    Returns:
        (cost, path) where path is a list of (i, j) pairs from (0, 0) to
        (N-1, M-1) representing the optimal alignment.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    N, M = x.shape[0], y.shape[0]
    D = _dtw_matrix(x, y)

    # Backtrack, rederiving the predecessor from D at each step.
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            up, left, diag = D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
            if up <= left and up <= diag:
                i -= 1
            elif left <= diag:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    return float(D[N - 1, M - 1]), path


def dtw_cost(x: np.ndarray, y: np.ndarray) -> float:
    """Standard DTW cost only (no path) — cheaper for medoid computation."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return float(_dtw_matrix(x, y)[-1, -1])


# ---------------------------------------------------------------------------
# Medoid (init)
# ---------------------------------------------------------------------------

def medoid(sequences: List[np.ndarray]) -> int:
    """Return the index of the sequence with the smallest sum of DTW distances
    to all other sequences.
    """
    K = len(sequences)
    total_cost = np.zeros(K)
    for i in range(K):
        for j in range(i + 1, K):
            c = dtw_cost(sequences[i], sequences[j])
            total_cost[i] += c
            total_cost[j] += c
    return int(np.argmin(total_cost))


# ---------------------------------------------------------------------------
# DBA — main algorithm
# ---------------------------------------------------------------------------

def _resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a (T, E) sequence to (length, E)."""
    T = seq.shape[0]
    if T == length:
        return seq.copy()
    idx = np.linspace(0, T - 1, length)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    frac = (idx - lo)[:, None]
    return seq[lo] * (1.0 - frac) + seq[hi] * frac


def dba(
    sequences: List[np.ndarray],
    max_iters: int = 20,
    tol: float = 1e-4,
    init: str = "medoid",
    barycenter_length: int = None,
) -> np.ndarray:
    """Compute the DBA barycenter of a list of sequences.

    Args:
        sequences: list of (T_i, E) arrays. Must all have the same E.
        max_iters: maximum iterations.
        tol: relative change tolerance for early stopping.
        init: one of "medoid", "random", "first". Determines initial barycenter.
        barycenter_length: length T of the output barycenter. If None, use the
            length of the init sequence.

    Returns:
        barycenter: (T, E) numpy array.

    Algorithm (Petitjean 2011, Algorithm 1): initialize b from one of the
    inputs, then repeat: align every input to b with full DTW, replace each
    b[t] by the mean of all input frames aligned to t, stop when the total
    DTW cost stops decreasing.
    """
    if len(sequences) == 0:
        raise ValueError("dba() needs at least one sequence")
    out_dtype = np.asarray(sequences[0]).dtype
    sequences = [np.asarray(s, dtype=np.float64) for s in sequences]
    if len(sequences) == 1:
        b = sequences[0]
        b = _resample(b, barycenter_length) if barycenter_length else b.copy()
        return b.astype(out_dtype, copy=False)

    if init == "medoid":
        b = sequences[medoid(sequences)]
    elif init == "random":
        b = sequences[random.randrange(len(sequences))]
    elif init == "first":
        b = sequences[0]
    else:
        raise ValueError(f"Unknown init: {init}")
    b = _resample(b, barycenter_length) if barycenter_length else b.copy()

    prev_total_cost = np.inf
    for _ in range(max_iters):
        aligned_frames = [[] for _ in range(len(b))]
        total_cost = 0.0
        for seq in sequences:
            cost, path = full_dtw_with_path(b, seq)
            total_cost += cost
            for (t, j) in path:
                aligned_frames[t].append(seq[j])

        new_b = np.array(
            [np.mean(np.stack(frames), axis=0) for frames in aligned_frames]
        )

        if np.isfinite(prev_total_cost) and (
            prev_total_cost - total_cost
        ) <= tol * max(abs(prev_total_cost), 1e-12):
            break
        b = new_b
        prev_total_cost = total_cost

    return b.astype(out_dtype, copy=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_dtw_simple():
    """Sanity check on the worked example from PRIMER.md §5.3."""
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[1.0], [3.0]])
    cost, path = full_dtw_with_path(x, y)
    assert abs(cost - 1.0) < 1e-6, f"Expected cost 1.0, got {cost}"
    # Path should be (0,0) → (1,1) → (2,1)
    assert path == [(0, 0), (1, 1), (2, 1)], f"Path mismatch: {path}"
    print("test_dtw_simple PASSED")


def _test_dba_trivial():
    """Sanity check on the worked example from PRIMER.md §8.3."""
    seq1 = np.array([[1.0], [5.0], [9.0]])
    seq2 = np.array([[1.0], [1.0], [5.0], [9.0], [9.0]])
    barycenter = dba([seq1, seq2], init="first", max_iters=10)
    expected = np.array([[1.0], [5.0], [9.0]])
    assert barycenter.shape == expected.shape, f"Shape mismatch: {barycenter.shape}"
    assert np.allclose(barycenter, expected, atol=1e-3), f"Barycenter mismatch:\n{barycenter}\nvs\n{expected}"
    print("test_dba_trivial PASSED")


def _test_dba_converges_to_mean_for_aligned():
    """Two equally-long, perfectly-aligned sequences → barycenter = elementwise mean."""
    seq1 = np.array([[1.0], [5.0], [9.0]])
    seq2 = np.array([[3.0], [5.0], [7.0]])
    barycenter = dba([seq1, seq2], init="first", max_iters=10)
    expected = np.array([[2.0], [5.0], [8.0]])
    assert np.allclose(barycenter, expected, atol=1e-3), f"Mean barycenter wrong:\n{barycenter}\nvs\n{expected}"
    print("test_dba_converges_to_mean_for_aligned PASSED")


def _test_medoid():
    """Middle sequence of a 1-D family should be the medoid."""
    seqs = [
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[1.0], [1.0], [1.0]]),   # central
        np.array([[2.0], [2.0], [2.0]]),
    ]
    assert medoid(seqs) == 1, f"Expected medoid index 1, got {medoid(seqs)}"
    print("test_medoid PASSED")


def _test_dba_monotone_cost_on_random():
    """On random (T, 8) sequences, the within-cluster DTW cost must not increase."""
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(rng.integers(20, 40), 8)) for _ in range(5)]

    def total_cost(b):
        return sum(full_dtw_with_path(b, s)[0] for s in seqs)

    b = seqs[medoid(seqs)].copy()
    prev = total_cost(b)
    bary = dba(seqs, max_iters=20, tol=0.0, init="medoid")
    final = total_cost(bary)
    assert final <= prev + 1e-9, f"DBA increased cost: {prev} -> {final}"
    print(f"test_dba_monotone_cost_on_random PASSED (cost {prev:.3f} -> {final:.3f})")


if __name__ == "__main__":
    _test_dtw_simple()
    _test_dba_trivial()
    _test_dba_converges_to_mean_for_aligned()
    _test_medoid()
    _test_dba_monotone_cost_on_random()
    print("\nAll tests passed.")
