"""
dba.py — DTW Barycenter Averaging (Petitjean 2011) for embedding sequences.

This is the SKELETON you fill in for Phase 1a deliverable 5.1.1–5.1.3 (see
rmd/RESEARCH_PLAN.md). Implements:

  - full_dtw_with_path: standard DTW with alignment path reconstruction.
  - medoid: the "central" sequence under DTW distance, used as init.
  - dba: iterative barycenter averaging.

REFERENCE:
  Petitjean, Ketterlin, Gançarski.
  "A global averaging method for dynamic time warping, with applications to
   clustering." Pattern Recognition, 44(3), 2011.

DESIGN NOTES:

1. WHY NOT REUSE partial_DTW FROM server/model/classify.py?
   partial_DTW returns COSTS only, not the alignment path. For DBA we need the
   path — to know which input frames aligned to which barycenter frames so we
   can average them. We implement standard (non-subsequence) DTW here because
   DBA averages whole sequences, not subsequences.

2. NUMERICAL STABILITY.
   For large embedding dimensions (E=256), squared distances can grow. Consider
   normalizing embeddings before DBA (L2-normalize each per-frame vector).

3. CONVERGENCE.
   DBA monotonically decreases the sum of within-cluster DTW costs (Petitjean
   §3). Track this sum each iteration; stop when relative decrease < tol.

4. BARYCENTER LENGTH.
   The barycenter has a FIXED length (you choose). Common choices:
     - median length across input sequences (default)
     - mean length, rounded
     - max length (preserves detail; can over-fit to longest)
   For WLASL with WINDOW_FRAMES=30 cropped inputs, just use 30.

5. INIT STRATEGIES.
   - "medoid": pick the sequence with smallest sum of DTW distances to others
   - "random": pick a random input sequence
   - "first": use the first sequence (simplest, deterministic)
   Medoid is the most common choice in DBA literature. Random gives variability.

WORKED EXAMPLE (from PRIMER.md §8.3):
    seq1 = np.array([[1], [5], [9]])         # T=3, E=1
    seq2 = np.array([[1], [1], [5], [9], [9]])  # T=5, E=1
    barycenter = dba([seq1, seq2], init='first')
    # Expected: [[1], [5], [9]] — DBA converges in one iteration on this case.

USAGE:
    from experiments.dba import dba, full_dtw_with_path

    # For each WLASL class:
    embeddings = [embed_sequence(model, ld) for ld in class_recordings]
    barycenter = dba(embeddings, max_iters=20, tol=1e-4, init='medoid')
"""

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# DTW with alignment path
# ---------------------------------------------------------------------------

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

    Algorithm:
        1. Fill DP matrix D[i,j] = ||x_i - y_j|| + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
           with base case D[0,0] = ||x_0 - y_0|| and standard boundary fills.
        2. Backtrack from (N-1, M-1) to (0, 0), choosing the predecessor cell
           with the smallest value at each step.

    TODO:
        - Allocate D of shape (N, M).
        - Allocate predecessor matrix (you can use np.int8 with values:
              0 = came from diagonal (i-1, j-1)
              1 = came from up (i-1, j)
              2 = came from left (i, j-1)
          OR just rederive from D in the backtrack).
        - Fill D row by row.
        - Backtrack to produce path.
        - Return (D[N-1, M-1], path).

    Hint: looking at server/model/classify.py:7-39 for the recurrence pattern
    is useful; ignore the subsequence-DTW initialization differences.
    """
    raise NotImplementedError("TODO: implement full_dtw_with_path")


# ---------------------------------------------------------------------------
# Medoid (init)
# ---------------------------------------------------------------------------

def medoid(sequences: List[np.ndarray]) -> int:
    """Return the index of the sequence with the smallest sum of DTW distances
    to all other sequences.

    Args:
        sequences: list of (T_i, E) arrays.

    Returns:
        Index i* in [0, len(sequences)) minimizing
            sum_j DTW(sequences[i*], sequences[j])

    TODO:
        - For each i, compute total_cost[i] = sum_j full_dtw_with_path(seq_i, seq_j)[0]
          (skip i == j; or include it; the comparison is unchanged).
        - Return argmin(total_cost).

    Complexity: O(K² × N × M × E) where K = len(sequences). For K ≤ 20, this
    is fine. For larger K, consider sampling.
    """
    raise NotImplementedError("TODO: implement medoid")


# ---------------------------------------------------------------------------
# DBA — main algorithm
# ---------------------------------------------------------------------------

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
            length of the init sequence (which is whatever init returns).

    Returns:
        barycenter: (T, E) numpy array.

    Algorithm (Petitjean 2011, Algorithm 1):
        1. Initialize b ← (one of the input sequences, per init)
        2. Repeat:
            a. For each input seq_i, compute alignment path between b and seq_i
               using full_dtw_with_path(b, seq_i).
            b. For each frame t of b, collect all input frames seq_i[j] such
               that (t, j) is in seq_i's alignment path. The new b[t] is the
               mean of these collected frames (across all sequences).
            c. If total DTW cost decreased by less than tol relative to previous
               iteration, break.
        3. Return b.

    TODO:
        - Initialize b based on `init`.
        - Track prev_total_cost (initialize to np.inf).
        - For iter in range(max_iters):
            - aligned_frames = [[] for _ in range(len(b))]
              # aligned_frames[t] will hold all (seq_idx, j) tuples aligning to b[t]
            - total_cost = 0.0
            - for seq_idx, seq in enumerate(sequences):
                - cost, path = full_dtw_with_path(b, seq)
                - total_cost += cost
                - for (t, j) in path:
                    - aligned_frames[t].append(seq[j])
            - new_b = np.array([np.mean(np.stack(frames), axis=0) for frames in aligned_frames])
            - if (prev_total_cost - total_cost) / prev_total_cost < tol: break
            - b = new_b
            - prev_total_cost = total_cost
        - Return b.

    Notes:
        - This algorithm is guaranteed to decrease the within-cluster DTW cost
          monotonically (Petitjean §3 proof). If your implementation INCREASES
          cost, there's a bug.
        - For sequences with E=256, mean(np.stack(frames), axis=0) is fast.
    """
    raise NotImplementedError("TODO: implement dba")


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


if __name__ == "__main__":
    _test_dtw_simple()
    _test_dba_trivial()
    _test_dba_converges_to_mean_for_aligned()
    print("\nAll tests passed.")
