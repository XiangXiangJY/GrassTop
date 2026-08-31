"""Grassmann projection (chordal) distance -- the sole distance this framework uses.

    d(S,R)^2 = r - ||U_S^T U_R||_F^2

Invariant to any right-multiplication U_S -> U_S O by an orthogonal O: a genuine
subspace/Grassmannian distance. Singular-value magnitude never enters (contrast with the
companion "psd_core_bures_r10" magnitude-sensitive distance evaluated elsewhere in the
same paper, deliberately NOT included in this release -- this framework isolates the
pure-projection arm only, gr_core_proj_r10).
"""

from __future__ import annotations

import numpy as np


def grassmann_distance(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Single-pair distance. U_a, U_b: (D, r) orthonormal (same r)."""
    r = min(U_a.shape[1], U_b.shape[1])
    overlap = U_a[:, :r].T @ U_b[:, :r]
    d2 = r - float(np.square(overlap).sum())
    return float(np.sqrt(max(d2, 0.0)))


def grassmann_distance_matrix(U_all: np.ndarray, q_blk: int = 128) -> np.ndarray:
    """All-pairs distance matrix for one stacked array of subspaces.

    U_all: (n, D, r). Returns (n, n), symmetric, zero diagonal, non-negative.

    Block-batched over query rows (q_blk rows per BLAS call), with the reference side
    flattened ONCE outside the loop -- matches the original `grassmannian_gnn_v1/
    evaluate.py::grassmann_distance_matrix` that actually produced this paper's
    classification numbers. An earlier row-at-a-time version of this function (one
    sequence per outer-loop iteration, re-flattening the ENTIRE remaining array on every
    iteration) was a real performance bug, not an inherent cost of the math: it made
    ncbi2024full (13645 sequences) exceed a 6h sbatch time limit without finishing the
    distance-matrix step, when the same computation completes in a fraction of that time
    batched properly (verified this session).
    """
    n, D, r = U_all.shape
    U_all = np.ascontiguousarray(U_all)
    ref_flat = np.ascontiguousarray(U_all.transpose(0, 2, 1).reshape(n * r, D))
    d2 = np.empty((n, n), dtype=np.float64)
    for s in range(0, n, q_blk):
        e = min(s + q_blk, n)
        qb = e - s
        qflat = np.ascontiguousarray(U_all[s:e].transpose(1, 0, 2).reshape(D, qb * r))
        Pbig = ref_flat @ qflat
        P = Pbig.reshape(n, r, qb, r)
        fro2 = np.sum(P ** 2, axis=(1, 3))
        d2[s:e] = np.clip(r - fro2.T, 0.0, None)
    d = np.sqrt(d2)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return d


def grassmann_distances_to_pool(U_query: np.ndarray, U_pool: np.ndarray) -> np.ndarray:
    """Distances from one query subspace to every subspace in a pool.

    U_query: (D, r). U_pool: (n, D, r). Returns (n,).
    """
    r = U_query.shape[1]
    C = np.einsum("ndr,ds->nrs", U_pool, U_query, optimize=True)
    fro2 = np.sum(C ** 2, axis=(1, 2))
    d2 = np.clip(r - fro2, 0.0, None)
    return np.sqrt(d2)
