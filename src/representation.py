"""Feature blocks -> one point on the Grassmann manifold Gr(r, D).

For a chosen channel set {g} (e.g. the core 4-channel set B, E, MEANP, STDP), each block
G_{S,g}^{(k)} is first normalized by an RMS scale s_{g,k} and an eq.16 weight
w_k = 2^-(K_MAX-k), then vertically stacked over (g,k) into one matrix Z_S in R^{D x T}.
The rank-r (r=10 by default) truncated SVD Z_S = U_S Sigma_S V_S^T is taken and only the
left singular vectors U_S are kept -- the subspace span(U_S) is the Grassmann point.
Singular values (magnitude information) are discarded entirely.

This module is the SINGLE shared implementation used by all three evaluation tasks
(classification, phylogenetics, stability). Verified equivalent, line-by-line, to the
three previously-separate implementations that originally produced this paper's numbers
(`grassmannian_gnn_v1/representation.py` at alpha=1, `grassmannian_cakl_figure2/src/
representation.py`, `paper_reproduction/stability/representation.py` -- the latter two
were already byte-identical to each other). The only place the three tasks legitimately
differ is WHICH population `fit_scales` is called on: classification fits on the
training fold only (supervised k-fold CV); phylogenetics/stability fit on the whole
unlabeled pool (no train/test split exists for those tasks) -- see each driver script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_MAX = 5
FLOOR = 1e-12


def eq16_weights(ks, k_max: int = K_MAX) -> dict:
    return {k: 1.0 / 2.0 ** (k_max - k) for k in ks}


def fit_scales(blocks_list: list[dict], channels, ks) -> dict:
    """s_{g,k} = RMS over ALL entries of that (channel,k) block, pooled over every
    sequence in `blocks_list`."""
    ssq: dict = {}
    cnt: dict = {}
    for blocks in blocks_list:
        for g in channels:
            for k in ks:
                arr = blocks[(g, k)]
                key = (g, k)
                ssq[key] = ssq.get(key, 0.0) + float(np.square(arr, dtype=np.float64).sum())
                cnt[key] = cnt.get(key, 0) + arr.size
    return {key: max(float(np.sqrt(ssq[key] / cnt[key])), FLOOR) for key in ssq}


def block_coeffs(scales: dict, channels, ks, k_max: int = K_MAX) -> dict:
    """c_{g,k} = sqrt(w_k) / s_{g,k}."""
    w = eq16_weights(ks, k_max)
    return {(g, k): float(np.sqrt(w[k]) / scales[(g, k)]) for g in channels for k in ks}


def build_Z(blocks: dict, coeffs: dict, channels, ks) -> np.ndarray:
    """Z_S in R^{D x T}: vertical stack of c_{g,k} * G_{S,g}^{(k)}, channel-outer/k-inner."""
    parts = [coeffs[(g, k)] * blocks[(g, k)] for g in channels for k in ks]
    return np.concatenate(parts, axis=0)


@dataclass(frozen=True)
class GrassmannPoint:
    U: np.ndarray                 # (D, r), orthonormal columns
    singular_values: np.ndarray    # (T,), all of them, descending -- diagnostics only,
                                    # never used in the projection distance
    r: int
    deficient: bool


def svd_left_factors(Z: np.ndarray, r: int) -> GrassmannPoint:
    """Top-r LEFT singular vectors of Z (D x T), via the T x T Gram matrix (cheap since
    T=50 regardless of D), with a dense-SVD fallback on rank deficiency.
    """
    Z = np.asarray(Z, dtype=np.float64)
    D, T = Z.shape
    r_eff = int(min(r, D, T))
    G = Z.T @ Z
    w, V = np.linalg.eigh(G)
    w = w[::-1]
    V = V[:, ::-1]
    SV = np.sqrt(np.maximum(w, 0.0))
    Vr = np.ascontiguousarray(V[:, :r_eff])
    sr = SV[:r_eff]
    tol = SV[0] * max(D, T) * np.finfo(float).eps if SV.size else 0.0
    deficient = bool(np.any(sr <= tol))
    sr_safe = np.where(sr > FLOOR, sr, 1.0)
    U = (Z @ Vr) / sr_safe[None, :]
    if deficient:
        u, sv, _ = np.linalg.svd(Z, full_matrices=False)
        U = np.ascontiguousarray(u[:, :r_eff])
        SV = np.zeros(T, dtype=np.float64)
        SV[:len(sv)] = sv
    return GrassmannPoint(U=U.astype(np.float64), singular_values=SV, r=r_eff, deficient=deficient)


def sequence_to_point(blocks: dict, coeffs: dict, channels, ks, rank: int) -> GrassmannPoint:
    Z = build_Z(blocks, coeffs, channels, ks)
    return svd_left_factors(Z, rank)
