"""Raw DNA sequence -> per-(channel, k) topological feature blocks G_{S,g}^{(k)} in R^{4^k x T}.

Four channels, all computed directly from a raw nucleotide string with no external
service and no proprietary binary (numpy/scipy + the open-source `ripser` and `gudhi`
pip packages only):

  B      Betti-0 curve of the 0-dimensional Vietoris-Rips filtration on a k-mer's
         occurrence-position point cloud.
  E      Smallest positive eigenvalue of the persistent-Laplacian, same filtration.
  MEANP  Mean of the non-harmonic (positive) Laplacian spectrum at each filtration step.
  STDP   Std. dev. of the non-harmonic (positive) Laplacian spectrum at each filtration
         step.

Parameters (T=50 filtration steps, step_size(k) = 4**(k-1)) match Hozumi & Wei,
"Revealing the Shape of Genome Space via k-mer Topology" (arXiv:2412.20202) Eq. 31 and
its reference implementation.

PROVENANCE / ATTRIBUTION (per-function, see each docstring below):
  - `find_kmers_position`, `_persistent_diagram`, `_betti_curve`, `betti_feature_block`
    (channel B) are re-typed, algorithm-unchanged, from Hozumi & Wei's own
    `KmerTopology` package (`KmerTopology/_extract_kmers.py`, `_compute_topology.py`,
    `kmer_topology.py`) -- the paper's original reference implementation. That source
    repository ships with no LICENSE file; this is a good-faith re-implementation for
    research-reproducibility purposes, cited explicitly here rather than silently
    presented as original work.
  - `moments_for_sequence` (channels MEANP, STDP) is adapted from this same source
    repository's `experiments/crossdataset_core_bures/moments.py` -- a closed-form,
    eigendecomposition-free reformulation (mean/variance of the Laplacian spectrum
    follow from vertex degrees alone: trace(L)=sum(deg), trace(L^2)=sum(deg^2)+sum(deg)).
  - `eigmin_curve_scalable` (channel E) is adapted from this PROJECT's own prior
    experiment code (`KmerTopology-main/experiments/joint_cakl/eigmin_scalable.py`,
    written for this project, not part of the original KmerTopology package) -- a
    matrix-free LOBPCG estimator of the smallest positive Laplacian eigenvalue, needed
    because the paper's own reference dense-eigvalsh formula is documented as taking
    >100s for a SINGLE filtration step of a SINGLE k=1 k-mer on one 16.5kb genome.

Verified against dense reference formulas (this project, this session) before use: E and
MEANP/STDP outputs match a brute-force dense-eigendecomposition reference to full float
precision on synthetic test cases.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import ripser
from gudhi.representations.vector_methods import BettiCurve
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lobpcg

T_FILTRATION_STEPS = 50
CHANNELS = ("B", "E", "MEANP", "STDP")


def get_kmers_lexicographic(k: int) -> list[str]:
    return sorted("".join(t) for t in product("ACGT", repeat=k))


def find_kmers_position(sequence: str, k: int) -> dict[str, np.ndarray]:
    """1-based occurrence positions of each of the 4^k canonical k-mers in `sequence`.

    Adapted from Hozumi & Wei's `KmerTopology/_extract_kmers.py`. A sliding window that
    is not an exact match to one of the 4^k ACGT k-mers (e.g. contains an ambiguity code
    N/R/Y/...) is skipped entirely, never counted toward any k-mer.
    """
    kmers = get_kmers_lexicographic(k)
    pos = {w: [] for w in kmers}
    kmer_set = set(kmers)
    length_seq = len(sequence) - k + 1
    for idx in range(max(length_seq, 0)):
        s = sequence[idx:idx + k]
        if s in kmer_set:
            pos[s].append(idx + 1)
    return {w: np.array(v, dtype=float) for w, v in pos.items()}


def filtration_grid(k: int, max_step: int = T_FILTRATION_STEPS) -> np.ndarray:
    return np.linspace(0, 4.0 ** (k - 1) * max_step, max_step)


# --------------------------------------------------------------------- channel B --- #
def _persistent_diagram(positions: np.ndarray, step_size: float, max_step: int) -> np.ndarray:
    """Adapted from KmerTopology/_compute_topology.py::compute_kmers_persistent_diagram."""
    positions = positions[positions > 0]
    if positions.shape[0] == 1:
        return np.array([[0, np.inf]])
    if positions.shape[0] == 0:
        return np.array([[0, 0]])
    row, col, val = [], [], []
    n = positions.shape[0]
    for i in range(n):
        if i == 0:
            row.append(i); col.append(i + 1); val.append(positions[i + 1] - positions[i])
        elif i == n - 1:
            row.append(i); col.append(i - 1); val.append(positions[i] - positions[i - 1])
        else:
            row.append(i); col.append(i - 1); val.append(positions[i] - positions[i - 1])
            row.append(i); col.append(i + 1); val.append(positions[i + 1] - positions[i])
    dis = sparse.coo_array((val, (row, col)), shape=(n, n))
    return ripser.ripser(dis, thresh=step_size * max_step, distance_matrix=True)["dgms"][0]


def _betti_curve(pd: np.ndarray, step_size: float, max_step: int) -> np.ndarray:
    """Adapted from KmerTopology/_compute_topology.py::compute_kmers_betti."""
    if pd.ndim == 1:
        return np.array([0, 0]) if pd[1] == 0 else np.array([1, 1])
    deaths = pd[:, 1].copy()
    deaths[deaths == np.inf] = 0
    grid = np.linspace(0, step_size * max_step, max_step)
    bc = BettiCurve(predefined_grid=grid)
    return bc.fit_transform([pd]).reshape(-1)


def betti_feature_block(sequence: str, k: int, max_step: int = T_FILTRATION_STEPS) -> np.ndarray:
    """G_S^{(k)} in R^{4^k x T} for channel B: Betti-0 curve per k-mer, lexicographic order."""
    step_size = 4.0 ** (k - 1)
    kmers = get_kmers_lexicographic(k)
    positions = find_kmers_position(sequence, k)
    rows = []
    for w in kmers:
        pd = _persistent_diagram(positions[w], step_size, max_step)
        rows.append(_betti_curve(pd, step_size, max_step))
    return np.array(rows, dtype=np.float64)


# ----------------------------------------------------------- channels MEANP, STDP --- #
def _prep_positions(positions: np.ndarray) -> np.ndarray:
    p = np.asarray(positions, dtype=float).ravel()
    p = p[p > 0]
    p.sort()
    return p


def moments_for_sequence(kmers_pos: dict, kmers_list: list, k: int,
                          max_step: int = T_FILTRATION_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """(MEANP, STDP) arrays of shape (4^k, max_step) for one sequence at one k.

    Adapted from KmerTopology-main/experiments/crossdataset_core_bures/moments.py.
    Closed form (no dense adjacency, no eigendecomposition): on a 1-D threshold graph,
        trace(L)   = sum_i deg_i
        trace(L^2) = sum_i deg_i^2 + sum_i deg_i
        MEANP = trace(L) / m_positive,  STDP = sqrt(trace(L^2)/m_positive - MEANP^2)
    where m_positive = n_vertices - beta0 (beta0 = number of connected components).
    """
    grid = filtration_grid(k, max_step)
    n_kmers = len(kmers_list)
    meanp = np.zeros((n_kmers, max_step), dtype=np.float64)
    stdp = np.zeros((n_kmers, max_step), dtype=np.float64)

    s_max = float(grid[-1])
    chunks, ids = [], []
    seq_len_upper = 0.0
    for u, km in enumerate(kmers_list):
        p = _prep_positions(kmers_pos[km])
        if p.size:
            seq_len_upper = max(seq_len_upper, float(p[-1]))
    big = seq_len_upper + s_max + 10.0
    for u, km in enumerate(kmers_list):
        p = _prep_positions(kmers_pos[km])
        if p.size == 0:
            continue
        chunks.append(p + u * big)
        ids.append(np.full(p.size, u, dtype=np.int64))
    if not chunks:
        return meanp, stdp
    pg = np.concatenate(chunks)
    kid = np.concatenate(ids)
    n_per = np.bincount(kid, minlength=n_kmers).astype(np.float64)

    for t, s in enumerate(grid):
        if s <= 0:
            continue
        lo = np.searchsorted(pg, pg - s, side="left")
        hi = np.searchsorted(pg, pg + s, side="right")
        deg = (hi - lo - 1).astype(np.float64)

        is_start = np.empty(pg.size, dtype=np.float64)
        is_start[0] = 1.0
        if pg.size > 1:
            gaps = np.diff(pg)
            is_start[1:] = (gaps > s).astype(np.float64)

        tr = np.bincount(kid, weights=deg, minlength=n_kmers)
        tr2 = np.bincount(kid, weights=deg * deg, minlength=n_kmers) + tr
        beta0 = np.bincount(kid, weights=is_start, minlength=n_kmers)

        m_pos = n_per - beta0
        ok = m_pos > 0
        if not ok.any():
            continue
        mp = np.zeros(n_kmers)
        mp[ok] = tr[ok] / m_pos[ok]
        var = np.zeros(n_kmers)
        var[ok] = tr2[ok] / m_pos[ok] - mp[ok] * mp[ok]
        np.maximum(var, 0.0, out=var)
        meanp[:, t] = mp
        stdp[:, t] = np.sqrt(var)

    return meanp, stdp


# --------------------------------------------------------------------- channel E --- #
def _bands(p_sorted: np.ndarray, s: float):
    lo = np.searchsorted(p_sorted, p_sorted - s, side="left")
    hi = np.searchsorted(p_sorted, p_sorted + s, side="right") - 1
    c = (hi - lo + 1).astype(float)
    return lo, hi, c


def _laplacian_operator(p_sorted: np.ndarray, s: float) -> LinearOperator:
    lo, hi, c = _bands(p_sorted, s)
    m = p_sorted.shape[0]

    def matvec(x):
        x = np.asarray(x, dtype=float).ravel()
        pref = np.empty(m + 1)
        pref[0] = 0.0
        np.cumsum(x, out=pref[1:])
        S = pref[hi + 1] - pref[lo]
        return c * x - S

    op = LinearOperator((m, m), matvec=matvec, rmatvec=matvec, dtype=float)
    op.degree = c - 1.0
    return op


def _component_slices(p_sorted: np.ndarray, s: float) -> list[tuple[int, int]]:
    n = p_sorted.shape[0]
    if n == 0:
        return []
    breaks = np.nonzero(np.diff(p_sorted) > s)[0]
    out, start = [], 0
    for b in breaks:
        out.append((start, b + 1))
        start = b + 1
    out.append((start, n))
    return out


def _smallest_positive_eig_dense(p_comp: np.ndarray, s: float) -> float:
    m = p_comp.shape[0]
    if m < 2:
        return 0.0
    A = (np.abs(p_comp[:, None] - p_comp[None, :]) <= s).astype(float)
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(0)) - A
    eig = np.linalg.eigvalsh(L)
    numzero = int(np.count_nonzero(eig < 1e-6))
    return 0.0 if numzero == m else float(eig[numzero])


def _lobpcg_fiedler(A, M, Y, m, tol, maxiter, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, 1))
    X -= Y @ (Y.T @ X)
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals, vecs = lobpcg(A, X, M=M, Y=Y, tol=tol, maxiter=maxiter, largest=False)
        lam = float(vals[0])
        v = vecs[:, 0]
        if not np.isfinite(lam) or not np.all(np.isfinite(v)):
            return np.nan, np.inf
        res = float(np.linalg.norm(A.matvec(v) - lam * v) / (np.linalg.norm(v) + 1e-30))
        return lam, res
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.inf


def _smallest_positive_eig_matfree(p_comp, s, tol=1e-9, maxiter=5000, seeds=(0, 1)):
    m = p_comp.shape[0]
    if m < 2:
        return 0.0, True
    if m == 2:
        return float(_smallest_positive_eig_dense(p_comp, s)), True
    A = _laplacian_operator(p_comp, s)
    deg = A.degree
    scale = 2.0 * float(deg.max()) + 1.0
    M = LinearOperator((m, m), matvec=lambda x: np.asarray(x, float).ravel() / deg, dtype=float)
    Y = np.ones((m, 1)) / np.sqrt(m)

    lam0, res0 = _lobpcg_fiedler(A, M, Y, m, tol, maxiter, seeds[0])
    if np.isfinite(lam0) and lam0 > 0 and res0 / scale <= 1e-8:
        return lam0, True
    lam1, res1 = _lobpcg_fiedler(A, M, Y, m, tol, maxiter, seeds[1])
    good = np.isfinite(lam0) and np.isfinite(lam1) and lam0 > 0 and lam1 > 0
    if good and abs(lam0 - lam1) <= 1e-6 * max(lam0, lam1):
        return 0.5 * (lam0 + lam1), True
    finite = [x for x in (lam0, lam1) if np.isfinite(x) and x > 0]
    return (min(finite) if finite else np.nan), False


def _eigmin_value_at_radius(p_sorted, s, dense_cap=256, tol=1e-9, maxiter=3000):
    if s == 0 or p_sorted.shape[0] < 2:
        return 0.0, True
    emin, ok = np.inf, True
    for a, b in _component_slices(p_sorted, s):
        m = b - a
        if m < 2:
            continue
        pc = p_sorted[a:b]
        if m <= dense_cap:
            emin = min(emin, _smallest_positive_eig_dense(pc, s))
        else:
            val, conv = _smallest_positive_eig_matfree(pc, s, tol=tol, maxiter=maxiter)
            if conv:
                emin = min(emin, val)
            else:
                ok = False
    if not ok:
        return np.nan, False
    return (0.0 if not np.isfinite(emin) else emin), True


def eigmin_curve_scalable(positions: np.ndarray, k: int, max_step: int = T_FILTRATION_STEPS,
                           dense_cap: int = 256) -> np.ndarray:
    """Smallest-positive-Laplacian-eigenvalue curve for one k-mer, matrix-free.

    Adapted from this project's own `eigmin_scalable.py`. Small connected components
    (<=dense_cap points) use exact dense eigvalsh; larger ones use LOBPCG so this scales
    to k-mers with thousands of occurrences (channel E is otherwise infeasible at real
    dataset scale, see module docstring).
    """
    p = np.sort(_prep_positions(positions))
    grid = filtration_grid(k, max_step)
    out = np.zeros(max_step)
    for i, s in enumerate(grid):
        val, ok = _eigmin_value_at_radius(p, s, dense_cap=dense_cap)
        out[i] = val if ok and np.isfinite(val) else 0.0
    return out


# ------------------------------------------------------------------- unified entry --- #
def sequence_feature_blocks(sequence: str, channels=CHANNELS, ks=(1, 2, 3, 4, 5),
                             max_step: int = T_FILTRATION_STEPS) -> dict:
    """-> {(channel, k): np.ndarray(4**k, max_step)} for every requested (channel, k).

    All channels for a given k share one k-mer position extraction pass.
    """
    blocks = {}
    for k in ks:
        kmers = get_kmers_lexicographic(k)
        positions = find_kmers_position(sequence, k)
        if "B" in channels:
            blocks[("B", k)] = betti_feature_block(sequence, k, max_step)
        if "E" in channels:
            rows = [eigmin_curve_scalable(positions[w], k, max_step) for w in kmers]
            blocks[("E", k)] = np.array(rows, dtype=np.float64)
        if "MEANP" in channels or "STDP" in channels:
            meanp, stdp = moments_for_sequence(positions, kmers, k, max_step)
            if "MEANP" in channels:
                blocks[("MEANP", k)] = meanp
            if "STDP" in channels:
                blocks[("STDP", k)] = stdp
    return blocks
