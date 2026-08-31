"""Self-implemented baseline alignment-free methods for the purity comparison: NVM,
FFP-JS, FFP-KL, Markov k-string, and FPS (Fourier power spectrum).

Motivation: the paper's Table `tab:purity` originally cited these 5 methods' purity
numbers from published tables/bar-charts elsewhere (CAKL's own Fig. 2d-f and, separately,
Hozumi & Wei's Table 2), which disagree with each other and were never computed by this
project under the SAME purity definition used for GrassTop. This module implements each
method from its standard published formula and returns a distance matrix that is scored
by the exact SAME `phylogenetics/upgma.py` + `phylogenetics/purity.py` pipeline used for
GrassTop (`run_purity_eval.py`), so all 6 methods (GrassTop + 5 baselines) are compared
under one internally-consistent metric.

Each `*_distance_matrix` function takes `seqs: dict[str, str]` (accession -> uppercased
ACGT/ambiguity-code sequence, as returned by `src/io_utils.parse_fasta`) and returns
`(D, accs)`: a square symmetric zero-diagonal numpy distance matrix and the accession
order (== `list(seqs.keys())`, i.e. dict insertion order) that indexes its rows/cols.

IUPAC ambiguity codes (N, R, Y, S, W, K, M, ...) are present in 2 of the 4 datasets
(rhinovirus, mammalian mitochondria -- see driver script docstring). Handling, chosen to
be simple and consistent across all 5 methods: a base/window is counted only when every
character in it is a canonical A/C/G/T; ambiguous positions are silently skipped for
counting purposes but still occupy a real index in the sequence (so sequence length L and
inter-nucleotide spacing are computed on the true, full-length sequence, not a filtered
one). This matches how such positions are conventionally handled in the underlying
literature (skip/ignore rather than impute a base).

Provenance of the exact formula/parameter choices (also repeated per-function below):
  - NVM: Deng, Yu, Liang, He, Yau (2011) PLoS ONE e17293 [ref 32 in 25m1786957.pdf] /
    Yu et al. (2013) PLoS ONE e64328 [ref 31] -- classic SINGLE-NUCLEOTIDE natural vector
    (4 letters, not generalized to k-mers), moments up to order 5. The paper's own results
    narrative labels this "NVM using k=5"; cross-checked against the bibliography (refs
    31/32 are both single-nucleotide NVM papers, no generalized-to-k-mers NVM reference is
    cited), so "k=5" is read here as "moments computed up to the 5th central moment"
    (matching the classic 24-dim vector: n_i, mu_i, D_2^i..D_5^i for each of A,C,G,T), NOT
    as a 5-mer word length. This is documented as an interpretive choice, not a certainty.
  - FFP-JS / FFP-KL: Feature Frequency Profile, k=3 (confirmed against 25m1786957.pdf's
    own results narrative, e.g. "FFP-JS using k = 3" repeated for every dataset).
  - Markov: order-3 Markov k-string model (confirmed against "Markov K-String using k = 3"
    in the same narrative); ref [25] T.-J. Wu, Y.-C. Hsieh, L.-A. Li, "Statistical
    measures of DNA sequence dissimilarity under Markov chain models of base
    composition" is the disambiguating citation for "Markov k-string" in this literature.
  - FPS: Hoang, Yin, Zheng, Yu, He, Yau (2015), "A new method to cluster DNA sequences
    using Fourier power spectrum" [ref 40 in 25m1786957.pdf].
"""
from __future__ import annotations

from itertools import product

import numpy as np

NUCLEOTIDES = ("A", "C", "G", "T")
BASE_IDX = {b: i for i, b in enumerate(NUCLEOTIDES)}


# ---------------------------------------------------------------------------
# 1. NVM -- Natural Vector Method
# ---------------------------------------------------------------------------

def natural_vector(seq: str, max_moment_order: int = 5) -> np.ndarray:
    """24-dim natural vector (Deng/Yu/Yau/Yau family): for each nucleotide i in
    {A,C,G,T} with 1-indexed occurrence positions p_1..p_{n_i} in a length-L sequence,

        n_i    = count of nucleotide i
        mu_i   = (1/n_i) * sum_k p_k                                    (mean position)
        D_j^i  = sum_k (p_k - mu_i)^j / (n_i^(j-1) * L^(j-1)),  j = 2..max_moment_order

    Vector = concat_i (n_i, mu_i, D_2^i, ..., D_{max_moment_order}^i)  -> 4*(2+(order-1))
    dims; with max_moment_order=5 this is the standard 4*6=24-dim vector. n_i=0 (nucleotide
    absent) and n_i=1 (no spread) both yield all-zero moment terms for that nucleotide
    beyond n_i itself, since sum_k(p_k-mu_i)^j is identically 0 in both cases.

    L = len(seq) (the TRUE full sequence length, including any non-ACGT/ambiguity-code
    characters) -- consistent with the original NVM definition, which treats the sequence
    as a fixed-length line and positions of each base as a subset of {1..L}.
    """
    L = len(seq)
    out = []
    for b in NUCLEOTIDES:
        positions = np.fromiter((i + 1 for i, c in enumerate(seq) if c == b), dtype=float)
        n_i = positions.size
        out.append(float(n_i))
        if n_i == 0:
            out.append(0.0)
            out.extend([0.0] * (max_moment_order - 1))
            continue
        mu_i = float(positions.mean())
        out.append(mu_i)
        centered = positions - mu_i
        for j in range(2, max_moment_order + 1):
            if n_i > 1:
                Dj = float(np.sum(centered ** j) / (n_i ** (j - 1) * L ** (j - 1)))
            else:
                Dj = 0.0
            out.append(Dj)
    return np.array(out, dtype=float)


def nvm_distance_matrix(seqs: dict) -> tuple[np.ndarray, list[str]]:
    """Pairwise Euclidean distance between natural vectors (standard NVM comparison)."""
    from scipy.spatial.distance import pdist, squareform
    accs = list(seqs.keys())
    V = np.stack([natural_vector(seqs[a]) for a in accs])
    D = squareform(pdist(V, metric="euclidean"))
    return D, accs


# ---------------------------------------------------------------------------
# 2/3. FFP -- Feature Frequency Profile (k-mer probability distribution), compared by
#            Jensen-Shannon divergence (FFP-JS) or symmetrized KL divergence (FFP-KL).
# ---------------------------------------------------------------------------

def _all_kmers(k: int) -> list[str]:
    return ["".join(p) for p in product(NUCLEOTIDES, repeat=k)]


def kmer_profile(seq: str, k: int, pseudocount: float = 1.0) -> np.ndarray:
    """Normalized k-mer frequency profile over all 4^k canonical (pure-ACGT) k-mers.

    Sliding window of length k over `seq`; any window containing a non-ACGT character is
    skipped (not counted). Add-`pseudocount` (default 1, i.e. Laplace/add-one) smoothing
    is applied to every one of the 4^k bins before renormalizing to a probability
    distribution, so no bin is ever exactly 0 -- required for KL/JS divergence (log of a
    zero-probability bin is undefined otherwise). Documented choice: pseudocount=1.
    """
    kmers = _all_kmers(k)
    idx = {w: i for i, w in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=float)
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        j = idx.get(w)
        if j is not None:
            counts[j] += 1.0
    counts += pseudocount
    return counts / counts.sum()


def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P || Q) = sum P * ln(P/Q), natural log (documented convention; log2 would only
    rescale JSD/KL by a constant ln(2) factor and not change UPGMA topology or purity)."""
    return float(np.sum(P * np.log(P / Q)))


def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = (P+Q)/2, natural log."""
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


def ffp_distance_matrices(seqs: dict, k: int = 3, pseudocount: float = 1.0
                           ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """-> (D_js, D_kl, accs). D_kl uses the symmetrized KL divergence
    0.5*(KL(P||Q)+KL(Q||P)) (a true JS-style symmetrization, distinct from JSD itself)."""
    accs = list(seqs.keys())
    profiles = [kmer_profile(seqs[a], k, pseudocount) for a in accs]
    n = len(accs)
    D_js = np.zeros((n, n))
    D_kl = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            P, Q = profiles[i], profiles[j]
            d_js = js_divergence(P, Q)
            d_kl = 0.5 * (kl_divergence(P, Q) + kl_divergence(Q, P))
            D_js[i, j] = D_js[j, i] = d_js
            D_kl[i, j] = D_kl[j, i] = d_kl
    return D_js, D_kl, accs


# ---------------------------------------------------------------------------
# 4. Markov k-string model (order-3)
# ---------------------------------------------------------------------------

def markov_deviation_vector(seq: str, order: int = 3) -> np.ndarray:
    """Per-sequence 'deviation from its own fitted Markov model' feature vector, a
    standard alignment-free technique in the Wu-Hsieh-Li Markov k-string family (ref [25]
    of 25m1786957.pdf: statistical measures of DNA sequence dissimilarity under Markov
    chain models of base composition).

    For a sequence, fit an order-`order` (default 3) Markov chain from ITS OWN base
    composition by counting: for every window of length order+1 (a "word") made of pure
    ACGT characters, let ctx = word[:order] (the length-`order` context) and
    b = word[order] (the next base). Then:

        P(b | ctx)   = (count(ctx,b) + 1) / (count(ctx) + 4)      [Laplace-smoothed]
        P(ctx)       = count(ctx) / sum_ctx' count(ctx')          [empirical context marginal]
        E[word]      = P(ctx) * P(b | ctx)                        [Markov-expected word freq]
        O[word]      = count(word) / total_words                  [observed word freq]

    over all 4^(order+1) canonical words (256 for order=3). The feature vector returned
    is the deviation d[word] = O[word] - E[word] -- how much the sequence's actual
    (order+1)-mer usage departs from what its own order-`order` Markov model would predict.
    Two sequences are then compared via Euclidean distance between their deviation
    vectors (see `markov_distance_matrix`). This differs from Wu-Hsieh-Li's original
    formulation, which computes a single pairwise dissimilarity statistic directly from
    two sequences' word/context counts rather than per-sequence feature vectors compared
    by a generic metric; the per-sequence-vector + Euclidean-distance version used here
    was chosen to fit this project's existing distance-matrix -> UPGMA -> purity pipeline
    (`upgma_linkage`/`average_purity` expect a plain distance matrix from any method).
    """
    ctx_len = order
    word_len = order + 1
    contexts = _all_kmers(ctx_len)
    words = _all_kmers(word_len)
    ctx_idx = {c: i for i, c in enumerate(contexts)}
    word_idx = {w: i for i, w in enumerate(words)}
    n_ctx = len(contexts)

    trans_counts = np.zeros((n_ctx, 4), dtype=float)
    ctx_counts = np.zeros(n_ctx, dtype=float)
    word_counts = np.zeros(len(words), dtype=float)

    for i in range(len(seq) - word_len + 1):
        w = seq[i:i + word_len]
        wi = word_idx.get(w)
        if wi is None:
            continue
        word_counts[wi] += 1.0
        ci = ctx_idx[w[:ctx_len]]
        trans_counts[ci, BASE_IDX[w[ctx_len]]] += 1.0
        ctx_counts[ci] += 1.0

    trans_probs = (trans_counts + 1.0) / (ctx_counts[:, None] + 4.0)  # Laplace smoothing
    ctx_total = ctx_counts.sum()
    ctx_marg = ctx_counts / ctx_total if ctx_total > 0 else np.full(n_ctx, 1.0 / n_ctx)

    total_words = word_counts.sum()
    obs_freq = word_counts / total_words if total_words > 0 else np.zeros(len(words))

    exp_freq = np.zeros(len(words))
    for ci, ctx in enumerate(contexts):
        for bi, b in enumerate(NUCLEOTIDES):
            exp_freq[word_idx[ctx + b]] = ctx_marg[ci] * trans_probs[ci, bi]

    return obs_freq - exp_freq


def markov_distance_matrix(seqs: dict, order: int = 3) -> tuple[np.ndarray, list[str]]:
    from scipy.spatial.distance import pdist, squareform
    accs = list(seqs.keys())
    V = np.stack([markov_deviation_vector(seqs[a], order) for a in accs])
    D = squareform(pdist(V, metric="euclidean"))
    return D, accs


# ---------------------------------------------------------------------------
# 5. FPS -- Fourier Power Spectrum
# ---------------------------------------------------------------------------

def power_spectrum(seq: str) -> np.ndarray:
    """Map `seq` to 4 binary indicator sequences (1 at positions of that base, 0
    elsewhere, incl. 0 at any ambiguity-code position for all 4 indicators), DFT each
    with `numpy.fft.fft`, and sum the squared magnitudes across the 4 transforms:

        S(f) = sum_{b in ACGT} |FFT(indicator_b)(f)|^2

    Returns the full-length (length-L) power spectrum (numpy.fft.fft's output for a
    real-valued input is Hermitian-symmetric, i.e. the second half mirrors the first;
    this full vector is kept rather than only the first half since resampling to a
    common length and the correlation-based comparison below are insensitive to that
    redundancy, and this is the simpler, direct reading of "sum of squared-magnitude DFTs"
    in Hoang et al. (2015)).
    """
    L = len(seq)
    total = np.zeros(L, dtype=float)
    for b in NUCLEOTIDES:
        indicator = np.fromiter((1.0 if c == b else 0.0 for c in seq), dtype=float, count=L)
        F = np.fft.fft(indicator)
        total += (F.real ** 2 + F.imag ** 2)
    return total


def resample_spectrum(spectrum: np.ndarray, n_points: int) -> np.ndarray:
    """Linear interpolation onto `n_points` points evenly spaced over the spectrum's
    normalized [0,1] index range (`numpy.interp`) -- makes power spectra from
    different-length sequences directly comparable coordinate-by-coordinate."""
    L = len(spectrum)
    if L == n_points:
        return spectrum.copy()
    x_old = np.linspace(0.0, 1.0, L)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, spectrum)


def fps_distance_matrix(seqs: dict, n_points: int | None = None
                         ) -> tuple[np.ndarray, list[str], int]:
    """Resample every sequence's power spectrum to a common length (default: the
    SHORTEST sequence length in this dataset, documented choice -- avoids inventing
    high-frequency detail for short sequences via upsampling, at the cost of discarding
    some high-frequency detail from longer ones) and compare pairs by the standard
    correlation-based distance for this method, `1 - Pearson_correlation(spec1, spec2)`.

    -> (D, accs, n_points_used).
    """
    accs = list(seqs.keys())
    lengths = [len(seqs[a]) for a in accs]
    if n_points is None:
        n_points = min(lengths)
    specs = np.stack([resample_spectrum(power_spectrum(seqs[a]), n_points) for a in accs])
    n = len(accs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(specs[i], specs[j])[0, 1]
            d = 1.0 - r
            D[i, j] = D[j, i] = d
    return D, accs, n_points
