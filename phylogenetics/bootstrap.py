"""Bootstrap confidence intervals for small phylogenetic-purity datasets."""
from __future__ import annotations

import numpy as np


def bootstrap_ci_labels(values_by_label: dict[str, float], n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the average purity, resampling LABELS with replacement
    (the natural unit here, since purity is defined per-label and datasets have few
    labels -- 8, 8, 4 -- so this CI should be read as indicative, not precise; see report).

    Returns (point_estimate, lo, hi).
    """
    labels = list(values_by_label)
    vals = np.array([values_by_label[l] for l in labels])
    point = float(vals.mean())
    if len(labels) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(labels)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = vals[idx].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_ci_leaves(D: np.ndarray, labels: list[str], purity_fn, n_boot: int = 500,
                         seed: int = 0, alpha: float = 0.05, min_per_label: int = 2
                         ) -> tuple[float, float, float]:
    """Percentile bootstrap CI resampling SEQUENCES (leaves) within each label (stratified,
    so every bootstrap replicate keeps every label represented), rebuilding the UPGMA tree
    and average purity each time. `purity_fn(D_sub, labels_sub) -> float`.
    """
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    uniq = sorted(set(labels))
    idx_by_label = {l: np.where(labels_arr == l)[0] for l in uniq}
    usable = [l for l in uniq if len(idx_by_label[l]) >= min_per_label]
    if not usable:
        p = purity_fn(D, labels)
        return p, p, p
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(idx_by_label[l], size=len(idx_by_label[l]), replace=True)
                               for l in uniq])
        D_sub = D[np.ix_(idx, idx)]
        labels_sub = [labels[i] for i in idx]
        boot.append(purity_fn(D_sub, labels_sub))
    boot = np.array(boot)
    point = purity_fn(D, labels)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
