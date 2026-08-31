"""CAKL's exact label-based phylogenetic tree purity metric (paper eq. 22-24, verified
against the local PDF, page 18-19: "4.5 Purity metrics for assessing the performance of
phylogenetic analysis methods").

For label l with n(l) leaves, find the maximal subtrees whose leaves are exclusively
labeled l (a partition of S(l) into pure subtrees S_1..S_m). Then:

    purity(l)  = sum_i (|S_i| / n(l))^2                      (eq. 23)
    avg purity = (1 / |L|) * sum_l purity(l)                 (eq. 24)

This is NOT the same as:
  - grassmannian/evaluation.py's lca_precision (single-LCA contamination) in
    KmerTopology-main, or
  - KmerTopology-main/test/run_phylogeny.py's cut-into-k-clusters majority purity.
Neither of those match eq.22-24; this module implements eq.22-24 fresh, per the
discrepancy noted during framework inspection.
"""
from __future__ import annotations

from upgma import node_members, node_parent


def maximal_pure_subtrees(Z, n_leaves: int, labels: list[str], label: str) -> list[set[int]]:
    """Maximal leaf-index sets whose members are ALL `label`, partitioning S(label)."""
    members = node_members(Z, n_leaves)
    parent = node_parent(Z, n_leaves)
    label_set = {i for i, l in enumerate(labels) if l == label}

    def is_pure(node_id: int) -> bool:
        m = members[node_id]
        return len(m) > 0 and m.issubset(label_set)

    maximal = []
    covered = set()
    # Leaves first (guarantees every label(l) leaf is covered even if isolated),
    # then internal nodes largest-membership-first so a big pure clade is recorded
    # before its (now-covered) pure children would otherwise also qualify.
    all_nodes = sorted(members.keys(), key=lambda nid: -len(members[nid]))
    for nid in all_nodes:
        if not is_pure(nid):
            continue
        m = members[nid]
        if m & covered:
            continue  # already inside a larger recorded pure clade
        par = parent.get(nid)
        if par is not None and is_pure(par):
            continue  # not maximal -- its parent is also pure, will be recorded instead
        maximal.append(m)
        covered |= m
    return maximal


def label_purity(Z, n_leaves: int, labels: list[str], label: str) -> float:
    n_l = labels.count(label)
    subtrees = maximal_pure_subtrees(Z, n_leaves, labels, label)
    assert sum(len(s) for s in subtrees) == n_l, (
        f"maximal pure subtrees do not partition S({label}): "
        f"{sum(len(s) for s in subtrees)} != {n_l}"
    )
    return sum((len(s) / n_l) ** 2 for s in subtrees)


def average_purity(Z, n_leaves: int, labels: list[str],
                    exclude_labels: tuple[str, ...] = ()) -> tuple[float, dict[str, float]]:
    """Returns (avg purity, per-label purity dict), eq.24, over labels not in exclude_labels."""
    uniq = sorted(set(labels) - set(exclude_labels))
    per_label = {l: label_purity(Z, n_leaves, labels, l) for l in uniq}
    avg = sum(per_label.values()) / len(uniq) if uniq else float("nan")
    return avg, per_label
