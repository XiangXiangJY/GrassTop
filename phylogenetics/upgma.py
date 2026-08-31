"""UPGMA (average-linkage) tree construction, Newick export, and a labelled tree figure.

Uses scipy.cluster.hierarchy (average linkage IS UPGMA) rather than a bespoke
implementation, and exposes the linkage matrix directly since src/purity.py's CAKL
purity metric (eq.22-24) needs to walk the tree's internal nodes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def upgma_linkage(D: np.ndarray) -> np.ndarray:
    """Average-linkage (UPGMA) clustering on a symmetric zero-diagonal distance matrix."""
    condensed = squareform(D, checks=False)
    return linkage(condensed, method="average")


def node_members(Z: np.ndarray, n_leaves: int) -> dict[int, set[int]]:
    """leaf-index-set for every node id (0..n-1 = leaves, n..2n-2 = internal merges)."""
    members = {i: {i} for i in range(n_leaves)}
    for i, (a, b, _dist, _cnt) in enumerate(Z):
        members[n_leaves + i] = members[int(a)] | members[int(b)]
    return members


def node_parent(Z: np.ndarray, n_leaves: int) -> dict[int, int]:
    parent = {}
    for i, (a, b, _dist, _cnt) in enumerate(Z):
        parent[int(a)] = n_leaves + i
        parent[int(b)] = n_leaves + i
    return parent


def _newick_recursive(node_id: int, n_leaves: int, Z: np.ndarray, labels: list[str],
                       node_height: dict[int, float]) -> str:
    if node_id < n_leaves:
        return labels[node_id]
    a, b, dist, _cnt = Z[node_id - n_leaves]
    a, b = int(a), int(b)
    h = node_height[node_id]
    ha = node_height.get(a, h)
    hb = node_height.get(b, h)
    left = _newick_recursive(a, n_leaves, Z, labels, node_height)
    right = _newick_recursive(b, n_leaves, Z, labels, node_height)
    return f"({left}:{h - ha:.6f},{right}:{h - hb:.6f})"


def to_newick(Z: np.ndarray, labels: list[str]) -> str:
    n_leaves = len(labels)
    node_height = {i: 0.0 for i in range(n_leaves)}
    for i, (_a, _b, dist, _cnt) in enumerate(Z):
        node_height[n_leaves + i] = float(dist)
    root_id = n_leaves + len(Z) - 1
    return _newick_recursive(root_id, n_leaves, Z, labels, node_height) + ";"


def save_tree(D: np.ndarray, labels: list[str], groups: list[str], out_prefix: Path,
              title: str = "") -> np.ndarray:
    """Compute UPGMA linkage, write .nwk, and save a label-colored dendrogram (.pdf + .png).

    Returns the linkage matrix Z.
    """
    Z = upgma_linkage(D)
    newick = to_newick(Z, labels)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".nwk")).write_text(newick)

    uniq = sorted(set(groups))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    color_of = {g: cmap(i % cmap.N) for i, g in enumerate(uniq)}
    leaf_colors = [color_of[g] for g in groups]

    fig_h = max(6, 0.22 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    dn = dendrogram(Z, labels=[f"{l} [{g}]" for l, g in zip(labels, groups)],
                     orientation="left", ax=ax, color_threshold=0, above_threshold_color="black")
    ax.set_title(title)
    ordered_labels = dn["ivl"]
    for tick, lab_text in zip(ax.get_yticklabels(), ordered_labels):
        g = lab_text.split("[")[-1].rstrip("]")
        tick.set_color(color_of.get(g, "black"))
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Z


def _leaf_order(node_id: int, n_leaves: int, children: dict[int, tuple[int, int]]) -> list[int]:
    """Non-crossing left-to-right leaf order induced by the linkage's own (a,b) child order."""
    if node_id < n_leaves:
        return [node_id]
    a, b = children[node_id]
    return _leaf_order(a, n_leaves, children) + _leaf_order(b, n_leaves, children)


def _circular_layout(Z: np.ndarray, n_leaves: int):
    """-> (angle, radius, children, leaf_order, root_id).

    angle[node] in [0, 2*pi): leaves are evenly spaced around the circle in
    non-crossing order; each internal node's angle is the midpoint of its two
    children's angles. radius[node] in [r_root, r_leaf]: leaves sit on the
    outer ring (radius=1.0) and radius shrinks linearly toward the center as
    merge height increases, so the root sits near the center.
    """
    children: dict[int, tuple[int, int]] = {}
    height = {i: 0.0 for i in range(n_leaves)}
    for i, (a, b, dist, _cnt) in enumerate(Z):
        node_id = n_leaves + i
        children[node_id] = (int(a), int(b))
        height[node_id] = float(dist)
    root = n_leaves + len(Z) - 1
    order = _leaf_order(root, n_leaves, children)

    angle: dict[int, float] = {}
    for rank, leaf in enumerate(order):
        angle[leaf] = 2.0 * np.pi * (rank + 0.5) / n_leaves
    for i, (a, b, _dist, _cnt) in enumerate(Z):
        node_id = n_leaves + i
        angle[node_id] = 0.5 * (angle[int(a)] + angle[int(b)])

    max_h = max(height.values()) if height else 1.0
    max_h = max_h if max_h > 0 else 1.0
    r_leaf, r_root = 1.0, 0.05
    radius = {nid: r_leaf - (h / max_h) * (r_leaf - r_root) for nid, h in height.items()}
    return angle, radius, children, order, root


def _short_display_labels(labels: list[str], max_len: int = 8) -> list[str]:
    """Strip any prefix shared by every label (e.g. GISAID's 'EPI_ISL_'), then truncate
    the remainder to the SHORTEST length >= max_len that keeps all labels distinct in
    this tree (grown one character at a time if `max_len` collides -- e.g. sequential
    RefSeq ids like NC_005275.1/NC_005270.1 only start differing after 8 characters).
    Falls back to the untouched original labels if even the full remainder collides
    (duplicate accessions) -- this function only ever removes ambiguity, never adds it.
    """
    if not labels:
        return []
    lcp = labels[0]
    for s in labels[1:]:
        k = min(len(lcp), len(s))
        while k and lcp[:k] != s[:k]:
            k -= 1
        lcp = lcp[:k]
        if not lcp:
            break
    cut = len(lcp)
    remainders = [s[cut:] if s[cut:] else s for s in labels]
    max_full = max((len(s) for s in remainders), default=0)
    length = min(max_len, max_full) if max_full else max_len
    while length <= max_full:
        candidate = [s[:length] for s in remainders]
        if len(set(candidate)) == len(candidate):
            return candidate
        length += 1
    return list(labels)


EMBED_WIDTH_IN = 3.05  # target print width in the paper (0.47*textwidth of an 11pt article)
TARGET_LEAF_PT = 4.0   # apparent leaf-label size once embedded -- small enough to sit
                       # inside the angular gap between branch tips without crowding them
TARGET_TITLE_PT = 6.5  # apparent title size once embedded


def save_circular_tree(D: np.ndarray, labels: list[str], groups: list[str], out_prefix: Path,
                        title: str = "", label_fontsize: float | None = None,
                        max_label_chars: int = 5, embed_width_in: float = EMBED_WIDTH_IN
                        ) -> np.ndarray:
    """Compute UPGMA linkage, write .nwk, and save a circular (iTOL-style) dendrogram.

    Leaves sit on the outer ring in non-crossing order; branches are colored by the
    label of the largest pure (monophyletic) clade they belong to (matching CAKL's own
    Fig. 2a-c / Supp. Fig. S1-S7 style), falling back to black wherever a merge crosses
    between different labels.
    """
    Z = upgma_linkage(D)
    newick = to_newick(Z, labels)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".nwk")).write_text(newick)

    n = len(labels)
    angle, radius, children, order, root = _circular_layout(Z, n)

    uniq = sorted(set(groups))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    color_of = {g: cmap(i % cmap.N) for i, g in enumerate(uniq)}
    black = (0.0, 0.0, 0.0, 1.0)

    node_label: dict[int, str | None] = {i: groups[i] for i in range(n)}
    node_color: dict[int, tuple] = {i: color_of[groups[i]] for i in range(n)}
    for i, (a, b, _dist, _cnt) in enumerate(Z):
        node_id, a, b = n + i, int(a), int(b)
        la, lb = node_label.get(a), node_label.get(b)
        if la is not None and la == lb:
            node_label[node_id] = la
            node_color[node_id] = color_of[la]
        else:
            node_label[node_id] = None
            node_color[node_id] = black

    def xy(a_: float, r_: float) -> tuple[float, float]:
        return r_ * np.cos(a_), r_ * np.sin(a_)

    short_labels = _short_display_labels(labels, max_len=max_label_chars)
    side = min(20.0, max(11.0, 8.0 + 0.07 * n))
    # Scale source point-sizes with the source canvas so that, once every dataset's
    # figure is embedded at the SAME print width (embed_width_in), the apparent
    # (on-page) font size is the same across datasets regardless of leaf count. Pass a
    # larger embed_width_in for a dataset given more page width (e.g. a dense, high-n
    # tree given a full-textwidth figure instead of a quarter-page slot) so its labels
    # end up the same apparent size as the others without being more cramped.
    if label_fontsize is None:
        label_fontsize = TARGET_LEAF_PT * side / embed_width_in
    title_fontsize = TARGET_TITLE_PT * side / embed_width_in

    fig, ax = plt.subplots(figsize=(side, side))
    ax.set_aspect("equal")
    for i, (a, b, _dist, _cnt) in enumerate(Z):
        node_id, a, b = n + i, int(a), int(b)
        r_parent = radius[node_id]
        for child in (a, b):
            x0, y0 = xy(angle[child], radius[child])
            x1, y1 = xy(angle[child], r_parent)
            ax.plot([x0, x1], [y0, y1], color=node_color[child], linewidth=1.1,
                     solid_capstyle="round")
        a_lo, a_hi = sorted((angle[a], angle[b]))
        n_pts = max(int((a_hi - a_lo) / 0.01) + 2, 2)
        thetas = np.linspace(a_lo, a_hi, n_pts)
        ax.plot(r_parent * np.cos(thetas), r_parent * np.sin(thetas),
                 color=node_color[node_id], linewidth=1.1, solid_capstyle="round")

    # Stagger label start radius in two tiers (odd/even leaf rank) so that, at high leaf
    # counts, adjacent labels are not competing for the exact same radial band -- this
    # trades a slightly larger plot radius for real separation instead of shrinking text.
    stagger = 0.05 if n > 70 else 0.0
    label_r = [1.04 + (stagger if i % 2 else 0.0) for i in range(n)]

    for rank, leaf in enumerate(order):
        a_ = angle[leaf]
        x, y = xy(a_, label_r[rank])
        deg = np.degrees(a_)
        right_side = -90.0 < deg <= 90.0
        ax.text(x, y, short_labels[leaf], color=node_color[leaf],
                fontsize=label_fontsize, rotation=deg if right_side else deg + 180,
                rotation_mode="anchor", ha="left" if right_side else "right", va="center")

    lim = 1.32 + stagger
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=title_fontsize * 1.4)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Z
