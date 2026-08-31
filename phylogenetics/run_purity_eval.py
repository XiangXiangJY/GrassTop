"""Driver: raw FASTA + labels.csv -> gr_core_proj_r10 subspaces -> UPGMA tree -> CAKL's
label-based purity metric (eq. 22-24 of arXiv:2508.09406).

Unsupervised: labels are used only to score the tree after it is built, never to build
the representation or the tree itself. The RMS normalization scale is pool-wide (fit on
every sequence in the dataset -- there is no train/test split for this task).

Usage:
    python -m phylogenetics.run_purity_eval --fasta data/rhinovirus.fasta \\
        --labels data/rhinovirus_labels.csv --out results/rhinovirus_purity.json \\
        [--tree-figure results/rhinovirus_tree]

The paper's primary phylogenetics configuration uses channel B only (see
`paper/grassmannian_kmer_topology.tex` Sec. 4); pass `--channels B,E,MEANP,STDP` to try
the full 4-channel set instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src import kmer_features as KF   # noqa: E402
from src import representation as R   # noqa: E402
from src import distances as DI       # noqa: E402
from src import io_utils as IO        # noqa: E402
from upgma import upgma_linkage, save_tree, save_circular_tree   # noqa: E402
from purity import average_purity            # noqa: E402
from bootstrap import bootstrap_ci_labels, bootstrap_ci_leaves  # noqa: E402

KS = (1, 2, 3, 4, 5)
RANK = 10


def load_labels(path: str) -> dict:
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:2] == ["accession", "family"], f"expected 'accession,family' header, got {header}"
    return {r[0]: r[1] for r in body}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default="B")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--tree-figure", default=None, help="optional output prefix for a .nwk/.pdf/.png dendrogram")
    ap.add_argument("--circular-tree-figure", default=None,
                     help="optional output prefix for a circular (iTOL-style) .nwk/.pdf/.png dendrogram")
    ap.add_argument("--dataset-title", default=None,
                     help="human-readable dataset name shown above the circular tree (e.g. 'SARS-CoV-2'); omit for no title")
    ap.add_argument("--embed-width-in", type=float, default=None,
                     help="target print width (inches) this circular figure will be embedded at in the paper; "
                          "controls label/title font scaling so apparent size matches other figures embedded "
                          "at a different width (e.g. a full-textwidth figure vs. a quarter-page one)")
    args = ap.parse_args()
    channels = tuple(args.channels.split(","))
    ks = tuple(int(k) for k in args.ks.split(","))

    all_seqs = IO.parse_fasta(args.fasta)
    labels = load_labels(args.labels)
    accs = sorted(a for a in labels if a in all_seqs)
    print(f"{len(accs)} labeled sequences with FASTA records "
          f"({len(set(labels[a] for a in accs))} labels)", flush=True)

    t0 = time.time()
    blocks_by_seq = {acc: KF.sequence_feature_blocks(all_seqs[acc], channels, ks) for acc in accs}
    print(f"features built in {time.time()-t0:.0f}s", flush=True)

    scales = R.fit_scales(list(blocks_by_seq.values()), channels, ks)
    coeffs = R.block_coeffs(scales, channels, ks)

    U_list = [R.sequence_to_point(blocks_by_seq[acc], coeffs, channels, ks, args.rank).U[:, : args.rank]
              for acc in accs]
    U_all = np.stack(U_list)
    D = DI.grassmann_distance_matrix(U_all)

    group_labels = [labels[a] for a in accs]
    Z = upgma_linkage(D)
    avg_purity, per_label = average_purity(Z, len(accs), group_labels)
    print(f"average purity: {avg_purity:.4f}")
    for lab, p in sorted(per_label.items()):
        print(f"  {lab}: {p:.4f}")

    def purity_fn(D_sub, labels_sub):
        Z_sub = upgma_linkage(D_sub)
        avg, _ = average_purity(Z_sub, len(labels_sub), labels_sub)
        return avg

    ci_labels_point, ci_labels_lo, ci_labels_hi = bootstrap_ci_labels(per_label)
    ci_leaves_point, ci_leaves_lo, ci_leaves_hi = bootstrap_ci_leaves(
        D, group_labels, purity_fn, n_boot=300)
    print(f"95% leaf-level bootstrap CI: [{ci_leaves_lo:.4f}, {ci_leaves_hi:.4f}]")

    if args.tree_figure:
        save_tree(D, accs, group_labels, Path(args.tree_figure),
                  title=f"UPGMA, GrassTop, channels={channels}")
    if args.circular_tree_figure:
        kwargs = {}
        if args.embed_width_in is not None:
            kwargs["embed_width_in"] = args.embed_width_in
        save_circular_tree(D, accs, group_labels, Path(args.circular_tree_figure),
                            title=args.dataset_title or "", **kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "n": len(accs), "n_labels": len(set(group_labels)), "channels": list(channels),
            "ks": list(ks), "rank": args.rank,
            "average_purity": avg_purity, "per_label_purity": per_label,
            "bootstrap_ci_over_labels": {
                "point": ci_labels_point, "lo": ci_labels_lo, "hi": ci_labels_hi, "alpha": 0.05,
            },
            "bootstrap_ci_over_leaves": {
                "point": ci_leaves_point, "lo": ci_leaves_lo, "hi": ci_leaves_hi, "alpha": 0.05,
                "n_boot": 300,
            },
        }, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
