"""Driver: raw FASTA + labels.csv -> 5 self-implemented alignment-free baseline methods
(NVM, FFP-JS, FFP-KL, Markov k-string, FPS) -> UPGMA tree -> CAKL's exact label-based
purity metric (eq. 22-24 of arXiv:2508.09406) -- the SAME `upgma_linkage` +
`average_purity` functions used to score GrassTop in `run_purity_eval.py`, so all 6
methods are scored under one internally-consistent metric rather than mixing numbers
copied from different published sources with possibly different tree-building/scoring
conventions.

Formula/parameter choices for each method are documented in `baseline_methods.py`'s
module docstring and per-function docstrings.

Usage:
    python -m phylogenetics.run_baseline_purity_eval \\
        --fasta .../ebolavirus.fasta --labels data/labels/ebolavirus_labels.csv \\
        --out results/phylogenetics_baselines_selfcomputed/ebolavirus_baselines.json \\
        --dataset-name ebolavirus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src import io_utils as IO          # noqa: E402
from upgma import upgma_linkage         # noqa: E402
from purity import average_purity       # noqa: E402
import baseline_methods as BM           # noqa: E402

FFP_K = 3
MARKOV_ORDER = 3
NVM_MAX_MOMENT = 5


def load_labels(path: str) -> dict:
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:2] == ["accession", "family"], f"expected 'accession,family' header, got {header}"
    return {r[0]: r[1] for r in body}


def score(D, accs, labels_by_acc):
    group_labels = [labels_by_acc[a] for a in accs]
    Z = upgma_linkage(D)
    avg_purity, per_label = average_purity(Z, len(accs), group_labels)
    return avg_purity, per_label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", required=True)
    args = ap.parse_args()

    all_seqs = IO.parse_fasta(args.fasta)
    labels = load_labels(args.labels)
    accs_sorted = sorted(a for a in labels if a in all_seqs)
    seqs = {a: all_seqs[a] for a in accs_sorted}
    print(f"{args.dataset_name}: {len(seqs)} labeled sequences with FASTA records "
          f"({len(set(labels[a] for a in accs_sorted))} labels)", flush=True)

    results = {
        "dataset": args.dataset_name,
        "n": len(seqs),
        "n_labels": len(set(labels[a] for a in accs_sorted)),
        "fasta": os.path.abspath(args.fasta),
        "labels_csv": os.path.abspath(args.labels),
        "methods": {},
    }

    t0 = time.time()
    D, accs = BM.nvm_distance_matrix(seqs)
    avg, per_label = score(D, accs, labels)
    results["methods"]["NVM"] = {
        "average_purity": avg, "per_label_purity": per_label,
        "params": {"max_moment_order": NVM_MAX_MOMENT, "vector_dim": 24,
                    "unit": "single nucleotide (A/C/G/T)", "compare": "euclidean"},
        "formula_note": ("classic Deng/Yu/Yau/Yau natural vector, single-nucleotide "
                          "(NOT k-mer) moments up to order 5; see baseline_methods.py "
                          "natural_vector() docstring for the exact interpretive choice"),
    }
    print(f"  NVM done ({time.time()-t0:.1f}s): avg_purity={avg:.4f}", flush=True)

    t0 = time.time()
    D_js, D_kl, accs = BM.ffp_distance_matrices(seqs, k=FFP_K, pseudocount=1.0)
    avg_js, per_label_js = score(D_js, accs, labels)
    avg_kl, per_label_kl = score(D_kl, accs, labels)
    results["methods"]["FFP-JS"] = {
        "average_purity": avg_js, "per_label_purity": per_label_js,
        "params": {"k": FFP_K, "pseudocount": 1.0, "log_base": "natural (ln)",
                    "compare": "Jensen-Shannon divergence"},
    }
    results["methods"]["FFP-KL"] = {
        "average_purity": avg_kl, "per_label_purity": per_label_kl,
        "params": {"k": FFP_K, "pseudocount": 1.0, "log_base": "natural (ln)",
                    "compare": "symmetrized KL divergence 0.5*(KL(P||Q)+KL(Q||P))"},
    }
    print(f"  FFP-JS/KL done ({time.time()-t0:.1f}s): "
          f"avg_purity JS={avg_js:.4f} KL={avg_kl:.4f}", flush=True)

    t0 = time.time()
    D, accs = BM.markov_distance_matrix(seqs, order=MARKOV_ORDER)
    avg, per_label = score(D, accs, labels)
    results["methods"]["Markov"] = {
        "average_purity": avg, "per_label_purity": per_label,
        "params": {"order": MARKOV_ORDER, "word_len": MARKOV_ORDER + 1,
                    "smoothing": "Laplace add-1 on transition counts", "compare": "euclidean",
                    "feature": "observed minus Markov-expected (k+1)-mer frequency deviation vector"},
    }
    print(f"  Markov done ({time.time()-t0:.1f}s): avg_purity={avg:.4f}", flush=True)

    t0 = time.time()
    D, accs, n_points = BM.fps_distance_matrix(seqs, n_points=None)
    avg, per_label = score(D, accs, labels)
    results["methods"]["FPS"] = {
        "average_purity": avg, "per_label_purity": per_label,
        "params": {"resample_n_points": n_points,
                    "resample_rule": "shortest sequence length in this dataset",
                    "compare": "1 - Pearson correlation of resampled power spectra"},
    }
    print(f"  FPS done ({time.time()-t0:.1f}s, resampled to {n_points} pts): "
          f"avg_purity={avg:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
