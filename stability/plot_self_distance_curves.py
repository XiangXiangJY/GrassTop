"""Plot normalized distance to the original sequence versus perturbation rate.

Direct test of the paper's own stated mechanism (Section 5): the Grassmann projection
distance is bounded (max = sqrt(rank)) and should saturate as mutation load grows,
whereas the flat Frobenius distance is unbounded and should grow faster than linearly.
Unlike the discretized 5-NN preservation metrics (Table 8 / Figure 2), this uses the raw
continuous distance from each mutant to its own unmutated original, normalized by the
typical (mean) pairwise distance within that dataset's unmutated pool -- so both arms are
on a comparable "fraction of a typical between-sequence gap" scale despite G being
bounded by sqrt(rank) and F being an unbounded Frobenius norm on ~10^5-dim vectors.

Requires results/stability_dist/*.json (produced by a rerun of run_stability_eval.py that
additionally records self_dist_g/self_dist_flat per record and pool_reference_distance at
the top level -- the original results/stability/*.json used for Table 8 / Figure 2 do not
have these fields).

Usage:
    python plot_self_distance_curves.py --out /path/to/self_distance_curves.pdf
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, "..", "results", "stability_dist")

DATASETS = [
    ("sars_cov_2", "SARS-CoV-2"),
    ("mammalian_mitochondria", "Mammalian\nmitochondrial"),
    ("ebolavirus", "Ebolavirus"),
    ("rhinovirus", "Rhinovirus"),
]

G_COLOR = "#1f77b4"
F_COLOR = "#d95f02"

LABEL_PT = 12  # one consistent size for title, axis labels, tick labels, and legend


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, f"{name}_stability.json")) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True)

    for col, (key, title) in enumerate(DATASETS):
        d = load(key)
        ref = d["pool_reference_distance"]
        by_rate = defaultdict(lambda: ([], []))
        for r in d["records"]:
            g, f = by_rate[r["rate"]]
            g.append(r["self_dist_g"] / ref["g_mean"])
            f.append(r["self_dist_flat"] / ref["flat_mean"])
        rates = sorted(by_rate)
        g_mean = [float(np.mean(by_rate[r][0])) for r in rates]
        f_mean = [float(np.mean(by_rate[r][1])) for r in rates]
        g_lo = [float(np.percentile(by_rate[r][0], 25)) for r in rates]
        g_hi = [float(np.percentile(by_rate[r][0], 75)) for r in rates]
        f_lo = [float(np.percentile(by_rate[r][1], 25)) for r in rates]
        f_hi = [float(np.percentile(by_rate[r][1], 75)) for r in rates]

        ax = axes[col]
        ax.fill_between(rates, g_lo, g_hi, color=G_COLOR, alpha=0.15, linewidth=0)
        ax.fill_between(rates, f_lo, f_hi, color=F_COLOR, alpha=0.15, linewidth=0)
        ax.plot(rates, g_mean, color=G_COLOR, marker="o", ms=4, lw=1.8, label="G (GrassTop)")
        ax.plot(rates, f_mean, color=F_COLOR, marker="s", ms=4, lw=1.8, ls="--", label="F (flat)")

        ax.set_xscale("log")
        ax.set_title(title, fontsize=LABEL_PT)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Perturbation rate", fontsize=LABEL_PT)
        ax.tick_params(labelsize=LABEL_PT)

    axes[0].set_ylabel("Distance to original /\nmean distance in original set", fontsize=LABEL_PT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=LABEL_PT)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
