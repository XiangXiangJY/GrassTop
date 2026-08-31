"""Plot label and self preservation versus perturbation rate across the full rate grid.

Table stability in the paper only shows 4 selected rates and is hard to read as a trend;
this script renders the full grid from results/stability/*.json as line plots instead, one
column per dataset, so the G-vs-F pattern over the whole rate range is visible at a glance.

Usage:
    python plot_stability_curves.py --out /path/to/stability_curves.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS = os.path.join(HERE, "..", "results", "stability")

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

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True)

    for col, (key, title) in enumerate(DATASETS):
        d = load(key)
        rows = sorted(d["summary_by_rate"], key=lambda r: r["rate"])
        rates = [r["rate"] for r in rows]
        g_label = [r["label_preservation"] for r in rows]
        f_label = [r["flat_label_preservation"] for r in rows]
        g_self = [r["self_preservation"] for r in rows]
        f_self = [r["flat_self_preservation"] for r in rows]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(rates, g_label, color=G_COLOR, marker="o", ms=4, lw=1.6, label="G (GrassTop)")
        ax_top.plot(rates, f_label, color=F_COLOR, marker="s", ms=4, lw=1.6, ls="--", label="F (flat)")
        ax_bot.plot(rates, g_self, color=G_COLOR, marker="o", ms=4, lw=1.6, label="G (GrassTop)")
        ax_bot.plot(rates, f_self, color=F_COLOR, marker="s", ms=4, lw=1.6, ls="--", label="F (flat)")

        for ax in (ax_top, ax_bot):
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.05)
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=LABEL_PT)

        ax_top.set_title(title, fontsize=LABEL_PT)

    axes[0, 0].set_ylabel("Label\npreservation", fontsize=LABEL_PT)
    axes[1, 0].set_ylabel("Self\npreservation", fontsize=LABEL_PT)
    for col in range(4):
        axes[1, col].set_xlabel("Perturbation rate", fontsize=LABEL_PT)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=LABEL_PT)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
