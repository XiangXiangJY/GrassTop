"""Driver: synthetic point-mutation stability ablation for gr_core_proj_r10.

For a stratified sample of "seed" sequences, generates mutated copies at several
substitution rates and scores, via a k-NN majority vote against the full labeled pool,
whether the mutant's neighbors still recover (a) the seed's own unmutated original
(self-preservation) and (b) the seed's ground-truth label (label-preservation). This is
the SAME evaluation rule as `classification/run_5nn_eval.py` (k=5 by default), so the
stability result is directly comparable to, not a different protocol from, the
classification benchmark.

Also scores a "flat" control in parallel: the SAME evaluation rule, but distances are the
raw Frobenius distance between un-projected Z_S feature stacks (`src.representation.
build_Z`), never reduced to a Grassmann subspace. This isolates the SVD/subspace step's
own marginal contribution to noise robustness, per `paper/grassmannian_kmer_topology.tex`
Section 5's framing. Ported from `paper_reproduction/stability/run_all_5nn.py`'s
`d_kmertopology_flat_from_original` logic, adapted to this release's label-stratified
seed sampling (that script used a fixed, non-stratified 4-seeds/dataset subsample --
see README.md's "Stability flat baseline" note).

Usage:
    python -m stability.run_stability_eval --fasta data/rhinovirus.fasta \\
        --labels data/rhinovirus_labels.csv --out results/rhinovirus_stability.json \\
        --dataset-name rhinovirus
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src import kmer_features as KF   # noqa: E402
from src import representation as R   # noqa: E402
from src import distances as DI       # noqa: E402
from src import io_utils as IO        # noqa: E402
from mutate import mutate_sequence     # noqa: E402

KS = (1, 2, 3, 4, 5)
RANK = 10
RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4]


def load_labels(path: str) -> dict:
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:2] == ["accession", "family"], f"expected 'accession,family' header, got {header}"
    return {r[0]: r[1] for r in body}


def stratified_seeds(accs, labels, n_per_label, seed):
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for a in accs:
        by_label[labels[a]].append(a)
    seeds = []
    for lab in sorted(by_label):
        pool = sorted(by_label[lab])
        k = min(n_per_label, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        seeds.extend(pool[i] for i in idx)
    return seeds


def knn_majority_label(sorted_dists, labels, k):
    topk = sorted_dists[:k]
    cand = [labels[acc] for _, acc in topk]
    counts = Counter(cand)
    top = max(counts.values())
    tied = {l for l, c in counts.items() if c == top}
    if len(tied) == 1:
        return next(iter(tied))
    for _, acc in topk:
        if labels[acc] in tied:
            return labels[acc]
    return cand[0]


def knn_includes_self(sorted_dists, self_acc, k):
    return self_acc in {acc for _, acc in sorted_dists[:k]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", required=True, help="used only to derive deterministic mutation seeds")
    ap.add_argument("--channels", default="B,MEANP,STDP",
                     help="E is excluded by default: too expensive to recompute per mutant at scale")
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--n-seeds-per-label", type=int, default=1)
    ap.add_argument("--n-replicates", type=int, default=5)
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    channels = tuple(args.channels.split(","))
    ks = tuple(int(k) for k in args.ks.split(","))
    rates = [float(r) for r in args.rates.split(",")]

    all_seqs = IO.parse_fasta(args.fasta)
    labels = load_labels(args.labels)
    accs = sorted(a for a in labels if a in all_seqs)
    print(f"pool: {len(accs)} sequences, {len(set(labels[a] for a in accs))} labels", flush=True)

    t0 = time.time()
    pool_blocks = {acc: KF.sequence_feature_blocks(all_seqs[acc], channels, ks) for acc in accs}
    scales = R.fit_scales(list(pool_blocks.values()), channels, ks)
    coeffs = R.block_coeffs(scales, channels, ks)
    pool_U = {acc: R.sequence_to_point(pool_blocks[acc], coeffs, channels, ks, args.rank).U[:, : args.rank]
              for acc in accs}
    pool_Z = {acc: R.build_Z(pool_blocks[acc], coeffs, channels, ks) for acc in accs}
    print(f"pool representations built in {time.time()-t0:.0f}s", flush=True)

    U_stack = np.stack([pool_U[a] for a in accs])

    # Reference scale: typical (mean/median) pairwise distance within the unmutated pool,
    # for both arms -- lets self-distance be read as "what fraction of the way toward a
    # random other pool sequence has this mutant drifted" rather than a raw, differently
    # scaled number for G (bounded, max sqrt(rank)) vs F (unbounded Frobenius norm).
    D_pool_g = DI.grassmann_distance_matrix(U_stack)
    iu = np.triu_indices(len(accs), k=1)
    ref_mean_g = float(D_pool_g[iu].mean())
    ref_median_g = float(np.median(D_pool_g[iu]))
    # Gram-matrix trick (avoids an O(n^2 x D) broadcast tensor for the ~200k-dim flat
    # feature vectors): ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    Z_stack_flat = np.stack([pool_Z[a] for a in accs]).reshape(len(accs), -1).astype(np.float64)
    sq_norms = np.sum(Z_stack_flat ** 2, axis=1)
    gram = Z_stack_flat @ Z_stack_flat.T
    D2_pool_flat = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram, 0.0)
    D_pool_flat = np.sqrt(D2_pool_flat)
    ref_mean_flat = float(D_pool_flat[iu].mean())
    ref_median_flat = float(np.median(D_pool_flat[iu]))
    print(f"pool reference distance -- G: mean={ref_mean_g:.4f} median={ref_median_g:.4f}; "
          f"F: mean={ref_mean_flat:.4f} median={ref_median_flat:.4f}", flush=True)

    def dists_to_pool(U_query):
        d = DI.grassmann_distances_to_pool(U_query, U_stack)
        return sorted(zip(d.tolist(), accs), key=lambda x: x[0])

    def flat_dists_to_pool(Z_query):
        return sorted(((float(np.linalg.norm(Z_query - pool_Z[a])), a) for a in accs),
                       key=lambda x: x[0])

    def self_dist(sorted_pairs, self_acc):
        for dist, acc in sorted_pairs:
            if acc == self_acc:
                return dist
        raise KeyError(self_acc)

    seeds = stratified_seeds(accs, labels, args.n_seeds_per_label, args.seed)
    print(f"{len(seeds)} seed sequences", flush=True)

    all_results = []
    for seed_acc in seeds:
        seed_label = labels[seed_acc]
        for rate in rates:
            for rep in range(args.n_replicates):
                mutated = mutate_sequence(all_seqs[seed_acc], rate, args.dataset_name, seed_acc, rep)
                blocks_mut = KF.sequence_feature_blocks(mutated, channels, ks)
                point_mut = R.sequence_to_point(blocks_mut, coeffs, channels, ks, args.rank)
                Z_mut = R.build_Z(blocks_mut, coeffs, channels, ks)
                dists = dists_to_pool(point_mut.U[:, : args.rank])
                flat_dists = flat_dists_to_pool(Z_mut)
                all_results.append({
                    "seed_acc": seed_acc, "label": seed_label, "rate": rate, "replicate": rep,
                    "knn_majority_same_label": knn_majority_label(dists, labels, args.knn) == seed_label,
                    "knn_includes_self": knn_includes_self(dists, seed_acc, args.knn),
                    "flat_knn_majority_same_label":
                        knn_majority_label(flat_dists, labels, args.knn) == seed_label,
                    "flat_knn_includes_self": knn_includes_self(flat_dists, seed_acc, args.knn),
                    "self_dist_g": self_dist(dists, seed_acc),
                    "self_dist_flat": self_dist(flat_dists, seed_acc),
                })

    agg = defaultdict(list)
    for r in all_results:
        agg[r["rate"]].append(r)
    summary = [{
        "rate": rate,
        "n": len(rows),
        "label_preservation": float(np.mean([r["knn_majority_same_label"] for r in rows])),
        "self_preservation": float(np.mean([r["knn_includes_self"] for r in rows])),
        "flat_label_preservation": float(np.mean([r["flat_knn_majority_same_label"] for r in rows])),
        "flat_self_preservation": float(np.mean([r["flat_knn_includes_self"] for r in rows])),
        "mean_self_dist_g": float(np.mean([r["self_dist_g"] for r in rows])),
        "mean_self_dist_flat": float(np.mean([r["self_dist_flat"] for r in rows])),
        "median_self_dist_g": float(np.median([r["self_dist_g"] for r in rows])),
        "median_self_dist_flat": float(np.median([r["self_dist_flat"] for r in rows])),
    } for rate, rows in sorted(agg.items())]
    for row in summary:
        print(f"  rate={row['rate']}: label_preservation={row['label_preservation']:.3f} "
              f"self_preservation={row['self_preservation']:.3f} "
              f"flat_label_preservation={row['flat_label_preservation']:.3f} "
              f"flat_self_preservation={row['flat_self_preservation']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "config": {"channels": list(channels), "ks": list(ks), "rank": args.rank,
                       "knn": args.knn, "n_seeds_per_label": args.n_seeds_per_label,
                       "n_replicates": args.n_replicates, "rates": rates, "seed": args.seed},
            "pool_reference_distance": {
                "g_mean": ref_mean_g, "g_median": ref_median_g,
                "flat_mean": ref_mean_flat, "flat_median": ref_median_flat,
            },
            "summary_by_rate": summary, "records": all_results,
        }, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
