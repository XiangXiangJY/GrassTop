"""Driver: gr_core_proj_r10 subspaces -> full pairwise projection-distance matrix ->
Hozumi & Wei's 5-NN classification protocol.

Two feature-source modes:

  --cache-dataset NAME (default path, matches how this paper's published classification
    numbers were actually produced): load precomputed B/E/MEANP/STDP blocks from this
    workshop's on-disk `data/<dataset>/features*/` caches (originally extracted by
    KmerTopology-main's own pipeline; this project's Grassmannian framework was always
    built on top of those, not a from-scratch re-extraction at NCBI scale -- channel E
    in particular is far too expensive to recompute per run over thousands of real NCBI
    sequences, some exceeding 1 Mb). The SVD / eq.16 weighting / projection-distance /
    5-NN-eval steps below are the SAME unified `src/` code used by phylogenetics and
    stability -- only the raw feature INPUT comes from a cache here instead of being
    recomputed from raw sequence.

  --fasta/--labels (recompute from scratch): use `src/kmer_features.py` to build every
    channel from a raw FASTA, exactly like phylogenetics/stability do. Fine for small
    datasets or smoke tests; impractical for a full NCBI dataset's channel E (budget
    real multi-day compute if you actually want this path at full scale).

Usage:
    python -m classification.run_5nn_eval --cache-dataset ncbi2020 \\
        --out results/ncbi2020.json
    python -m classification.run_5nn_eval --fasta data/ncbi2020.fasta \\
        --labels data/ncbi2020_labels.csv --out results/ncbi2020.json

Expects a two-column `labels.csv` with header `accession,family`. The RMS normalization
scale is fit ONCE over the whole dataset (not per-CV-fold): not a train/test leakage
concern, since the scale is a fixed, label-independent numeric normalization (no learned
parameter ever sees a label) -- see `src/representation.py` and this project's original
`grassmannian_gnn_v1`/`grassmannian/` packages, which use the same "fit once" discipline
for this exact reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src import kmer_features as KF   # noqa: E402
from src import representation as R   # noqa: E402
from src import distances as DI       # noqa: E402
from src import io_utils as IO        # noqa: E402
import paper_5nn_eval as PEV          # noqa: E402
import multidata as MD                # noqa: E402

CHANNELS = ("B", "E", "MEANP", "STDP")
KS = (1, 2, 3, 4, 5)
RANK = 10


def load_labels(path: str) -> dict:
    import csv
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:2] == ["accession", "family"], f"expected 'accession,family' header, got {header}"
    return {r[0]: r[1] for r in body}


def load_from_cache(dataset: str, channels, ks, limit: int):
    """-> (accs, labels_dict, blocks_by_seq) from this workshop's on-disk feature caches."""
    ds = MD.load_labels(dataset)
    data, info = MD.load_channels(dataset, channels, ks=ks, verbose=True)
    accs = list(ds.accessions)
    labels = dict(zip(ds.accessions.tolist(), ds.families.tolist()))
    if limit > 0:
        accs = accs[:limit]
    acc_to_idx = {a: i for i, a in enumerate(ds.accessions.tolist())}
    blocks_by_seq = {}
    for acc in accs:
        idx = acc_to_idx[acc]
        blocks_by_seq[acc] = {(c, k): data[c][k][idx] for c in channels for k in ks}
    return accs, labels, blocks_by_seq


def load_from_fasta(fasta_path: str, labels_path: str, channels, ks, limit: int):
    all_seqs = IO.parse_fasta(fasta_path)
    labels = load_labels(labels_path)
    accs = [a for a in labels if a in all_seqs]
    if limit > 0:
        accs = accs[:limit]
    t0 = time.time()
    blocks_by_seq = {}
    for i, acc in enumerate(accs):
        blocks_by_seq[acc] = KF.sequence_feature_blocks(all_seqs[acc], channels, ks)
        if (i + 1) % 200 == 0:
            print(f"  features {i+1}/{len(accs)} ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"features built in {time.time()-t0:.0f}s", flush=True)
    return accs, labels, blocks_by_seq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dataset", default=None,
                     help="load precomputed features for this dataset name (e.g. ncbi2020) "
                          "instead of recomputing from FASTA")
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default=",".join(CHANNELS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--min-family-size", type=int, default=15)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--limit", type=int, default=-1, help="use only the first N sequences (smoke test)")
    args = ap.parse_args()
    channels = tuple(args.channels.split(","))
    ks = tuple(int(k) for k in args.ks.split(","))

    if args.cache_dataset:
        accs, labels, blocks_by_seq = load_from_cache(args.cache_dataset, channels, ks, args.limit)
    elif args.fasta and args.labels:
        accs, labels, blocks_by_seq = load_from_fasta(args.fasta, args.labels, channels, ks, args.limit)
    else:
        raise SystemExit("must pass either --cache-dataset or both --fasta and --labels")
    print(f"{len(accs)} labeled sequences "
          f"({len(set(labels[a] for a in accs))} families)", flush=True)

    scales = R.fit_scales(list(blocks_by_seq.values()), channels, ks)
    coeffs = R.block_coeffs(scales, channels, ks)

    t0 = time.time()
    U_list = []
    for acc in accs:
        point = R.sequence_to_point(blocks_by_seq[acc], coeffs, channels, ks, args.rank)
        U_list.append(point.U[:, : args.rank])
    U_all = np.stack(U_list)
    print(f"subspaces built in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    D_full = DI.grassmann_distance_matrix(U_all)
    print(f"{D_full.shape[0]}x{D_full.shape[0]} distance matrix in {time.time()-t0:.0f}s", flush=True)

    classes = sorted(set(labels[a] for a in accs))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[labels[a]] for a in accs])

    result = PEV.paper_5nn_eval(D_full, y, k=5, min_family_size=args.min_family_size,
                                 n_splits=5, n_seeds=args.n_seeds, seed0=0)
    print(f"n_sequences_used={result['n_sequences_used']} n_classes_used={result['n_classes_used']}")
    for k_, v_ in result["mean"].items():
        print(f"  {k_}: {v_:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
