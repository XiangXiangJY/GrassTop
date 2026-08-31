"""Dataset-parameterised labels and channel loading for all 4 NCBI datasets used in the
paper's 5-NN classification benchmark (ncbi2020, ncbi2022auth, ncbi2024, ncbi2024full).

Own copy for paper_reproduction/ (reads the SAME on-disk feature caches under
<workshop_root>/data/, never copies them -- those are input data, not framework code),
trimmed from psd_v2/multidata.py to just the CORE_CHANNELS (B, E, MEANP, STDP) this
package's two arms (gr_core_proj_r10, psd_core_bures_r10) both use.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
WS = os.path.abspath(os.path.join(HERE, "..", ".."))  # grassmann_psd_workshop root

MAX_STEP = 50
KS = (1, 2, 3, 4, 5)
K_MAX = 5
DATASETS = ("ncbi2020", "ncbi2022auth", "ncbi2024", "ncbi2024full")
CORE_CHANNELS = ("B", "E", "MEANP", "STDP")


@dataclass(frozen=True)
class Dataset:
    name: str
    accessions: np.ndarray
    families: np.ndarray
    y: np.ndarray
    family_names: np.ndarray


def features_dir(ds: str) -> str:
    return os.path.join(WS, "data", ds, "features")


def betti_cache(ds: str, k: int) -> str:
    return os.path.join(features_dir(ds), f"features_k{k}_ms50.npy")


def labels_csv(ds: str) -> str:
    return os.path.join(features_dir(ds), "labels.csv")


def eig_cache(ds: str, k: int) -> str:
    return os.path.join(WS, "data", ds, "features_eigmin", f"features_eigmin_k{k}_ms50.npy")


def moment_cache(ds: str, channel: str, k: int) -> str:
    if ds == "ncbi2020":
        return os.path.join(WS, "data", "ncbi2020", "features_specstats",
                             f"features_specstats_k{k}_ms50.npy")
    return os.path.join(WS, "data", ds, "features_moments", f"features_{channel.lower()}_k{k}_ms50.npy")


def load_labels(ds: str) -> Dataset:
    with open(labels_csv(ds)) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header[:2] == ["accession", "family"], header
    accessions = np.array([r[0] for r in body])
    families = np.array([r[1] for r in body])
    family_names, y = np.unique(families, return_inverse=True)
    return Dataset(ds, accessions, families, y.astype(int), family_names)


def load_channel(ds: str, channel: str, k: int, chunk=512, dtype=np.float32):
    """(n, 4^k, 50) for one channel of one dataset at one k."""
    if channel == "B":
        X = np.load(betti_cache(ds, k), mmap_mode="r")
    elif channel == "E":
        X = np.load(eig_cache(ds, k), mmap_mode="r")
    elif channel in ("MEANP", "STDP"):
        if ds == "ncbi2020":
            X = np.load(moment_cache(ds, channel, k), mmap_mode="r")
            idx, take_sqrt = (3, False) if channel == "MEANP" else (5, True)
            n = X.shape[0]
            out = np.empty((n, 4 ** k, MAX_STEP), dtype=dtype)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                v = np.asarray(X[s:e, :, :, idx], dtype=np.float64)
                if take_sqrt:
                    v = np.sqrt(np.maximum(v, 0.0))
                out[s:e] = v.astype(dtype)
            return out
        return np.asarray(np.load(moment_cache(ds, channel, k), mmap_mode="r"), dtype=dtype)
    else:
        raise ValueError(f"unknown channel {channel!r} -- must be one of {CORE_CHANNELS}")
    n = X.shape[0]
    out = np.empty((n, 4 ** k, MAX_STEP), dtype=dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.asarray(X[s:e], dtype=dtype).reshape(e - s, 4 ** k, MAX_STEP)
    return out


def load_channels(ds: str, channels=CORE_CHANNELS, ks=KS, verbose=False):
    """{channel: {k: (n, 4^k, 50)}} plus an info dict recording any imputation."""
    import time
    channels = tuple(channels)
    out = {c: {} for c in channels}
    info = {"dataset": ds, "channels": list(channels), "nonfinite_imputed": 0}
    for k in ks:
        t0 = time.perf_counter()
        for c in channels:
            out[c][k] = load_channel(ds, c, k)
        if verbose:
            gib = sum(out[c][k].nbytes for c in channels) / 2 ** 30
            print(f"    [load] {ds} k={k} {len(channels)} channels {gib:.2f} GiB "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    for c in channels:
        for k in ks:
            bad = ~np.isfinite(out[c][k])
            nb = int(bad.sum())
            if nb:
                out[c][k][bad] = 0.0
                info["nonfinite_imputed"] += nb
    if verbose:
        print(f"    [load] {info}", flush=True)
    return out, info
