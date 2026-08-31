"""Deterministic sequence mutation for the stability experiment.

Every mutated copy is generated from a seed derived only from
(dataset, seq_id, rate, replicate_index) -- reproducible without a global RNG state.
"""
from __future__ import annotations

import hashlib

import numpy as np

BASES = "ACGT"


def _seed_for(dataset: str, seq_id: str, rate: float, replicate: int) -> int:
    key = f"{dataset}|{seq_id}|{rate}|{replicate}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def mutate_sequence(sequence: str, rate: float, dataset: str, seq_id: str, replicate: int,
                     indel_rate: float = 0.0) -> str:
    """Substitute each base independently w.p. `rate` with one of the other 3 bases;
    optionally, independently w.p. `indel_rate`, insert or delete (50/50) a random base
    at that position. Deterministic given (dataset, seq_id, rate, replicate).
    """
    rng = np.random.default_rng(_seed_for(dataset, seq_id, rate, replicate))
    out = []
    for c in sequence:
        if c in BASES and rng.random() < indel_rate:
            if rng.random() < 0.5:
                continue  # deletion
            out.append(rng.choice(list(BASES)))
            out.append(c)
            continue
        if c in BASES and rng.random() < rate:
            choices = [b for b in BASES if b != c]
            out.append(str(rng.choice(choices)))
        else:
            out.append(c)
    return "".join(out)
