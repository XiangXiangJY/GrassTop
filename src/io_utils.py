"""FASTA parsing, shared by all three evaluation scripts. No external dependency."""

from __future__ import annotations

NUCLEOTIDES = ("A", "C", "G", "T")


def parse_fasta(path: str) -> dict:
    """-> {accession: sequence}. accession = first whitespace-delimited token after '>'.
    Sequence is uppercased and RNA U is mapped to DNA T. A duplicated accession keeps the
    last record seen.
    """
    seqs, acc, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(chunks).upper().replace("U", "T")
                acc = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(chunks).upper().replace("U", "T")
    return seqs


def is_pure_acgt(seq: str) -> bool:
    return set(seq) <= set(NUCLEOTIDES)
