# Data provenance

No raw sequence data is included in this release -- `data/labels/*.csv` contain only
`accession,family` metadata (safe to redistribute; no sequence content).

## NCBI datasets (classification)

| Dataset | Raw FASTA (on this filesystem) | Source |
|---|---|---|
| ncbi2020 | `/mnt/gs21/scratch/wangx306/project6/KmerTopology-main/data/ncbi2020/ncbi2020.fasta` | NCBI Virus, curated by Hozumi & Wei (arXiv:2412.20202) |
| ncbi2022auth | `.../data/ncbi2022auth/ncbi2022auth.fasta` | same |
| ncbi2024 | `.../data/ncbi2024/ncbi2024.fasta` | same |
| ncbi2024full | `.../data/ncbi2024full/ncbi2024full.fasta` | same |

Public NCBI data, no redistribution restriction. Anyone without access to this exact
filesystem path can re-fetch the same accessions (listed in `data/labels/*.csv`) from
NCBI Virus directly.

## CAKL datasets (phylogenetics, stability)

| Dataset | Raw FASTA (on this filesystem) | Source |
|---|---|---|
| rhinovirus | `<workshop>/grassmannian_cakl_figure2/data/raw/rhinovirus/rhinovirus.fasta` | GenBank, accession-verified against CAKL's own GitHub repo |
| mammalian_mitochondria | `.../mammalian_mitochondria/mitochondria.fasta` | same |
| ebolavirus | `.../ebolavirus/ebolavirus.fasta` | same |
| sars_cov_2 | `.../sars_cov_2/sars_cov_2.fasta` | **real GISAID data -- see below** |

## SARS-CoV-2: GISAID restriction (important)

The 44 SARS-CoV-2 genomes used in the paper's phylogenetics/stability tables were
downloaded directly from GISAID's hCoV-19 database via an authenticated login
(`grassmannian_cakl_figure2/data/raw/sars_cov_2/ACQUISITION_NOTE.md`). GISAID's terms of
use forbid third-party redistribution of raw sequence data without a separate agreement.
**This release never copies that FASTA anywhere, and neither should any fork of it.**

- `data/labels/sars_cov_2_labels.csv` contains only the accession/label metadata (the
  EPI_ISL accession numbers), which is safe to share.
- To reproduce the SARS-CoV-2 arm yourself: obtain a GISAID account, fetch the 44
  EPI_ISL accessions listed in that CSV, and point `--fasta` at your own copy.
- A public-data NCBI substitute (not the paper's exact GISAID sequences, purity was
  NOT verified to match on it) exists at `grassmannian_cakl_figure2/data/raw/sars_cov_2/
  SUBSTITUTE_NCBI_BACKUP_2026-08-05/` for anyone who wants a runnable end-to-end demo
  without GISAID access -- clearly label any results from it as using substitute data,
  not the paper's reported numbers.
