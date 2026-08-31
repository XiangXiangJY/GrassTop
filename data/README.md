# Data provenance

No raw sequence data is included in this repository. `data/labels/*.csv` contain only
`accession,family` metadata (safe to redistribute; no sequence content).

## NCBI datasets (classification)

| Dataset | Source |
|---|---|
| ncbi2020 | NCBI Virus, curated by Hozumi & Wei |
| ncbi2022auth | NCBI Virus, curated by Hozumi & Wei |
| ncbi2024 | NCBI Virus, curated by Hozumi & Wei |
| ncbi2024full | NCBI Virus, curated by Hozumi & Wei |

Public NCBI data with no redistribution restriction. The accessions listed in
`data/labels/*.csv` can be re-fetched from NCBI Virus directly.

## Phylogenetic-clustering / perturbation datasets

| Dataset | Source |
|---|---|
| rhinovirus | GenBank |
| mammalian_mitochondria | GenBank |
| ebolavirus | GenBank |
| sars_cov_2 | GISAID (see below) |

## SARS-CoV-2: GISAID restriction

The 44 SARS-CoV-2 genomes used in the paper's phylogenetics and perturbation
experiments were obtained from GISAID under its terms of use, which forbid
third-party redistribution of raw sequence data. This repository does not include
that FASTA file, and no fork of it should either.

- `data/labels/sars_cov_2_labels.csv` contains only the EPI_ISL accession numbers and
  lineage labels, which is safe to share.
- To reproduce the SARS-CoV-2 results, obtain a GISAID account, retrieve the 44
  EPI_ISL accessions listed in that file, and point `--fasta` at your own copy.
