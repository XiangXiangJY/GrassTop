# GrassTop

Reference implementation for the paper **"GrassTop: A Grassmannian $k$-mer Topology
Representation of Genomes"** (Wang & Wei). GrassTop represents each genome by a
low-rank subspace on a Grassmannian, built from multiscale topological and spectral
descriptors of $k$-mer positional patterns, and compares genomes with a Grassmannian
distance. This repository reproduces the paper's three experiments: 5-NN viral-family
classification, UPGMA phylogenetic-clustering purity, and a sequence-perturbation
stability comparison.

## Method

For a sequence $S$:

1. **Feature extraction.** Compute the $k$-mer topology feature blocks $G_{S,g}^{(k)}$
   for each $k$-mer order $k$ and descriptor type $g$ -- persistent-homology and
   persistent-Laplacian statistics over a common filtration (Hozumi & Wei, 2025).
2. **Normalization and stacking.** Normalize each block by an RMS scale $s_{g,k}$ and a
   block weight $w_k$, then stack over $(g,k)$ into $Z_S \in \mathbb{R}^{n\times T}$.
3. **Grassmannian embedding.** Take a rank-$r$ truncated SVD
   $Z_S = U_S\Sigma_S V_S^\top$ and keep only $\mathrm{span}(U_S)\in\mathrm{Gr}(r,n)$.
4. **Distance.** Compare two genomes with the Grassmannian chordal distance
   $d_{\mathrm{chord}}(S,R) = \tfrac{1}{\sqrt2}\lVert U_SU_S^\top-U_RU_R^\top\rVert_F.$

See `src/kmer_features.py`, `src/representation.py`, and `src/distances.py` for the
implementation, and Sections 2-3 of the paper for the full derivation.

**Configuration used in the paper's experiments:** $k$-mer orders
$\mathcal{K}=\{1,\dots,5\}$, $T=50$ filtration steps, $w_k = 2^{-(5-k)}$, rank $r=10$,
chordal distance. Descriptor types: classification uses all four channels ($B$ =
Betti-0 curve, $E$ = smallest positive persistent-Laplacian eigenvalue,
$\mathrm{MEANP}$/$\mathrm{STDP}$ = mean/std. of the positive Laplacian spectrum);
phylogenetic clustering uses $B$ alone; the perturbation experiment uses $B$,
$\mathrm{MEANP}$, $\mathrm{STDP}$.

## Repository layout

```
src/            Core implementation: kmer_features.py, representation.py, distances.py, io_utils.py
classification/ 5-NN viral-family classification (run_5nn_eval.py)
phylogenetics/  UPGMA tree construction and purity evaluation (run_purity_eval.py, upgma.py, purity.py, bootstrap.py)
stability/      Sequence-perturbation experiment (run_stability_eval.py, mutate.py)
data/labels/    accession,family metadata only -- no raw sequences (see data/README.md)
slurm/          Batch scripts for the full-scale reproduction runs
results/        Output produced by the scripts above, included for the runs reported in the paper
```

## Reproducing the paper's results

### Classification (Section 4.2.1)

| Dataset | Sequences | Families | ACC | BA |
|---|---|---|---|---|
| NCBI 2020 | 6,810 | 57 | 0.9131 | 0.8909 |
| NCBI 2022 | 11,066 | 78 | 0.9040 | 0.8229 |
| NCBI 2024 | 11,635 | 109 | 0.8805 | 0.8135 |
| NCBI 2024 (All) | 13,160 | 120 | 0.8788 | 0.8165 |

GrassTop has the largest value for each of the six reported metrics (ACC, BA, F1,
AUC-ROC, recall, precision) on all four datasets.

```bash
sbatch slurm/run_classification_all.sbatch
```

### Phylogenetic clustering (Section 4.2.2)

Average UPGMA tree purity is 1.0 on all four datasets: human rhinovirus ($n=116$),
mammalian mitochondrial genomes ($n=41$), Ebolavirus ($n=59$), and SARS-CoV-2
($n=44$).

```bash
sbatch slurm/run_phylogenetics_all.sbatch
# or, for a single dataset:
python -m phylogenetics.run_purity_eval \
    --fasta <path-to-rhinovirus.fasta> --labels data/labels/rhinovirus_labels.csv \
    --out results/phylogenetics/rhinovirus_purity.json
```

### Sequence perturbation (Section 4.2.3)

Across 32 dataset-rate-measure combinations, GrassTop matches or exceeds the flat
(pre-projection) descriptor in 29 of 32.

```bash
sbatch slurm/run_stability_all.sbatch
```

## Data

No raw sequence data is included in this repository; `data/labels/*.csv` contain only
`accession,family` metadata. See `data/README.md` for how to retrieve each dataset.
SARS-CoV-2 sequences were obtained from GISAID under its terms of use and are not
redistributed here -- only the EPI_ISL accession list is provided, consistent with the
paper's Data and Code Availability statement.

The pre-computed feature caches used to speed up reruns are too large for GitHub and
are hosted separately (link to be added once published).

## Environment

```bash
pip install -r requirements.txt
```

Python 3.10. Core dependencies: numpy, scipy, scikit-learn, ripser, gudhi, matplotlib.
`full_environment_freeze.txt` is a complete `pip freeze` of the environment the
reported results were computed in, kept for reference; most transitive dependencies
listed there are not required to run this code.

## Citation

If you use this code, please cite:

> Xiang Xiang Wang and Guo-Wei Wei. "GrassTop: A Grassmannian $k$-mer Topology
> Representation of Genomes."
