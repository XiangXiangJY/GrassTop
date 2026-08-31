# grassmannian_framework_release

One unified implementation of the **gr_core_proj_r10** framework -- a Grassmannian
subspace representation of persistent-homology "k-mer topology" features -- used to
reproduce all three result sections of `paper/grassmannian_kmer_topology.tex`:
5-NN viral-family classification (NCBI), UPGMA phylogenetic-clustering purity (CAKL
Fig. 2), and a synthetic point-mutation noise-stability ablation.

## The framework

For a chosen channel set (B = Betti-0 curve, E = smallest positive persistent-Laplacian
eigenvalue, MEANP/STDP = mean/std. dev. of the positive Laplacian spectrum) and k-mer
orders k=1..5, each sequence's feature blocks are:

1. normalized by an RMS scale `s_{g,k}` and an eq.16 weight `w_k = 2^-(5-k)`,
2. vertically stacked into `Z_S in R^{D x T}` (T=50 filtration steps),
3. reduced to its top-`r` (r=10) left singular vectors `U_S` via SVD -- the Grassmann
   point `span(U_S) in Gr(10, D)`.

Sequences are compared with the pure subspace-projection (chordal) distance
`d(S,R)^2 = r - ||U_S^T U_R||_F^2`. No singular-value magnitude ever enters -- that is
what makes this "gr_core_proj_r10" rather than the paper's companion magnitude-sensitive
`psd_core_bures_r10` arm, which is deliberately not included here.

See `src/kmer_features.py`, `src/representation.py`, `src/distances.py`.

## Framework equivalence (why "unified" is a true claim, not an assumption)

This paper's three result sections were **originally computed by three separate
codebases** that turn out to implement the identical mathematical construction:

| Section | Original code | Scale population |
|---|---|---|
| Classification | `grassmannian_gnn_v1/` (untrained-anchor mode, alpha≡1) | train fold only, per CV fold |
| Phylogenetics | `grassmannian_cakl_figure2/src/` | whole unlabeled pool |
| Stability | `paper_reproduction/stability/` | whole unlabeled pool |

Verified during this project (not assumed): `grassmannian_cakl_figure2/src/
representation.py` and `paper_reproduction/stability/representation.py` are
byte-identical. `grassmannian_gnn_v1/representation.py`'s GNN encoder only computes a
per-node weight score; the actual matrix fed into the SVD is
`Z_tilde = diag(sqrt(alpha)) @ Z` where `Z` is the RAW (un-encoded) feature stack -- so
at `alpha == 1` (the untrained/zero-init state, verified at runtime via
`assert np.allclose(alpha, 1.0, atol=1e-8)` in `test/export_gnn_v1_untrained_anchor.py`)
it reduces algebraically, not approximately, to the same `Z -> SVD -> U` pipeline as the
other two. The chordal distance formula and `K_MAX=5`/`rank=10`/`T=50` constants also
match exactly across all three.

The only two REAL, disclosed differences across tasks:
1. **Scale population** -- classification fits `s_{g,k}` on the training fold only
   (supervised k-fold CV); phylogenetics/stability fit it on the whole unlabeled pool
   (no train/test split exists for those unsupervised tasks). Same formula, different
   population, a necessary adaptation, not a different framework.
2. **Channel count** -- classification uses all 4 channels; phylogenetics uses B only
   (the paper's primary config) and stability uses B+MEANP+STDP. Channel E is excluded
   from phylogenetics/stability because it must be recomputed from scratch for every
   mutated/pool sequence and is prohibitively expensive at those tasks' scale (see
   `src/kmer_features.py`'s module docstring) -- already disclosed in the paper.

`src/` in this release is the single, literal implementation all three driver scripts
import -- not three near-duplicate copies.

## Layout

```
src/                    kmer_features.py, representation.py, distances.py, io_utils.py
classification/         run_5nn_eval.py (+ paper_5nn_eval.py, Hozumi & Wei's protocol)
phylogenetics/          run_purity_eval.py (+ upgma.py, purity.py -- CAKL eq.22-24,
                         bootstrap.py -- leaf-level 95% CI, see below)
stability/              run_stability_eval.py (+ mutate.py) -- scores Grassmannian (G)
                         AND flat/pre-projection (F) arms in one pass, see below
data/                   labels/*.csv (accession,family only -- NO raw sequences, see data/README.md)
slurm/                  sbatch scripts for the real, full-scale reproduction runs
results/                driver-script output (created on run)
```

## Environment

Python 3.10.20. `pip install -r requirements.txt` installs the 6 packages this code
actually imports, pinned to the exact versions verified working this session (numpy
2.2.6, scipy 1.15.3, scikit-learn 1.7.2, ripser 0.6.15, gudhi 3.13.0, matplotlib
3.10.9) -- not loose minimum bounds. `full_environment_freeze.txt` is a complete `pip
freeze` of the original conda env (`/mnt/gs21/scratch/wangx306/conda_envs/kmer-topology`)
for reference if some transitive dependency version ever matters; most of it (pandas,
biopython, pypdf, ...) belongs to unrelated projects sharing that env, not to this code.
No `torch` dependency -- this release deliberately drops `grassmannian_gnn_v1`'s GNN
training machinery, since it is provably a no-op for the gr_core_proj_r10 numbers (see
above).

Do not install this now if you don't need to run the code immediately -- the package
list is here so you *can* reinstall later without having to rediscover which exact
versions were verified to work.

## Running

Each driver script takes `--fasta`, `--labels` (two-column `accession,family` CSV),
`--out`. See each script's own docstring/`--help` for task-specific flags.

Quick check (already run once, see `results/phylogenetics/`):
```bash
python -m phylogenetics.run_purity_eval \
    --fasta <path-to-rhinovirus.fasta> --labels data/labels/rhinovirus_labels.csv \
    --out results/phylogenetics/rhinovirus_purity.json
# -> average purity: 1.0000, matches the paper's Table purity exactly
# -> 95% leaf-level bootstrap CI: [0.9765, 1.0000] (n_boot=300, see below)
```

Full reproduction of every number in the paper (real compute, submit via sbatch --
**do not** run these interactively; see `memory/feedback_dev_amd20_background_process_
killed_gotcha.md` for why a bare background process on a login/dev node is not safe for
anything over ~10-15 minutes):
```bash
sbatch slurm/run_classification_all.sbatch   # NCBI 2020/2022/2024/2024-All, ~hours (channel E)
sbatch slurm/run_phylogenetics_all.sbatch    # 4 CAKL datasets, minutes
sbatch slurm/run_stability_all.sbatch        # 4 CAKL datasets, tens of minutes
```

## Verification performed in this release (this session)

- `src/kmer_features.py`'s E-channel and MEANP/STDP outputs checked against brute-force
  dense-eigendecomposition reference formulas on synthetic test cases: exact match.
- `src/distances.py`'s batched distance matrix checked against the single-pair formula:
  exact match, symmetric, zero diagonal.
- `phylogenetics/run_purity_eval.py` run end-to-end on all 4 real CAKL datasets
  (rhinovirus n=116, mammalian mitochondrial n=41, ebolavirus n=59, SARS-CoV-2 n=44,
  channel B): average purity 1.0000 on every dataset, exactly matching the paper's
  Table purity, now with `bootstrap.py`'s leaf-level 95% CI wired in and written to
  each output JSON (see "Purity confidence intervals" above).
- `stability/run_stability_eval.py` run end-to-end on all 4 real CAKL datasets, full
  11-rate grid, 5 replicates/label, both the Grassmannian (G) and flat (F) arms (see
  "Stability ablation sample size" and "Stability flat (F) baseline" above): results in
  `results/stability/*.json`.
- `classification/run_5nn_eval.py` run end-to-end via `slurm/run_classification_all.sbatch`
  on all 4 real, full-scale NCBI corpora (4 channels incl. channel E), matching the
  paper's Tables: `results/classification/*.json`. Sizes and mean 5-NN accuracy:

  | Dataset | n sequences | n classes | ACC | BA |
  |---|---|---|---|---|
  | ncbi2020 | 6,810 | 57 | 0.9131 | 0.8909 |
  | ncbi2022auth | 11,066 | 78 | 0.9040 | 0.8229 |
  | ncbi2024 | 11,635 | 109 | 0.8805 | 0.8135 |
  | ncbi2024full | 13,160 | 120 | 0.8788 | 0.8165 |

## Data

No raw sequence data is included in this release -- see `data/README.md` for exact
provenance and how to obtain each dataset (all are on this filesystem already; nothing
needs to be re-downloaded to run the sbatch scripts above). One dataset (the CAKL
SARS-CoV-2 phylogenetics/stability arm) is real GISAID data and is genuinely not
redistributable; `data/README.md` explains the substitute available for anyone without
GISAID access.

## Purity confidence intervals

`paper/grassmannian_kmer_topology.tex` (Table purity's caption, Sec. 4) states "95% CIs
are leaf-level bootstrap." `phylogenetics/bootstrap.py` (copied verbatim from
`gr_core_proj_r10/phylogenetics/bootstrap.py`, which is byte-identical to
`paper_reproduction/phylogenetics/bootstrap.py`) implements this: `bootstrap_ci_leaves`
resamples sequences (leaves) with replacement, stratified per label, rebuilds the UPGMA
tree and re-scores average purity each time (n_boot=300), and reports the 2.5/97.5
percentiles. `run_purity_eval.py` now calls it automatically and writes
`bootstrap_ci_over_leaves` (the one the paper cites) and `bootstrap_ci_over_labels` (a
coarser label-resampling CI, kept for parity with the original three codebases) into
every purity JSON. All 4 datasets were re-run after wiring this in (2026-08-24); average
purity is unchanged (still 1.0000 everywhere) -- only the CI fields are new.

## Stability ablation sample size

The paper text should describe `n` per-rate-per-dataset as **varying by dataset**, not a
single value. `stability/run_stability_eval.py` draws `--n-seeds-per-label` (default 1)
representative sequence(s) per label, then mutates each at `--n-replicates` (5) draws per
rate. So `n = n_labels x n_replicates`, which differs across the 4 datasets because they
have different label counts:

| Dataset | Labels | n (= labels x 5) |
|---|---|---|
| Rhinovirus | 4 | 20 |
| Ebolavirus | 5 | 25 |
| Mammalian mitochondrial | 8 | 40 |
| SARS-CoV-2 | 8 | 40 |

(Verified directly from `results/stability/*.json`'s `config` + per-rate record counts.)
A single "n=20" figure in the paper text is only correct for rhinovirus.

## Stability flat (F) baseline: now implemented, and it changes a conclusion

`stability/run_stability_eval.py` now also computes the flat/pre-projection (F) arm --
raw Frobenius distance between un-projected `Z_S` feature stacks (`src.representation.
build_Z`), no SVD -- in the same pass as the Grassmannian (G) arm, so `results/
stability/*.json`'s `summary_by_rate` entries carry both `label_preservation`/
`self_preservation` (G) and `flat_label_preservation`/`flat_self_preservation` (F).

This F logic was ported from `paper_reproduction/stability/run_all_5nn.py`'s
`d_kmertopology_flat_from_original` computation, but that script sampled a **fixed,
non-stratified 4-seed-sequences-per-dataset subsample** -- by its own docstring, an
"INTERACTIVE-SESSION SUBSAMPLE... not the full task-spec design" -- and every earlier
version of the paper's Table stability was built from that subsample's numbers, never
superseded by a full run. This release's `run_stability_eval.py` instead samples one seed
per label (stratified; see "Stability ablation sample size" above), and the two designs
produce materially different results on Ebolavirus specifically: the subsample had flat
topology at or above the Grassmannian subspace at rate=0.25/0.4; the label-stratified
full run instead shows the two arms tied on label-preservation across nearly the whole
rate grid, with the Grassmannian subspace ahead on self-preservation throughout and ahead
on both metrics at the highest rate. `paper/grassmannian_kmer_topology.tex` Section 5 and
Section 6 (Reproducibility) were updated 2026-08-24 to report the new, label-stratified
numbers and explicitly flag the Ebolavirus reversal as superseded, not as a robust
property of that dataset.

## Mammalian mitochondrial dataset: n=41, not 42

Hozumi & Wei's own SIAM paper text (Section 2.3.1) says "42 complete mitochondrial
genomes," but their own released data
(`KmerTopology-main/data/mitochondria/mito_accessions.csv`, and the FASTA in the same
directory) contains exactly **41** accessions -- diffed byte-for-byte against this
release's `data/labels/mammalian_mitochondria_labels.csv` and found identical. CAKL's own
GitHub repo data (`CAKL/data2/mammalianMT_record.csv`) also has 41, matching. This
release's n=41 is therefore correct and matches both primary data sources; "42" in the
published paper's prose is very likely an off-by-one in that paper's own text, not an
error introduced by this codebase or by CAKL.

## What is deliberately NOT in this release

- `psd_core_bures_r10` (the companion magnitude-sensitive Bures-distance arm) -- this
  release isolates gr_core_proj_r10 only, per the paper's own ablation structure.
- `grassmannian_gnn_v1`'s trained-GNN / `channel_split` pretext-task line -- a separate,
  more complex experimental finding in this project, not part of the paper this release
  accompanies.
