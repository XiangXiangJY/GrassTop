#!/usr/bin/env bash
#SBATCH --job-name=gr_unified_phylo
#SBATCH --account=guowei-search
#SBATCH --partition=guowei-search
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.err

# Full reproduction of the paper's gr_core_proj_r10 phylogenetic-purity numbers
# (Table purity), using the SINGLE unified src/ implementation. Channel B only, matching
# the paper's primary config. All 4 CAKL datasets are small (n<=116) -- this is cheap,
# included here mainly for a one-command "reproduce everything" workflow, not because it
# strictly needs a batch job.
#
# Array index 3 (sars_cov_2) reads its raw FASTA from grassmannian_cakl_figure2's
# ORIGINAL data directory, NOT from anywhere inside this release folder -- those 44
# sequences are real GISAID data (see ../data/README.md) and are never copied into this
# release. Running this array task requires that original data still being present on
# this filesystem; it is not something a fresh clone of this release can reproduce
# without separately obtaining GISAID access.
#
# Submit with: sbatch run_phylogenetics_all.sbatch

set -euo pipefail

WS=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop
REL=$WS/grassmannian_framework_release
PY=/mnt/gs21/scratch/wangx306/conda_envs/kmer-topology/bin/python3
CAKL_RAW=$WS/grassmannian_cakl_figure2/data/raw

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

DATASETS=(rhinovirus mammalian_mitochondria ebolavirus sars_cov_2)
FASTA_SUBDIR=(rhinovirus mammalian_mitochondria ebolavirus sars_cov_2)
FASTA_NAME=(rhinovirus.fasta mitochondria.fasta ebolavirus.fasta sars_cov_2.fasta)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SUBDIR=${FASTA_SUBDIR[$SLURM_ARRAY_TASK_ID]}
FNAME=${FASTA_NAME[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$REL/results/phylogenetics"

echo "=================== RESOLVED CONFIG ==================="
echo "host    = $(hostname)"
echo "job     = ${SLURM_JOB_ID:-none} array_task = ${SLURM_ARRAY_TASK_ID:-none}"
echo "dataset = $DS"
echo "python  = $PY"
echo "======================================================="

cd "$REL"
"$PY" -m phylogenetics.run_purity_eval \
  --fasta "$CAKL_RAW/$SUBDIR/$FNAME" \
  --labels "$REL/data/labels/${DS}_labels.csv" \
  --out "$REL/results/phylogenetics/${DS}_purity.json" \
  --channels B

echo "exit=$?"
