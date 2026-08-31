#!/usr/bin/env bash
#SBATCH --job-name=gr_unified_stab
#SBATCH --account=guowei-search
#SBATCH --partition=guowei-search
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.err

# Full reproduction of the paper's noise-stability numbers (Table stability, 5-NN
# scoring), using the SINGLE unified src/ implementation. Channels B+MEANP+STDP (E
# excluded: too expensive to recompute per mutant, see stability/run_stability_eval.py's
# docstring). Same GISAID caveat as run_phylogenetics_all.sbatch for array index 3.
#
# Submit with: sbatch run_stability_all.sbatch

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

mkdir -p "$REL/results/stability"

echo "=================== RESOLVED CONFIG ==================="
echo "host    = $(hostname)"
echo "job     = ${SLURM_JOB_ID:-none} array_task = ${SLURM_ARRAY_TASK_ID:-none}"
echo "dataset = $DS"
echo "python  = $PY"
echo "======================================================="

cd "$REL"
"$PY" -m stability.run_stability_eval \
  --fasta "$CAKL_RAW/$SUBDIR/$FNAME" \
  --labels "$REL/data/labels/${DS}_labels.csv" \
  --out "$REL/results/stability/${DS}_stability.json" \
  --dataset-name "$DS"

echo "exit=$?"
