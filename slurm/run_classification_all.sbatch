#!/usr/bin/env bash
#SBATCH --job-name=gr_unified_cls
#SBATCH --account=general
#SBATCH --partition=general-long
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.out
#SBATCH --error=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop/grassmannian_framework_release/log/%x_%A_%a.err

# Full-scale reproduction of the paper's gr_core_proj_r10 classification numbers
# (Table ncbi2020..ncbi2024full), using the SINGLE unified src/ implementation
# (verified equivalent to grassmannian_gnn_v1's untrained-anchor path -- see
# ../README.md "Framework equivalence" section). Loads precomputed B/E/MEANP/STDP
# feature caches (--cache-dataset) rather than recomputing channel E from raw sequence:
# recomputing E from scratch was measured this session at <200 sequences/30min on real
# NCBI sequence lengths (some exceed 1 Mb) -- utterly infeasible at 6810-13160 sequences
# per dataset. This project's Grassmannian framework was always built on top of these
# caches (see classification/multidata.py, copied from paper_reproduction/), not a
# from-scratch re-extraction at this scale -- only the SVD/eq.16/projection-distance/
# 5-NN-eval steps are the "unified code" being verified here, matching the framework
# equivalence documented in ../README.md.
#
# One array task per dataset (0=ncbi2020, 1=ncbi2022auth, 2=ncbi2024, 3=ncbi2024full).
# Submit with: sbatch run_classification_all.sbatch

set -euo pipefail

WS=/mnt/gs21/scratch/wangx306/project6/grassmann_psd_workshop
REL=$WS/grassmannian_framework_release
PY=/mnt/gs21/scratch/wangx306/conda_envs/kmer-topology/bin/python3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

DATASETS=(ncbi2020 ncbi2022auth ncbi2024 ncbi2024full)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$REL/results/classification"

echo "=================== RESOLVED CONFIG ==================="
echo "host    = $(hostname)"
echo "job     = ${SLURM_JOB_ID:-none} array_task = ${SLURM_ARRAY_TASK_ID:-none}"
echo "dataset = $DS"
echo "python  = $PY"
"$PY" -c "import numpy,scipy,sklearn,ripser,gudhi; print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"
echo "======================================================="

cd "$REL"
"$PY" -m classification.run_5nn_eval \
  --cache-dataset "$DS" \
  --out "$REL/results/classification/${DS}.json"

echo "exit=$?"
