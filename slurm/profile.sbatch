#!/bin/bash
# Throughput micro-benchmark sweep for WSS retraction methods (perf branch).
# One independent single-GPU job per config (SLURM array 0-19; run `--mode list` for the map).
# Each task writes outputs/perf/profile_cfg<idx>_cuda.csv (+ profiler trace with --profiler).
#
# >>> EDIT the four USER lines below for your cluster before submitting. <<<

#SBATCH --job-name=wss-prof
#SBATCH --output=slurm/logs/prof_%A_%a.out
#SBATCH --error=slurm/logs/prof_%A_%a.err
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1               # one GPU per task
#SBATCH --time=06:00:00                 # ceiling; profiling itself takes minutes
#SBATCH --partition=gpu                 # the A100 partition. REQUIRED -- omitting it lands you on
                                        # the default partition (no A100) -> "node config not available"

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"

# Activate conda/mamba. Adjust the module name(s) to your site if needed.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on CUDA; keeps the code Mac-runnable too
cd "$REPO"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/complex/experiments/profile_retraction.py \
    --mode profile --config "${SLURM_ARRAY_TASK_ID}" \
    --steps 200 --warmup 30 --batch_size 128 --device cuda --profiler
