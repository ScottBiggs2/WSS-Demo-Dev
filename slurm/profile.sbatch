#!/bin/bash
# Throughput micro-benchmark sweep for WSS retraction methods (perf branch; CIFAR-100 shapes).
# One independent single-GPU job per config (SLURM array over the 30-config grid; `--mode list`
# for the map). Each task writes outputs/perf/profile_cfg<idx>_cuda.csv (+ profiler trace).
#
# >>> EDIT the four USER lines below for your cluster before submitting. <<<

#SBATCH --job-name=wss-prof
#SBATCH --output=slurm/logs/prof_%A_%a.out
#SBATCH --error=slurm/logs/prof_%A_%a.err
# Trimmed subset of the 30-config grid (full map: `--mode list`): per tier {dense, wss-qr, wss-ns,
# wss-none, wss-ns_K4}. 100k=0,5,6,7,9 ; 1m=10,15,16,17,19 ; 10m=20,25,26,27,29.
# CAP: the gpu QOS allows only 8 jobs in-system / 4 running per user, so a 15-task array is REJECTED
# at submit (QOSMaxSubmitJobPerUserLimit) -- even with an empty queue. Keep each submit <= 8.
# Default = the 100k headline tier (5 tasks). Submit 1m/10m as SEPARATE arrays after this drains:
#   sbatch --array=10,15,16,17,19 slurm/profile.sbatch     # 1m  (re-add --gres=gpu:a100:1 for memory)
#   sbatch --array=20,25,26,27,29 slurm/profile.sbatch     # 10m (re-add --gres=gpu:a100:1 for memory)
#SBATCH --array=0,5,6,7,9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                       # 10m-tier ViTs (~12.5M params) need more headroom than 1m
#SBATCH --gres=gpu:1                    # 100k: ANY free gpu (a100s are saturated; v100s sit idle).
                                        # For 1m/10m or strictly comparable it/s, pin gpu:a100:1.
#SBATCH --time=02:00:00                 # ceiling; profiling itself takes minutes even at 10m
#SBATCH --partition=gpu                 # REQUIRED -- the default partition has no GPUs, so omitting
                                        # this fails with "node config not available".

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"

# Activate conda. We deliberately do NOT `module load anaconda3`: on this cluster it prepends the
# system anaconda to PATH, which SHADOWS the activated env's python -- the job then runs a python
# WITHOUT geoopt and dies on `import geoopt`. Source the env's OWN conda base directly instead.
# cuda/12.1.1 matches torch's cu121 build (bare `module load cuda` pulls the default 13.x mismatch).
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
CONDA_BASE="${WSS_CONDA_BASE:-/home/biggs.s/miniconda}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on CUDA; keeps the code Mac-runnable too
cd "$REPO"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --amp --allow_tf32 profile the bf16 path the scaling suite actually runs (drop --amp for an
# fp32 profile). Linalg (retraction, eigvalsh) stays fp32 regardless.
python src/complex/experiments/profile_retraction.py \
    --mode profile --config "${SLURM_ARRAY_TASK_ID}" \
    --steps 200 --warmup 30 --batch_size 128 --device cuda --amp --allow_tf32 --profiler
