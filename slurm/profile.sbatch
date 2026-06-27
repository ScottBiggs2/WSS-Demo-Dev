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
# wss-none, wss-ns_K4}. 100k=0,5,6,7,9 ; 1m=10,15,16,17,19 ; 10m=20,25,26,27,29. Drops
# Cayley/single_rank/K2 (low throughput value). Use --array=0-29 for the full grid.
#SBATCH --array=0,5,6,7,9,10,15,16,17,19,20,25,26,27,29
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                       # 10m-tier ViTs (~12.5M params) need more headroom than 1m
#SBATCH --gres=gpu:a100:1               # one GPU per task (A100 or H200)
#SBATCH --time=02:00:00                 # ceiling; profiling itself takes minutes even at 10m
#SBATCH --partition=gpu                 # the A100 partition. REQUIRED -- omitting it lands you on
                                        # the default partition (no A100) -> "node config not available"

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"

# Activate conda/mamba. Adjust the module name(s) to your site if needed.
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
