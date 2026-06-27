#!/bin/bash
# Stage-2+ iso-param SCALING sweeps (perf branch, CIFAR-100, bf16): depth<->width and J<->r.
# One single-GPU job per scaling_suite config; the tier is chosen via WSS_TIER (100k|1m|10m).
# Each task writes outputs/perf/scaling_<tier>_cfg<idx>_cuda.csv.
#
# Config counts per tier (run `scaling_suite.py --mode list --tier <t>`): 100k=15, 1m=16, 10m=14.
# CAP: the gpu QOS allows only 8 jobs in-system / 4 running per user, so a full tier (>8) is REJECTED
# at submit (QOSMaxSubmitJobPerUserLimit). CHUNK it: submit --array=0-7, then --array=8-N once that
# drains. These are multi-hour training runs; the gpu partition (8h wall) is the right home.
#
# >>> EDIT the four USER lines (+ WSS_TIER and --array) below for your cluster before submitting. <<<

#SBATCH --job-name=wss-scale
#SBATCH --output=slurm/logs/scale_%A_%a.out
#SBATCH --error=slurm/logs/scale_%A_%a.err
#SBATCH --array=0-7                       # <= 8 (gpu QOS cap). 100k: then 8-14 ; 1m: 0-7,8-15 ; 10m: 0-7,8-13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                      # 100k: any free gpu (a100s saturated). 1m/10m: pin gpu:a100:1
#SBATCH --time=08:00:00                   # 8h cap; 10m configs may need the epoch-cap mitigation
#SBATCH --partition=gpu                   # the A100 partition. REQUIRED (see profile.sbatch note)

set -euo pipefail

REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-60}"
TIER="${WSS_TIER:-100k}"

# Activate conda. We deliberately do NOT `module load anaconda3`: on this cluster it prepends the
# system anaconda to PATH, which SHADOWS the activated env's python -- the job then runs a python
# WITHOUT geoopt and dies on `import geoopt`. Source the env's OWN conda base directly instead.
# cuda/12.1.1 matches torch's cu121 build (bare `module load cuda` pulls the default 13.x mismatch).
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
CONDA_BASE="${WSS_CONDA_BASE:-/home/biggs.s/miniconda}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}  tier=${TIER}  epochs=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/complex/experiments/scaling_suite.py \
    --mode convergence --tier "${TIER}" --config "${SLURM_ARRAY_TASK_ID}" \
    --epochs "${EPOCHS}" --batch_size 128 --device cuda --amp --allow_tf32
