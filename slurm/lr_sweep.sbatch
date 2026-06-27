#!/bin/bash
# Stage-0.5 PER-METHOD LR CALIBRATION (perf branch, CIFAR-100, bf16) -- RUN BEFORE the Stage-1
# headline. LR is not a fair shared axis across geometries: WSS carries its weight in the Stiefel
# frames, so it needs a larger lr_riemann than a dense model's lr_euclid (W=U S V^T is bilinear ->
# the frames must rotate a finite angle/step). This sweeps each method's PRIMARY lr group and holds
# the other at the baseline; collect_results.py prints the best LR per method. Pin those into the
# Stage-1 headline. One single-GPU job per (method, lr) point; tier chosen via WSS_TIER (100k|1m|10m).
# Each task writes outputs/perf/lr_sweep_<tier>_cfg<idx>_cuda.csv.
#
# Config counts per tier (run `lr_sweep.py --mode list --tier <t>`): default 3 methods x 4 LRs = 12.
# CAP: the gpu QOS allows only 8 jobs in-system / 4 running per user, so >8 configs are REJECTED at
# submit (QOSMaxSubmitJobPerUserLimit). CHUNK it: submit --array=0-7, then --array=8-11 once it drains.
# Calibration is cheap (a fraction of the headline epoch budget) -- WSS_EPOCHS defaults to 30.
#
# >>> EDIT the four USER lines (+ WSS_TIER and --array) below for your cluster before submitting. <<<

#SBATCH --job-name=wss-lrsweep
#SBATCH --output=slurm/logs/lrsweep_%A_%a.out
#SBATCH --error=slurm/logs/lrsweep_%A_%a.err
#SBATCH --array=0-7                       # <= 8 (gpu QOS cap). default 100k: then 8-11 once it drains
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100-sxm2:1            # PINNED to one GPU model so runs are comparable (abundant -> fast). 1m/10m: gpu:a100:1
#SBATCH --time=08:00:00                   # 8h cap; calibration is short, but 1m/10m points are larger
#SBATCH --partition=gpu                   # the A100 partition. REQUIRED (see profile.sbatch note)

set -euo pipefail

REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-30}"
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

python src/complex/experiments/lr_sweep.py \
    --mode convergence --tier "${TIER}" --config "${SLURM_ARRAY_TASK_ID}" \
    --epochs "${EPOCHS}" --batch_size 128 --device cuda --amp --allow_tf32
