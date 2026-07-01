#!/bin/bash
# ViT-B/16 CIFAR-10 headline over 5 seeds (perf branch). One single-GPU job PER SEED; each task
# trains all four conditions -- dense (ceiling) | single_rank_Jr (factorized control) |
# dense_matched (THE equal-param control) | wss (the method) -- on the real torchvision vit_b_16
# (768d/12L/12h, weights=None, CIFAR-10 upscaled to 224px). Each task writes
# outputs/summary_b16_cifar10_e<E>_s<seed>.csv + histories_*.json and logs per-epoch to WandB
# (one WandB run per condition, grouped so seeds overlay). Aggregate across seeds afterwards with
# slurm/b16_aggregate.sbatch (or python .../aggregate_b16_seeds.py --epochs <E>).
#
# WandB: set WANDB_API_KEY for online logging, or WANDB_MODE=offline to disable remote logging.
# LR: overridable via env (dense/dense_matched use lr_euclid; wss/single_rank use lr_riemann):
#   WSS_LR_RIEMANN=<best> WSS_LR_EUCLID=<best> sbatch slurm/b16_cifar10_seeds.sbatch
# >>> EDIT the USER lines (REPO / CONDA_ENV / CONDA_BASE) below for your cluster before submitting. <<<

#SBATCH --job-name=wss-b16
#SBATCH --output=slurm/logs/b16_%A_%a.out
#SBATCH --error=slurm/logs/b16_%A_%a.err
#SBATCH --array=0-4                      # 5 tasks (<= 8 QOS in-system cap); one seed per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # B/16 @224px is memory-hungry
#SBATCH --gres=gpu:a100:1                # B/16 @224px needs the A100's memory + throughput
#SBATCH --time=08:00:00                  # 8h cap; the wss/single_rank retraction runs are the time hog
#SBATCH --partition=gpu                  # REQUIRED (see profile.sbatch note)

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-50}"
# WandB: set WANDB_API_KEY or WANDB_MODE=offline before submitting
WANDB_MODE="${WANDB_MODE:-online}"
# dense/dense_matched use lr_euclid; wss/single_rank use lr_riemann (both groups run in one task).
LR_RIEMANN="${WSS_LR_RIEMANN:-1e-3}"
LR_EUCLID="${WSS_LR_EUCLID:-1e-3}"

# One seed per array task.
SEEDS=(1 2 3 4 5)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

# Activate conda. We deliberately do NOT `module load anaconda3`: on this cluster it prepends the
# system anaconda to PATH, which SHADOWS the activated env's python -- the job then runs a python
# WITHOUT geoopt and dies on `import geoopt`. Source the env's OWN conda base directly instead.
# cuda/12.1.1 matches torch's cu121 build (bare `module load cuda` pulls the default 13.x mismatch).
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
CONDA_BASE="${WSS_CONDA_BASE:-/home/biggs.s/miniconda}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export WANDB_MODE="${WANDB_MODE}"
cd "$REPO"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}  seed=${SEED}  epochs=${EPOCHS}  wandb_mode=${WANDB_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/complex/experiments/headline_vit_b16_cifar10.py \
    --seed "${SEED}" --epochs "${EPOCHS}" --batch_size 128 --device cuda --amp --allow_tf32 \
    --lr_riemann "${LR_RIEMANN}" --lr_euclid "${LR_EUCLID}"
