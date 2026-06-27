#!/bin/bash
# CIFAR-100 convergence runs (perf branch). Stage 1 headline = "is WSS more than a regularizer?":
# does the gated J-structure beat an EQUAL-PARAM dense (dense_matched)? Also the retraction-parity
# check (qr/ns vs canonical) and the no-retraction collapse floor. One single-GPU job/config; bf16.
# Each task writes outputs/perf/convergence_cfg<idx>_cuda.csv.
#
# Stage-1 headline array (100k tier, the cheapest place to answer the question):
#   0 dense (ceiling) | 1 dense_matched (THE equal-param control) | 3 single_rank_Jr-ns
#   (fancy-LoRA control) | 6 wss-newton_schulz (the method) | 7 wss-none (collapse floor)
# Add 4 (wss-canonical) / 5 (wss-qr) for the fast-retraction parity reference; 9 for lazy-K cost.
# Promote to 1m by adding 10,11,13,16 (and 17) -- run with WSS_EPOCHS bumped. 10m: 20,21,23,26.
#
# >>> EDIT the four USER lines below for your cluster before submitting. <<<

#SBATCH --job-name=wss-conv
#SBATCH --output=slurm/logs/conv_%A_%a.out
#SBATCH --error=slurm/logs/conv_%A_%a.err
#SBATCH --array=0,1,3,6,7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                       # headroom for the 1m/10m tiers
#SBATCH --gres=gpu:1                    # 100k: any free gpu (a100s saturated). 1m/10m: pin gpu:a100:1
#SBATCH --time=08:00:00                 # 8h cap; 1m/10m real training is the time hog
#SBATCH --partition=gpu                 # the A100 partition. REQUIRED (see profile.sbatch note)

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-50}"

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

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}  epochs=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/complex/experiments/profile_retraction.py \
    --mode convergence --config "${SLURM_ARRAY_TASK_ID}" \
    --epochs "${EPOCHS}" --batch_size 128 --device cuda --amp --allow_tf32
