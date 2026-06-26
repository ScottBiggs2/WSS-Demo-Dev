#!/bin/bash
# Short CIFAR-10 convergence runs per retraction method (perf branch).
# Confirms qr / newton_schulz reach canonical's accuracy & orthonormality, and quantifies the
# accuracy cost of the NON-FAITHFUL controls (none, retract_every>1). One single-GPU job/config.
# Each task writes outputs/perf/convergence_cfg<idx>_cuda.csv.
#
# Default array (both tiers): dense baseline + dense_matched (equal-param control) +
# single_rank_Jr-canonical + wss {canonical, newton_schulz, none}. This is the param-matched
# accuracy comparison + the "does fast faithful retraction hurt accuracy?" check + the collapse
# floor. Add the lazy-K accuracy cost with idx 8,9 / 18,19, or use --array=0-19 for everything.
#
# >>> EDIT the four USER lines below for your cluster before submitting. <<<

#SBATCH --job-name=wss-conv
#SBATCH --output=slurm/logs/conv_%A_%a.out
#SBATCH --error=slurm/logs/conv_%A_%a.err
#SBATCH --array=0,1,2,4,6,7,10,11,12,14,16,17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu                 # the A100 partition. REQUIRED (see profile.sbatch note)

set -euo pipefail

# Repo path + conda env PREFIX on the cluster (overridable via env vars).
REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-10}"

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$REPO"

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-NA}  epochs=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/complex/experiments/profile_retraction.py \
    --mode convergence --config "${SLURM_ARRAY_TASK_ID}" \
    --epochs "${EPOCHS}" --batch_size 128 --device cuda
