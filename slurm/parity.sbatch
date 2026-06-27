#!/bin/bash
# Stage-0c TRUST GATE: fp32-vs-bf16 convergence parity at 100K (perf branch, CIFAR-100).
# Trains the 100k-wss-newton_schulz config twice (fp32 then bf16, same seed) and reports the
# accuracy / orthonormality deltas. If |Δacc| is small (target <= 1.0%) and ortho stays the same
# order of magnitude, the whole bf16 scaling suite is trustworthy; otherwise run the suite in fp32.
# Writes outputs/perf/parity_cfg<idx>_cuda.csv (default config = 100k-wss-newton_schulz, idx 6).
#
# >>> EDIT the four USER lines below for your cluster before submitting. <<<

#SBATCH --job-name=wss-parity
#SBATCH --output=slurm/logs/parity_%A_%a.out
#SBATCH --error=slurm/logs/parity_%A_%a.err
#SBATCH --array=6                        # 100k-wss-newton_schulz (see `--mode list`)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                     # 100k: any free gpu (a100s saturated)
#SBATCH --time=04:00:00                  # trains twice (fp32 + bf16); 100k is fast
#SBATCH --partition=gpu                  # the A100 partition. REQUIRED (see profile.sbatch note)

set -euo pipefail

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

# --allow_tf32 applies only to the bf16 run inside parity_one (the 'fast' path it certifies).
python src/complex/experiments/profile_retraction.py \
    --mode parity --config "${SLURM_ARRAY_TASK_ID}" \
    --epochs "${EPOCHS}" --batch_size 128 --device cuda --allow_tf32
