#!/bin/bash
# Roll the 5 per-seed ViT-B/16 CIFAR-10 runs up into mean+/-std stats + a report figure. CPU-only
# and fast (seconds) -- reads outputs/summary_b16_cifar10_e<E>_s*.csv + histories_*.json and writes
# outputs/agg_b16_cifar10_e<E>.csv + report_b16_cifar10_e<E>.png.
#
# Submit AFTER the seed array with a dependency so it runs once all seeds finish:
#   ARR=$(sbatch --parsable slurm/b16_cifar10_seeds.sbatch)
#   sbatch --dependency=afterok:$ARR slurm/b16_aggregate.sbatch
# Or just run the python by hand once the array drains (no GPU needed):
#   python src/complex/experiments/aggregate_b16_seeds.py --epochs 50
# >>> EDIT the USER lines (REPO / CONDA_ENV / CONDA_BASE) below for your cluster before submitting. <<<

#SBATCH --job-name=wss-b16-agg
#SBATCH --output=slurm/logs/b16_agg_%j.out
#SBATCH --error=slurm/logs/b16_agg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=cpu                  # CPU queue; no GPU needed. EDIT to your cluster's CPU partition.

set -euo pipefail

REPO="${WSS_REPO:-/home/biggs.s/WSS-Demo-Dev}"
CONDA_ENV="${WSS_ENV:-/scratch/biggs.s/conda_envs/wss}"
EPOCHS="${WSS_EPOCHS:-50}"

CONDA_BASE="${WSS_CONDA_BASE:-/home/biggs.s/miniconda}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO"
echo "host=$(hostname)  aggregating epochs=${EPOCHS}"

python src/complex/experiments/aggregate_b16_seeds.py --epochs "${EPOCHS}"
