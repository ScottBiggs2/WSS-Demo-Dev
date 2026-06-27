#!/bin/bash
# Submit the FUNNEL's Stage-0 gates (profiling + fp32/bf16 parity), then the Stage-1 headline
# convergence array. EDIT slurm/*.sbatch (account, partition, repo path, conda env) FIRST. Each
# array task is an independent single-GPU job, so they schedule concurrently as the cluster allows.
#
# Funnel order (gate each stage before submitting the next -- see PERF_NOTES / the plan):
#   Stage 0:  profile.sbatch (pick fast retraction) + parity.sbatch (GATE 0c: trust bf16)
#   Stage 1:  convergence.sbatch (100k headline: wss vs dense_matched)
#   Stage 2+: scaling.sbatch (WSS_TIER=100k|1m|10m) -- submit AFTER the headline gate passes.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
mkdir -p slurm/logs              # sbatch needs the log dir to exist at submit time

echo "Stage 0 — profiling array (see --array in profile.sbatch; minutes each)..."
sbatch slurm/profile.sbatch

echo "Stage 0c — fp32-vs-bf16 parity (the bf16 trust gate)..."
sbatch slurm/parity.sbatch

echo "Stage 1 — convergence array (100k headline; see --array in convergence.sbatch)..."
# NOTE: if this fails with QOSMaxSubmitJobPerUserLimit, the earlier arrays already fill your
# per-user queue quota. A pending dependent job still occupies a slot, so just submit this AFTER
# they drain:  squeue --me   # wait until empty, then:  sbatch slurm/convergence.sbatch
sbatch slurm/convergence.sbatch

echo
echo "Watch:     squeue --me"
echo "Collect (merges per-config CSVs + prints markdown tables for PERF_NOTES.md):"
echo "  python src/complex/experiments/collect_results.py"
echo
echo "Stage 2+ (only after the headline gate passes): the iso-param scaling sweeps, per tier:"
echo "  WSS_TIER=100k sbatch slurm/scaling.sbatch   # --array=0-14"
echo "  WSS_TIER=1m   sbatch --array=0-15 slurm/scaling.sbatch"
echo "  WSS_TIER=10m  sbatch --array=0-13 slurm/scaling.sbatch"
