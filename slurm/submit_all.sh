#!/bin/bash
# Submit the Stage-0 PROFILING array ONLY. Each array task is an independent single-GPU job.
# EDIT slurm/*.sbatch (account, partition, repo path, conda env) FIRST.
#
# This script intentionally submits ONE stage. The `gpu` QOS caps you at 8 jobs in-system / 4 running
# per user, so submitting profiling + parity + convergence together = 11 > 8 and the extras are
# rejected (QOSMaxSubmitJobPerUserLimit). Run the later stages BY HAND, one at a time, each only once
# `squeue --me` is empty. PROFILING and SCALING are separate jobs and are never chained here.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
mkdir -p slurm/logs              # sbatch needs the log dir to exist at submit time

echo "Stage 0 — profiling array (100k tier, 5 tasks)..."
sbatch slurm/profile.sbatch

cat <<'NEXT'

Submitted. Watch:  squeue --me

Run the rest by hand, each only when `squeue --me` is empty (8-job cap), each <= 8 tasks:

  # Stage 0c + Stage 1  (parity = 1 task, convergence = 5 tasks; 6 <= 8):
  sbatch slurm/parity.sbatch && sbatch slurm/convergence.sbatch

  # Collect results -> markdown tables for PERF_NOTES.md:
  python src/complex/experiments/collect_results.py

  # Stage 2 SCALING -- a SEPARATE job, only after reviewing Stage 1. Chunk to <= 8:
  WSS_TIER=100k sbatch --array=0-7 slurm/scaling.sbatch     # then --array=8-14 once it drains
NEXT
