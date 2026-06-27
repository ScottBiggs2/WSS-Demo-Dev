#!/bin/bash
# Submit the full profiling + convergence sweep. EDIT slurm/*.sbatch (account, partition, repo
# path, conda env) FIRST. Each array task is an independent single-GPU job, so they schedule
# concurrently as the cluster allows.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
mkdir -p slurm/logs              # sbatch needs the log dir to exist at submit time

echo "Submitting profiling array (see --array in profile.sbatch; minutes each)..."
sbatch slurm/profile.sbatch

echo "Submitting convergence array (see --array in convergence.sbatch)..."
sbatch slurm/convergence.sbatch

echo
echo "Watch:    squeue --me"
echo "Aggregate after completion:"
echo "  python - <<'PY'"
echo "  import csv, glob"
echo "  rows=[]"
echo "  for f in sorted(glob.glob('src/complex/experiments/outputs/perf/profile_cfg*_cuda.csv')):"
echo "      rows += list(csv.DictReader(open(f)))"
echo "  cols=list(dict.fromkeys(k for r in rows for k in r))"
echo "  w=csv.DictWriter(open('src/complex/experiments/outputs/perf/profile_all_cuda.csv','w'),fieldnames=cols)"
echo "  w.writeheader(); w.writerows(rows)"
echo "  print('merged', len(rows), 'rows')"
echo "  PY"
