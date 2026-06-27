# SLURM jobs — WSS retraction GPU profiling

Single-GPU (A100/H200), ≤8h jobs that profile and convergence-test the pluggable Stiefel
retraction methods on small ViTs. The codebase was developed on M1/MPS; these run it on CUDA.

## What runs

The arrays ship **trimmed** to fit cluster job/time limits; the full 20-config grid stays in
`build_configs()`, so widen the `--array` lists to add anything back.

- **`profile.sbatch`** — trimmed `--array=0,5,6,7,9,10,16,17` (100k {dense, qr, ns, none, ns_K4} +
  1m {dense, ns, none}). Throughput + per-phase wall-clock attribution (forward / diversity /
  backward / optimizer-step = the retraction) + peak memory + orthonormality, via
  `experiments/profile_retraction.py --mode profile`. Minutes per task. `--profiler` also dumps a
  `torch.profiler` kernel table + chrome trace. (Cayley/single_rank/K2 dropped; `--array=0-19` for all.)
- **`convergence.sbatch`** — trimmed `--array=0,1,3,6,7` (100k tier only — the 1m tier trains
  ~1M-param ViTs and is the time/size hog). The param-matched accuracy comparison (dense baseline
  vs equal-param dense vs single_rank_Jr vs wss-newton_schulz) + the `none` collapse floor.

The config matrix (sizes × layer_type × retraction method), **with live param counts**, is:

```bash
python src/complex/experiments/profile_retraction.py --mode list
```

**20 configs, indices 0-19.** Each tier has TWO dense models — a full-size baseline and a
`dense_matched` shrunk to the factorized param count (the equal-param control that separates
"WSS helps" from "smaller models just regularize better") — plus `single_rank_Jr` and `wss`:

| tier | dense baseline | dense_matched | single_rank_Jr / wss (r) |
|------|----------------|---------------|--------------------------|
| `100k` (dim 56, depth 4, r 4) | ~110K (idx 0) | ~58K (idx 1) | ~60K (idx 2-9) |
| `1m`   (dim 128, depth 6, r 16) | ~811K (idx 10) | ~714K (idx 11) | ~715K (idx 12-19) |

Within each tier, wss spans the retraction sweep: canonical / qr / newton_schulz / none + NS at
`retract_every` 2,4. The `100k` tier uses r=4 (so WSS is genuinely ~half of dense); the `1m` tier
keeps r=16 for continuity with prior runs.

## Cluster settings (pre-filled for this cluster)

Both `.sbatch` files are set up for the current cluster: `--partition=gpu`, `--gres=gpu:a100:1`,
`--nodes=1 --ntasks-per-node=1`, `REPO=/home/biggs.s/WSS-Demo-Dev`,
`WSS_ENV=/scratch/biggs.s/conda_envs/wss`. `--partition` is **required** — omitting it lands you on
the default partition (no A100) and `sbatch` fails with *"Requested node configuration is not available."*

A teammate on a different cluster edits, in BOTH files:

| Directive / var        | Set to                                                        |
|------------------------|---------------------------------------------------------------|
| `#SBATCH --partition`  | your A100/H200 partition                                      |
| `#SBATCH --gres`       | site GPU syntax (some clusters use `--gpus=1`)                |
| `#SBATCH --account`    | your allocation/account, if your site requires one           |
| `WSS_REPO`             | repo path on the cluster (or `export WSS_REPO=...`)          |
| `WSS_ENV`              | conda env prefix path / name (or `export WSS_ENV=...`)       |

The env is activated via `conda activate "$WSS_ENV"` (a prefix path works); adjust the `module load`
lines if your site names modules differently. `PYTORCH_ENABLE_MPS_FALLBACK=1` is a no-op on CUDA.

## Run

```bash
# one-off sanity check (interactive GPU node) -- idx 6 = 100k-wss-newton_schulz:
python src/complex/experiments/profile_retraction.py --mode profile --config 6 --steps 50 --device cuda

# full sweep:
bash slurm/submit_all.sh          # creates slurm/logs/, submits both arrays
squeue --me
```

Outputs land in `src/complex/experiments/outputs/perf/`
(`profile_cfg<idx>_cuda.csv`, `convergence_cfg<idx>_cuda.csv`, `profiler_*.txt`, `trace_*.json`).
`submit_all.sh` prints a one-liner to merge the per-config CSVs into `profile_all_cuda.csv`.

Record the verdict (speedup per method, accuracy parity, drift) in `PERF_NOTES.md` at the repo root.
