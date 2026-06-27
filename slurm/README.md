# SLURM jobs — WSS scaling + ablation flight (CIFAR-100)

Single-GPU (A100/H200), ≤8h jobs for the WSS scaling/ablation study on **CIFAR-100** ViTs at three
param tiers (~100K / 1M / 10M). The codebase was developed on M1/MPS; these run it on CUDA in **bf16
autocast** (matmuls bf16, linalg fp32) with TF32 GEMM. The funnel screens cheap and promotes winners
(see `PERF_NOTES.md` / the plan): Stage 0 gates → 100K headline → scaling sweeps → 1M → 10M.

## What runs

Arrays ship **trimmed** to fit job/time limits; the full grid stays in `build_configs()` /
`build_scaling_configs()`, so widen the `--array` lists to add anything back.

- **`profile.sbatch`** — throughput + per-phase attribution (forward / diversity / backward /
  optimizer-step = the retraction) + peak memory + orthonormality, via `profile_retraction.py
  --mode profile`. Trimmed to per-tier {dense, wss-qr, wss-ns, wss-none, wss-ns_K4}. Minutes per
  task; `--profiler` dumps a kernel table + chrome trace. `--array=0-29` for the full grid.
- **`parity.sbatch`** — **Stage-0c trust gate**: `--mode parity` trains 100k-wss-newton_schulz twice
  (fp32 then bf16, same seed) and reports Δacc / Δortho. Pass this gate before trusting any bf16 run.
- **`convergence.sbatch`** — the **Stage-1 headline** (100k tier): does wss beat the EQUAL-PARAM
  dense (`dense_matched`)? `--array=0,1,3,6,7` = {dense, dense_matched, single_rank_Jr-ns, wss-ns,
  wss-none}. Promote to 1m/10m by widening the array (10,11,13,16 / 20,21,23,26) with `WSS_EPOCHS` set.
- **`scaling.sbatch`** — **Stage-2+ iso-param sweeps** (`scaling_suite.py`): depth↔width and J↔r at a
  tier chosen by `WSS_TIER` (100k|1m|10m). One job per sweep config.

The retraction-sweep matrix (sizes × layer_type × retraction method), **with live param counts**:

```bash
python src/complex/experiments/profile_retraction.py --mode list           # 30 configs, idx 0-29
python src/complex/experiments/scaling_suite.py --mode list --tier 100k     # the scaling sweep
```

**30 configs, indices 0-29 (10 per tier).** Each tier has a full-size dense baseline + a
`dense_matched` shrunk to the factorized param count (the equal-param control that separates "WSS
helps" from "smaller models just regularize better"), plus `single_rank_Jr` and `wss`. CIFAR-100
counts (`param_budget` == live ViT count; matched dim is derived, not hand-tuned):

| tier | anchor (dim, depth, J·r) | dense baseline | dense_matched | single_rank_Jr / wss |
|------|--------------------------|----------------|---------------|----------------------|
| `100k` (dim 72, depth 4, J·r 24) | idx 0: 184,780 | idx 1: 115,068 | 116,236 (idx 2-9) |
| `1m`   (dim 192, depth 6, J·r 48) | idx 10: 1,823,908 | idx 11: 797,806 | 830,308 (idx 12-19) |
| `10m`  (dim 384, depth 12, J·r 192) | idx 20: 14,289,892 | idx 21: 12,567,340 | 12,534,244 (idx 22-29) |

Within each tier, wss spans the retraction sweep: canonical / qr / newton_schulz / none + NS at
`retract_every` 2,4. Tier index layout is stable: tier `t` occupies `10*t .. 10*t+9`.

## Cluster settings (pre-filled for this cluster)

All four `.sbatch` files are set up for the current cluster: `--partition=gpu`, `--gres=gpu:a100:1`,
`--nodes=1 --ntasks-per-node=1`, `REPO=/home/biggs.s/WSS-Demo-Dev`,
`WSS_ENV=/scratch/biggs.s/conda_envs/wss`, `WSS_CONDA_BASE=/home/biggs.s/miniconda`. `--partition`
is **required** — omitting it lands you on the default partition (no A100) and `sbatch` fails with
*"Requested node configuration is not available."*

A teammate on a different cluster edits, in ALL FOUR files:

| Directive / var        | Set to                                                        |
|------------------------|---------------------------------------------------------------|
| `#SBATCH --partition`  | your A100/H200 partition                                      |
| `#SBATCH --gres`       | site GPU syntax (some clusters use `--gpus=1`)                |
| `#SBATCH --account`    | your allocation/account, if your site requires one           |
| `WSS_REPO`             | repo path on the cluster (or `export WSS_REPO=...`)          |
| `WSS_ENV`              | conda env prefix path / name (or `export WSS_ENV=...`)       |
| `WSS_CONDA_BASE`       | base install whose `etc/profile.d/conda.sh` is sourced       |

**Do NOT `module load anaconda3` to get conda** — that was the original bug: on this cluster the
anaconda module prepends `/shared/.../anaconda3/bin` to `PATH`, which **shadows** the activated env's
python, so the job runs a system python *without geoopt* and dies on `import geoopt` (the GPU is
allocated, then the script crashes on the first import — check the `.err`). The jobs source
`$WSS_CONDA_BASE/etc/profile.d/conda.sh` directly instead, then `conda activate "$WSS_ENV"` (a prefix
path works). `cuda/12.1.1` is loaded to match torch's cu121 build; a bare `module load cuda` pulls
this cluster's default 13.x, a mismatch. `PYTORCH_ENABLE_MPS_FALLBACK=1` is a no-op on CUDA.

## Run

```bash
# one-off sanity check (interactive GPU node) -- idx 6 = 100k-wss-newton_schulz:
python src/complex/experiments/profile_retraction.py --mode profile --config 6 --steps 50 --device cuda --amp

# the funnel (gate each stage before the next):
bash slurm/submit_all.sh          # Stage 0 (profile + parity) + Stage 1 (100k headline)
squeue --me
# after the headline gate passes, the iso-param scaling sweeps, per tier:
WSS_TIER=100k sbatch slurm/scaling.sbatch                  # --array=0-14
WSS_TIER=1m   sbatch --array=0-15 slurm/scaling.sbatch
WSS_TIER=10m  sbatch --array=0-13 slurm/scaling.sbatch     # may need the 8h epoch-cap mitigation
```

Outputs land in `src/complex/experiments/outputs/perf/` (`profile_cfg<idx>_cuda.csv`,
`convergence_cfg<idx>_cuda.csv`, `parity_cfg<idx>_cuda.csv`, `scaling_<tier>_cfg<idx>_cuda.csv`,
`profiler_*.txt`, `trace_*.json`).

Merge + format them with the collector (pure stdlib, runs anywhere):

```bash
python src/complex/experiments/collect_results.py    # writes *_all_cuda.csv, prints MD tables
```

It prints markdown tables ready to paste into `PERF_NOTES.md`: throughput/phase-attribution (it/s,
`x vs dense`, the forward/diversity/backward/**retraction** split, peak MB, ortho err, retraction
speedup), the **fp32-vs-bf16 parity** trust gate (Δacc, Δortho), the param-matched convergence table
(final acc, dtype, ortho err), and the scaling-sweep table. Record the verdict per method there.

**8h cap at 10m:** a 10m config may straddle the 8h wall-clock cap. Mitigate by capping `WSS_EPOCHS`
(report at the epoch reached), running only the headline pair {wss, dense_matched} + 1 ceiling dense,
or raising `--batch_size` (re-tune LR). bf16 is the default; if the parity gate fails, drop `--amp`
and re-budget (fp32 ~doubles 10m wall-clock).

**QOS submit limit:** if `submit_all.sh`'s second `sbatch` fails with `QOSMaxSubmitJobPerUserLimit`,
the profiling array already fills your per-user queue quota (pending + running). A dependent job
still occupies a slot, so submit convergence *after* profiling drains: watch `squeue --me`, then
`sbatch slurm/convergence.sbatch`. Check your ceiling with
`sacctmgr -n show assoc user=$USER format=QOS,MaxSubmitJobs,MaxJobs`.
