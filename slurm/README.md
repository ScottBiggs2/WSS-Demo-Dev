# SLURM jobs — WSS scaling + ablation flight (CIFAR-100)

Single-GPU (A100/H200), ≤8h jobs for the WSS scaling/ablation study on **CIFAR-100** ViTs at three
param tiers (~100K / 1M / 10M). The codebase was developed on M1/MPS; these run it on CUDA in **bf16
autocast** (matmuls bf16, linalg fp32) with TF32 GEMM. The funnel screens cheap and promotes winners
(see `PERF_NOTES.md` / the plan): Stage 0 gates → LR calibration → 100K headline → scaling sweeps → 1M → 10M.

## What runs

Arrays ship **trimmed** to fit job/time limits; the full grid stays in `build_configs()` /
`build_scaling_configs()`, so widen the `--array` lists to add anything back.

- **`profile.sbatch`** — throughput + per-phase attribution (forward / diversity / backward /
  optimizer-step = the retraction) + peak memory + orthonormality, via `profile_retraction.py
  --mode profile`. Trimmed to per-tier {dense, wss-qr, wss-ns, wss-none, wss-ns_K4}. Minutes per
  task; `--profiler` dumps a kernel table + chrome trace. `--array=0-29` for the full grid.
- **`parity.sbatch`** — **Stage-0c trust gate**: `--mode parity` trains 100k-wss-newton_schulz twice
  (fp32 then bf16, same seed) and reports Δacc / Δortho. Pass this gate before trusting any bf16 run.
- **`lr_sweep.sbatch`** — **Stage-0.5 LR calibration** (`lr_sweep.py`), run BEFORE the Stage-1
  headline. LR is not a fair shared axis across geometries: WSS carries its weight in the Stiefel
  frames, so it needs a larger `lr_riemann` than dense's `lr_euclid` (W=U·S·Vᵀ is bilinear → the
  frames must rotate a finite angle per step). Sweeps each method's primary LR group (default 3
  methods × 4 LRs = 12 jobs/tier; chunk `--array=0-7` then `8-11`); `collect_results.py` prints the
  best LR per method to pin into the headline. Tier via `WSS_TIER`.
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

All five `.sbatch` files are set up for the current cluster: `--partition=gpu`, `--gres=gpu:1`,
`--nodes=1 --ntasks-per-node=1`, `REPO=/home/biggs.s/WSS-Demo-Dev`,
`WSS_ENV=/scratch/biggs.s/conda_envs/wss`, `WSS_CONDA_BASE=/home/biggs.s/miniconda`. The 100k jobs
ask for a generic `--gres=gpu:1` (the a100s are usually saturated; the abundant v100s schedule
immediately) — re-add `--gres=gpu:a100:1` for the 1m/10m tiers or for strictly comparable it/s.
`--partition=gpu` is **required** — the default partition has no GPUs, so omitting it fails with
*"Requested node configuration is not available."*

A teammate on a different cluster edits, in ALL FIVE files:

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

# the funnel -- ONE stage at a time, each <= 8 jobs, next only when `squeue --me` is empty:
bash slurm/submit_all.sh          # Stage 0: profiling only (5 tasks). Prints the follow-ups.
squeue --me                       # wait until empty, then:
sbatch slurm/parity.sbatch        # Stage 0c trust gate (1 task)
WSS_TIER=100k sbatch slurm/lr_sweep.sbatch   # Stage 0.5 LR calibration (--array=0-7 then 8-11)
# read the best LR per method (collect_results -> "LR calibration" table), pin into the headline:
sbatch slurm/convergence.sbatch   # Stage 1 headline (5 tasks), each method at its calibrated LR
# after the headline gate passes, the iso-param scaling sweeps -- chunk each tier to <= 8:
WSS_TIER=100k sbatch --array=0-7 slurm/scaling.sbatch      # then --array=8-14 once it drains
WSS_TIER=1m   sbatch --array=0-7 slurm/scaling.sbatch      # then --array=8-15
WSS_TIER=10m  sbatch --array=0-7 slurm/scaling.sbatch      # then --array=8-13 (+ 8h epoch-cap)
```

Outputs land in `src/complex/experiments/outputs/perf/` (`profile_cfg<idx>_cuda.csv`,
`convergence_cfg<idx>_cuda.csv`, `parity_cfg<idx>_cuda.csv`, `scaling_<tier>_cfg<idx>_cuda.csv`,
`lr_sweep_<tier>_cfg<idx>_cuda.csv`, `profiler_*.txt`, `trace_*.json`).

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

**QOS submit limit (the big gotcha):** the `gpu` QOS caps you at **8 jobs in-system (pending +
running) and 4 running**, per user — verify with
`sacctmgr -n show qos gpu format=MaxSubmitJobsPerUser,MaxJobsPerUser`. So a single array of > 8
tasks, *or* several arrays totalling > 8, is rejected at submit with `QOSMaxSubmitJobPerUserLimit`
**even when `squeue --me` is empty** (the array size itself exceeds the cap). Dependencies don't help
— a pending dependent job still occupies a submit slot. Work within it:

- Keep every submit **≤ 8 tasks**, and submit stages one at a time — wait for `squeue --me` to empty
  before the next. `submit_all.sh` fires only the profiling array and prints the follow-up commands.
- The trimmed `profile.sbatch` ships the **100k tier only** (5 tasks). Add 1m/10m as separate arrays
  after it drains (`sbatch --array=10,15,16,17,19 slurm/profile.sbatch`, etc.).
- `scaling.sbatch` tiers are 14–16 configs, so **chunk** them (`--array=0-7` then `8-N`).
- Dropping `--gres=gpu:a100:1` → generic `--gres=gpu:1` lets jobs schedule on the idle v100s instead
  of queueing behind the saturated a100s — it speeds *scheduling* but does **not** raise the submit
  cap. (The 100k `.sbatch` files already use `gpu:1`.)
