# WSS GPU throughput — perf notes (branch `perf/gpu-retraction`)

Goal: speed up WSS training on A100/H200 for scaling ablations **without breaking math
faithfulness**. The team suspected Python overhead + QR-retraction cost.

## Diagnosis (verified against the code, not assumed)

1. **The forward/backward is already fully einsum-vectorized over `J`** (`superposition.py`
   forward) — no Python loop over components. The "Python relaunch overhead" hypothesis is
   largely misdirected for the math itself; the residual per-step Python cost (double
   `zero_grad`/`step`, `.item()` syncs) is minor.
2. **The per-step Stiefel retraction is the real cost.** geoopt's default `CanonicalStiefel`
   (`stiefel_canonical=True`) does an `n×n` `torch.linalg.solve` (Cayley) every step — and on M1
   `solve`/`qr` have no MPS kernel, so they fall back to CPU. Terminology note vs the original
   ask: the default is *already* Cayley/`solve`; QR is the *existing faster* alternative.
3. **The diversity `eigvalsh` was hard-routed to CPU every step** (`diversity.py`) — an MPS
   workaround. `diversity_loss()` runs every step, so **on CUDA this was a pure GPU→CPU→GPU sync
   each step** (eigvalsh is native on CUDA).

## What changed

- **New `src/complex/retraction/` subpackage** — a matmul-only **Newton–Schulz** retraction
  (`NewtonSchulzStiefel`), a no-retraction control (`NoRetractionStiefel`), and an
  every-K-steps `LazyRetractionStiefel`. See that package's `README.md` for the math (convergent
  *cubic* iteration, spectral normalization, why **not** the Muon quintic).
- **`retraction_method` / `retract_every`** config knobs on `LayerConfig` and `ViTConfig`,
  threaded through `make_stiefel_param` → `superposition` → `vit`. Default `"auto"` / `1`
  reproduces the prior behavior byte-for-byte (all 63 existing gates still green).
- **CUDA-native diversity**: `device.needs_cpu_linalg()` keeps `eigvalsh` on CPU only for MPS;
  on CUDA/CPU it stays on-device, removing the per-step host sync. MPS numerics unchanged.
- **Profiling harness** `experiments/profile_retraction.py` (+ `slurm/`): per-phase wall-clock
  attribution (forward / diversity / backward / optimizer-step=retraction) via CUDA events,
  peak memory, orthonormality drift, `torch.profiler` traces, and a short CIFAR-10 convergence mode.

## Experimental grid (20 configs, `--mode list`)

Two size tiers, each with **two dense models** + the factorized models:

| tier | dense baseline | dense_matched (equal-param control) | single_rank_Jr / wss |
|---|---|---|---|
| `100k` (dim 56, depth 4, **r 4**) | ~110K | ~58K (dim 40) | ~60K |
| `1m` (dim 128, depth 6, **r 16**, continuity) | ~811K | ~714K (dim 120) | ~715K |

`dense_matched` is a dense ViT shrunk to ~the factorized param count — the equal-param control that
separates "WSS's structure helps" from "smaller models just regularize / scale differently". The
`100k` tier uses **r=4** so WSS is genuinely ~half of dense (at r=8, `J·r=dim/2` makes WSS ≈ dense
in params and the control is moot). `wss` spans the full retraction sweep
(canonical / qr / newton_schulz / none + NS at `retract_every` 2,4); `single_rank_Jr` gets
canonical + newton_schulz.

## Faithfulness ledger

| change | verdict |
|---|---|
| CUDA-native `eigvalsh` (device guard) | **faithful** — identical math, MPS path byte-identical |
| `qr` retraction | **faithful** — valid Stiefel retraction (different metric), already in geoopt |
| `newton_schulz` retraction | **faithful** — polar retraction (≈ geoopt `projx`), different trajectory than Cayley; OFF by default |
| `none` | **control** — non-faithful, geometry drifts; for ablation only |
| `retract_every>1` | **control** — non-faithful, drift corrupts spectrum/gate/diversity invariants; reported, not shipped |

## Results — local sanity (M1 CPU, tiny, NOT the headline)

Per-step retraction cost, 100k wss ViT (dim 56, r 4, 60K params), bs 64 (CPU; GPU picture differs
— NS is pure GEMM with no CPU fallback, and canonical's `n×n` solve scales far worse at the 1M
size where the real win should show):

| method | optimizer-step (retraction) ms | orthonormality |
|---|---|---|
| canonical (Cayley/solve) | ~23 | 9.5e-7 |
| qr | ~9 | 4.8e-7 |
| newton_schulz | ~15 | 3.6e-7 |
| none | ~6 | 1.4e-5 (drifting) |

## Results — A100/H200 (FILL IN after `slurm/submit_all.sh`)

Merge per-config CSVs (`outputs/perf/profile_all_cuda.csv`) and fill:

### Throughput + attribution (`--mode profile`)

| idx | config | params | it/s | ms fwd | ms div | ms bwd | ms retr | peak MB | ortho |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100k-dense | | | | | | | | |
| 4 | 100k-wss-canonical | | | | | | | | |
| 5 | 100k-wss-qr | | | | | | | | |
| 6 | 100k-wss-newton_schulz | | | | | | | | |
| 8,9 | 100k-wss-ns_K2 / ns_K4 | | | | | | | | |
| 10 | 1m-dense | | | | | | | | |
| 14 | 1m-wss-canonical | | | | | | | | |
| 16 | 1m-wss-newton_schulz | | | | | | | | |

### Convergence (`--mode convergence`, CIFAR-10)

The headline is the **param-matched 4-way at each tier** (do the factorized models beat an
equal-param dense, and does fast retraction preserve it?):

| idx | config | params | final acc | ortho |
|---|---|---|---|---|
| 0 | 100k-dense (baseline) | ~110K | | |
| 1 | 100k-dense_matched (control) | ~58K | | |
| 2 | 100k-single_rank_Jr-canonical | ~60K | | |
| 4 | 100k-wss-canonical | ~60K | | |
| 6 | 100k-wss-newton_schulz | ~60K | | |
| 7 | 100k-wss-none (collapse floor) | ~60K | | |
| 9 | 100k-wss-ns_K4 (lazy cost) | ~60K | | |

### Verdict (FILL IN)

- Speedup of `newton_schulz` vs `canonical` at 100K / 1M: __×__ / __×__.
- Did the CUDA-native diversity remove the eigvalsh sync? (compare `ms div` on CUDA): ____.
- Accuracy parity `newton_schulz`/`qr` vs `canonical`: ____ (within noise?).
- **WSS vs equal-param dense** (idx 4/6 vs idx 1) and vs single_rank_Jr (idx 2): does the gated
  J-component structure beat a same-size dense and the single-subspace control? ____.
- Cost of the `retract_every∈{2,4}` controls (Δacc, drift): ____.
- Recommended default for scaling ablations: ____.
