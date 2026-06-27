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
  peak memory, orthonormality drift, `torch.profiler` traces, and a short convergence mode.

## Scaling-flight additions (CIFAR-100 study)

For the 100K/1M/10M scaling + ablation flight (see the plan / the funnel below), on top of the
retraction work:

- **bf16 autocast** over the matmul-heavy forward+loss (`train.py` `train_epoch`, profile/scaling
  via `--amp`), with **all linalg forced fp32**: the diversity Gram + `eigvalsh` re-enable fp32 at
  `diversity.gram()`, and the optimizer step / retraction run outside autocast on fp32 master params
  (NS also self-casts). No GradScaler (bf16). A Stage-0c `--mode parity` job certifies bf16 vs fp32.
- **TF32** matmul/cudnn flags via `device.setup_backend(allow_tf32)` (CUDA-only no-op), `--allow_tf32`.
- **`weight_decay`** (TrainConfig) on the Euclidean Adam; passed to RiemannianAdam too but ~inert on
  the Stiefel frames (verified: ortho stays <1e-4 with WD on). **`attn_dropout`/`mlp_dropout`**
  (ViTConfig), default 0.0 ⇒ `nn.Dropout` identity.
- **`experiments/param_budget.py`** — closed-form ViT param count (== live count, tested) + a
  `solve_dim_depth` / `dense_matched_dim` solver, so tier dims and the equal-param control are
  derived (correctly counting CIFAR-100's `dim*100` head) rather than hand-tuned.
- **`experiments/scaling_suite.py`** — iso-param depth↔width and J↔r sweeps (retraction fixed to the
  Stage-0 winner). A factorized proj costs `J·r·(n+m)+J·r+b·m`, so J↔r at fixed `J·r` is exactly
  iso-param and wss == single_rank_Jr by construction.

## Experimental grid (30 configs, `--mode list`; CIFAR-100, num_classes=100)

Three tiers, each with a dense baseline + a `dense_matched` equal-param control + the factorized
models. Counts verified (`param_budget` == live ViT count); `dense_matched_dim` is **derived**:

| tier | anchor (dim, depth, J·r) | dense baseline | dense_matched | single_rank_Jr / wss |
|---|---|---|---|---|
| `100k` (dim 72, depth 4, **J·r 24**) | 184,780 | 115,068 (dim 56) | 116,236 |
| `1m`   (dim 192, depth 6, **J·r 48**) | 1,823,908 | 797,806 (dim 126) | 830,308 |
| `10m`  (dim 384, depth 12, **J·r 192**) | 14,289,892 | 12,567,340 (dim 360) | 12,534,244 |

`dense_matched` separates "WSS's structure helps" from "smaller models just regularize / scale
differently" — **the headline comparator**. `wss` spans the retraction sweep (canonical / qr /
newton_schulz / none + NS at `retract_every` 2,4); `single_rank_Jr` gets canonical + newton_schulz.
Tier index layout is stable: tier `t` occupies `10*t .. 10*t+9` (100k 0-9, 1m 10-19, 10m 20-29).

## Funnel (screen cheap, promote winners)

Stage 0 (100K): profile → pick fast retraction; `--mode parity` → **GATE 0c** trust bf16; retraction
parity vs canonical. → Stage 1 (100K headline, 3 seeds): **GATE 1** wss > dense_matched? → Stage 2
(100K iso-param axes): keep winning regions. → Stage 3 (1M headline): **GATE 3** wss > dense_matched
at 100K **and** 1M? → Stage 4 (10M, 2 seeds) + lr/λ_div/retract_every/wd/dropout ablations at 100K/1M.

## Faithfulness ledger

| change | verdict |
|---|---|
| CUDA-native `eigvalsh` (device guard) | **faithful** — identical math, MPS path byte-identical |
| `qr` retraction | **faithful** — valid Stiefel retraction (different metric), already in geoopt |
| `newton_schulz` retraction | **faithful** — polar retraction (≈ geoopt `projx`), different trajectory than Cayley; OFF by default |
| bf16 autocast (fwd+loss only; eigvalsh + NS retraction + optimizer step kept fp32) | **faithful-with-tolerance** — quantified by the `--mode parity` gate; default off (fp32) |
| TF32 matmul/cudnn flags | **faithful-with-tolerance** — reduced-precision GEMM accumulation; default off, on for the suite |
| `weight_decay` on Euclidean params (dense W, spectrum, bias) | **faithful** — standard L2 |
| `weight_decay` on Stiefel `U`/`V` | **near-inert / labeled** — radial part projected out by `egrad2rgrad` (ortho stays <1e-4); kept for API symmetry |
| `attn_dropout`/`mlp_dropout` (p>0) | **faithful** — standard ViT regularization; p=0 default is exact identity |
| CIFAR-100 (own mean/std, reused 32×32 augment pipeline) | **faithful** |
| `none` | **control** — non-faithful, geometry drifts; for ablation only |
| `retract_every>1` | **control** — non-faithful, drift corrupts spectrum/gate/diversity invariants; reported, not shipped |

### Precision policy (autocast boundary)

bf16 autocast wraps **only** the forward + loss (`train_epoch`). Everything that needs fp32 is kept
fp32: `diversity.gram()` re-enables fp32 (`torch.autocast(enabled=False)` + `.float()`) so `eigvalsh`
sees fp32; the Newton–Schulz retraction self-casts (`Y = X.float()`); the optimizer step (and thus
the Stiefel retraction) runs outside autocast on fp32 master params. bf16 needs no GradScaler. Eval
runs in fp32. Any result with `init_scale != 1.0` combined with `--amp` is untested and non-headline.

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
