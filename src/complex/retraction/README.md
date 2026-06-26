# Pluggable Stiefel retraction (perf branch)

This subpackage adds GPU-friendly, swappable retraction methods for the Stiefel frames `U_j`,
`V_j` of WSS. It exists because the per-step **retraction is the throughput bottleneck**, not the
WSS forward/backward (which is already fully einsum-vectorized over `J`, no Python loop).

Defaults are unchanged: `retraction_method="auto"`, `retract_every=1` reproduce the pre-perf
behavior byte-for-byte, so every existing call site and faithfulness gate is untouched.

## The methods

| `retraction_method` | how `retr(x,u)` works | faithful? | notes |
|---|---|---|---|
| `auto` (default) | `canonical` if `stiefel_canonical` else `qr` | yes | exact legacy behavior |
| `canonical` | Cayley transform via `torch.linalg.solve` on an `n×n` skew matrix (geoopt `CanonicalStiefel`) | yes (agent_guide default) | **bottleneck**: `O(n³)` solve; no MPS kernel → CPU fallback on M1 |
| `qr` | reduced `qr(x+u)` with sign fix (geoopt `EuclideanStiefel`) | yes (different metric) | `O(nr²)`; QR has limited GPU/MPS parallelism |
| `newton_schulz` | matmul-only polar orthonormalization of `x+u` | yes (polar retraction) | **GPU-ideal**: pure GEMM, no qr/svd/solve, no host sync, batched over `J` |
| `none` | `x+u`, no projection | **NO — control** | geometry drifts; pair with `stabilize=None` |
| + `retract_every=K>1` | wrap qr/newton_schulz, re-orthonormalize every K steps | **NO — control** | retraction-frequency vs speed tradeoff |

## Why Newton–Schulz, and which iteration

To re-project a frame onto the Stiefel manifold we drive its singular values to 1, i.e. compute
the orthogonal **polar factor** `U Vᵀ` of `Y = U Σ Vᵀ` — exactly what geoopt's SVD-based `projx`
does, but with matmuls only. `polar(x+u)` is a textbook first-order Stiefel retraction (same
family as geoopt's QR `retr` and SVD `projx`), so `newton_schulz` is a *faithful* alternative
trajectory, just not the canonical Cayley one — hence OFF by default.

We use the **convergent cubic** Newton–Schulz iteration

```
Y ← 1.5·Y − 0.5·Y (YᵀY)
```

whose singular-value map `s ↦ s(3−s²)/2` has a stable fixed point at `s=1` with `f'(1)=0`
(**quadratic** convergence), valid for `s ∈ (0, √3)`.

> ⚠️ This is **not** the Muon quintic `(3.4445, −4.7750, 2.0315)`. Those coefficients are tuned to
> orthogonalize *gradients* fast and deliberately leave singular values in a loose band around 1 —
> they never reach the `‖YᵀY − I‖ < 1e-4` the Stiefel orthonormality gates require. We verified
> this empirically (residual stuck at ~0.3); the cubic is what maintains the invariant.

**Normalization matters.** We must guarantee `s_max < √3` for convergence. Per-frame *Frobenius*
scaling (`Y/‖Y‖_F`) divides an orthonormal `(n,r)` frame by `√r`, pushing its singular values to
`1/√r` and **destroying the warm-start** (then ~8 iters are needed). Instead we estimate `s_max`
with a couple of **power-iteration** steps on the small `r×r` Gram `A = YᵀY` (matmul-only,
deterministic from a ones vector) and divide by it. Near the manifold `s_max ≈ 1`, so this is
~a no-op and the cubic converges quadratically from the warm start.

**Measured residual `‖UᵀU − I‖∞`** (M1, fp32), confirming the iteration counts we ship:

| start | `r=8` | `r=16` | `r=32` |
|---|---|---|---|
| warm (`x + 1e-2·noise`), **5 iters** (`retr` default) | ~1e-7 | ~3e-7 | ~3e-7 |
| cold (random), **8 iters** (`projx` default) | ~1e-7 | ~2e-7 | ~4e-7 |

and over 500 `RiemannianAdam` steps: `canonical`/`auto` ≈ 1.8e-6, `qr` ≈ 7e-7,
`newton_schulz` ≈ 2e-7, `none` ≈ 2.7e-3 (drifts, as a control should).

## Integration (no custom optimizer)

`NewtonSchulzStiefel` / `NoRetractionStiefel` subclass geoopt's `EuclideanStiefel` and override
**only** `retr` (and `projx`). `RiemannianAdam` calls `retr_transp`, whose base implementation
composes our `retr` + the inherited Euclidean `transp`; the inherited
`proju`/`egrad2rgrad`/`inner` provide the unchanged tangent-grad projection and metric. So the
adaptive optimizer machinery (moments, transport, `stabilize`) is reused verbatim — only the
retraction step changes. All ops are batched over the leading `J` dimension of `(J, n, r)`.

## The `retract_every=K` tradeoff (NON-FAITHFUL)

`LazyRetractionStiefel` takes `K−1` cheap additive (`x+u`) steps then a real orthonormalization on
the K-th. This is **distinct from geoopt `stabilize`** (an *extra* periodic `projx` on top of the
per-step retraction). Between projections the frame is off-manifold **by design**, which does not
just relax a float tolerance — it silently corrupts three invariants that assume orthonormal
frames, so any `K>1` trades correctness for speed and must be *reported*, never shipped:

- **Spectrum scale**: the He init `σ₀=√(2Jm/r)` assumes unit-norm columns; drift gets re-absorbed
  by the separately-trained log-spectrum, breaking `σ_j` identifiability.
- **Gate energy**: `‖xU_j‖²/‖x‖² ∈ [0,1]` only when `U_j` is orthonormal; off-manifold it can
  exceed 1 and blow up `exp`/`pow` gates.
- **Diversity**: the von Neumann entropy bounds / `ENC ∈ [1,J]` assume `trace((1/J)UᵀU)=r`.

Sample orthonormality only **immediately after** a projection (at multiples of `K`). The
convergence jobs quantify the accuracy cost at `K∈{2,4}`.

## Use

```python
ViTConfig(layer_type="wss", retraction_method="newton_schulz")          # faithful, GPU-fast
ViTConfig(layer_type="wss", retraction_method="newton_schulz", retract_every=4)  # control
make_stiefel_param(n, r, J, retraction_method="qr")                     # low-level
```

See `experiments/profile_retraction.py` and `slurm/` for the benchmarking harness.
