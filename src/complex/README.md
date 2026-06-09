# Weight Subspace Superposition (WSS) — core prototype

Implementation of the WSS layer and Riemannian training loop, per
[`agent_guide.md`](agent_guide.md) (the authoritative contract; its §0 corrections override
the PDF sketch). Scope: Phase-1 numerics core + the MNIST accuracy goalpost.

A `SuperpositionLinear` layer maps `R^n → R^m` as a **gated superposition of `J` rank-`r`
factorized components**:

```
W(X) = c · Σ_j  g_j(X) ⊙ (X U_j) S_j V_jᵀ  +  b
```

with `U_j ∈ St(r,n)`, `V_j ∈ St(r,m)` on Stiefel manifolds, `S_j = diag(exp(s_j)) > 0`, a
content gate `g_j` that reads the **left** frame `U`, and the normalization-XOR prefactor
`c` (`1/J` for non-normalized gates, `1` for softmax). The dense `n×m` weight is never
materialized in the forward/backward hot path.

## Why this exists (vs `src/simple/`)

The `src/simple/` attempt used plain Adam + a gradient-projection hook to keep `V`
orthonormal and the geometry collapsed. The fixes, all in this package:

- **Riemannian optimizer** (`geoopt.RiemannianAdam`) does retraction + moment transport — the
  manifold step is load-bearing, not a post-hoc gradient patch. (Guarded by the Remark-8
  no-retraction regression.)
- Gate reads `U`, not `V` (`agent_guide` §0.1).
- Spectrum is `σ = exp(s)`, always positive.
- Corrected He init `σ₀ = sqrt(2·J·m/r)` (variance ≈ `2/n`).
- Diversity penalty (von Neumann entropy of the stacked frames) keeps components spread.

## Layout

```
src/complex/
  config.py          # dataclasses; LayerConfig.validate() enforces r<=n,m and the norm-XOR rule
  device.py          # device selection + MPS->CPU fallback chokepoint
  manifold.py        # Haar (sign-fixed QR) init, Stiefel ManifoldParameter, orthonormality checks
  spectrum.py        # log-parametrized positive diagonal Σ
  gate.py            # energy u=||XU_j||²/||X||², phi variants, softmax-XOR normalization, detach
  superposition.py   # SuperpositionLinear (forward via batched einsum; materialize_weight for tests)
  diversity.py       # stacked Gram -> CPU eigvalsh -> von Neumann entropy + ENC
  grads_reference.py # closed-form Euclidean gradient oracle (TEST ONLY) + diversity fallback
  models.py          # MLP with dense / single_rank_Jr / wss layers (readout stays dense)
  train.py           # two param groups (RiemannianAdam + Adam), retraction toggle, fit/eval/smoke
  data.py            # MNIST / Fashion-MNIST loaders (-> <repo>/data)
  tests/             # 6 pytest files: invariants, normalization, grads, init, diversity, smoke
  experiments/headline_mnist.py   # the Phase-2 goalpost run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch 2.3.1, geoopt 0.5.1, torchvision, etc.
export PYTORCH_ENABLE_MPS_FALLBACK=1      # MPS lacks qr/solve/eigvalsh -> CPU fallback
```

## Run the tests (Phase-1 gates)

From the repo root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest src/complex/tests -q
```

The suite runs on CPU **and** MPS where applicable; `test_grads.py` is CPU + float64 only
(MPS cannot allocate float64). All gates: orthonormality after 10k RiemannianAdam steps
(< 1e-4); softmax sums to 1 + init scale invariant to `J`; analytic grads match autograd to
1e-5; init variance ≈ `2/n`; entropy/ENC bounds `[log r, log Jr]` / `[1, J]`; smoke-train loss
drops for all three layer types.

## Run the MNIST goalpost

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/complex/experiments/headline_mnist.py --epochs 10
# fast sanity:
python src/complex/experiments/headline_mnist.py --quick
```

Trains `dense`, `single_rank_Jr`, and `wss` (plus a `λ_div=0` ENC-collapse check and a
no-retraction Remark-8 control), printing a summary table and the key verdicts, and writing
`summary_*.csv` / `report_*.png` to `experiments/outputs/`.

## Results (MNIST, MLP 784→256→128→10, J=4, r=32, softmax gate)

Phase-1: all 20 pytest gates green on CPU + MPS.

Phase-2 (25 epochs, RiemannianAdam lr=1e-3; readout dense for all):

| run | params | test acc | ‖UᵀU−I‖∞ |
|---|---|---|---|
| dense (baseline) | 235,146 | 98.29% (overfits, train loss 7e-4) | — |
| single_rank_Jr (rank 4·32=128) | 184,202 | 95.63% | 2.5e-6 |
| **wss (J=4, r=32)** | 184,202 | **96.14%** | 5.5e-6 |

- **Headline (§4.2.1): wss beats single-rank-Jr by +0.51%** at exactly matched params — stable
  across 10-epoch (+0.71%) and 25-epoch runs. Gated J>1 helps at MNIST scale.
- **wss vs dense: −2.15%** at 78% of the params (a rank bottleneck on layer 1: rank 128 of 256)
  while dense overfits. A characterized, capacity-bound gap, just outside the ≤2% target.
- **Remark-8 (manifold step is load-bearing):** disabling the retraction (plain SGD on the raw
  frames) degrades orthonormality from ~5e-6 to ~1.3e-2 and climbing — the geometry invariant
  is maintained only by the retraction. (Accuracy impact is mild at this small scale.)
- **Diversity penalty:** at MNIST scale ENC stays near J (~3.5/4) with *and* without the penalty
  (λ_div=0 gives the same ENC) — Haar init + the softmax gate keep components diverse on their
  own, so the explicit penalty is redundant here. Whether it matters at scale is a Phase-3 question.

Reproduce: `python src/complex/experiments/headline_mnist.py --epochs 25 --runs dense,single_rank_Jr,wss`
(add `wss_div0,wss_no_retraction` to the `--runs` list for the diversity / Remark-8 controls).

## MPS notes (M1)

- Forward/backward (einsum/bmm) run on MPS. The Stiefel **retraction** (`solve`/`qr`) and the
  diversity `eigvalsh` are unimplemented on MPS and fall back to CPU (set the env var). The
  factors are tiny so this is correct, just slower per step than dense.
- The MNIST MLP is small enough that `--device cpu` is also fine.

## Deferred (clean seams left, not implemented)

Fashion-MNIST sweep, full λ_div/J/gate ablation matrix, interpretability probe,
`SuperpositionLinear.maybe_refactor()` global re-factorization, and the LDAdam second-moment
transport correction. See `agent_guide.md` §6.
