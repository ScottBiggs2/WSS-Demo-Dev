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

## Phase 2.5 tooling (scaling + interpretability)

Three additions, all isolated to `memory.py` + the experiment scripts (the numerics core is
untouched):

**Memory breakdown** ([memory.py](memory.py)) — separates **weight / gradient / optimizer /
activation** memory. Weight/grad/optimizer are exact (shapes + `opt.state`; both RiemannianAdam
and Adam keep `exp_avg`+`exp_avg_sq`, so optimizer ≈ 2× weights). Activation is empirical
(`current_allocated_memory()` delta around the forward — MPS has no peak counter) cross-checked
against an analytic per-layer model `∝ B`. Surfaced in `headline_mnist.py` as CSV columns
(`mem_*`), a printed table, and a 4th stacked-bar plot panel. The scaling signal: at matched
params, **wss carries the largest activation footprint** (`J·B·(2r+m)` intermediates) while its
weight/grad/optimizer match `single_rank_Jr` — a parameter-vs-activation tradeoff to weigh when
scaling.

**Fashion-MNIST** — `--dataset fmnist` (the harder task). Outputs are dataset-tagged
(`summary_fmnist_*`, `report_fmnist_*`) so they don't collide with MNIST.

**Subspace interpretability** ([experiments/subspace_interp.py](experiments/subspace_interp.py))
— trains a low-rank wss MLP and visualizes per-component gate weights as an aggregate heatmap
(rows = `(layer, subspace)`, columns = class, color = mean gate weight − uniform; a per-class
accuracy strip on top). Gate weights are extracted by a **read-only forward hook** that recomputes
`compute_gate` on the captured layer input (no core edit; asserted: hooked logits == clean logits).

```
python src/complex/experiments/subspace_interp.py --J 10 --r 2 --epochs 8   # one-hot hypothesis
python src/complex/experiments/subspace_interp.py --J 7  --r 2 --epochs 8   # 7-segment hypothesis
```

*Finding (MNIST, r=2, softmax gate):* the gate does **not** specialize per class. The selectivity
metric — effective #subspaces per class, `exp(entropy)` of the per-class gate distribution — comes
out at **10.00/10 (J=10)** and **7.00/7 (J=7)**, i.e. essentially uniform. The deviation heatmap
shows only faint structure (a few mildly "preferred" generalist subspaces, no diagonal/combinatorial
code), despite ~93% accuracy. So neither the one-hot (J=10) nor 7-segment (J=7) hypothesis holds
under the energy-softmax gate at low rank — the accuracy comes from the frames/spectrum, not gate
selection. (The probe takes `--gate_phi` to retry with sharper gates as follow-up.)

## Phase 3 — WSS ViT on CIFAR-10

A tiny pre-norm Vision Transformer ([vit.py](vit.py)) whose **attention Q/K/V/O and MLP fc1/fc2
are WSS layers** (patch-embed Conv2d and the classification head stay dense). One `layer_type`
(dense / single_rank_Jr / wss) drives every factorized projection — the same matched three-way
comparison as the MNIST MLP — and an independent **`attn_type`** selector swaps the attention
implementation:

- `wss_separate` — separate WSS Q/K/V/O (each its own Stiefel frames + gate + spectrum). Built now.
- `dense` — conventional MHA (nn.Linear Q/K/V/O), e.g. to pair a WSS MLP with dense attention.
- `wss_fused`, `wss_folded` — **reserved seams** (fused-QKV; the gate-folded "idea 1" attention),
  left as commented stubs in [superposition.py](superposition.py) for future experiments.

Default config (`dim=128, depth=6, heads=4, mlp_ratio=2, J=4, r=16`) — every model **<1M params**:

| model | attn_type | params |
|---|---|---|
| dense ViT (baseline) | dense | 811,146 |
| single_rank_Jr ViT | wss_separate | 715,146 |
| **wss ViT** | wss_separate | 715,146 (matched to single_rank) |

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/complex/experiments/headline_vit.py --epochs 20
python src/complex/experiments/headline_vit.py --quick          # fast 2-epoch sanity
```

**Faithfulness note:** this build keeps the WSS contract's init `σ₀=sqrt(2Jm/r)` (`init_scale=1.0`)
and per-step diversity — it is **not** tuned for M1 speed. The `(...,n)` generalization of
`SuperpositionLinear.forward` (flatten leading dims) is a *mathematically-identical* per-token
application, labeled as such in the code; the gate therefore fires **per patch token**. Attention
uses manual `softmax(QKᵀ/√d_h)V` (the explicit definition, not fused SDPA). `init_scale<1` is an
explicitly non-faithful stability probe, not a default.

**Status:** all 52 tests green (incl. 25 ViT gates). Pipeline verified end-to-end on real CIFAR-10
via a 2-epoch smoke (`dim64/depth2`): dense/single_rank_Jr/wss all train, orthonormality holds
(~2e-6), ENC stays diverse (~3.3/4), and the memory panel (log-scale grouped bars) shows activation
(the J token-stream intermediates) dominating weight/grad/optimizer by ~100-300×. A full headline
accuracy comparison at the default config is left to a longer run (the toronto CIFAR mirror throttles
the first-time download to ~50 KB/s; the tarball is cached after that).

## MPS notes (M1)

- Forward/backward (einsum/bmm) run on MPS. The Stiefel **retraction** (`solve`/`qr`) and the
  diversity `eigvalsh` are unimplemented on MPS and fall back to CPU (set the env var). The
  factors are tiny so this is correct, just slower per step than dense.
- The MNIST MLP is small enough that `--device cpu` is also fine.

### M1 speedups (all faithful to the math — commented in code)

The retraction and `eigvalsh` CPU-fallbacks dominate WSS step time on M1. Two **math-preserving**
pickups (profiled on the full ViT, `dim128/depth6`, B128):

1. **Batched diversity** (`diversity.summed_diversity`, on by default): the training-loss path
   stacks all equally-sized `Jr×Jr` Grams and does **one** batched `eigvalsh` after **one**
   MPS→CPU sync, instead of one per frame (~72 for a depth-6 ViT). Bit-identical result (the
   `1/J` Gram scaling cancels in the unit-trace normalization). **MPS 0.80 → 1.03 it/s.**
2. **Euclidean (QR) Stiefel retraction** (`--euclidean` / `ViTConfig.stiefel_canonical=False`,
   opt-in): the canonical (Cayley/`solve`) retraction is the single biggest cost; the QR
   retraction is **~2.7× faster on MPS** and keeps `UᵀU=I` *more* tightly (it's a QR). Both are
   valid Stiefel retractions — they differ only in the manifold metric/trajectory — so this is
   faithful to the WSS *idea*. Default stays canonical (the agent_guide choice). **+24% on MPS.**

Combined (`--euclidean`, batched diversity on): **MPS 0.80 → ~1.24 it/s (~1.55×)**, and with these
on the optimized MPS path edges out pure CPU. None of this changes the WSS mathematics; all
substitutions are labeled in code so future versions stay faithful.

## Deferred (clean seams left, not implemented)

Fashion-MNIST sweep, full λ_div/J/gate ablation matrix, interpretability probe,
`SuperpositionLinear.maybe_refactor()` global re-factorization, and the LDAdam second-moment
transport correction. See `agent_guide.md` §6.
