# Weight Subspace Superposition — Core Prototype Build Spec

**Target:** a working, tested WSS core (the `SuperpositionLinear` layer + Riemannian
training loop) validated on an MNIST / Fashion-MNIST MLP, runnable on an M1 MacBook Pro.

**Audience:** coding agents and SWE staff. Research-level decisions are already made and
encoded below as a *contract*; the job is faithful implementation + the verification gates.

**In scope (this doc):** Phase 1 (numerics core) and Phase 2 (MNIST/FMNIST MLP).
**Deferred (later docs / after research meetings):** ViT, small-LLM scaling, attention
heads-as-components, global re-factorization. Do **not** build these now.

---

## 0. Ground rules — corrections already baked in

These differ from earlier draft notes. Implement the versions here, not anything contradictory.

1. **The content gate reads the LEFT frame `U`, not `V`.** The input is read by `X @ U_j`;
   `V_j` lives in the output space. The gate energy is `‖X U_j‖²_F / ‖X‖²_F`, which also
   reuses the forward-pass quantity `H_j = X U_j`. Any formula gating on `V` is wrong.
2. **No per-component re-canonicalization.** With a Riemannian optimizer (geoopt) the
   retraction keeps `UᵀU = I` and `Σ` diagonal every step, so the SVD form is
   self-maintaining. The per-step SVD-of-Σ "regrounding" is a no-op and is omitted.
3. **No global re-factorization in Phases 1–2.** Not needed at MNIST scale. Leave a clean
   seam for it later (see §6) but do not implement.
4. **Initialization variance:** `σ₀² = 2·J·m / r` to match He fan-in (square layer: `2·J·n/r`).
   The earlier `J/r` value is wrong by a factor of ~`2n` and will start training near zero.
5. **Normalization is XOR:** either the `1/J` prefactor (for non-normalized gates) **or** a
   softmax-over-`j` gate (whose weights already sum to 1) — **never both**, or you double-suppress.

A correct build must, at all times, satisfy these invariants (they are unit-tested):

- `U_jᵀ U_j = I_r` and `V_jᵀ V_j = I_r` to float tolerance after every optimizer step.
- `Σ_j` is positive and diagonal.
- The layer never materializes the dense `n × m` weight in the forward/backward hot path
  (only in tests and diagnostics).
- The analytic data-fit gradients (§2.4) match autograd to ~1e-5 when the gate is detached.

---

## 1. Environment & framework

### 1.1 Decision: PyTorch + geoopt (not JAX/NumPy) for this phase

The "we need to strictly regulate gradients/updates" concern is fully satisfiable in PyTorch:
geoopt provides a `Stiefel` manifold and `RiemannianAdam`/`RiemannianSGD` as drop-in
subclasses of the standard optimizers, which handle the retraction **and** the tangent-space
transport of Adam moments internally. Stop-grad on the gate is `.detach()`. The closed-form
gradients are available as test oracles. On the M1 specifically, PyTorch's MPS backend is the
mature path; JAX-on-Metal is still experimental and not worth debugging underneath a research
prototype. Revisit JAX/MLX only if `vmap`-over-`J` kernel fusion becomes the bottleneck at
scale (a Phase-3+ concern), and only by porting the inner layer, not the stack.

### 1.2 Install (Apple Silicon)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision            # MPS-enabled wheels
pip install geoopt                        # Riemannian manifolds + optimizers
pip install numpy matplotlib pytest tqdm  # tooling, tests, viz
```

Device selection and MPS fallback (some linalg ops, e.g. `eigh`/`qr`, may be unimplemented on
MPS — let them fall back to CPU rather than crash):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

```python
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

**MPS gotcha to design around:** run the small `Jr × Jr` diversity eigendecomposition and any
manifold QR on **CPU** explicitly (`.cpu()` → op → `.to(device)`). These factors are tiny, so
the transfer cost is negligible and it sidesteps backend gaps. Do the bulk matmuls on MPS.

### 1.3 Repo skeleton

```
wss/
  __init__.py
  config.py            # dataclasses for layer/model/train config
  manifold.py          # Stiefel param construction, Haar init, retraction check
  spectrum.py          # log-parametrized positive diagonal Σ
  gate.py              # energy u_j, φ variants, normalization, detach flag
  superposition.py     # SuperpositionLinear (forward; relies on autograd for backward)
  diversity.py         # density-matrix von Neumann entropy penalty + ENC diagnostic
  grads_reference.py   # closed-form Euclidean gradients (TEST ORACLE ONLY)
  models.py            # MLP variants (dense baseline, single-rank-Jr, WSS)
  train.py             # training loop, RiemannianAdam wiring, logging
  data.py              # MNIST / Fashion-MNIST loaders
tests/
  test_invariants.py   # orthonormality, Σ positivity, no-dense-materialization
  test_grads.py        # analytic vs autograd (gate detached)
  test_init.py         # init variance ≈ 2/n
  test_normalization.py# softmax-XOR-prefactor scale invariant
  test_diversity.py    # entropy bounds, ENC ∈ [1, J]
experiments/
  mnist_ablation.py    # the Phase-2 experiment matrix
  interpret_probe.py   # per-subspace specialization viz
```

---

## 2. Mathematical contract

All shapes for a single layer mapping `R^n → R^m`, batch `B`, components `j = 1..J`, rank `r`.

### 2.1 Parametrization & shapes

| object | shape | manifold / constraint |
|---|---|---|
| `U_j` | `n × r` | Stiefel `St(r,n)`: `U_jᵀU_j = I_r` |
| `V_j` | `m × r` | Stiefel `St(r,m)`: `V_jᵀV_j = I_r` |
| `σ_j` (spectrum vector) | `r` | positive; store as `s_j ∈ R^r`, `σ_j = exp(s_j)` |
| `b` (bias) | `m` | Euclidean |

`S_j := diag(σ_j)`. Component `j` represents `W_j = U_j S_j V_jᵀ`. Stack `U = [U_1 … U_J] ∈ R^{n×Jr}`
for diversity.

### 2.2 Gate (default: per-sample)

`H_j = X U_j ∈ R^{B×r}` (computed once in the forward).

Per-sample energy (row `b`): `u_{b,j} = ‖x_b U_j‖² / ‖x_b‖² = ‖H_{b,j,:}‖² / ‖x_b‖²  ∈ [0,1]`.

> **Granularity note for the team:** the draft's `‖X‖_F` definition is *batch-level* (one
> scalar gate per component per batch). Per-sample is strictly more general and is required
> for the interpretability / data-utility story (a component "fires" for specific inputs), so
> it is the default here. Expose `gate_granularity ∈ {"sample","batch"}`; batch mode just uses
> the Frobenius norm over the whole batch.

`φ` variants (config `gate_phi`):

- `"linear"`: `φ(u) = u`                          → non-normalized, use `1/J` prefactor
- `"exp"`: `φ(u) = exp(u)`                         → non-normalized, use `1/J` prefactor
- `"sigmoid"`: `φ(u) = sigmoid(α u + β)`           → non-normalized, use `1/J` prefactor (α,β learnable scalars)
- `"pow"`: `φ(u) = uᵞ`, γ>0                        → non-normalized, use `1/J` prefactor
- `"softmax"`: `g_{b,:} = softmax_j(u_{b,:})`      → normalized over j, **drop** `1/J` prefactor

`gate_detach: bool` — if true, `g` is `.detach()`-ed before entering the forward (removes the
gate gradient path; restores full per-component decoupling). Default `False`; the ablation
flips it.

### 2.3 Forward

Let `c = 1/J` for non-normalized gates, `c = 1` for softmax. With `g_j ∈ R^B` the realized
per-sample gate and `⊙` row-broadcast scaling:

```
Ã = c · Σ_j  (g_j ⊙ (X U_j)) S_j V_jᵀ  +  1 bᵀ
```

Cost `O(J B (n+m) r)`; never form `n × m`. Implementation: batched `einsum`/`bmm` over a
stacked `U ∈ R^{J×n×r}` so the `J` contractions are one kernel. The only cross-`j` sync is the
softmax normalizer (a reduction over `J` scalars per row) — cheap.

### 2.4 Analytic gradients — TEST ORACLE (gate detached, ambient/Euclidean)

These are validated against autograd; they are **not** the training path (RiemannianAdam
consumes `U_j.grad`/`V_j.grad` and does projection+transport+retraction itself). With
`δ = ∂L/∂Ã ∈ R^{B×m}` and `H̃_j = g_j ⊙ H_j`:

```
∂L/∂V_j = c · (δᵀ H̃_j) S_j                        # (m×r)
∂L/∂σ_j = c · diag( H̃_jᵀ δ V_j )                  # (r,)  take diagonal
∂L/∂U_j = c · Xᵀ ( g_j ⊙ (δ V_j S_j) )             # (n×r)
```

These reduce to the draft's (5)–(7) when `g_j` is a constant scalar and `c = 1/J`. The gate
path (active when `gate_detach=False`) is left to autograd — do not hand-code it.

### 2.5 Diversity penalty (autograd through `eigh`)

Left/right density matrices via the small Gram (do **not** form `n×n`):

```
G_L = (1/J) · Uᵀ U      ∈ R^{Jr×Jr}     # U is the stacked n×Jr
λ_L = eigvalsh(G_L)      (on CPU)
p_L = clamp(λ_L, min=ε) ; p_L = p_L / p_L.sum()      # unit trace, ε≈1e-12
S_L = -(p_L * log p_L).sum()
```

Same for `G_R = (1/J) Vᵀ V`. Penalty `D = -(S_L + S_R)`; total loss `L = L_pred + λ_div · D`.
Autograd differentiates through `eigvalsh` (the closed form `∂S/∂U_j = -(2/Jr)(I+log ρ̄)U_j` is
available in `grads_reference.py` as a fallback/oracle).

Bounds (asserted in tests, `Jr ≤ n`): `S` ranges in `[log r, log(Jr)]`; max at mutually
orthogonal subspaces, min at coincident.

> **Numerical risk:** `eigvalsh` backward is ill-conditioned when eigenvalues are degenerate
> (terms `1/(λ_i−λ_j)`). At init subspaces are near-orthogonal so the `Jr` eigenvalues are all
> ≈ `1/(Jr)` — nearly degenerate. Mitigate with the `ε` floor, and if instability appears,
> switch that layer to the closed-form gradient. This is the single most likely numerical
> footgun in the build; surface it in logs.

### 2.6 Effective number of components (diagnostic)

`ENC = exp(S) / r ∈ [1, J]` per side. Track per layer over training; it is the primary
"is `J>1` doing anything" signal and a later adaptive-`J` trigger.

### 2.7 Initialization

`U_j`, `V_j` Haar-uniform on their Stiefels (QR of an `N(0,I)` matrix, or `geoopt.Stiefel().random`).
`s_j = log(σ₀)` with `σ₀ = sqrt(2·J·m / r)`. Bias zero.

---

## 3. Phase 1 — numerics core

Build order and the verifiable gate for each module. Each gate is a `pytest` that an agent can
run without understanding the research.

### 3.1 `manifold.py`
- `make_stiefel_param(n, r) -> geoopt.ManifoldParameter` (Haar init).
- `check_orthonormal(U, atol) -> bool`.
- **Gate (`test_invariants.py`):** after 10k random RiemannianAdam steps on a dummy loss,
  `‖U_jᵀU_j − I‖_∞ < 1e-4` for all `j`, both frames.

### 3.2 `spectrum.py`
- Log-parametrized positive diagonal; `sigma()` returns `exp(s)`.
- **Gate:** `σ > 0` always; standard Adam updates `s`; round-trips `s → σ → s`.

### 3.3 `gate.py`
- `energy(X, U) -> u` (per-sample and batch modes), `phi(u, kind, params)`, `normalize`,
  `detach` handling. Returns `g ∈ R^B` per component (stacked `R^{B×J}`).
- **Gate (`test_normalization.py`):** softmax mode → `g.sum(dim=j) == 1`; with matched
  normalization, output scale at init is invariant to `J` (no double-suppression).

### 3.4 `superposition.py`
- `SuperpositionLinear(n, m, J, r, gate_cfg, ...)`, forward per §2.3, batched einsum.
- `materialize_weight()` for tests/diagnostics only (asserted unused in hot path).
- **Gate (`test_grads.py`):** with `gate_detach=True`, autograd `U_j.grad/V_j.grad/s_j.grad`
  match `grads_reference.py` to `1e-5` on random small inputs (`n=m=16, r=4, J=3, B=8`,
  float64 for the check).
- **Gate (`test_init.py`):** materialized `W̃` at init (set `f≡1`, `c=1/J`) has entry variance
  `≈ 2/n` within 5% over ≥20 seeds (e.g. `n=m=512, r=32, J=4` → var ≈ 2/512).

### 3.5 `diversity.py`
- Gram → CPU `eigvalsh` → entropy (both sides) + `ENC`.
- **Gate (`test_diversity.py`):** orthogonal subspaces → `S ≈ log(Jr)`, `ENC ≈ J`; identical
  subspaces → `S ≈ log r`, `ENC ≈ 1`; gradient finite under the `ε` floor.

### 3.6 `train.py` wiring
- Two parameter groups: Stiefel (`U`,`V`) → `geoopt.optim.RiemannianAdam`; Euclidean
  (`s`, `b`, gate scalars) → `torch.optim.Adam`. (geoopt's RiemannianAdam handles Euclidean
  params too, but keep groups explicit for clarity/control.)
- Loss = task loss + `λ_div · D`.

### Phase 1 exit criteria
All five test files green on CPU **and** MPS; one smoke-train step on a random target reduces
loss; orthonormality holds post-step on MPS (with CPU fallback for QR/eigh).

---

## 4. Phase 2 — MNIST / Fashion-MNIST MLP

### 4.1 Models (`models.py`)
A 2–3 layer MLP (e.g. `784 → 256 → 128 → 10`), with three interchangeable layer types so the
core ablation is a config flip:

- **`dense`** — plain `nn.Linear` (baseline).
- **`single_rank_Jr`** — one rank-`Jr` factorization `U S Vᵀ` (the `J=1`, rank-`Jr` control;
  this is the honest comparison from the draft's §14.4).
- **`wss`** — `SuperpositionLinear`, `J` components of rank `r` with gate + diversity.

Match parameter budgets as closely as practical across the three (report exact counts).

### 4.2 Experiment matrix (`experiments/mnist_ablation.py`)
Sweep, logging final test accuracy, ENC trajectories, and wall-clock/peak memory:

1. **Core question — does gated `J>1` beat a single rank-`Jr`?** `wss(J=4, r=R)` vs
   `single_rank_Jr(rank=4R)` at matched params. Success = WSS ≥ control (or a characterized,
   explained gap).
2. **Gate ablation:** `gate_detach ∈ {off→gate disabled (f≡1), detached, on}` × `gate_phi ∈
   {linear, softmax}`.
3. **Diversity sweep:** `λ_div ∈ {0, 1e-3, 1e-2, 1e-1}`; expect higher `λ_div` → higher ENC,
   bounded principal angles, and (hopefully) no accuracy collapse.
4. **`J` sweep:** `J ∈ {2,4,8,10}` at fixed `r` to watch ENC saturate.

Keep `r` modest (`r ∈ {8,16,32}`) so `Jr` stays well under layer width — that's where any
memory story lives.

### 4.3 Interpretability probe (`experiments/interpret_probe.py`)
With `J=10` on MNIST, log per-class average gate activation `g_{·,j}` and check whether
components specialize toward digits; visualize the top right-singular directions of each
component. (Pretty pictures + a first ENC-over-training plot; this is the seed of the
interpretability contribution, not a deliverable to gate the phase on.)

### 4.4 Metrics & logging
Per run: test acc/loss curves; per-layer `ENC_L`, `ENC_R` over training; min pairwise principal
angle between components; param count; peak MPS memory; steps/sec. Persist as CSV + a small
matplotlib report.

### Phase 2 exit criteria
- `wss` trains stably on both MNIST and Fashion-MNIST and reaches within a small margin (target
  ≤1–2% abs) of the `dense` baseline at comparable params.
- The core ablation (4.2.1) produces a clear, logged verdict on `J>1` vs single rank-`Jr`.
- ENC behaves sensibly: rises off 1 with training, responds monotonically to `λ_div`, and
  collapses toward 1 when diversity is off (confirming the penalty does real work).
- Remark-8 regression check: an SGD-without-retraction variant should visibly degrade
  orthonormality and accuracy — confirming the manifold step is load-bearing.

---

## 5. Agent task tickets (ordered, independently testable)

1. Repo skeleton + config dataclasses + device/MPS-fallback util.
2. `manifold.py` + orthonormality test.
3. `spectrum.py` + positivity test.
4. `gate.py` + normalization test.
5. `grads_reference.py` (Euclidean oracle, §2.4).
6. `superposition.py` forward + gradcheck test (uses #5) + init-variance test.
7. `diversity.py` + entropy/ENC test.
8. `train.py` wiring (param groups, RiemannianAdam) + Phase-1 smoke-train.
9. `models.py` (dense / single-rank-Jr / wss) + `data.py`.
10. `experiments/mnist_ablation.py` (matrix in §4.2) + logging.
11. `experiments/interpret_probe.py` (§4.3).

Tickets 2–7 are pure-function modules with closed-form invariants — ideal for parallel
ownership by SWE staff with no research context. Tickets 8–11 need a little more model literacy.

---

## 6. Known risks & deferred seams

- **`eigvalsh` degeneracy** (§2.5) — most likely numerical issue; `ε` floor + closed-form
  fallback. Watch the diversity gradient norm in logs.
- **MPS op gaps** — keep QR / eigh on CPU; `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Normalization double-count** — enforce the softmax-XOR-prefactor rule in `gate.py`/forward;
  add an assertion, not just docs.
- **Gate granularity** — per-sample default; batch mode behind the same flag for parity with
  the draft definition.
- **Init blow-up** — the corrected `σ₀² = 2Jm/r` is essential; the init-variance test guards it.
- **Deferred seams (do not implement now):** global re-factorization hook (a `maybe_refactor()`
  no-op stub with the `ENC < J/2` trigger commented), attention/heads-as-components, and the
  LDAdam-style projection-aware second-moment correction (the principled fix for transport
  error noted in the review). Leave clearly-marked stubs so later work has an obvious entry
  point.

---

*Phases 1–2 only. ViT, small-LLM scaling, and the attention treatment are deferred pending the
working core and the upcoming research meetings on attention (current heads-as-components idea
is a reparametrization, not a compression — to be revisited).*