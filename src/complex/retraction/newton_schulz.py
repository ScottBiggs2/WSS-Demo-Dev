"""Matmul-only Stiefel retraction via the cubic Newton-Schulz polar iteration.

WHY THIS EXISTS (perf, agent_guide §6 deferred perf work; see [[wss-faithfulness-over-perf]]):
geoopt's stock Stiefel retractions are the bottleneck on GPU. ``CanonicalStiefel.retr``
(the agent_guide default) solves an ``n x n`` linear system (``torch.linalg.solve``) every
step; ``EuclideanStiefel.retr`` does a ``qr`` on ``(n, r)``. Both are poorly parallelized on
A100/H200 (and ``qr``/``solve`` have no MPS kernel, so they fall back to CPU on M1).

Newton-Schulz orthonormalization is built from batched matmuls ONLY -- no qr/svd/solve and
no host sync -- so it runs natively on CUDA and MPS, saturates the GEMM units, and batches
trivially over the leading ``J`` dimension of our ``(J, n, r)`` frames.

THE MATH. To re-orthonormalize ``Y`` (re-project onto the Stiefel manifold) we drive its
singular values to 1, i.e. we compute the orthogonal polar factor ``U V^T`` of ``Y = U S V^T``
(this is exactly geoopt's ``Stiefel.projx``). The *cubic* Newton-Schulz iteration

    Y <- 1.5 * Y - 0.5 * Y (Y^T Y)

maps each singular value ``s -> s(3 - s^2)/2``, whose stable fixed point is ``s = 1`` with
``f'(1) = 0`` (QUADRATIC convergence). It converges for ``s in (0, sqrt(3))``, which we
guarantee by spectral-normalizing ``Y`` first (``s_max < 1``).

IMPORTANT -- this is NOT the Muon quintic. The popular quintic coefficients
``(3.4445, -4.7750, 2.0315)`` used to orthogonalize *gradients* in the Muon optimizer
deliberately leave the singular values in a loose band around 1 and DO NOT reach the
``||Y^T Y - I|| < 1e-4`` that the Stiefel orthonormality gates (test_invariants.py) require.
We verified this empirically; the convergent cubic iteration is what maintains the invariant.

NORMALIZATION. Per-frame Frobenius scaling (``Y / ||Y||_F``) divides an already-orthonormal
``(n, r)`` frame by ``sqrt(r)``, pushing its singular values to ``1/sqrt(r)`` and DESTROYING
the warm-start (then ~8 iters are needed). Instead we estimate ``s_max`` with a couple of
power-iteration steps on the small ``r x r`` Gram ``A = Y^T Y`` (matmul-only, deterministic
from a ones vector) and divide by it. Near the manifold ``s_max ~ 1`` so this is ~a no-op and
preserves quadratic warm-start convergence: 5 iters reach ~1e-7 for r in {8,16,32}.
"""

from __future__ import annotations

import torch
from geoopt.manifolds.stiefel import EuclideanStiefel, Stiefel


def ns_orthonormalize(
    X: torch.Tensor,
    steps: int = 5,
    *,
    power_iters: int = 2,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Orthonormalize the columns of each ``(..., n, r)`` frame (``n >= r``) via the cubic
    Newton-Schulz polar iteration. Returns a tensor with ``Y^T Y ~= I_r`` to a tolerance set
    by ``steps``. Pure batched matmul; safe on CUDA/MPS; batched over all leading dims (J).

    ``steps``: 5 is plenty for the per-step (warm-start) retraction; 8 for a cold projx.
    ``power_iters``: power-iteration steps to estimate ``s_max`` for spectral normalization
    (2 near the manifold; bump to ~3 for arbitrary/cold inputs).
    """
    orig_dtype = X.dtype
    Y = X.float()  # the iteration's convergence/coefficients assume fp32
    # Frames are tall (r <= n) by the Stiefel constraint; this guard only matters for the
    # defensive wide-input case and keeps the heavy matmul on the smaller r x r Gram.
    transpose = Y.shape[-2] < Y.shape[-1]
    if transpose:
        Y = Y.transpose(-2, -1)

    # Spectral normalization: estimate s_max via power iteration on A = Y^T Y (r x r),
    # batched over leading dims. Deterministic ones-vector init.
    A = Y.transpose(-2, -1) @ Y                                   # (..., r, r)
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype)  # (..., r, 1)
    for _ in range(power_iters):
        v = A @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp_min(eps)
    lam = (v.transpose(-2, -1) @ (A @ v)).clamp_min(eps)          # ~ s_max^2, (..., 1, 1)
    Y = Y / (lam.sqrt() * 1.01).clamp_min(eps)                   # s_max < 1 < sqrt(3) (1% margin)

    for _ in range(steps):
        A = Y.transpose(-2, -1) @ Y
        Y = 1.5 * Y - 0.5 * (Y @ A)

    if transpose:
        Y = Y.transpose(-2, -1)
    return Y.to(orig_dtype)


class _CustomStiefelMixin:
    """Bypass ``Stiefel.__new__``'s canonical/euclidean dispatch.

    ``Stiefel.__new__(cls, canonical=True)`` hard-routes to ``CanonicalStiefel`` /
    ``EuclideanStiefel`` and only accepts a ``canonical`` kwarg. Our subclasses take their own
    constructor args, so we skip straight to ``Manifold.__new__`` (the class above ``Stiefel``
    in the MRO).
    """

    def __new__(cls, *args, **kwargs):
        return super(Stiefel, cls).__new__(cls)


class NewtonSchulzStiefel(_CustomStiefelMixin, EuclideanStiefel):
    """Stiefel manifold whose retraction/projection are matmul-only Newton-Schulz.

    A FAITHFUL alternative retraction (it approximates the polar retraction ``polar(x + u)``,
    a textbook first-order Stiefel retraction -- the same family as geoopt's QR ``retr`` and
    SVD ``projx``), but a different trajectory than the canonical Cayley default, so it is OFF
    by default. Subclasses ``EuclideanStiefel`` and overrides ONLY ``retr`` and ``projx``; the
    Euclidean ``proju``/``egrad2rgrad``/``transp``/``inner`` and the base ``retr_transp``
    (which composes our ``retr`` + inherited ``transp``) are reused so ``RiemannianAdam`` works
    unchanged.
    """

    name = "Stiefel(newton_schulz)"
    reversible = False

    def __init__(self, retr_steps: int = 5, projx_steps: int = 8, power_iters: int = 2):
        super().__init__()
        self.retr_steps = retr_steps
        self.projx_steps = projx_steps
        self.power_iters = power_iters

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # warm start: x is orthonormal, u is a small tangent step -> few iters suffice
        return ns_orthonormalize(x + u, self.retr_steps, power_iters=self.power_iters)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        # cold-ish projection (used by RiemannianAdam's periodic `stabilize`): more iters
        return ns_orthonormalize(x, self.projx_steps, power_iters=max(self.power_iters, 3))


class NoRetractionStiefel(_CustomStiefelMixin, EuclideanStiefel):
    """NON-FAITHFUL control: additive update with NO projection (``retr(x, u) = x + u``).

    A cleaner Remark-8 control than ``TrainConfig.retraction=False`` (plain SGD): it keeps
    RiemannianAdam's adaptive preconditioning and the tangent-grad projection, isolating the
    effect of dropping the retraction alone. Orthonormality WILL drift. Default OFF; excluded
    from the orthonormality gates. NB: pair with ``stabilize=None`` (else geoopt's periodic
    ``projx`` would silently re-orthonormalize and this would no longer be a pure control).
    """

    name = "Stiefel(none)"
    reversible = False

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u
