"""Lazy retraction: re-orthonormalize only every K optimizer steps (the retraction-frequency
vs speed tradeoff the team asked to explore).

NON-FAITHFUL knob -- default OFF. Between the K-step re-projections the frame takes cheap
additive (``x + u``) steps and drifts OFF the Stiefel manifold. That drift is not just a
float-tolerance issue; it silently corrupts three model invariants that ASSUME orthonormal
frames, so any K > 1 trades correctness for speed and must be reported, never shipped silently:

  * Spectrum scale (spectrum.py / superposition.py:43): the He-fan-in init ``sigma0=sqrt(2Jm/r)``
    assumes ``||U_j e_k|| = 1``. Off-manifold column-norm drift gets silently re-absorbed by the
    separately-Adam-trained log-spectrum ``s_j``, breaking the identifiability of ``sigma_j``.
  * Gate energy (gate.py): ``||x U_j||^2 / ||x||^2`` is a subspace-energy fraction in [0,1] ONLY
    when ``U_j`` is orthonormal; off-manifold it can exceed 1 and blow up exp/pow gates.
  * Diversity (diversity.py): the von Neumann entropy bounds and ``ENC in [1, J]`` assume
    ``trace((1/J) U^T U) = r``.

This is DISTINCT from geoopt's ``RiemannianAdam(stabilize=K)``: ``stabilize`` does an *extra*
periodic ``projx`` on top of the per-step ``retr``; here we replace the per-step retraction
itself with an every-K-steps one. Sample orthonormality only IMMEDIATELY AFTER a re-projection
(at multiples of ``retract_every``) -- in between it is expected to be off-manifold by design.

Counter correctness: ``RiemannianAdam`` calls ``retr`` exactly once per parameter per step
(via ``retr_transp``), and ``make_stiefel_param`` builds a fresh manifold instance per frame
(U and V get separate instances), so the per-instance ``_step`` counter tracks optimizer steps
unambiguously. Do NOT share one lazy-manifold instance across multiple parameters.
"""

from __future__ import annotations

import torch
from geoopt.manifolds.stiefel import EuclideanStiefel

from .newton_schulz import _CustomStiefelMixin, ns_orthonormalize


def _qr_orthonormalize(y: torch.Tensor) -> torch.Tensor:
    """Sign-fixed reduced QR orthonormalization (the EuclideanStiefel retraction's projection)."""
    q, r = torch.linalg.qr(y)
    sign = r.diagonal(dim1=-2, dim2=-1).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(-2)


class LazyRetractionStiefel(_CustomStiefelMixin, EuclideanStiefel):
    """Re-orthonormalize every ``retract_every`` steps; additive steps in between.

    ``base_method`` in {"newton_schulz", "qr"} selects the orthonormalizer used on the K-th
    step (and in ``projx``). ``retract_every == 1`` reduces to retracting every step (i.e.
    identical to the chosen base retraction).
    """

    name = "Stiefel(lazy)"
    reversible = False

    def __init__(
        self,
        base_method: str = "newton_schulz",
        retract_every: int = 4,
        *,
        retr_steps: int = 5,
        power_iters: int = 2,
    ):
        super().__init__()
        if base_method not in ("newton_schulz", "qr"):
            raise ValueError(f"lazy base_method must be 'newton_schulz' or 'qr', got {base_method!r}")
        self.base_method = base_method
        self.retract_every = max(1, int(retract_every))
        self.retr_steps = retr_steps
        self.power_iters = power_iters
        self._step = 0

    def _real_retr(self, y: torch.Tensor) -> torch.Tensor:
        if self.base_method == "newton_schulz":
            return ns_orthonormalize(y, self.retr_steps, power_iters=self.power_iters)
        return _qr_orthonormalize(y)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        self._step += 1
        y = x + u
        if self._step % self.retract_every == 0:
            return self._real_retr(y)
        return y                                   # off-manifold by design between projections

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return self._real_retr(x)
