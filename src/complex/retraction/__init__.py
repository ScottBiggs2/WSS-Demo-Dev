"""Pluggable Stiefel retraction methods for GPU throughput (perf branch, see README.md here).

One factory, ``make_stiefel_manifold``, returns the geoopt manifold a ``ManifoldParameter``
should carry, selected by a string:

  * "canonical"     -> geoopt.Stiefel(canonical=True)  (Cayley/solve; agent_guide default)
  * "qr"            -> geoopt.Stiefel(canonical=False)  (Euclidean QR retraction)
  * "newton_schulz" -> NewtonSchulzStiefel              (matmul-only; faithful, GPU-ideal)
  * "none"          -> NoRetractionStiefel              (NON-FAITHFUL control: no projection)
  * "auto"          -> "canonical" if stiefel_canonical else "qr"  (exact legacy behavior)

``retract_every > 1`` wraps the {qr, newton_schulz} orthonormalizer in a LazyRetractionStiefel
that only re-projects every K steps (NON-FAITHFUL; see lazy.py).

Defaults (``retraction_method="auto"``, ``retract_every=1``) reproduce the pre-perf behavior
byte-for-byte, so every existing call site and test is unchanged.
"""

from __future__ import annotations

import geoopt

from .lazy import LazyRetractionStiefel
from .newton_schulz import NewtonSchulzStiefel, NoRetractionStiefel, ns_orthonormalize

__all__ = [
    "make_stiefel_manifold",
    "NewtonSchulzStiefel",
    "NoRetractionStiefel",
    "LazyRetractionStiefel",
    "ns_orthonormalize",
    "RETRACTION_METHODS",
]

RETRACTION_METHODS = ("auto", "canonical", "qr", "newton_schulz", "none")


def resolve_method(retraction_method: str, stiefel_canonical: bool) -> str:
    """Map 'auto' to the legacy boolean's meaning; validate the rest."""
    if retraction_method == "auto":
        return "canonical" if stiefel_canonical else "qr"
    if retraction_method not in RETRACTION_METHODS:
        raise ValueError(
            f"unknown retraction_method {retraction_method!r}, expected one of {RETRACTION_METHODS}"
        )
    return retraction_method


def make_stiefel_manifold(
    retraction_method: str = "auto",
    *,
    stiefel_canonical: bool = True,
    retract_every: int = 1,
    retr_steps: int = 5,
    projx_steps: int = 8,
    power_iters: int = 2,
) -> geoopt.manifolds.Manifold:
    """Build the geoopt Stiefel manifold for the requested retraction method."""
    method = resolve_method(retraction_method, stiefel_canonical)

    if retract_every and int(retract_every) > 1:
        if method not in ("qr", "newton_schulz"):
            raise ValueError(
                "retract_every > 1 requires retraction_method in {'qr','newton_schulz'} "
                f"(got resolved method {method!r}); Cayley/canonical has no lazy form."
            )
        return LazyRetractionStiefel(
            base_method=method,
            retract_every=int(retract_every),
            retr_steps=retr_steps,
            power_iters=power_iters,
        )

    if method == "canonical":
        return geoopt.Stiefel(canonical=True)
    if method == "qr":
        return geoopt.Stiefel(canonical=False)
    if method == "newton_schulz":
        return NewtonSchulzStiefel(retr_steps=retr_steps, projx_steps=projx_steps, power_iters=power_iters)
    if method == "none":
        return NoRetractionStiefel()
    raise ValueError(f"unhandled retraction method {method!r}")
