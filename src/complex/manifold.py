"""Stiefel manifold parameters, Haar initialization, and orthonormality checks.

A single ManifoldParameter of shape (J, n, r) holds all J left frames; geoopt batches
its manifold operations over the leading J dimension, so RiemannianAdam needs no Python
loop over components (verified against geoopt 0.5.1).

Haar init: we roll our own sign-corrected QR rather than geoopt's ``Stiefel.random``
(== ``random_naive``), which is QR-of-Gaussian WITHOUT the sign fix and is therefore not
uniform on the Stiefel manifold.
"""

from __future__ import annotations

import geoopt
import torch


def haar_init(
    n: int,
    r: int,
    J: int | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Haar-uniform orthonormal r-frames in R^n.

    Returns (n, r) if J is None, else stacked (J, n, r). Uses QR of an N(0, I) matrix
    with the diagonal-sign correction that makes the distribution genuinely uniform.
    Built on CPU (QR is an MPS gap) then moved to ``device``.
    """
    shape = (n, r) if J is None else (J, n, r)
    a = torch.randn(*shape, dtype=dtype, generator=generator)  # CPU
    q, rmat = torch.linalg.qr(a)                                # reduced QR -> q: (..., n, r)
    # Sign-fix: multiply each column of q by sign(diag(R)) so the result is Haar-uniform.
    sign = rmat.diagonal(dim1=-2, dim2=-1).sign()               # (..., r)
    q = q * sign.unsqueeze(-2)
    if device is not None:
        q = q.to(device)
    return q


def make_stiefel_param(
    n: int,
    r: int,
    J: int | None = None,
    *,
    canonical: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> geoopt.ManifoldParameter:
    """Create a Haar-initialized Stiefel ManifoldParameter of shape (J, n, r) (or (n, r))."""
    data = haar_init(n, r, J, device=device, dtype=dtype, generator=generator)
    manifold = geoopt.Stiefel(canonical=canonical)
    return geoopt.ManifoldParameter(data, manifold=manifold)


def orthonormality_error(*frames: torch.Tensor) -> float:
    """Max over given frames of ||U^T U - I_r||_inf. Supports (n, r) and (J, n, r)."""
    worst = 0.0
    for U in frames:
        r = U.shape[-1]
        gram = U.transpose(-1, -2) @ U                          # (..., r, r)
        eye = torch.eye(r, device=U.device, dtype=U.dtype)
        err = (gram - eye).abs().amax().item()
        worst = max(worst, err)
    return worst


def check_orthonormal(*frames: torch.Tensor, atol: float = 1e-4) -> bool:
    """True iff every frame's columns are orthonormal to within ``atol``."""
    return orthonormality_error(*frames) < atol
