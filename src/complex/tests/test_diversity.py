"""Gate for diversity.py: entropy/ENC bounds and finite gradients (agent_guide §3.5).

Orthogonal subspaces -> S = log(Jr), ENC = J. Identical subspaces -> S = log r, ENC = 1.
The von Neumann entropy gradient is finite under the eps floor even at the near-degenerate
init spectrum (the §2.5 numerical-footgun check).
"""

import math

import pytest
import torch

from complex.diversity import von_neumann
from complex.manifold import haar_init


def _orthogonal_frames(n, r, J):
    """J mutually-orthogonal r-frames: split one (n, Jr) orthonormal basis into J blocks."""
    Q = haar_init(n, J * r)                      # (n, Jr) orthonormal columns
    return Q.reshape(n, J, r).permute(1, 0, 2).contiguous()  # (J, n, r)


def _identical_frames(n, r, J):
    U0 = haar_init(n, r)                          # (n, r)
    return U0.unsqueeze(0).expand(J, n, r).contiguous()


def test_orthogonal_subspaces_maximize_entropy():
    n, r, J = 64, 4, 4
    U = _orthogonal_frames(n, r, J)
    S, ENC = von_neumann(U, J, r)
    assert S.item() == pytest.approx(math.log(J * r), abs=1e-4)
    assert ENC.item() == pytest.approx(J, abs=1e-3)


def test_identical_subspaces_minimize_entropy():
    n, r, J = 64, 4, 4
    U = _identical_frames(n, r, J)
    S, ENC = von_neumann(U, J, r)
    assert S.item() == pytest.approx(math.log(r), abs=1e-4)
    assert ENC.item() == pytest.approx(1.0, abs=1e-3)


def test_entropy_within_bounds_for_random_frames():
    torch.manual_seed(0)
    n, r, J = 64, 4, 4
    U = haar_init(n, r, J)
    S, ENC = von_neumann(U, J, r)
    assert math.log(r) - 1e-3 <= S.item() <= math.log(J * r) + 1e-3
    assert 1.0 - 1e-3 <= ENC.item() <= J + 1e-3


def test_entropy_gradient_finite_at_near_degenerate_init():
    n, r, J = 64, 4, 4
    U = haar_init(n, r, J).clone().requires_grad_(True)   # near-orthogonal -> near-degenerate spectrum
    S, _ = von_neumann(U, J, r)
    S.backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all(), "diversity gradient went non-finite at degeneracy"
