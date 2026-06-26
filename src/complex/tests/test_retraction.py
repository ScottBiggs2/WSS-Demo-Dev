"""Gates for the pluggable retraction subpackage (perf branch, src/complex/retraction/).

Covers the new code WITHOUT touching the existing faithfulness gates (which still exercise the
default "auto" -> canonical path). Verifies:
  * ns_orthonormalize converges to the Stiefel manifold (warm + cold start) for our r range;
  * the factory's "auto" path is byte-identical to the legacy geoopt.Stiefel(canonical=...);
  * newton_schulz / qr hold orthonormality under RiemannianAdam (faithful);
  * the inherited Euclidean tangent ops (egrad2rgrad/transp) are reused unchanged by NS;
  * lazy retract_every=K re-orthonormalizes post-projection; "none" drifts (negative control);
  * a wss ViT built with newton_schulz trains and stays orthonormal.
"""

import geoopt
import pytest
import torch

from complex.config import ViTConfig
from complex.manifold import make_stiefel_param, orthonormality_error
from complex.retraction import (
    LazyRetractionStiefel,
    NewtonSchulzStiefel,
    NoRetractionStiefel,
    make_stiefel_manifold,
    ns_orthonormalize,
)
from complex.train import _orthonormality, smoke_train_step
from complex.config import TrainConfig
from complex.vit import ViT

from conftest import available_devices


def _ortho_err(U):
    g = U.transpose(-1, -2) @ U
    eye = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype)
    return (g - eye).abs().amax().item()


@pytest.mark.parametrize("r", [8, 16, 32])
def test_ns_orthonormalize_warm_start(r):
    """From a near-orthonormal frame (the per-step retraction regime), few iters reach <1e-4."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(4, 64, r))
    y = ns_orthonormalize(q + 1e-2 * torch.randn(4, 64, r), steps=5)
    assert _ortho_err(y) < 1e-4


@pytest.mark.parametrize("r", [8, 16, 32])
def test_ns_orthonormalize_cold_start(r):
    """From an arbitrary matrix (projx regime), the spectral-normalized cubic still converges."""
    torch.manual_seed(0)
    y = ns_orthonormalize(torch.randn(4, 64, r), steps=8, power_iters=3)
    assert _ortho_err(y) < 1e-4


def test_auto_matches_legacy_canonical():
    """make_stiefel_manifold('auto', stiefel_canonical=True/False) reproduces the stock manifolds."""
    assert type(make_stiefel_manifold("auto", stiefel_canonical=True)) is type(geoopt.Stiefel(canonical=True))
    assert type(make_stiefel_manifold("auto", stiefel_canonical=False)) is type(geoopt.Stiefel(canonical=False))
    assert isinstance(make_stiefel_manifold("newton_schulz"), NewtonSchulzStiefel)
    assert isinstance(make_stiefel_manifold("none"), NoRetractionStiefel)
    assert isinstance(make_stiefel_manifold("newton_schulz", retract_every=4), LazyRetractionStiefel)
    with pytest.raises(ValueError):
        make_stiefel_manifold("canonical", retract_every=4)   # Cayley has no lazy form
    with pytest.raises(ValueError):
        make_stiefel_manifold("bogus")


@pytest.mark.parametrize("method", ["newton_schulz", "qr"])
def test_faithful_methods_stay_orthonormal_under_radam(method):
    torch.manual_seed(0)
    n, r, J = 64, 16, 4
    U = make_stiefel_param(n, r, J, retraction_method=method)
    V = make_stiefel_param(n, r, J, retraction_method=method)
    target_u, target_v = torch.randn(J, n, r), torch.randn(J, n, r)
    opt = geoopt.optim.RiemannianAdam([U, V], lr=1e-2, stabilize=50)
    for _ in range(1000):
        opt.zero_grad()
        (((U - target_u) ** 2).sum() + ((V - target_v) ** 2).sum()).backward()
        opt.step()
    err = orthonormality_error(U.detach(), V.detach())
    assert err < 1e-4, f"{method} drifted to {err:.2e}"


def test_ns_reuses_euclidean_tangent_ops():
    """NewtonSchulzStiefel must inherit EuclideanStiefel's proju/egrad2rgrad/transp unchanged so
    RiemannianAdam's grad projection + transport are identical -- only the retraction differs."""
    torch.manual_seed(0)
    ns = NewtonSchulzStiefel()
    eu = geoopt.Stiefel(canonical=False)  # EuclideanStiefel
    x, _ = torch.linalg.qr(torch.randn(3, 16, 4))
    u = torch.randn(3, 16, 4)
    assert torch.allclose(ns.proju(x, u), eu.proju(x, u), atol=1e-6)
    assert torch.allclose(ns.egrad2rgrad(x, u), eu.egrad2rgrad(x, u), atol=1e-6)
    y = ns.retr(x, 1e-3 * u)
    assert torch.allclose(ns.transp(x, y, u), eu.transp(x, y, u), atol=1e-6)


def test_lazy_reorthonormalizes_post_projection():
    """retract_every=K: orthonormality is restored at multiples of K (sampled post-projection)."""
    torch.manual_seed(0)
    n, r, J, K = 64, 16, 4, 4
    U = make_stiefel_param(n, r, J, retraction_method="newton_schulz", retract_every=K)
    target = torch.randn(J, n, r)
    opt = geoopt.optim.RiemannianAdam([U], lr=1e-2, stabilize=None)
    for _ in range(8 * K):                       # land on a projection step
        opt.zero_grad()
        ((U - target) ** 2).sum().backward()
        opt.step()
    assert orthonormality_error(U.detach()) < 1e-4


def test_none_control_drifts():
    """The 'none' control (no projection) must visibly leave the manifold -- the negative control."""
    torch.manual_seed(0)
    n, r, J = 64, 16, 4
    U = make_stiefel_param(n, r, J, retraction_method="none")
    target = torch.randn(J, n, r)
    opt = geoopt.optim.RiemannianAdam([U], lr=1e-2, stabilize=None)
    for _ in range(500):
        opt.zero_grad()
        ((U - target) ** 2).sum().backward()
        opt.step()
    assert orthonormality_error(U.detach()) > 1e-3, "no-retraction control should drift off-manifold"


@pytest.mark.parametrize("device", available_devices())
def test_newton_schulz_vit_trains_and_stays_orthonormal(device):
    cfg = ViTConfig(layer_type="wss", dim=64, depth=2, heads=4, J=4, r=8, mlp_ratio=2,
                    retraction_method="newton_schulz")
    torch.manual_seed(0)
    model = ViT(cfg)
    X = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    losses = smoke_train_step(model, X, y, TrainConfig(lr_riemann=1e-2, lambda_div=1e-3),
                              n_steps=20, device=device)
    assert losses[-1] < losses[0], "newton_schulz wss ViT failed to reduce loss"
    assert _orthonormality(model) < 1e-4
