"""Gate for manifold.py + the manifold-step invariant (agent_guide §3.1).

After many RiemannianAdam steps the Stiefel frames stay orthonormal to float tolerance,
and they remain orthonormal after a step taken on MPS (with CPU fallback for the retraction).
This is the property src/simple/ lacked: the retraction is load-bearing.
"""

import geoopt
import pytest
import torch

from complex.config import GateConfig, LayerConfig
from complex.manifold import make_stiefel_param, orthonormality_error
from complex.superposition import SuperpositionLinear

from conftest import available_devices


def test_stiefel_stays_orthonormal_under_riemannian_adam():
    torch.manual_seed(0)
    n, r, J = 16, 4, 3
    U = make_stiefel_param(n, r, J)
    V = make_stiefel_param(n, r, J)
    target_u = torch.randn(J, n, r)
    target_v = torch.randn(J, n, r)
    opt = geoopt.optim.RiemannianAdam([U, V], lr=1e-2, stabilize=50)
    for _ in range(10_000):
        opt.zero_grad()
        loss = ((U - target_u) ** 2).sum() + ((V - target_v) ** 2).sum()
        loss.backward()
        opt.step()
    err = orthonormality_error(U.detach(), V.detach())
    assert err < 1e-4, f"orthonormality drifted to {err:.2e} after 10k steps"


@pytest.mark.parametrize("device", available_devices())
def test_orthonormal_after_one_layer_step(device):
    torch.manual_seed(1)
    cfg = LayerConfig(in_dim=16, out_dim=12, J=3, r=4, gate=GateConfig(phi="softmax"))
    layer = SuperpositionLinear(cfg).to(device)
    opt = geoopt.optim.RiemannianAdam(layer.stiefel_params(), lr=1e-2, stabilize=50)
    X = torch.randn(8, 16, device=device)
    opt.zero_grad()
    out = layer(X)
    out.pow(2).sum().backward()
    opt.step()
    err = orthonormality_error(layer.U.detach(), layer.V.detach())
    assert err < 1e-4, f"orthonormality {err:.2e} after one step on {device}"
