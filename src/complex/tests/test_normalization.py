"""Gate for gate.py: softmax sums to 1 over j, and the XOR rule means init output scale
is invariant to J (no double-suppression) -- agent_guide §3.3, §0.5.
"""

import pytest
import torch

from complex.config import GateConfig, LayerConfig
from complex.gate import compute_gate
from complex.manifold import make_stiefel_param
from complex.superposition import SuperpositionLinear

from conftest import available_devices


@pytest.mark.parametrize("device", available_devices())
def test_softmax_gate_sums_to_one_over_components(device):
    torch.manual_seed(0)
    n, r, J, B = 16, 4, 5, 7
    U = make_stiefel_param(n, r, J).to(device)
    X = torch.randn(B, n, device=device)
    g, c = compute_gate(X, U, GateConfig(phi="softmax"))
    assert g.shape == (J, B)
    assert c == 1.0
    sums = g.sum(dim=0)                       # sum over components j
    assert torch.allclose(sums, torch.ones(B, device=device), atol=1e-5)


def test_non_normalized_gate_uses_one_over_J_prefactor():
    torch.manual_seed(0)
    U = make_stiefel_param(16, 4, 4)
    X = torch.randn(8, 16)
    for phi in ("linear", "exp", "pow"):
        _, c = compute_gate(X, U, GateConfig(phi=phi))
        assert c == pytest.approx(1.0 / 4)


def test_init_output_scale_invariant_to_J():
    """With matched config (linear phi, c=1/J, He sigma0), materialized-weight entry
    variance is ~independent of J: c exactly cancels the sum-over-J growth (no double-suppress).
    """
    torch.manual_seed(0)
    n = m = 256
    r = 16
    variances = {}
    for J in (1, 2, 4, 8):
        cfg = LayerConfig(in_dim=n, out_dim=m, J=J, r=r,
                          gate=GateConfig(phi="linear", disabled=True))
        # average over a few seeds to denoise
        vs = []
        for seed in range(5):
            gen = torch.Generator().manual_seed(seed)
            layer = SuperpositionLinear(cfg, generator=gen)
            W = layer.materialize_weight(summed=True)
            vs.append(W.var().item())
        variances[J] = sum(vs) / len(vs)
    target = 2.0 / n
    for J, v in variances.items():
        assert v == pytest.approx(target, rel=0.15), f"J={J}: var {v:.5f} vs target {target:.5f}"
    # cross-J spread should be small (scale invariant to J)
    vals = list(variances.values())
    assert max(vals) / min(vals) < 1.3, f"variance not J-invariant: {variances}"
