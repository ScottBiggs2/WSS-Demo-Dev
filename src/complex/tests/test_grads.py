"""Gate for superposition.py: analytic Euclidean gradients match autograd (gate detached),
plus spectrum positivity / round-trip -- agent_guide §3.2, §3.4.

CPU + float64 ONLY: MPS cannot allocate float64.
"""

import torch

from complex.config import GateConfig, LayerConfig
from complex.gate import compute_gate
from complex.grads_reference import euclidean_grads
from complex.spectrum import Spectrum
from complex.superposition import SuperpositionLinear


def test_analytic_grads_match_autograd_gate_detached():
    torch.manual_seed(0)
    n, m, r, J, B = 16, 16, 4, 3, 8
    cfg = LayerConfig(in_dim=n, out_dim=m, J=J, r=r,
                      gate=GateConfig(phi="linear", detach=True))
    layer = SuperpositionLinear(cfg, dtype=torch.float64)  # CPU default
    X = torch.randn(B, n, dtype=torch.float64)
    G = torch.randn(B, m, dtype=torch.float64)             # so dL/dout == G (delta)

    out = layer(X)
    loss = (out * G).sum()
    loss.backward()

    # realized (detached) gate + prefactor, fed to the oracle
    sigma = layer.spectrum.sigma().detach()
    g, c = compute_gate(X, layer.U.detach(), cfg.gate)
    ref = euclidean_grads(X, layer.U.detach(), layer.V.detach(), sigma, g, c, G)

    assert torch.allclose(layer.U.grad, ref["dU"], atol=1e-5), \
        (layer.U.grad - ref["dU"]).abs().max()
    assert torch.allclose(layer.V.grad, ref["dV"], atol=1e-5), \
        (layer.V.grad - ref["dV"]).abs().max()
    assert torch.allclose(layer.spectrum.s.grad, ref["ds"], atol=1e-5), \
        (layer.spectrum.s.grad - ref["ds"]).abs().max()


def test_spectrum_positive_and_round_trip():
    torch.manual_seed(0)
    spec = Spectrum(J=3, r=4, sigma0=2.0)
    assert (spec.sigma() > 0).all()
    # round-trip s -> sigma -> s
    sig = spec.sigma().detach()
    s2 = Spectrum.s_from_sigma(sig)
    assert torch.allclose(s2, spec.s.detach(), atol=1e-6)
    # stays positive after Euclidean Adam steps on a dummy loss
    opt = torch.optim.Adam([spec.s], lr=1e-1)
    for _ in range(50):
        opt.zero_grad()
        (spec.sigma() - 0.1).pow(2).sum().backward()
        opt.step()
    assert (spec.sigma() > 0).all()
