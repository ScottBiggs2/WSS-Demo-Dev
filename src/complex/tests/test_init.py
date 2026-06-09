"""Gate for the corrected initialization (agent_guide §3.4, §0.4).

With f == 1 and c = 1/J, the materialized weight has entry variance ~= 2/n (He fan-in).
This guards against the earlier wrong sigma0 = J/r value that started training near zero.
"""

import torch

from complex.config import GateConfig, LayerConfig
from complex.superposition import SuperpositionLinear


def test_materialized_weight_variance_is_two_over_n():
    n = m = 512
    r, J = 32, 4
    cfg = LayerConfig(in_dim=n, out_dim=m, J=J, r=r,
                      gate=GateConfig(phi="linear", disabled=True))  # f==1, c=1/J
    vs = []
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        layer = SuperpositionLinear(cfg, generator=gen)
        W = layer.materialize_weight(summed=True)         # (n, m)
        vs.append(W.var().item())
    mean_var = sum(vs) / len(vs)
    target = 2.0 / n
    assert mean_var == __import__("pytest").approx(target, rel=0.05), \
        f"init variance {mean_var:.6f} vs target {target:.6f} (2/n)"
