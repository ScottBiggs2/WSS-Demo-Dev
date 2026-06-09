"""Phase-1 exit smoke test (agent_guide §3.6, §Phase-1 exit criteria).

One short training run on a random target reduces loss, for all three layer types, and the
Stiefel frames stay orthonormal post-step on every available device.
"""

import pytest
import torch

from complex.config import GateConfig, ModelConfig, TrainConfig
from complex.models import MLP
from complex.train import _orthonormality, smoke_train_step

from conftest import available_devices


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("layer_type", ["dense", "single_rank_Jr", "wss"])
def test_smoke_train_reduces_loss(device, layer_type):
    torch.manual_seed(0)
    cfg = ModelConfig(layer_type=layer_type, dims=[64, 48, 10], J=4, r=8,
                      gate=GateConfig(phi="softmax"), lambda_div=1e-2)
    model = MLP(cfg)
    X = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    tcfg = TrainConfig(lr_riemann=5e-2, lr_euclid=5e-2, lambda_div=cfg.lambda_div, device=device)
    losses = smoke_train_step(model, X, y, tcfg, n_steps=30, device=torch.device(device))
    assert losses[-1] < losses[0], f"{layer_type}/{device}: loss did not drop ({losses[0]:.3f}->{losses[-1]:.3f})"
    # orthonormality preserved post-training for the factorized types
    if layer_type != "dense":
        err = _orthonormality(model)
        assert err < 1e-4, f"{layer_type}/{device}: orthonormality {err:.2e} after smoke train"
