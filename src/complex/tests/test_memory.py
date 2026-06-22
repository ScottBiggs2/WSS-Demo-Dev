"""Gate for memory.py + the gate-extraction hook (Phase 2.5 tooling).

Exact accounting: weights == params x dtype-bytes; gradient == trainable weights; optimizer ==
2x weights (Adam/RiemannianAdam keep exp_avg + exp_avg_sq); analytic activation scales linearly
with batch B. Plus: the interpretability forward-hook captures the correct layer input and is
read-only (hooked logits == clean logits).
"""

import torch

from complex.config import GateConfig, LayerConfig, ModelConfig, TrainConfig
from complex.gate import compute_gate
from complex.memory import (
    activation_bytes_analytic,
    grad_bytes,
    measure_breakdown,
    optimizer_breakdown,
    param_breakdown,
)
from complex.models import MLP
from complex.superposition import SuperpositionLinear


def _wss_mlp():
    cfg = ModelConfig(layer_type="wss", dims=[64, 32, 16, 10], J=4, r=4,
                      gate=GateConfig(phi="softmax"), lambda_div=1e-2)
    return MLP(cfg)


def test_param_breakdown_sums_to_params():
    model = _wss_mlp()
    pb = param_breakdown(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elem = next(model.parameters()).element_size()
    assert abs(pb["total"] - n_params * elem / 1e6) < 1e-9
    parts = sum(v for k, v in pb.items() if k != "total")
    assert abs(parts - pb["total"]) < 1e-9       # categories partition the total
    assert pb["U"] > 0 and pb["V"] > 0 and pb["spectrum"] > 0


def test_grad_equals_weight_and_optimizer_is_two_moments():
    model = _wss_mlp()
    tcfg = TrainConfig(device="cpu", lambda_div=1e-2)
    x = torch.randn(8, 1, 8, 8)                  # 64-dim flattened input
    y = torch.randint(0, 10, (8,))
    bd = measure_breakdown(model, tcfg, (x, y), device=torch.device("cpu"))
    # gradient buffers == trainable weight bytes
    assert abs(bd["mem_grad_mb"] - bd["mem_weight_mb"]) < 1e-9
    # optimizer state == 2x weights (exp_avg + exp_avg_sq), step is negligible
    assert abs(bd["mem_optim_mb"] - 2.0 * bd["mem_weight_mb"]) < 1e-3


def test_optimizer_breakdown_keys():
    model = _wss_mlp()
    tcfg = TrainConfig(device="cpu")
    x = torch.randn(8, 1, 8, 8); y = torch.randint(0, 10, (8,))
    bd = measure_breakdown(model, tcfg, (x, y), device=torch.device("cpu"))
    ob = bd["optim_breakdown_mb"]
    assert "exp_avg" in ob and "exp_avg_sq" in ob
    assert abs(ob["exp_avg"] - ob["exp_avg_sq"]) < 1e-9     # same shape as params


def test_activation_scales_linearly_with_batch():
    model = _wss_mlp()
    a1 = activation_bytes_analytic(model, 16)
    a2 = activation_bytes_analytic(model, 32)
    a4 = activation_bytes_analytic(model, 64)
    assert abs(a2 - 2 * a1) < 1e-6
    assert abs(a4 - 4 * a1) < 1e-6


def test_grad_bytes_zero_before_backward():
    model = _wss_mlp()
    assert grad_bytes(model) == 0.0             # no .grad populated yet


def test_hook_captures_correct_layer_input():
    """A forward hook recomputing compute_gate on input[0] must match a direct recompute."""
    lcfg = LayerConfig(in_dim=20, out_dim=12, J=5, r=3, gate=GateConfig(phi="softmax"))
    layer = SuperpositionLinear(lcfg)
    x = torch.randn(7, 20)
    g_ref, _ = compute_gate(x, layer.U, layer.cfg.gate, layer.gate_alpha, layer.gate_beta)

    captured = {}

    def hook(mod, inp, _out):
        g, _ = compute_gate(inp[0], mod.U, mod.cfg.gate, mod.gate_alpha, mod.gate_beta)
        captured["g"] = g.detach()

    h = layer.register_forward_hook(hook)
    try:
        layer(x)
    finally:
        h.remove()
    assert torch.allclose(captured["g"], g_ref, atol=1e-6)
    assert captured["g"].shape == (5, 7)        # (J, B)


def test_extraction_hooks_are_read_only():
    """Registering the gate-extraction hooks must not change the model output."""
    model = _wss_mlp()
    x = torch.randn(8, 1, 8, 8)
    model.eval()
    with torch.no_grad():
        clean = model(x)

    wss = [(i, l) for i, l in enumerate(model.layers)
           if isinstance(l, SuperpositionLinear) and l.J > 1]
    cache = {}

    def make_hook(idx):
        def hook(mod, inp, _out):
            cache[idx] = compute_gate(inp[0], mod.U, mod.cfg.gate,
                                      mod.gate_alpha, mod.gate_beta)[0].detach()
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in wss]
    try:
        with torch.no_grad():
            hooked = model(x)
    finally:
        for hd in handles:
            hd.remove()
    assert torch.allclose(clean, hooked, atol=1e-6)
    assert set(cache) == {i for i, _ in wss}    # every wss layer fired
