"""Gate for the WSS ViT (Phase 3): shapes, param budget, faithfulness of the (...,n) generalization,
manifold invariants, diversity/diagnostics wiring, smoke training, and the per-token gate hook.

Tiny config (dim=32, depth=2, heads=4, J=2, r=4, patch=8, img=16 -> seq=5) for speed; one check
uses the real default config to verify the <1M budget and the wss==single_rank_Jr match.
"""

import pytest
import torch

from conftest import available_devices

from complex.config import GateConfig, LayerConfig, ViTConfig
from complex.memory import measure_breakdown, param_breakdown
from complex.superposition import SuperpositionLinear, make_proj
from complex.train import TrainConfig, _orthonormality, build_optimizers, smoke_train_step
from complex.vit import ViT

_LAYER_TYPES = ["dense", "single_rank_Jr", "wss"]


def _tiny(layer_type="wss", attn_type="wss_separate", **kw):
    return ViTConfig(layer_type=layer_type, attn_type=attn_type, img_size=16, patch_size=8,
                     in_chans=3, num_classes=10, dim=32, depth=2, heads=4, mlp_ratio=2,
                     J=2, r=4, gate=GateConfig(phi="softmax"), **kw)


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("device", available_devices())
def test_forward_shape(layer_type, device):
    at = "dense" if layer_type == "dense" else "wss_separate"
    model = ViT(_tiny(layer_type, at)).to(device)
    x = torch.randn(4, 3, 16, 16, device=device)
    out = model(x)
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()


def test_leading_dim_generalization_is_faithful():
    """The (...,n) flatten in SuperpositionLinear.forward must equal looping the 2D path per row."""
    layer = make_proj("wss", 16, 12, J=3, r=4, gate=GateConfig(phi="softmax"))
    X = torch.randn(5, 7, 16)
    out3d = layer(X)
    ref = layer(X.reshape(-1, 16)).reshape(5, 7, 12)
    assert torch.allclose(out3d, ref, atol=1e-6)
    assert out3d.shape == (5, 7, 12)
    # 2D path still works and is unchanged in shape
    assert layer(torch.randn(8, 16)).shape == (8, 12)


def test_param_budget_and_matched_counts():
    """Default config: dense & wss < 1M params; wss exactly matches single_rank_Jr."""
    def n(cfg):
        return sum(p.numel() for p in ViT(cfg).parameters() if p.requires_grad)
    dense = n(ViTConfig(layer_type="dense", attn_type="dense"))
    single = n(ViTConfig(layer_type="single_rank_Jr"))
    wss = n(ViTConfig(layer_type="wss"))
    assert dense < 1_000_000 and wss < 1_000_000
    assert wss == single                      # matched by construction (J comps rank r == 1 comp rank Jr)


@pytest.mark.parametrize("layer_type", ["single_rank_Jr", "wss"])
@pytest.mark.parametrize("device", available_devices())
def test_orthonormality_after_step(layer_type, device):
    at = "wss_separate"
    model = ViT(_tiny(layer_type, at)).to(device)
    tcfg = TrainConfig(device=str(device), lr_riemann=1e-3, lr_euclid=1e-3)
    opts = build_optimizers(model, tcfg)
    x = torch.randn(8, 3, 16, 16, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    for o in opts:
        o.zero_grad()
    (torch.nn.functional.cross_entropy(model(x), y) + tcfg.lambda_div * model.diversity_loss()).backward()
    for o in opts:
        o.step()
    assert _orthonormality(model) < 1e-4


def test_diversity_and_diagnostics():
    wss = ViT(_tiny("wss"))
    div = wss.diversity_loss()
    assert torch.isfinite(div) and div.requires_grad
    diag = wss.diagnostics()
    assert len(diag) == 2 * 6                  # depth=2 blocks * (q,k,v,o + fc1,fc2) = 12 wss layers
    for v in diag.values():
        assert 1.0 - 1e-6 <= v["ENC_L"] <= wss.cfg.J + 1e-6

    for lt in ("dense", "single_rank_Jr"):
        at = "dense" if lt == "dense" else "wss_separate"
        m = ViT(_tiny(lt, at))
        assert float(m.diversity_loss()) == 0.0
        assert m.diagnostics() == {}


def test_wss_mlp_with_dense_attention():
    """attn_type='dense' + layer_type='wss' -> only the 2 MLP layers per block are wss."""
    m = ViT(_tiny("wss", attn_type="dense"))
    assert m(torch.randn(2, 3, 16, 16)).shape == (2, 10)
    assert len(m.diagnostics()) == 2 * 2       # depth=2 * (fc1, fc2)


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("device", available_devices())
def test_smoke_train_decreases_loss(layer_type, device):
    at = "dense" if layer_type == "dense" else "wss_separate"
    model = ViT(_tiny(layer_type, at))
    tcfg = TrainConfig(device=str(device), lr_riemann=1e-2, lr_euclid=1e-2, lambda_div=1e-3)
    x = torch.randn(16, 3, 16, 16)
    y = torch.randint(0, 10, (16,))
    losses = smoke_train_step(model, x, y, tcfg, n_steps=30, device=device)
    assert losses[-1] < losses[0]


def test_config_validate_rejects_bad():
    with pytest.raises(AssertionError):
        ViTConfig(dim=30, heads=4).validate()              # dim % heads != 0
    with pytest.raises(AssertionError):
        ViTConfig(img_size=32, patch_size=5).validate()    # img % patch != 0
    with pytest.raises(AssertionError):
        ViTConfig(dim=32, J=8, r=8).validate()             # J*r=64 > dim=32
    with pytest.raises(AssertionError):
        ViTConfig(attn_type="wss_fused").validate()        # reserved seam, not a valid value yet


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
def test_param_breakdown_sums_to_total(layer_type):
    at = "dense" if layer_type == "dense" else "wss_separate"
    model = ViT(_tiny(layer_type, at))
    pb = param_breakdown(model)
    parts = sum(v for k, v in pb.items() if k != "total")
    assert abs(parts - pb["total"]) < 1e-9
    assert pb["conv_weight"] > 0 and pb["norm"] > 0 and pb["other"] > 0   # patch conv, LN, cls+pos
    # weight/grad/optimizer are exact on CPU even though analytic activation is nan for a ViT
    bd = measure_breakdown(model, TrainConfig(device="cpu"), (torch.randn(8, 3, 16, 16),
                                                              torch.randint(0, 10, (8,))),
                           device=torch.device("cpu"))
    assert bd["mem_weight_mb"] > 0 and bd["mem_grad_mb"] > 0 and bd["mem_optim_mb"] > 0


def test_per_token_gate_hook_is_read_only():
    """The interpretability hook recomputes the gate per token (J, B*N) and must not perturb output."""
    from complex.gate import compute_gate
    model = ViT(_tiny("wss"))
    model.eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        clean = model(x)

    cap = {}

    def hook(mod, inp, _out):
        g, _ = compute_gate(inp[0].reshape(-1, inp[0].shape[-1]), mod.U, mod.cfg.gate,
                            mod.gate_alpha, mod.gate_beta)
        cap["g"] = g.detach()

    h = model.blocks[0].attn.q_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            hooked = model(x)
    finally:
        h.remove()
    assert torch.allclose(clean, hooked, atol=1e-6)
    # (J, B*seq) = (2, 2*5) = (2, 10)
    assert cap["g"].shape == (model.cfg.J, 2 * model.cfg.seq_len)
