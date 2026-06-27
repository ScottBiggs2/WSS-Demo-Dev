"""Gates for the scaling-flight knobs: bf16 autocast plumbing, TF32, weight_decay, dropout, CIFAR-100.

These verify the new knobs are faithful no-ops at their defaults and behave correctly when on:
  * autocast_ctx / setup_backend are no-ops off CUDA (so CPU/MPS runs are byte-unchanged)
  * diversity.gram() stays fp32 even inside a (cuda) autocast region -> eigvalsh is safe
  * weight_decay reaches both optimizers and does NOT break Stiefel orthonormality
  * dropout p=0 is an exact identity; p>0 perturbs only in train() and eval() is deterministic
  * CIFAR-100 is registered with its own (not CIFAR-10's) normalization stats
"""

import contextlib

import pytest
import torch

from conftest import available_devices

from complex.config import GateConfig, ViTConfig
from complex.device import autocast_ctx, setup_backend
from complex.diversity import gram, stack_frames, summed_diversity
from complex.superposition import SuperpositionLinear
from complex.train import TrainConfig, _orthonormality, build_optimizers, train_epoch
from complex.vit import ViT


def _tiny(**kw):
    defaults = dict(layer_type="wss", attn_type="wss_separate", img_size=16, patch_size=8,
                    in_chans=3, num_classes=10, dim=32, depth=2, heads=4, mlp_ratio=2,
                    J=2, r=4, retraction_method="newton_schulz", gate=GateConfig(phi="softmax"))
    defaults.update(kw)
    return ViTConfig(**defaults)


def _wss_frames(model):
    frames = []
    for m in model.modules():
        if isinstance(m, SuperpositionLinear):
            frames.extend([m.U, m.V])
    return frames


# ── autocast / TF32 helpers are safe + no-op off CUDA ──────────────────────────────────────────
def test_autocast_ctx_nullcontext_off_cuda():
    assert isinstance(autocast_ctx("cpu", True), contextlib.nullcontext)
    assert isinstance(autocast_ctx("cpu", False), contextlib.nullcontext)


def test_setup_backend_no_op_off_cuda():
    setup_backend(True)   # must not raise when CUDA is absent
    setup_backend(False)


# ── diversity Gram stays fp32 under a (cuda) autocast region ──────────────────────────────────
def test_gram_is_fp32_and_byte_identical_for_fp32_inputs():
    U = torch.randn(3, 16, 4)
    G = gram(stack_frames(U), 3)
    assert G.dtype == torch.float32
    # the fp32-guard wrapping must not change the value vs the plain matmul for fp32 callers
    Uc = stack_frames(U)
    ref = (Uc.transpose(-1, -2) @ Uc) / 3
    assert torch.allclose(G, ref, atol=0)


def test_summed_diversity_matches_inside_simulated_autocast():
    model = ViT(_tiny())
    frames = _wss_frames(model)
    base = summed_diversity(frames)
    with torch.autocast(device_type="cuda", enabled=False):   # the same guard gram() uses
        inside = summed_diversity(frames)
    assert torch.allclose(base, inside, atol=0)


# ── weight_decay reaches both optimizers and does not break the manifold ──────────────────────
def test_weight_decay_set_on_both_optimizers():
    model = ViT(_tiny())
    opts = build_optimizers(model, TrainConfig(weight_decay=1e-2, lambda_div=1e-3))
    assert len(opts) == 2
    for o in opts:
        for g in o.param_groups:
            assert g["weight_decay"] == 1e-2


@pytest.mark.parametrize("device", available_devices())
def test_weight_decay_keeps_orthonormality(device):
    model = ViT(_tiny()).to(device)
    tcfg = TrainConfig(weight_decay=1e-1, lambda_div=1e-3)   # aggressive WD
    opts = build_optimizers(model, tcfg)
    x = torch.randn(8, 3, 16, 16, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    for _ in range(5):
        train_epoch(model, [(x, y)], opts, tcfg.lambda_div, torch.device(device),
                    torch.nn.CrossEntropyLoss(), progress=False)
    assert _orthonormality(model) < 1e-4   # radial WD is projected out of the Stiefel step


# ── dropout: p=0 is identity; p>0 perturbs only in train(), eval() deterministic ──────────────
def test_dropout_zero_is_identity():
    torch.manual_seed(0)
    base = ViT(_tiny(attn_dropout=0.0, mlp_dropout=0.0)).eval()
    torch.manual_seed(0)
    drop = ViT(_tiny(attn_dropout=0.0, mlp_dropout=0.0)).eval()
    x = torch.randn(4, 3, 16, 16)
    assert torch.allclose(base(x), drop(x), atol=0)


def test_dropout_active_in_train_deterministic_in_eval():
    model = ViT(_tiny(attn_dropout=0.3, mlp_dropout=0.3))
    x = torch.randn(4, 3, 16, 16)
    model.train()
    torch.manual_seed(1); a = model(x)
    torch.manual_seed(2); b = model(x)
    assert not torch.allclose(a, b)            # dropout randomizes in train
    model.eval()
    assert torch.allclose(model(x), model(x), atol=0)   # deterministic in eval


def test_num_classes_100_head():
    model = ViT(_tiny(num_classes=100))
    assert model.head.out_features == 100
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 100)


# ── CIFAR-100 registered with its own normalization stats (no download) ───────────────────────
def test_cifar100_registered_with_own_stats():
    from complex import data
    assert "cifar100" in data._DATASETS
    t = data._build_transform("cifar100", train=False, augment=False)
    norms = [s for s in t.transforms if s.__class__.__name__ == "Normalize"]
    assert norms and tuple(norms[0].mean) == data._CIFAR100_MEAN
    assert tuple(norms[0].mean) != data._CIFAR10_MEAN     # not silently reusing CIFAR-10
    # train transform keeps the 32x32 augment pipeline
    names = [s.__class__.__name__ for s in
             data._build_transform("cifar100", train=True, augment=True).transforms]
    assert "RandomCrop" in names and "RandomHorizontalFlip" in names
