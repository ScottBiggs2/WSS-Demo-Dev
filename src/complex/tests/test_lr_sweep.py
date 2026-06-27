"""Gate for the Stage-0.5 LR-calibration harness (experiments/lr_sweep.py).

The load-bearing invariants: each method sweeps its PRIMARY optimizer LR group (lr_riemann for the
factorized models whose weights live on the Stiefel manifold, lr_euclid for dense), the OTHER group
is held at the baseline, and the architecture is held fixed across a method's LR points (only the LR
varies). These are what make the resulting "best LR per method" a fair, apples-to-apples calibration.
"""

import pytest

from complex.config import ViTConfig
from complex.vit import ViT
from complex.experiments.lr_sweep import (
    LR_GRID, METHODS, build_lr_configs, _arch_for, _lrs_for_method,
)


def _live(cfg: dict) -> int:
    vcfg = ViTConfig(
        layer_type=cfg["layer_type"],
        attn_type=("dense" if cfg["layer_type"] == "dense" else "wss_separate"),
        num_classes=100, img_size=32, patch_size=4,
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], mlp_ratio=cfg["mlp_ratio"],
        J=cfg["J"], r=cfg["r"], retraction_method=cfg["retraction_method"],
        retract_every=cfg["retract_every"],
    )
    return sum(p.numel() for p in ViT(vcfg).parameters() if p.requires_grad)


# ── the (method x lr) matrix has the right shape and stable grouping ───────────────────────────
def test_default_grid_counts_and_grouping():
    cfgs = build_lr_configs("100k")
    assert len(cfgs) == len(METHODS) * len(LR_GRID)          # 3 methods x 4 LRs = 12
    assert len({c["label"] for c in cfgs}) == len(cfgs)      # labels unique
    # grouped by method, contiguous, in METHODS order
    seen = [c["method"] for c in cfgs]
    assert seen == [m for m in METHODS for _ in LR_GRID]
    # each method's points cover exactly the grid, ascending
    for m in METHODS:
        vals = [c["lr_value"] for c in cfgs if c["method"] == m]
        assert vals == sorted(LR_GRID)


# ── the primary-group split is the whole point: sweep one group, hold the other ────────────────
def test_lrs_for_method_pure_fn():
    # factorized: primary is lr_riemann -> (lr_value, base_euclid)
    assert _lrs_for_method("wss", 1e-2, 1e-3, 5e-4) == (1e-2, 5e-4)
    assert _lrs_for_method("single_rank_Jr", 3e-2, 1e-3, 5e-4) == (3e-2, 5e-4)
    # dense: primary is lr_euclid -> (base_riemann, lr_value)
    assert _lrs_for_method("dense_matched", 1e-2, 7e-4, 1e-3) == (7e-4, 1e-2)
    assert _lrs_for_method("dense", 3e-3, 7e-4, 1e-3) == (7e-4, 3e-3)


def test_factorized_sweep_riemann_hold_euclid():
    cfgs = build_lr_configs("100k", base_riemann=1e-3, base_euclid=2e-3)
    for m in ("wss", "single_rank_Jr"):
        pts = [c for c in cfgs if c["method"] == m]
        assert {c["lr_swept"] for c in pts} == {"lr_riemann"}
        assert [c["lr_riemann"] for c in pts] == sorted(LR_GRID)   # the swept group tracks the grid
        assert {c["lr_euclid"] for c in pts} == {2e-3}             # the held group is the baseline


def test_dense_sweep_euclid_hold_riemann():
    cfgs = build_lr_configs("100k", methods=("dense_matched",), base_riemann=1e-3, base_euclid=2e-3)
    assert {c["lr_swept"] for c in cfgs} == {"lr_euclid"}
    assert [c["lr_euclid"] for c in cfgs] == sorted(LR_GRID)
    assert {c["lr_riemann"] for c in cfgs} == {1e-3}


# ── only the LR varies within a method: the architecture is identical across its points ─────────
def test_arch_fixed_across_a_methods_lr_points():
    cfgs = build_lr_configs("100k")
    for m in METHODS:
        archs = {(c["dim"], c["depth"], c["heads"], c["J"], c["r"], c["mlp_ratio"], c["layer_type"])
                 for c in cfgs if c["method"] == m}
        assert len(archs) == 1, f"{m} arch must be constant across LR points, got {archs}"


def test_dense_matched_is_dense_and_shrunk():
    """dense_matched is a dense ViT shrunk toward the factorized (wss) count -> equal-param control."""
    size = {"dim": 72, "depth": 4, "heads": 4, "J": 4, "r": 6, "mlp_ratio": 2}
    lt_w, arch_w = _arch_for("wss", size)
    lt_d, arch_d = _arch_for("dense_matched", size)
    assert lt_w == "wss" and lt_d == "dense"
    assert arch_d["dim"] < arch_w["dim"]               # dense must shrink to match the factorized count
    wss_cfg = dict(layer_type="wss", retraction_method="newton_schulz", retract_every=1, **arch_w)
    dm_cfg = dict(layer_type="dense", retraction_method="auto", retract_every=1, **arch_d)
    assert abs(_live(dm_cfg) - _live(wss_cfg)) / _live(wss_cfg) < 0.05   # equal-param within a few %


# ── every emitted config builds a valid ViT (Stiefel r<=dim etc.) at a sane size ───────────────
@pytest.mark.parametrize("tier", ["100k", "1m"])
def test_all_configs_build_valid_vits(tier):
    for c in build_lr_configs(tier):
        assert _live(c) > 0
        if c["layer_type"] != "dense":
            assert c["retraction_method"] == "newton_schulz"   # faithful retraction for calibration
            assert c["retract_every"] == 1


# ── overrides ──────────────────────────────────────────────────────────────────────────────────
def test_grid_and_methods_overridable():
    cfgs = build_lr_configs("100k", lr_grid=(1e-3, 1e-2), methods=("wss", "dense_matched"))
    assert len(cfgs) == 2 * 2
    assert {c["lr_value"] for c in cfgs} == {1e-3, 1e-2}
    assert {c["method"] for c in cfgs} == {"wss", "dense_matched"}


def test_unknown_tier_and_method_raise():
    with pytest.raises(ValueError):
        build_lr_configs("999k")
    with pytest.raises(ValueError):
        build_lr_configs("100k", methods=("not_a_method",))
