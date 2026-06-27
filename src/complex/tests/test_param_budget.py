"""Gate for the closed-form param counter + solver (experiments/param_budget.py).

The load-bearing check: vit_param_count(...) must EXACTLY equal the live
sum(p.numel() for p in ViT(cfg).parameters()) -- it is the single source of truth for sizing the
100K/1M/10M tiers and deriving the equal-param dense_matched control. Also exercises the solver.
"""

import pytest

from complex.config import ViTConfig
from complex.vit import ViT
from complex.experiments.param_budget import (
    dense_matched_dim, solve_dim_depth, vit_param_count,
)


def _live(cfg: ViTConfig) -> int:
    return sum(p.numel() for p in ViT(cfg).parameters() if p.requires_grad)


# (layer_type, attn_type, dim, depth, heads, mlp_ratio, J, r, num_classes)
_CASES = [
    ("dense", "dense", 128, 6, 4, 2, 4, 16, 10),
    ("single_rank_Jr", "wss_separate", 128, 6, 4, 2, 4, 16, 10),
    ("wss", "wss_separate", 128, 6, 4, 2, 4, 16, 10),
    ("wss", "wss_separate", 72, 4, 4, 2, 4, 6, 100),
    ("dense", "dense", 56, 4, 4, 2, 4, 6, 100),
    ("wss", "wss_separate", 384, 12, 8, 2, 8, 24, 100),
    ("single_rank_Jr", "wss_separate", 384, 12, 8, 2, 8, 24, 100),
]


@pytest.mark.parametrize("lt,at,dim,depth,heads,mr,J,r,nc", _CASES)
def test_closed_form_matches_live(lt, at, dim, depth, heads, mr, J, r, nc):
    cfg = ViTConfig(layer_type=lt, attn_type=at, dim=dim, depth=depth, heads=heads, mlp_ratio=mr,
                    J=J, r=r, num_classes=nc, img_size=32, patch_size=4)
    formula = vit_param_count(dim=dim, depth=depth, heads=heads, mlp_ratio=mr, J=J, r=r,
                              num_classes=nc, patch_size=4, img_size=32, layer_type=lt, attn_type=at)
    assert formula == _live(cfg), f"{lt}: formula {formula} != live {_live(cfg)}"


def test_wss_equals_single_rank_and_depends_only_on_product():
    """wss(J,r) == single_rank_Jr (J=1, rank J*r), and both depend only on the product J*r."""
    common = dict(dim=192, depth=6, heads=6, mlp_ratio=2, num_classes=100, patch_size=4, img_size=32)
    wss = vit_param_count(J=4, r=12, layer_type="wss", attn_type="wss_separate", **common)
    single = vit_param_count(J=4, r=12, layer_type="single_rank_Jr", attn_type="wss_separate", **common)
    assert wss == single
    # different (J, r) split, same product J*r=48 -> identical count (iso-param J<->r axis)
    for J, r in [(2, 24), (8, 6), (16, 3)]:
        assert vit_param_count(J=J, r=r, layer_type="wss", attn_type="wss_separate", **common) == wss


@pytest.mark.parametrize("target,J,r,heads", [(100_000, 4, 6, 4), (1_000_000, 4, 12, 6)])
def test_solver_hits_target_and_round_trips(target, J, r, heads):
    s = solve_dim_depth(target, r=r, J=J, mlp_ratio=2, heads=heads, num_classes=100,
                        patch_size=4, img_size=32)
    assert s["within_tol"], f"solver missed target by {s['rel_err']:.1%}"
    assert s["dim"] % heads == 0 and s["dim"] >= J * r
    # the reported params must equal an actually-built ViT of that (dim, depth)
    cfg = ViTConfig(layer_type="wss", attn_type="wss_separate", dim=s["dim"], depth=s["depth"],
                    heads=heads, mlp_ratio=2, J=J, r=r, num_classes=100, img_size=32, patch_size=4)
    assert _live(cfg) == s["params"]


def test_dense_matched_is_close_and_head_divisible():
    """dense_matched_dim yields a dense ViT whose count is near a factorized target (equal-param)."""
    heads = 4
    target = vit_param_count(dim=72, depth=4, heads=heads, mlp_ratio=2, J=4, r=6, num_classes=100,
                             patch_size=4, img_size=32, layer_type="wss", attn_type="wss_separate")
    dm = dense_matched_dim(target, depth=4, heads=heads, mlp_ratio=2, num_classes=100,
                           patch_size=4, img_size=32)
    assert dm % heads == 0
    dense_count = vit_param_count(dim=dm, depth=4, heads=heads, mlp_ratio=2, J=4, r=6,
                                  num_classes=100, patch_size=4, img_size=32,
                                  layer_type="dense", attn_type="dense")
    assert abs(dense_count - target) / target < 0.05   # equal-param control within a few %
