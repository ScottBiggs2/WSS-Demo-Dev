"""Closed-form ViT parameter count + a (dim, depth) budget solver (scaling flight).

Pure integer arithmetic -- NO torch import, so ``--mode list`` and the SLURM array builders stay
instant. The formulas mirror ``complex/vit.py`` exactly; ``test_param_budget`` asserts
``vit_param_count(...) == sum(p.numel() for p in ViT(cfg).parameters())`` for several configs, so
this module is the single source of truth for sizing the 100K / 1M / 10M tiers.

Why this exists: the old CIFAR-10 ``dense_matched_dim`` literals were hand-tuned and silently
undercount the CIFAR-100 head (``dim * 100`` vs ``dim * 10``). Here the head term is in the formula,
so ``dense_matched_dim`` is derived correctly for any ``num_classes``.

Key structural fact: a factorized projection ``n->m`` costs ``J*r*(n+m) + J*r + b*m`` -- LINEAR in
``dim`` and dependent only on the PRODUCT ``J*r``. So (a) wss and single_rank_Jr are param-identical
by construction, and (b) the J<->r axis (hold ``J*r`` fixed, vary the split) is perfectly iso-param.
A dense projection costs ``n*m + b*m`` -- QUADRATIC in ``dim``. Both are monotone increasing in
``dim``, which is what the solver relies on.
"""

from __future__ import annotations

from math import gcd

FACTORIZED = ("wss", "single_rank_Jr")


def _proj_params(n: int, m: int, layer_type: str, J: int, r: int, use_bias: bool) -> int:
    """Param count of one projection n->m built by superposition.make_proj."""
    b = 1 if use_bias else 0
    if layer_type == "dense":
        return n * m + b * m
    # Factorized (SuperpositionLinear): U(J,n,r) + V(J,m,r) + spectrum(J,r) + bias(m).
    # single_rank_Jr is J=1 at rank J*r -> identical count (the product J*r is what matters).
    if layer_type == "single_rank_Jr":
        J, r = 1, J * r
    return J * r * (n + m) + J * r + b * m


def vit_param_count(dim: int, depth: int, heads: int, mlp_ratio: int, J: int, r: int,
                    num_classes: int, patch_size: int, img_size: int, in_chans: int = 3,
                    layer_type: str = "wss", attn_type: str = "wss_separate",
                    use_bias: bool = True) -> int:
    """Trainable-parameter count of complex.vit.ViT for the given config (exact, closed form).

    Mirrors vit.py: dense patch-embed Conv2d, cls token, pos embed, `depth` pre-norm blocks
    (2 LayerNorms + 4 attention projections + fc1/fc2), a final LayerNorm, and a dense head.
    The attention projection family is dense iff attn_type=="dense" (else it follows layer_type);
    the MLP always follows layer_type. heads does not affect the count (it is a reshape).
    """
    b = 1 if use_bias else 0
    hidden = mlp_ratio * dim
    seq_len = (img_size // patch_size) ** 2 + 1            # + cls token
    attn_lt = "dense" if attn_type == "dense" else layer_type

    # Always-dense / parameter-free-shape parts.
    patch_embed = in_chans * dim * patch_size * patch_size + dim   # Conv2d (bias=True)
    cls_token = dim
    pos_embed = seq_len * dim
    norms = (4 * depth + 2) * dim                          # 2 LayerNorms/block + final, each 2*dim
    head = dim * num_classes + num_classes

    per_block = (
        4 * _proj_params(dim, dim, attn_lt, J, r, use_bias)          # Q, K, V, O
        + _proj_params(dim, hidden, layer_type, J, r, use_bias)       # fc1
        + _proj_params(hidden, dim, layer_type, J, r, use_bias)       # fc2
    )
    return patch_embed + cls_token + pos_embed + norms + head + depth * per_block


def _round_to(x: int, multiple: int) -> int:
    """Nearest positive multiple of `multiple` to x (ties -> up)."""
    if multiple <= 1:
        return max(1, x)
    return max(multiple, int(round(x / multiple)) * multiple)


def _best_dim_for_depth(target: int, depth: int, *, heads: int, mlp_ratio: int, J: int, r: int,
                        num_classes: int, patch_size: int, img_size: int, in_chans: int,
                        layer_type: str, attn_type: str, use_bias: bool,
                        dim_max: int = 8192) -> tuple[int, int]:
    """Head-divisible dim (>= J*r for factorized) whose count is closest to target, at this depth.

    Count is monotone increasing in dim, so we scan multiples of `heads` from the smallest legal dim
    until we pass target, then return whichever of the bracketing candidates is closer. Returns
    (dim, params); (0, 0) if nothing legal under dim_max (shouldn't happen for sane targets).
    """
    step = heads
    floor = step
    if layer_type in FACTORIZED:
        floor = max(floor, _round_to(J * r, step))
        if floor < J * r:
            floor += step                                  # ensure dim >= J*r after snapping
    def count(d):
        return vit_param_count(d, depth, heads, mlp_ratio, J, r, num_classes, patch_size,
                               img_size, in_chans, layer_type, attn_type, use_bias)
    best_dim, best_params, best_err = 0, 0, None
    prev = None
    d = floor
    while d <= dim_max:
        c = count(d)
        err = abs(c - target)
        if best_err is None or err < best_err:
            best_dim, best_params, best_err = d, c, err
        if c >= target:
            # also consider the candidate just below (already seen via prev tracking)
            break
        prev = (d, c)
        d += step
    return best_dim, best_params


def solve_dim_depth(target: int, *, r: int, J: int, mlp_ratio: int, heads: int, num_classes: int,
                    patch_size: int, img_size: int, in_chans: int = 3, layer_type: str = "wss",
                    attn_type: str = "wss_separate", use_bias: bool = True,
                    depth_grid=range(2, 25), tol: float = 0.05) -> dict:
    """Find (dim, depth) whose vit_param_count is closest to `target`.

    dim is snapped to a multiple of `heads` and (for factorized) kept >= J*r. Returns the best
    (dim, depth) over `depth_grid` minimizing |params - target|, as a dict with keys
    dim, depth, params, rel_err, within_tol.
    """
    best = None
    for depth in depth_grid:
        dim, params = _best_dim_for_depth(
            target, depth, heads=heads, mlp_ratio=mlp_ratio, J=J, r=r, num_classes=num_classes,
            patch_size=patch_size, img_size=img_size, in_chans=in_chans, layer_type=layer_type,
            attn_type=attn_type, use_bias=use_bias)
        if dim == 0:
            continue
        rel = abs(params - target) / target
        cand = {"dim": dim, "depth": depth, "params": params, "rel_err": rel,
                "within_tol": rel <= tol}
        if best is None or rel < best["rel_err"]:
            best = cand
    if best is None:
        raise ValueError(f"no (dim, depth) found for target={target} on depth_grid")
    return best


def dense_matched_dim(target: int, *, depth: int, heads: int, mlp_ratio: int, num_classes: int,
                      patch_size: int, img_size: int, in_chans: int = 3, use_bias: bool = True,
                      dim_max: int = 8192) -> int:
    """Head-divisible dim whose DENSE ViT count is closest to `target` (the equal-param control).

    This is the dense baseline shrunk to match a factorized anchor's param count. Because the head
    term `dim * num_classes` is in the formula, this automatically accounts for CIFAR-100's larger
    head (no hand-tuning). Returns the dim minimizing |dense_count - target|.
    """
    dim, _ = _best_dim_for_depth(
        target, depth, heads=heads, mlp_ratio=mlp_ratio, J=1, r=1, num_classes=num_classes,
        patch_size=patch_size, img_size=img_size, in_chans=in_chans, layer_type="dense",
        attn_type="dense", use_bias=use_bias, dim_max=dim_max)
    return dim
