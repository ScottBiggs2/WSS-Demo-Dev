"""Iso-parameter scaling sweeps for the WSS flight: depth<->width and J<->r (Stage 2 onward).

A thin sibling of profile_retraction.py -- it REUSES that module's make_model_and_tcfg / count_params
/ convergence_one / _write_csv (so the CIFAR-100 + bf16 + dropout/WD plumbing is shared) and the
param_budget solver. It does NOT live in profile_retraction because that file's config matrix is the
*retraction-method* sweep whose indices are referenced by PERF_NOTES and the SLURM docs; the scaling
sweeps vary a different axis set and would pollute those indices.

Two indexable sweeps per tier (retraction fixed to the Stage-0 winner, newton_schulz):

  depth<->width (iso-param): fix the tier's param target, vary depth, solve dim to hold params ~const.
      Each point runs {wss, dense_matched} -- the headline pair at a different aspect ratio.
  J<->r (iso-param): fix the anchor dim/depth, hold J*r constant, vary the (J,r) split. Each runs
      {wss}; plus ONE shared dense_matched and the single_rank_Jr (J=1) endpoint. Params are
      identical across the whole axis by construction (a factorized proj depends only on J*r).

Usage (from repo root, in the venv):
    python src/complex/experiments/scaling_suite.py --mode list --tier 100k
    python src/complex/experiments/scaling_suite.py --mode convergence --tier 100k --config 0 --amp --allow_tf32
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
from pathlib import Path

import wandb

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from complex.device import get_device
from complex.experiments.param_budget import dense_matched_dim, solve_dim_depth, vit_param_count
from complex.experiments.profile_retraction import (
    DATASET, IMG_SIZE, NUM_CLASSES, OUT_DIR, PATCH_SIZE, SIZES,
    _write_csv, convergence_one, count_params, make_model_and_tcfg,
)

# Stage-0 winner: the fast faithful retraction used for every scaling/ablation run.
RETRACTION = "newton_schulz"

# Depth grid per tier for the depth<->width axis (sensible aspect ratios around each anchor depth).
DEPTH_GRIDS = {
    "100k": [2, 3, 4, 6, 8],
    "1m":   [3, 4, 6, 8, 12],
    "10m":  [6, 8, 12, 16],
}


def _cfg(label: str, tier: str, layer_type: str, dim: int, depth: int,
         heads: int, J: int, r: int, mlp_ratio: int) -> dict:
    """A config dict in the schema make_model_and_tcfg / convergence_one expect."""
    return dict(label=label, size_name=tier, layer_type=layer_type,
                retraction_method=("auto" if layer_type == "dense" else RETRACTION),
                retract_every=1, dim=dim, depth=depth, heads=heads, J=J, r=r, mlp_ratio=mlp_ratio)


def _jr_splits(Jr: int, candidates=(2, 4, 8, 16)) -> list[tuple[int, int]]:
    """(J, r) pairs holding the product Jr fixed, for J in `candidates` that divide Jr (r = Jr/J)."""
    return [(J, Jr // J) for J in candidates if Jr % J == 0 and Jr // J >= 1]


def build_scaling_configs(tier: str = "100k") -> list[dict]:
    """The indexable (depth<->width ++ J<->r) sweep for one tier, in stable order."""
    if tier not in SIZES:
        raise ValueError(f"unknown tier {tier!r}, expected one of {list(SIZES)}")
    size = SIZES[tier]
    target = vit_param_count(num_classes=NUM_CLASSES, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                             layer_type="wss", attn_type="wss_separate", **size)  # tier's factorized count
    heads, mr, J0, r0 = size["heads"], size["mlp_ratio"], size["J"], size["r"]
    Jr = J0 * r0
    common = dict(num_classes=NUM_CLASSES, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    configs: list[dict] = []

    # ── depth <-> width (iso-param): hold `target`, vary depth, solve dim ───────────────────────
    for depth in DEPTH_GRIDS[tier]:
        s = solve_dim_depth(target, r=r0, J=J0, mlp_ratio=mr, heads=heads, depth_grid=[depth], **common)
        dim = s["dim"]
        wssp = vit_param_count(dim=dim, depth=depth, heads=heads, mlp_ratio=mr, J=J0, r=r0,
                               layer_type="wss", attn_type="wss_separate", **common)
        dm = dense_matched_dim(wssp, depth=depth, heads=heads, mlp_ratio=mr, **common)
        configs.append(_cfg(f"{tier}-dw-d{depth}-dim{dim}-wss", tier, "wss", dim, depth, heads, J0, r0, mr))
        configs.append(_cfg(f"{tier}-dw-d{depth}-dim{dm}-dense_matched", tier, "dense", dm, depth, heads, J0, r0, mr))

    # ── J <-> r (iso-param): fix anchor dim/depth, hold J*r, vary split + endpoints ─────────────
    dim, depth = size["dim"], size["depth"]
    wssp = vit_param_count(dim=dim, depth=depth, heads=heads, mlp_ratio=mr, J=J0, r=r0,
                           layer_type="wss", attn_type="wss_separate", **common)
    dm = dense_matched_dim(wssp, depth=depth, heads=heads, mlp_ratio=mr, **common)
    configs.append(_cfg(f"{tier}-jr-dense_matched", tier, "dense", dm, depth, heads, J0, r0, mr))
    # single_rank_Jr is the J=1 endpoint (make_proj collapses to J=1 at rank J*r -> identical params).
    configs.append(_cfg(f"{tier}-jr-single_rank", tier, "single_rank_Jr", dim, depth, heads, J0, r0, mr))
    for J, r in _jr_splits(Jr):
        if J * r > dim:
            continue
        configs.append(_cfg(f"{tier}-jr-J{J}r{r}-wss", tier, "wss", dim, depth, heads, J, r, mr))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["convergence", "list"], default="convergence")
    ap.add_argument("--tier", choices=list(SIZES), default="100k")
    ap.add_argument("--config", type=int, default=-1, help="config index (-1 = all); see --mode list")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr_riemann", type=float, default=1e-3)
    ap.add_argument("--lr_euclid", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--mlp_dropout", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast forward+loss (CUDA)")
    ap.add_argument("--allow_tf32", action="store_true", help="enable TF32 matmul/cudnn (CUDA)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--out", default=None, help="output CSV (default: outputs/perf/scaling_<tier>_<tag>.csv)")
    args = ap.parse_args()

    configs = build_scaling_configs(args.tier)
    if args.mode == "list":
        print(f"{len(configs)} configs for tier {args.tier}:")
        for i, c in enumerate(configs):
            model, _ = make_model_and_tcfg(c, args)
            print(f"  [{i:2d}] {c['label']:<34} {count_params(model):>10,}p  "
                  f"dim={c['dim']:<4} depth={c['depth']:<2} J={c['J']} r={c['r']} layer={c['layer_type']}")
        return

    device = get_device(args.device)
    selected = configs if args.config < 0 else [configs[args.config]]
    print(f"device={device} | tier={args.tier} | {len(selected)} config(s) | epochs={args.epochs} "
          f"amp={args.amp} tf32={args.allow_tf32}")

    wandb.init(
        project="wss-perf",
        name=f"scaling_{args.tier}_cfg{args.config if args.config >= 0 else 'all'}_{device.type}",
        config={
            "tier": args.tier, "config_idx": args.config, "device": str(device),
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr_riemann": args.lr_riemann, "lr_euclid": args.lr_euclid,
            "amp": args.amp, "allow_tf32": args.allow_tf32,
        }
    )

    rows = [convergence_one(c, args, device) for c in selected]

    tag = (f"cfg{args.config}" if args.config >= 0 else "all") + f"_{device.type}"
    out = Path(args.out) if args.out else OUT_DIR / f"scaling_{args.tier}_{tag}.csv"
    _write_csv(rows, out)
    wandb.finish()


if __name__ == "__main__":
    main()
