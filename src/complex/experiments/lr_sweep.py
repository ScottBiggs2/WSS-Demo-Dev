"""Per-method learning-rate calibration sweep for the WSS flight (Stage 0.5, BEFORE the headline).

WHY THIS EXISTS. The headline gate ("is WSS more than a regularizer?" -- does the gated J-structure
beat an EQUAL-PARAM dense) is only meaningful if each method runs at *its own* best LR. LR is NOT a
fair shared axis between a Euclidean model and one whose weights live on a product of Stiefel
manifolds: the represented map W = U S V^T is bilinear, so the frames U/V must rotate a finite angle
per step to change the function. At the dense-tuned 1e-3 the frames rotate too slowly and WSS
underfits within a fixed epoch budget (it "loses"); at ~1e-2 it converges and wins. Holding LR
constant across the two geometries therefore *bakes in* whichever LR happens to suit one of them --
a confound that could make the study STOP an anchor for a purely optimization reason.

The fix is mechanical and already plumbed: build_optimizers runs TWO optimizers with INDEPENDENT
LRs -- RiemannianAdam(lr=lr_riemann) on the Stiefel frames U/V, and Adam(lr=lr_euclid) on every
Euclidean param (spectrum s, bias, gate scalars, dense W, conv, head). So the geometric LR need
lives specifically in `lr_riemann` (the manifold group), and dense_matched -- which has NO Stiefel
params -- only ever uses `lr_euclid`. This harness sweeps each method's PRIMARY group (the lever that
carries its weight) and holds the other at the shared baseline:

    factorized (wss, single_rank_Jr)  ->  sweep lr_riemann   (lr_euclid held at --lr_euclid)
    dense (dense, dense_matched)      ->  sweep lr_euclid     (lr_riemann irrelevant -- no frames)

It is a SEPARATE, self-contained job/set from scaling_suite.py so it is easy to launch and debug; it
REUSES profile_retraction's make_model_and_tcfg / convergence_one / _write_csv / count_params (so the
CIFAR-100 + bf16 + dropout/WD plumbing is shared) and the param_budget solver for dense_matched. The
faithful retraction (newton_schulz, retract_every=1) is fixed -- LR is calibrated at the faithful
geometry; the retract_every / 'none' controls get their own LR recheck later (their optimal LR shifts
with the inter-retraction drift, so calibrating them here would confound "lazy hurts" with "LR moved").

Read each method's best LR off the table (collect_results.py prints the argmax per method), then run
the Stage-1 headline with each method pinned to its own best LR. Report the LR ratio (wss/dense) and
the robustness envelope as first-class findings -- a ~10x higher optimal lr_riemann is direct evidence
FOR the geometric story.

NATURAL EXPANSIONS (kept out for now to stay simple/modular): a 2D lr_riemann x lr_euclid grid for
wss; per-tier recalibration on promotion (optimal LR need not transfer across width); a per-J/r recheck
along the J<->r axis (changing r reconditions S and shifts the optimal lr_riemann).

Usage (from repo root, in the venv):
    python src/complex/experiments/lr_sweep.py --mode list --tier 100k
    python src/complex/experiments/lr_sweep.py --mode convergence --tier 100k --config 8 --amp --allow_tf32
    python src/complex/experiments/lr_sweep.py --mode convergence --tier 100k --config -1   # all (local)
    python src/complex/experiments/lr_sweep.py --mode list --lr_grid 1e-3,3e-3,1e-2,3e-2 --methods wss,dense_matched
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from complex.device import get_device
from complex.experiments.param_budget import dense_matched_dim, vit_param_count
from complex.experiments.profile_retraction import (
    IMG_SIZE, NUM_CLASSES, OUT_DIR, PATCH_SIZE, SIZES,
    _write_csv, convergence_one, count_params, make_model_and_tcfg,
)

# Faithful retraction used for calibration (Stage-0 winner). LR is calibrated at the faithful
# geometry (retract_every=1); the lazy/none controls get a separate LR recheck (their optimal LR
# shifts with the inter-retraction drift, so calibrating them here would confound the two effects).
RETRACTION = "newton_schulz"

# Coarse log-spaced grid for the PRIMARY lr group of each method. Spans the dense-tuned 1e-3 (where
# WSS underfits) through ~3e-2 (where the frames rotate enough to converge). Overridable: --lr_grid.
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)

# Methods to calibrate (the headline trio). dense (full ceiling) can be added via --methods.
METHODS = ("dense_matched", "single_rank_Jr", "wss")

# Which optimizer LR group is PRIMARY for each method -- the one we sweep. The other is held at the
# shared baseline (--lr_riemann / --lr_euclid). dense* have no Stiefel params, so only lr_euclid bites;
# factorized models carry their weight in the Stiefel frames, so lr_riemann is the lever.
_PRIMARY_GROUP = {
    "dense": "lr_euclid", "dense_matched": "lr_euclid",
    "single_rank_Jr": "lr_riemann", "wss": "lr_riemann",
}
_GROUP_SHORT = {"lr_riemann": "rie", "lr_euclid": "euc"}


def _lrs_for_method(method: str, lr_value: float, base_riemann: float,
                    base_euclid: float) -> tuple[float, float]:
    """(lr_riemann, lr_euclid) for one sweep point: the primary group takes lr_value, other = baseline."""
    if _PRIMARY_GROUP[method] == "lr_riemann":
        return lr_value, base_euclid
    return base_riemann, lr_value


def _arch_for(method: str, size: dict) -> tuple[str, dict]:
    """(layer_type, arch dict) for a method at this tier's anchor. dense_matched shrinks dim to the
    factorized param count (the equal-param control); wss/single_rank_Jr keep the anchor arch."""
    heads, mr, J, r = size["heads"], size["mlp_ratio"], size["J"], size["r"]
    dim, depth = size["dim"], size["depth"]
    common = dict(num_classes=NUM_CLASSES, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    if method in ("dense", "dense_matched"):
        if method == "dense_matched":
            wssp = vit_param_count(dim=dim, depth=depth, heads=heads, mlp_ratio=mr, J=J, r=r,
                                   layer_type="wss", attn_type="wss_separate", **common)
            dim = dense_matched_dim(wssp, depth=depth, heads=heads, mlp_ratio=mr, **common)
        layer_type = "dense"
    else:
        layer_type = method   # "wss" or "single_rank_Jr"
    return layer_type, dict(dim=dim, depth=depth, heads=heads, J=J, r=r, mlp_ratio=mr)


def build_lr_configs(tier: str = "100k", lr_grid=LR_GRID, methods=METHODS,
                     base_riemann: float = 1e-3, base_euclid: float = 1e-3) -> list[dict]:
    """The indexable (method x lr) calibration matrix for one tier, in stable order.

    Grouped by method, then ascending lr. Each config dict is in the schema make_model_and_tcfg /
    convergence_one expect, plus bookkeeping keys (method / lr_swept / lr_value / lr_riemann /
    lr_euclid) carried through to the CSV row so collect_results can pick the best LR per method.
    The actual per-config LRs reach TrainConfig by setting args.lr_riemann/args.lr_euclid in main().
    """
    if tier not in SIZES:
        raise ValueError(f"unknown tier {tier!r}, expected one of {list(SIZES)}")
    size = SIZES[tier]
    configs: list[dict] = []
    for method in methods:
        if method not in _PRIMARY_GROUP:
            raise ValueError(f"unknown method {method!r}, expected one of {list(_PRIMARY_GROUP)}")
        layer_type, arch = _arch_for(method, size)
        group = _PRIMARY_GROUP[method]
        for lr in lr_grid:
            lr_r, lr_e = _lrs_for_method(method, lr, base_riemann, base_euclid)
            label = f"{tier}-lr-{method}-{_GROUP_SHORT[group]}{lr:.0e}"
            configs.append(dict(
                label=label, size_name=tier, layer_type=layer_type,
                retraction_method=("auto" if layer_type == "dense" else RETRACTION),
                retract_every=1, method=method, lr_swept=group, lr_value=lr,
                lr_riemann=lr_r, lr_euclid=lr_e, **arch))
    return configs


def _csv_floats(s: str | None) -> list[float] | None:
    return [float(x) for x in s.split(",") if x.strip()] if s else None


def _csv_strs(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["convergence", "list"], default="convergence")
    ap.add_argument("--tier", choices=list(SIZES), default="100k")
    ap.add_argument("--config", type=int, default=-1, help="config index (-1 = all); see --mode list")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30,
                    help="calibration budget; keep it a meaningful fraction of the headline schedule "
                         "(the LR ranking can shift with budget -- re-confirm finalists at full epochs)")
    ap.add_argument("--lr_grid", default=None, help="comma list overriding the swept grid (e.g. 1e-3,3e-3,1e-2)")
    ap.add_argument("--methods", default=None, help="comma list overriding the calibrated methods")
    # --lr_riemann / --lr_euclid are the BASELINES for the HELD (non-swept) group of each method.
    ap.add_argument("--lr_riemann", type=float, default=1e-3, help="baseline lr_riemann (held for dense* points)")
    ap.add_argument("--lr_euclid", type=float, default=1e-3, help="baseline lr_euclid (held for factorized points)")
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--mlp_dropout", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast forward+loss (CUDA)")
    ap.add_argument("--allow_tf32", action="store_true", help="enable TF32 matmul/cudnn (CUDA)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--out", default=None, help="output CSV (default: outputs/perf/lr_sweep_<tier>_<tag>.csv)")
    args = ap.parse_args()

    grid = _csv_floats(args.lr_grid) or LR_GRID
    methods = _csv_strs(args.methods) or METHODS
    configs = build_lr_configs(args.tier, grid, methods, args.lr_riemann, args.lr_euclid)

    if args.mode == "list":
        print(f"{len(configs)} lr-calibration configs for tier {args.tier} "
              f"(grid={list(grid)}, methods={list(methods)}):")
        for i, c in enumerate(configs):
            model, _ = make_model_and_tcfg(c, args)
            print(f"  [{i:2d}] {c['label']:<32} {count_params(model):>10,}p  "
                  f"{c['lr_swept']}={c['lr_value']:.0e} (lr_riemann={c['lr_riemann']:.0e} "
                  f"lr_euclid={c['lr_euclid']:.0e})  dim={c['dim']:<4} layer={c['layer_type']}")
        return

    device = get_device(args.device)
    selected = configs if args.config < 0 else [configs[args.config]]
    print(f"device={device} | tier={args.tier} | {len(selected)} config(s) | epochs={args.epochs} "
          f"amp={args.amp} tf32={args.allow_tf32}")

    rows = []
    for c in selected:
        # inject this point's LRs into TrainConfig (make_model_and_tcfg reads args.lr_riemann/lr_euclid)
        args.lr_riemann, args.lr_euclid = c["lr_riemann"], c["lr_euclid"]
        row = convergence_one(c, args, device)
        row.update({"method": c["method"], "lr_swept": c["lr_swept"], "lr_value": c["lr_value"],
                    "lr_riemann": c["lr_riemann"], "lr_euclid": c["lr_euclid"]})
        rows.append(row)
        print(f"  {row['label']:<32} {c['method']:<14} {c['lr_swept']}={c['lr_value']:.0e}  "
              f"acc={row['final_acc']:.3%}  ortho={row['final_ortho_err']:.1e}")

    tag = (f"cfg{args.config}" if args.config >= 0 else "all") + f"_{device.type}"
    out = Path(args.out) if args.out else OUT_DIR / f"lr_sweep_{args.tier}_{tag}.csv"
    _write_csv(rows, out)


if __name__ == "__main__":
    main()
