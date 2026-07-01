"""Headline ViT-B/16 experiment on CIFAR-10 (per-seed driver; aggregate across seeds separately).

Trains the REAL torchvision vit_b_16 (768d / 12L / 12h, weights=None, CIFAR-10 upscaled to 224px)
under four conditions, for ONE seed per invocation:

  dense           -- full ViT-B/16 (accuracy ceiling)
  single_rank_Jr  -- one rank-(J*r) factorization per replaced projection (honest factorized control)
  dense_matched   -- a SHRUNK dense ViT-B/16 whose param count ~matches the wss model (equal-param control)
  wss             -- the method: J gated rank-r components on Q/K/V/O + MLP fc1/fc2

This EXTENDS headline_torchvision_vit.py (whose TorchvisionViTWithDiagnostics / _image_loaders /
count_params / final_enc it reuses) by adding: the dense_matched equal-param control (built by an
empirical width search, since param_budget.py only counts the custom vit.py, not torchvision), per-epoch
wandb logging, and per-seed CSV/JSON artifacts that aggregate_b16_seeds.py rolls up into mean+/-std.

Run one seed with all four runs; the SLURM array (slurm/b16_cifar10_seeds.sbatch) fans out over seeds.

Usage (from repo root, inside the venv):
    WANDB_MODE=disabled python src/complex/experiments/headline_vit_b16_cifar10.py --quick --seed 1
    python src/complex/experiments/headline_vit_b16_cifar10.py --seed 3 --epochs 50 --amp --allow_tf32
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from complex.config import TrainConfig
from complex.device import get_device
from complex.memory import measure_breakdown
from complex.seed import seed_everything
from complex.train import fit
from complex.experiments.headline_torchvision_vit import (
    _DATASET_NUM_CLASSES, _MEM_KEYS, _image_loaders, _print_init_policy_summary,
    TorchvisionViTWithDiagnostics, count_params, final_enc,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATASET = "cifar10"
MODEL = "vit_b_16"
HEAD_DIM = 64  # ViT-B/16 head_dim (768/12); held fixed for the dense_matched width search


def matched_dense_arch(target_params: int, *, num_classes: int, J: int, r: int,
                       gate_phi: str) -> tuple[int, int]:
    """(num_heads, hidden_dim) of a shrunk dense ViT-B/16 whose param count is closest to target.

    Sweeps num_heads h in 1..12 with hidden_dim = HEAD_DIM*h (head_dim held at ViT-B/16's 64) and
    mlp_dim = 4*hidden_dim, instantiates each dense backbone on CPU, and returns the closest match.
    """
    best = None  # (abs_err, h, hidden, params)
    for h in range(1, 13):
        hidden = HEAD_DIM * h
        model = TorchvisionViTWithDiagnostics(
            MODEL, "dense", num_classes=num_classes, J=J, r=r, gate_phi=gate_phi,
            hidden_dim=hidden, num_heads_override=h, mlp_dim=4 * hidden,
        )
        p = count_params(model)
        del model
        err = abs(p - target_params)
        if best is None or err < best[0]:
            best = (err, h, hidden, p)
    _, h, hidden, p = best
    rel = (p - target_params) / max(target_params, 1)
    print(f"  dense_matched: heads={h} hidden_dim={hidden} mlp_dim={4 * hidden} -> {p:,}p "
          f"(target {target_params:,}p, {rel:+.1%})")
    return h, hidden


def build_run(name: str, args, tcfg_base: dict, *, matched: tuple[int, int] | None = None):
    """Build (model, tcfg) for one of the four run conditions."""
    if name == "dense":
        model = TorchvisionViTWithDiagnostics(
            MODEL, "dense", num_classes=args.num_classes, J=args.J, r=args.r, gate_phi=args.gate_phi)
        lambda_div = 0.0
    elif name == "dense_matched":
        heads, hidden = matched
        model = TorchvisionViTWithDiagnostics(
            MODEL, "dense", num_classes=args.num_classes, J=args.J, r=args.r, gate_phi=args.gate_phi,
            hidden_dim=hidden, num_heads_override=heads, mlp_dim=4 * hidden)
        lambda_div = 0.0
    elif name == "single_rank_Jr":
        model = TorchvisionViTWithDiagnostics(
            MODEL, "single_rank_Jr", num_classes=args.num_classes, J=args.J, r=args.r,
            gate_phi=args.gate_phi, stiefel_canonical=not args.euclidean)
        lambda_div = 0.0
    elif name == "wss":
        model = TorchvisionViTWithDiagnostics(
            MODEL, "wss", num_classes=args.num_classes, J=args.J, r=args.r,
            gate_phi=args.gate_phi, stiefel_canonical=not args.euclidean)
        _print_init_policy_summary(model.init_policy_log, "wss")
        lambda_div = args.lambda_div
    else:
        raise ValueError(f"unknown run {name!r}")
    tcfg = TrainConfig(**{**tcfg_base, "lambda_div": lambda_div, "retraction": True})
    return model, tcfg


def _wandb_log_history(run_name: str, seed: int, hist: dict, args, project: str) -> None:
    if os.environ.get("WANDB_MODE") == "disabled":
        return
    try:
        import wandb
    except ImportError:
        print("  [wandb] not installed; skipping logging")
        return
    tag = f"b16_{DATASET}_e{args.epochs}"
    run = wandb.init(
        project=project, group=tag, job_type=run_name, name=f"{run_name}_s{seed}",
        reinit=True,
        config={
            "run": run_name, "seed": seed, "model": MODEL, "dataset": DATASET,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr_riemann": args.lr_riemann, "lr_euclid": args.lr_euclid,
            "lambda_div": args.lambda_div, "J": args.J, "r": args.r,
            "amp": args.amp, "allow_tf32": args.allow_tf32,
        },
    )
    encs = [next(iter(d.values()))["ENC_L"] if d else float("nan") for d in hist["diagnostics"]]
    for i, epoch in enumerate(hist["epoch"]):
        wandb.log({
            "epoch": epoch,
            "train_loss": hist["train_loss"][i],
            "val_acc": hist["test_acc"][i],
            "val_loss": hist["test_loss"][i],
            "ortho_err": hist["ortho_err"][i],
            "iter_per_sec": hist["steps_per_sec"][i],
            "ENC_L": encs[i],
        }, step=epoch)
    run.summary["final_acc"] = hist["final_acc"]
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    # LR: CLI flag wins; else env var (WSS_LR_*); else 1e-3.
    ap.add_argument("--lr_riemann", type=float,
                    default=float(os.environ.get("WSS_LR_RIEMANN", 1e-3)))
    ap.add_argument("--lr_euclid", type=float,
                    default=float(os.environ.get("WSS_LR_EUCLID", 1e-3)))
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--gate_phi", default="softmax")
    ap.add_argument("--euclidean", action="store_true",
                    help="use QR Stiefel retraction for wss/single_rank instead of canonical")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast forward+loss (CUDA)")
    ap.add_argument("--allow_tf32", action="store_true", help="enable TF32 matmul/cudnn (CUDA)")
    ap.add_argument("--runs", default="dense,single_rank_Jr,dense_matched,wss",
                    help="comma-separated subset of {dense,single_rank_Jr,dense_matched,wss} or 'all'")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--train_subset", type=int, default=None, help="optional train subset for smoke runs")
    ap.add_argument("--test_subset", type=int, default=None, help="optional test subset for smoke runs")
    ap.add_argument("--quick", action="store_true", help="1 epoch on a tiny subset for a fast sanity run")
    ap.add_argument("--wandb_project", default="wss-perf")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 1
        args.train_subset = args.train_subset or 64
        args.test_subset = args.test_subset or 64

    all_runs = ["dense", "single_rank_Jr", "dense_matched", "wss"]
    selected = all_runs if args.runs == "all" else [s.strip() for s in args.runs.split(",")]
    unknown = [s for s in selected if s not in all_runs]
    if unknown:
        raise ValueError(f"unknown runs {unknown}; expected {all_runs}")

    if args.num_classes is None:
        args.num_classes = _DATASET_NUM_CLASSES[DATASET]

    seed_everything(args.seed)
    device = get_device(args.device)

    # Probe image_size + the wss param count (drives the dense_matched width search).
    probe = TorchvisionViTWithDiagnostics(MODEL, "wss", num_classes=args.num_classes,
                                          J=args.J, r=args.r, gate_phi=args.gate_phi)
    image_size = probe.image_size
    wss_params = count_params(probe)
    del probe
    print(f"device={device} | model={MODEL}(raw torchvision, weights=None) | dataset={DATASET} "
          f"num_classes={args.num_classes} image_size={image_size} J={args.J} r={args.r} "
          f"lambda_div={args.lambda_div} epochs={args.epochs} seed={args.seed} "
          f"lr_riemann={args.lr_riemann} lr_euclid={args.lr_euclid}")
    print(f"  wss param count = {wss_params:,}")

    matched = None
    if "dense_matched" in selected:
        matched = matched_dense_arch(wss_params, num_classes=args.num_classes, J=args.J, r=args.r,
                                     gate_phi=args.gate_phi)

    tcfg_base = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr_riemann=args.lr_riemann,
        lr_euclid=args.lr_euclid, dataset=DATASET, device=args.device, stabilize=50,
        seed=args.seed, weight_decay=args.weight_decay, amp=args.amp, allow_tf32=args.allow_tf32,
    )

    results, histories = [], {}
    for run_name in selected:
        seed_everything(args.seed)
        train_loader, test_loader = _image_loaders(
            DATASET, image_size, args.batch_size, root=args.data_root, augment=not args.no_augment,
            seed=args.seed, train_subset=args.train_subset, test_subset=args.test_subset,
        )
        model, tcfg = build_run(run_name, args, tcfg_base, matched=matched)
        n_params = count_params(model)
        print(f"\n=== {run_name} ({n_params:,} params) [seed {args.seed}] ===")
        hist = fit(model, train_loader, test_loader, tcfg, device=device)
        row = {
            "name": run_name, "model": MODEL, "dataset": DATASET, "seed": args.seed,
            "params": n_params, "final_acc": hist["final_acc"],
            "final_ortho_err": hist["ortho_err"][-1], "steps_per_sec": hist["steps_per_sec"][-1],
            "peak_mem_mb": hist.get("peak_mem_mb", float("nan")), **final_enc(hist),
        }
        try:
            seed_everything(args.seed)
            mem_loader, _ = _image_loaders(
                DATASET, image_size, args.batch_size, root=args.data_root, augment=not args.no_augment,
                seed=args.seed, train_subset=args.train_subset, test_subset=args.test_subset,
            )
            mem_batch = next(iter(mem_loader))
            mem = measure_breakdown(model, tcfg, mem_batch, device=device)
            row.update({k: mem[k] for k in _MEM_KEYS})
        except Exception as e:
            print(f"  [memory] breakdown failed: {e}")
            row.update({k: float("nan") for k in _MEM_KEYS})
        results.append(row)
        histories[run_name] = hist
        _wandb_log_history(run_name, args.seed, hist, args, args.wandb_project)

    # ── summary table ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"  {'run':<18} {'params':>12} {'acc':>8} {'ortho_err':>11} {'ENC_L':>7} {'ENC_R':>7} {'it/s':>7}")
    print("-" * 96)
    for r in results:
        print(f"  {r['name']:<18} {r['params']:>12,} {r['final_acc']:>8.3%} "
              f"{r['final_ortho_err']:>11.2e} {r.get('ENC_L', float('nan')):>7.3f} "
              f"{r.get('ENC_R', float('nan')):>7.3f} {r['steps_per_sec']:>7.2f}")
    print("=" * 96)

    # ── persist per-seed CSV + JSON ───────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"b16_{DATASET}_quick_s{args.seed}" if args.quick else f"b16_{DATASET}_e{args.epochs}_s{args.seed}"
    fieldnames = list(dict.fromkeys(k for row in results for k in row))
    with open(OUT_DIR / f"summary_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    with open(OUT_DIR / f"histories_{tag}.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "diagnostics"}
                   for k, v in histories.items()}, f, indent=2)
    print(f"\nWrote outputs to {OUT_DIR}/ (summary_{tag}.csv, histories_{tag}.json)")


if __name__ == "__main__":
    main()
