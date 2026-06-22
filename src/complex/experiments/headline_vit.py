"""Phase-3 headline experiment: tiny WSS ViT on CIFAR-10 (mirrors headline_mnist.py).

Runs, on CIFAR-10:
  1. dense              -- conventional ViT (dense attn + dense MLP); accuracy reference
  2. single_rank_Jr     -- every projection a single rank-(J*r) factorization (the honest control)
  3. wss (lambda_div>0)  -- the method: J components of rank r, gated, on Q/K/V/O + MLP fc1/fc2
  4. wss_div0            -- diversity off (ENC-collapse check)
  5. wss_no_retraction   -- Remark-8 control (no retraction -> orthonormality + acc degrade)

Faithfulness note: this run uses the faithful WSS init (sigma0 = sqrt(2*J*m/r), init_scale=1.0)
and per-step diversity. It is NOT tuned for M1 speed. Report instability honestly rather than
silently stabilizing (use --init_scale < 1 explicitly to probe stability).

Logs final test accuracy, exact param counts (<1M), per-layer ENC, orthonormality, steps/sec,
and the weight/activation/gradient/optimizer memory breakdown; writes CSV + a 4-panel report.

Usage (from repo root, inside the venv):
    python src/complex/experiments/headline_vit.py --quick
    python src/complex/experiments/headline_vit.py --epochs 20 --runs dense,single_rank_Jr,wss
"""

from __future__ import annotations

# MUST precede torch import so the MPS->CPU fallback for qr/solve/eigvalsh is active.
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

from complex.config import GateConfig, TrainConfig, ViTConfig
from complex.data import get_loaders
from complex.device import get_device
from complex.memory import measure_breakdown
from complex.train import fit
from complex.vit import ViT

OUT_DIR = Path(__file__).resolve().parent / "outputs"
_MEM_KEYS = ("mem_weight_mb", "mem_activation_mb", "mem_grad_mb", "mem_optim_mb")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_run(name, layer_type, attn_type, lambda_div, retraction, args, tcfg_base):
    cfg = ViTConfig(
        layer_type=layer_type, attn_type=attn_type,
        dim=args.dim, depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio,
        J=args.J, r=args.r, lambda_div=lambda_div, init_scale=args.init_scale,
        gate=GateConfig(phi=args.gate_phi),
    )
    model = ViT(cfg)
    tcfg = TrainConfig(**{**tcfg_base, "lambda_div": lambda_div, "retraction": retraction})
    return name, model, tcfg


def final_enc(history) -> dict:
    """Mean final ENC_L/ENC_R over wss layers (or {})."""
    diag = history["diagnostics"][-1] if history["diagnostics"] else {}
    if not diag:
        return {}
    enc_l = [v["ENC_L"] for v in diag.values()]
    enc_r = [v["ENC_R"] for v in diag.values()]
    ang = [v["min_principal_angle"] for v in diag.values()]
    return {"ENC_L": sum(enc_l) / len(enc_l), "ENC_R": sum(enc_r) / len(enc_r),
            "min_principal_angle": min(ang)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_ratio", type=int, default=2)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--init_scale", type=float, default=1.0,
                    help="faithful=1.0; <1.0 is an explicit NON-faithful stability probe")
    ap.add_argument("--gate_phi", default="softmax")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_augment", action="store_true", help="disable CIFAR train augmentation")
    ap.add_argument("--runs", default="all",
                    help="subset of {dense,single_rank_Jr,wss,wss_div0,wss_no_retraction} or 'all'")
    ap.add_argument("--quick", action="store_true", help="2 epochs, for a fast sanity check")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2

    torch.manual_seed(0)
    device = get_device(args.device)
    dataset = "cifar10"
    print(f"device={device} | dataset={dataset} | dim={args.dim} depth={args.depth} heads={args.heads} "
          f"mlp_ratio={args.mlp_ratio} J={args.J} r={args.r} lambda_div={args.lambda_div} "
          f"epochs={args.epochs} gate={args.gate_phi} init_scale={args.init_scale}")

    train_loader, test_loader = get_loaders(dataset, args.batch_size, augment=not args.no_augment)
    mem_batch = next(iter(train_loader))   # representative batch for the memory breakdown

    tcfg_base = dict(epochs=args.epochs, batch_size=args.batch_size,
                     lr_riemann=args.lr, lr_euclid=args.lr,
                     dataset=dataset, device=args.device, stabilize=50)

    # (layer_type, attn_type, lambda_div, retraction)
    all_runs = {
        "dense":             ("dense",          "dense",        0.0,             True),
        "single_rank_Jr":    ("single_rank_Jr", "wss_separate", 0.0,             True),
        "wss":               ("wss",            "wss_separate", args.lambda_div, True),
        "wss_div0":          ("wss",            "wss_separate", 0.0,             True),
        "wss_no_retraction": ("wss",            "wss_separate", args.lambda_div, False),
    }
    selected = list(all_runs) if args.runs == "all" else [s.strip() for s in args.runs.split(",")]
    runs = [build_run(name, *all_runs[name], args, tcfg_base) for name in selected]

    results, histories = [], {}
    for name, model, tcfg in runs:
        n_params = count_params(model)
        print(f"\n=== {name}  ({n_params:,} params) ===")
        hist = fit(model, train_loader, test_loader, tcfg, device=device)
        row = {
            "name": name, "dataset": dataset, "params": n_params,
            "final_acc": hist["final_acc"], "final_ortho_err": hist["ortho_err"][-1],
            "steps_per_sec": hist["steps_per_sec"][-1],
            "peak_mem_mb": hist.get("peak_mem_mb", float("nan")),
            **final_enc(hist),
        }
        try:
            mem = measure_breakdown(model, tcfg, mem_batch, device=device)
            row.update({k: mem[k] for k in _MEM_KEYS})
        except Exception as e:
            print(f"  [memory] breakdown failed: {e}")
            row.update({k: float("nan") for k in _MEM_KEYS})
        results.append(row)
        histories[name] = hist

    # ── summary table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"  {'run':<20} {'params':>9} {'acc':>8} {'ortho_err':>11} {'ENC_L':>7} {'ENC_R':>7} {'it/s':>7}")
    print("-" * 96)
    for r in results:
        print(f"  {r['name']:<20} {r['params']:>9,} {r['final_acc']:>8.3%} "
              f"{r['final_ortho_err']:>11.2e} {r.get('ENC_L', float('nan')):>7.3f} "
              f"{r.get('ENC_R', float('nan')):>7.3f} {r['steps_per_sec']:>7.1f}")
    print("=" * 96)

    # ── memory table (MB) ──────────────────────────────────────────────────────────
    print("\n  Memory utilization (MB)")
    print(f"  {'run':<20} {'weight':>9} {'activation':>11} {'gradient':>9} {'optimizer':>10} {'total':>9}")
    print("-" * 72)
    for r in results:
        w, a = r.get("mem_weight_mb", float("nan")), r.get("mem_activation_mb", float("nan"))
        g, o = r.get("mem_grad_mb", float("nan")), r.get("mem_optim_mb", float("nan"))
        tot = sum(v for v in (w, a, g, o) if v == v)
        print(f"  {r['name']:<20} {w:>9.3f} {a:>11.3f} {g:>9.3f} {o:>10.3f} {tot:>9.3f}")
    print("  (activation = empirical live-alloc delta on MPS/CUDA; nan on CPU)")

    # ── verdicts (robust to a subset of runs) ────────────────────────────────────
    acc = {r["name"]: r["final_acc"] for r in results}
    ortho = {r["name"]: r["final_ortho_err"] for r in results}
    enc = {r["name"]: r.get("ENC_L") for r in results}
    print("\nVerdicts:")
    if "dense" in acc and "wss" in acc:
        print(f"  wss vs dense gap:             {acc['dense'] - acc['wss']:+.3%}")
    if "wss" in acc and "single_rank_Jr" in acc:
        print(f"  wss vs single_rank_Jr:        {acc['wss'] - acc['single_rank_Jr']:+.3%} "
              f"(>0 => gated J>1 helps at matched params)")
    if enc.get("wss") is not None and enc.get("wss_div0") is not None:
        print(f"  ENC_L with/without diversity: {enc['wss']:.3f} / {enc['wss_div0']:.3f}")
    if "wss" in acc and "wss_no_retraction" in acc:
        print(f"  Remark-8 (retraction on/off): acc {acc['wss']:.3%}/{acc['wss_no_retraction']:.3%}  "
              f"ortho {ortho['wss']:.1e}/{ortho['wss_no_retraction']:.1e}")

    # ── persist CSV + JSON + plot ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_quick" if args.quick else f"{dataset}_e{args.epochs}_d{args.dim}_L{args.depth}_J{args.J}_r{args.r}"
    fieldnames = list(dict.fromkeys(k for row in results for k in row))
    with open(OUT_DIR / f"summary_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    with open(OUT_DIR / f"histories_{tag}.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "diagnostics"}
                   for k, v in histories.items()}, f, indent=2)
    _plot(histories, results, OUT_DIR / f"report_{tag}.png", args, dataset)
    print(f"\nWrote outputs to {OUT_DIR}/ (summary_{tag}.csv, report_{tag}.png)")


def _plot(histories, results, path, args, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    for name, h in histories.items():
        axes[0].plot(h["epoch"], h["test_acc"], marker="o", label=name)
    axes[0].set(title="Test accuracy", xlabel="epoch", ylabel="acc")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for name, h in histories.items():
        encs = [next(iter(d.values()))["ENC_L"] if d else float("nan") for d in h["diagnostics"]]
        if any(e == e for e in encs):
            axes[1].plot(h["epoch"], encs, marker="o", label=name)
    axes[1].axhline(args.J, ls="--", c="gray", alpha=0.6, label=f"J={args.J}")
    axes[1].axhline(1.0, ls=":", c="gray", alpha=0.6, label="J=1")
    axes[1].set(title="ENC_L (first wss layer)", xlabel="epoch", ylabel="effective #components")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    for name, h in histories.items():
        axes[2].plot(h["epoch"], h["ortho_err"], marker="o", label=name)
    axes[2].set(title="Orthonormality error", xlabel="epoch", ylabel="||UᵀU-I||∞", yscale="log")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    # Grouped bars on a LOG y-axis: for a ViT the activation footprint (J token-stream
    # intermediates) dwarfs weight/grad/optimizer by ~100-300x, so a stacked linear bar would
    # hide them. Grouped + log keeps all four categories legible across the dynamic range.
    import numpy as np
    names = [r["name"] for r in results]
    cats = [("mem_weight_mb", "weight"), ("mem_activation_mb", "activation"),
            ("mem_grad_mb", "gradient"), ("mem_optim_mb", "optimizer")]
    xpos = np.arange(len(names)); width = 0.2
    for i, (key, label) in enumerate(cats):
        vals = [max(r.get(key, 0.0) if r.get(key, 0.0) == r.get(key, 0.0) else 0.0, 1e-6) for r in results]
        axes[3].bar(xpos + (i - 1.5) * width, vals, width, label=label)
    axes[3].set(title="Memory breakdown (MB, log)", ylabel="MB", yscale="log")
    axes[3].set_xticks(xpos); axes[3].set_xticklabels(names, rotation=45, fontsize=7)
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3, axis="y")

    fig.suptitle(f"{dataset} ViT | dim={args.dim} depth={args.depth} heads={args.heads} "
                 f"J={args.J} r={args.r} epochs={args.epochs}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
