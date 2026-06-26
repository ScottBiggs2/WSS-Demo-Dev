"""Phase-2 goalpost experiment (agent_guide §4.2.1, §4.4, Phase-2 exit criteria).

Runs, on MNIST:
  1. dense                       -- accuracy reference
  2. single_rank_Jr (rank=J*r)   -- the honest control
  3. wss (J, r, lambda_div>0)    -- the method
  4. wss (lambda_div=0)          -- ENC-collapse check (diversity off -> ENC -> 1)
  5. wss (retraction=False)      -- Remark-8 control (no retraction -> orthonormality + acc degrade)

Logs final test accuracy, exact param counts, per-layer ENC trajectory, orthonormality, and
steps/sec; writes a CSV summary + a matplotlib report under experiments/outputs/.

Usage (from repo root, inside the venv):
    python src/complex/experiments/headline_mnist.py --epochs 10
    python src/complex/experiments/headline_mnist.py --quick     # fast sanity (2 epochs)
"""

from __future__ import annotations

# MUST precede torch import so the MPS->CPU fallback for qr/solve is active.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import sys
from pathlib import Path

# make `complex` importable (experiments -> complex -> src)
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from complex.config import GateConfig, ModelConfig, TrainConfig
from complex.data import get_loaders
from complex.device import get_device
from complex.memory import measure_breakdown
from complex.models import MLP
from complex.seed import seed_everything
from complex.train import fit

_MEM_KEYS = ("mem_weight_mb", "mem_activation_mb", "mem_grad_mb", "mem_optim_mb")

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_run(name, layer_type, dims, J, r, lambda_div, retraction, gate_phi, tcfg_base):
    gate = GateConfig(phi=gate_phi)
    mcfg = ModelConfig(layer_type=layer_type, dims=dims, J=J, r=r, gate=gate, lambda_div=lambda_div)
    model = MLP(mcfg)
    tcfg = TrainConfig(**{**tcfg_base, "lambda_div": lambda_div, "retraction": retraction})
    return name, model, tcfg


def final_enc(history) -> dict:
    """Mean final ENC_L/ENC_R over wss layers (or None)."""
    diag = history["diagnostics"][-1] if history["diagnostics"] else {}
    if not diag:
        return {}
    enc_l = [v["ENC_L"] for v in diag.values()]
    enc_r = [v["ENC_R"] for v in diag.values()]
    ang = [v["min_principal_angle"] for v in diag.values()]
    return {
        "ENC_L": sum(enc_l) / len(enc_l),
        "ENC_R": sum(enc_r) / len(enc_r),
        "min_principal_angle": min(ang),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--lr_euclid", type=float, default=1e-3)
    ap.add_argument("--lr_riemann", type=float, default=1e-3)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gate_phi", default="softmax")
    ap.add_argument("--runs", default="all",
                    help="comma-separated subset of {dense,single_rank_Jr,wss,wss_div0,wss_no_retraction,wss_trung,wss_trung_div0,wss_trung_1,wss_trung_2,wss_trung_3} or 'all'")
    ap.add_argument("--quick", action="store_true", help="2 epochs, smaller, for a fast sanity check")
    ap.add_argument("--seed", type=int, default=3, help="shared seed reset before every run")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2

    seed_everything(args.seed)
    device = get_device(args.device)
    dims = [784, 256, 128, 10]
    print(f"device={device} | dataset={args.dataset} | dims={dims} | J={args.J} r={args.r} "
          f"lambda_div={args.lambda_div} epochs={args.epochs} gate={args.gate_phi} seed={args.seed}")

    tcfg_base = dict(
        epochs=args.epochs, batch_size=args.batch_size,
        lr_riemann=args.lr_riemann, lr_euclid=args.lr_euclid,
        dataset=args.dataset, device=args.device, stabilize=50, seed=args.seed,
    )

    all_runs = {
        "dense": ("dense", 0.0, True),
        "single_rank_Jr": ("single_rank_Jr", 0.0, True),
        "wss": ("wss", args.lambda_div, True),
        "wss_div0": ("wss", 0.0, True),
        "wss_no_retraction": ("wss", args.lambda_div, False),
        "wss_trung": ("wss_trung", args.lambda_div, True),
        "wss_trung_div0": ("wss_trung", 0.0, True),
        "wss_trung_1": ("wss_trung_1", args.lambda_div, True),
        "wss_trung_2": ("wss_trung_2", args.lambda_div, True),
        "wss_trung_3": ("wss_trung_3", args.lambda_div, True),
    }
    selected = list(all_runs) if args.runs == "all" else [s.strip() for s in args.runs.split(",")]

    results = []
    histories = {}
    for run_name in selected:
        seed_everything(args.seed)
        train_loader, test_loader = get_loaders(args.dataset, args.batch_size, seed=args.seed)
        name, model, tcfg = build_run(run_name, *all_runs[run_name][:1], dims, args.J, args.r,
                                      all_runs[run_name][1], all_runs[run_name][2],
                                      args.gate_phi, tcfg_base)
        n_params = count_params(model)
        print(f"\n=== {name}  ({n_params:,} params) ===")
        hist = fit(model, train_loader, test_loader, tcfg, device=device)
        enc = final_enc(hist)
        row = {
            "name": name,
            "dataset": args.dataset,
            "params": n_params,
            "final_acc": hist["final_acc"],
            "final_ortho_err": hist["ortho_err"][-1],
            "steps_per_sec": hist["steps_per_sec"][-1],
            "peak_mem_mb": hist.get("peak_mem_mb", float("nan")),
            **enc,
        }
        # memory breakdown (once per run, after training -- never in the hot loop)
        try:
            seed_everything(args.seed)
            mem_loader, _ = get_loaders(args.dataset, args.batch_size, seed=args.seed)
            mem_batch = next(iter(mem_loader))
            mem = measure_breakdown(model, tcfg, mem_batch, device=device)
            row.update({k: mem[k] for k in _MEM_KEYS})
            row["mem_activation_analytic_mb"] = mem["mem_activation_analytic_mb"]
        except Exception as e:  # best-effort tooling; never kill the run
            print(f"  [memory] breakdown failed: {e}")
            row.update({k: float("nan") for k in _MEM_KEYS})
        results.append(row)
        histories[name] = hist

    # ── summary table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"  {'run':<20} {'params':>9} {'acc':>8} {'ortho_err':>11} {'ENC_L':>7} {'ENC_R':>7} {'it/s':>7} {'peak_mem_mb':>7}")
    print("-" * 92)
    for r in results:
        print(f"  {r['name']:<20} {r['params']:>9,} {r['final_acc']:>8.3%} "
              f"{r['final_ortho_err']:>11.2e} {r.get('ENC_L', float('nan')):>7.3f} "
              f"{r.get('ENC_R', float('nan')):>7.3f} {r['steps_per_sec']:>7.1f} "
              f"{r.get('peak_mem_mb', float('nan')):>7.3f} " # I added this, syntax might be off
              )
    print("=" * 92)

    # verdicts (robust to running a subset of runs)
    acc = {r["name"]: r["final_acc"] for r in results}
    ortho = {r["name"]: r["final_ortho_err"] for r in results}
    enc = {r["name"]: r.get("ENC_L") for r in results}
    print("\nVerdicts:")
    if "dense" in acc and "wss" in acc:
        print(f"  wss vs dense gap:             {acc['dense'] - acc['wss']:+.3%} (target |gap| <= ~1-2%)")
    if "wss" in acc and "single_rank_Jr" in acc:
        print(f"  wss vs single_rank_Jr:        {acc['wss'] - acc['single_rank_Jr']:+.3%} "
              f"(>0 => gated J>1 helps at matched params)")
    if enc.get("wss") is not None and enc.get("wss_div0") is not None:
        print(f"  ENC_L with/without diversity: {enc['wss']:.3f} / {enc['wss_div0']:.3f} "
              f"(diversity should keep ENC higher)")
    if "wss" in acc and "wss_no_retraction" in acc:
        print(f"  Remark-8 (retraction on/off): "
              f"acc {acc['wss']:.3%}/{acc['wss_no_retraction']:.3%}  "
              f"ortho {ortho['wss']:.1e}/{ortho['wss_no_retraction']:.1e}")
    if "wss_trung" in acc and "single_rank_Jr" in acc:
        print(f"  wss_trung vs single_rank_Jr:  {acc['wss_trung'] - acc['single_rank_Jr']:+.3%} "
              f"(two-factor Euclidean variant)")
    if enc.get("wss_trung") is not None and enc.get("wss_trung_div0") is not None:
        print(f"  wss_trung ENC_L div/on-off:   {enc['wss_trung']:.3f} / {enc['wss_trung_div0']:.3f}")

    # ── memory breakdown table (MB) ───────────────────────────────────────────────
    print("\n  Memory utilization (MB)")
    print(f"  {'run':<20} {'weight':>9} {'activation':>11} {'gradient':>9} {'optimizer':>10} {'total':>9}")
    print("-" * 72)
    for r in results:
        w = r.get("mem_weight_mb", float("nan")); a = r.get("mem_activation_mb", float("nan"))
        g = r.get("mem_grad_mb", float("nan")); o = r.get("mem_optim_mb", float("nan"))
        tot = sum(v for v in (w, a, g, o) if v == v)  # nan-safe
        print(f"  {r['name']:<20} {w:>9.3f} {a:>11.3f} {g:>9.3f} {o:>10.3f} {tot:>9.3f}")
    print("  (activation = empirical live-alloc delta on MPS/CUDA; analytic estimate in CSV)")

    # ── persist CSV + JSON + plot ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_quick" if args.quick else f"{args.dataset}_e{args.epochs}_J{args.J}_r{args.r}"
    fieldnames = list(dict.fromkeys(k for row in results for k in row))  # union, ordered
    with open(OUT_DIR / f"summary_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    with open(OUT_DIR / f"histories_{tag}.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "diagnostics"}
                   for k, v in histories.items()}, f, indent=2)
    _plot(histories, results, OUT_DIR / f"report_{tag}.png", args)
    print(f"\nWrote outputs to {OUT_DIR}/ (summary_{tag}.csv, report_{tag}.png)")


def _plot(histories, results, path, args):
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
        if any(e == e for e in encs):  # not all-nan
            axes[1].plot(h["epoch"], encs, marker="o", label=name)
    axes[1].axhline(args.J, ls="--", c="gray", alpha=0.6, label=f"J={args.J}")
    axes[1].axhline(1.0, ls=":", c="gray", alpha=0.6, label="J=1")
    axes[1].set(title="ENC_L (layer 0)", xlabel="epoch", ylabel="effective #components")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    for name, h in histories.items():
        ortho = [max(v, 1e-12) for v in h["ortho_err"]]
        axes[2].plot(h["epoch"], ortho, marker="o", label=name)
    axes[2].set(title="Orthonormality error", xlabel="epoch", ylabel="||UᵀU-I||∞", yscale="log")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    # Panel 3: stacked memory breakdown per run (weight / activation / gradient / optimizer).
    names = [r["name"] for r in results]
    cats = [("mem_weight_mb", "weight"), ("mem_activation_mb", "activation"),
            ("mem_grad_mb", "gradient"), ("mem_optim_mb", "optimizer")]
    bottom = [0.0] * len(names)
    for key, label in cats:
        vals = [r.get(key, 0.0) if r.get(key, 0.0) == r.get(key, 0.0) else 0.0 for r in results]
        axes[3].bar(names, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[3].set(title="Memory breakdown (MB)", ylabel="MB")
    axes[3].tick_params(axis="x", rotation=45, labelsize=7)
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3, axis="y")

    fig.suptitle(f"{args.dataset} | J={args.J} r={args.r} epochs={args.epochs} gate={args.gate_phi}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
