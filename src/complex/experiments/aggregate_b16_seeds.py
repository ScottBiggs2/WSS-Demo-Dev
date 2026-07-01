"""Aggregate the per-seed ViT-B/16 CIFAR-10 runs (headline_vit_b16_cifar10.py) into mean+/-std.

Reads every per-seed artifact pair for one epoch budget:
    outputs/summary_b16_cifar10_e{E}_s*.csv      (final metrics per run)
    outputs/histories_b16_cifar10_e{E}_s*.json   (per-epoch curves per run)

and writes:
    outputs/agg_b16_cifar10_e{E}.csv    (per-run mean/std of final_acc, ortho, ENC, params, ...)
    outputs/report_b16_cifar10_e{E}.png (test-acc curve mean+/-std; final-acc bars +/-std;
                                         ortho-err mean+/-std (log); final ENC_L bars +/-std)

Usage (from repo root):
    python src/complex/experiments/aggregate_b16_seeds.py --epochs 50
    python src/complex/experiments/aggregate_b16_seeds.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATASET = "cifar10"
# stable run order for the plots / table
RUN_ORDER = ["dense", "single_rank_Jr", "dense_matched", "wss"]
# (csv column, label) for the per-run final-metric aggregation
_FINAL_METRICS = [
    ("final_acc", "acc"), ("final_ortho_err", "ortho_err"),
    ("ENC_L", "ENC_L"), ("ENC_R", "ENC_R"), ("params", "params"),
    ("steps_per_sec", "it_s"), ("peak_mem_mb", "peak_mem_mb"),
]


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load(tag: str):
    """Return (summary_rows_by_seed, histories_by_seed) for a given tag ('e50' or 'quick')."""
    summ_paths = sorted(OUT_DIR.glob(f"summary_b16_{DATASET}_{tag}_s*.csv"))
    hist_paths = sorted(OUT_DIR.glob(f"histories_b16_{DATASET}_{tag}_s*.json"))
    if not summ_paths:
        raise FileNotFoundError(f"no summary_b16_{DATASET}_{tag}_s*.csv in {OUT_DIR}")
    summaries = {}  # seed -> {run: row}
    for p in summ_paths:
        seed = p.stem.rsplit("_s", 1)[1]
        with open(p) as f:
            summaries[seed] = {row["name"]: row for row in csv.DictReader(f)}
    histories = {}  # seed -> {run: history}
    for p in hist_paths:
        seed = p.stem.rsplit("_s", 1)[1]
        with open(p) as f:
            histories[seed] = json.load(f)
    return summaries, histories


def _runs_present(summaries) -> list[str]:
    seen = set()
    for by_run in summaries.values():
        seen.update(by_run)
    return [r for r in RUN_ORDER if r in seen] + [r for r in seen if r not in RUN_ORDER]


def aggregate(summaries) -> list[dict]:
    """Per-run mean/std over seeds for each final metric."""
    rows = []
    for run in _runs_present(summaries):
        vals = {label: [] for _, label in _FINAL_METRICS}
        for by_run in summaries.values():
            if run not in by_run:
                continue
            for col, label in _FINAL_METRICS:
                vals[label].append(_to_float(by_run[run].get(col)))
        n = max(len(v) for v in vals.values()) if vals else 0
        agg = {"run": run, "n_seeds": n}
        for _, label in _FINAL_METRICS:
            arr = np.array(vals[label], dtype=float)
            arr = arr[~np.isnan(arr)]
            agg[f"mean_{label}"] = float(arr.mean()) if arr.size else float("nan")
            agg[f"std_{label}"] = float(arr.std(ddof=0)) if arr.size else float("nan")
        rows.append(agg)
    return rows


def _curve_mean_std(histories, run: str, key: str):
    """(epochs, mean, std) of a per-epoch history key across seeds; truncated to the shortest run."""
    series = []
    for by_run in histories.values():
        h = by_run.get(run)
        if h and key in h and h[key]:
            series.append(np.asarray(h[key], dtype=float))
    if not series:
        return None
    n = min(len(s) for s in series)
    stacked = np.stack([s[:n] for s in series], axis=0)
    epochs = np.asarray(histories[next(iter(histories))][run]["epoch"][:n], dtype=float)
    return epochs, stacked.mean(0), stacked.std(0)


def _plot(summaries, histories, agg_rows, path: Path, epochs_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _runs_present(summaries)
    n_seeds = max((a["n_seeds"] for a in agg_rows), default=0)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))

    # 1) test-accuracy curve, mean +/- std band across seeds
    for run in runs:
        c = _curve_mean_std(histories, run, "test_acc")
        if c is None:
            continue
        ep, mean, std = c
        axes[0].plot(ep, mean, marker="o", ms=3, label=run)
        axes[0].fill_between(ep, mean - std, mean + std, alpha=0.2)
    axes[0].set(title="Test accuracy (mean±std)", xlabel="epoch", ylabel="acc")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # 2) final accuracy bars with std error bars
    means = [next((a["mean_acc"] for a in agg_rows if a["run"] == r), float("nan")) for r in runs]
    stds = [next((a["std_acc"] for a in agg_rows if a["run"] == r), float("nan")) for r in runs]
    xpos = np.arange(len(runs))
    axes[1].bar(xpos, means, yerr=stds, capsize=4)
    axes[1].set(title="Final test accuracy (mean±std)", ylabel="acc")
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(runs, rotation=30, fontsize=8, ha="right")
    axes[1].grid(alpha=0.3, axis="y")

    # 3) orthonormality error curve (log), mean +/- std
    for run in runs:
        c = _curve_mean_std(histories, run, "ortho_err")
        if c is None:
            continue
        ep, mean, std = c
        mean = np.maximum(mean, 1e-12)
        axes[2].plot(ep, mean, marker="o", ms=3, label=run)
        axes[2].fill_between(ep, np.maximum(mean - std, 1e-12), mean + std, alpha=0.2)
    axes[2].set(title="Orthonormality error (mean±std)", xlabel="epoch", ylabel="||UᵀU-I||∞", yscale="log")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    # 4) final ENC_L bars with std (only meaningful for factorized runs; nan bars are skipped)
    enc_means = [next((a["mean_ENC_L"] for a in agg_rows if a["run"] == r), float("nan")) for r in runs]
    enc_stds = [next((a["std_ENC_L"] for a in agg_rows if a["run"] == r), float("nan")) for r in runs]
    axes[3].bar(xpos, np.nan_to_num(enc_means, nan=0.0),
                yerr=np.nan_to_num(enc_stds, nan=0.0), capsize=4)
    axes[3].set(title="Final ENC_L (mean±std)", ylabel="effective #components")
    axes[3].set_xticks(xpos); axes[3].set_xticklabels(runs, rotation=30, fontsize=8, ha="right")
    axes[3].grid(alpha=0.3, axis="y")

    fig.suptitle(f"ViT-B/16 CIFAR-10 | {epochs_label} | {n_seeds} seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--quick", action="store_true", help="aggregate the --quick smoke artifacts")
    args = ap.parse_args()

    tag = "quick" if args.quick else f"e{args.epochs}"
    summaries, histories = _load(tag)
    agg_rows = aggregate(summaries)

    print(f"\nAggregated {len(summaries)} seed(s): {sorted(summaries)}")
    print(f"  {'run':<18} {'n':>3} {'mean_acc':>9} {'std_acc':>8} {'mean_params':>13} {'mean_ENC_L':>10}")
    for a in agg_rows:
        print(f"  {a['run']:<18} {a['n_seeds']:>3} {a['mean_acc']:>9.3%} {a['std_acc']:>8.3%} "
              f"{a['mean_params']:>13,.0f} {a['mean_ENC_L']:>10.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg_csv = OUT_DIR / f"agg_b16_{DATASET}_{tag}.csv"
    fieldnames = list(dict.fromkeys(k for row in agg_rows for k in row))
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\nWrote {agg_csv}")

    _plot(summaries, histories, agg_rows, OUT_DIR / f"report_b16_{DATASET}_{tag}.png",
          epochs_label=("quick" if args.quick else f"{args.epochs} epochs"))


if __name__ == "__main__":
    main()
