"""Merge + pretty-print the per-config CSVs from profile_retraction.py (perf branch).

The SLURM arrays write one CSV per config (outputs/perf/profile_cfg<idx>_<dev>.csv and
convergence_cfg<idx>_<dev>.csv). This merges each group into a single CSV and prints
markdown tables ready to paste into PERF_NOTES.md, with the derived columns you actually
want for the verdict:

  profile:      it/s, ms/step, the per-phase split (forward / diversity / backward /
                optim_step=retraction), peak MB, ortho err, plus "x vs dense" throughput
                and retraction cost normalized to the slowest wss method present.
  convergence:  final acc, ortho err, it/s -- the param-matched accuracy comparison.

Pure stdlib (csv) so it runs anywhere, including locally on the CPU/MPS smoke CSVs.

Usage (from repo root):
    python src/complex/experiments/collect_results.py                 # auto: cuda if present
    python src/complex/experiments/collect_results.py --device cpu    # local smoke CSVs
    python src/complex/experiments/collect_results.py --dir some/perf/dir
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent / "outputs" / "perf"

# phase columns emitted by profile_one (the parenthesized retraction key is intentional)
PHASES = ["ms_forward", "ms_diversity", "ms_backward", "ms_optim_step(retraction)"]
_CFG_IDX = re.compile(r"_cfg(\d+)_")


def _cfg_idx(path: str) -> int:
    m = _CFG_IDX.search(path)
    return int(m.group(1)) if m else 1 << 30


def _load_group(perf_dir: Path, mode: str, dev: str) -> list[dict]:
    """Read every <mode>_cfg<idx>_<dev>.csv, ordered by config index."""
    files = sorted(glob.glob(str(perf_dir / f"{mode}_cfg*_{dev}.csv")), key=_cfg_idx)
    rows: list[dict] = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _merge_write(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {path}")


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(header)
    lines = [" | ".join(header), " | ".join(sep)] + [" | ".join(r) for r in rows]
    return "\n".join(f"| {l} |" for l in lines)


def _profile_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no profile CSVs found)_"
    # references: dense throughput ceiling, and slowest wss retraction present
    dense_its = max((_f(r, "steps_per_sec") for r in rows if r.get("layer_type") == "dense"),
                    default=float("nan"))
    wss_retr = [_f(r, "ms_optim_step(retraction)") for r in rows if r.get("layer_type") == "wss"]
    wss_retr = [v for v in wss_retr if v == v]  # drop NaN
    slowest_retr = max(wss_retr) if wss_retr else float("nan")

    body = []
    for r in rows:
        its = _f(r, "steps_per_sec")
        retr = _f(r, "ms_optim_step(retraction)")
        vs_dense = f"{its / dense_its:.2f}x" if dense_its == dense_its and its == its else "-"
        # retraction speedup only means something where there ARE Stiefel params -- dense's
        # optimizer step is plain Adam, so leave it blank there.
        has_stiefel = r.get("layer_type") in ("wss", "single_rank_Jr")
        retr_norm = (f"{slowest_retr / retr:.1f}x" if has_stiefel and slowest_retr == slowest_retr
                     and retr == retr and retr > 0 else "-")
        body.append([
            r.get("label", "?"),
            f"{int(_f(r, 'params')):,}" if _f(r, "params") == _f(r, "params") else "?",
            f"{its:.1f}", vs_dense,
            f"{_f(r, 'ms_per_step'):.2f}",
            f"{_f(r, 'ms_forward'):.2f}",
            f"{_f(r, 'ms_diversity'):.2f}",
            f"{_f(r, 'ms_backward'):.2f}",
            f"{retr:.2f}", retr_norm,
            f"{_f(r, 'peak_mem_mb'):.0f}",
            f"{_f(r, 'ortho_err'):.1e}",
        ])
    header = ["config", "params", "it/s", "vs dense", "ms/step", "fwd", "div",
              "bwd", "retr", "retr speedup", "peak MB", "ortho"]
    note = ("\n\n_`retr` = optimizer step = the Stiefel retraction. `retr speedup` is normalized "
            "to the slowest wss method here. `div` is the diversity/eigvalsh phase (CUDA-native on "
            "this branch). Phase sums slightly exceed ms/step (per-phase syncs)._")
    return _md_table(header, body) + note


def _convergence_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no convergence CSVs found)_"
    body = []
    for r in rows:
        body.append([
            r.get("label", "?"),
            f"{int(_f(r, 'params')):,}" if _f(r, "params") == _f(r, "params") else "?",
            r.get("dtype", "-"),
            f"{_f(r, 'final_acc') * 100:.2f}%" if _f(r, "final_acc") == _f(r, "final_acc") else "-",
            f"{_f(r, 'final_ortho_err'):.1e}",
            f"{_f(r, 'steps_per_sec'):.1f}",
            f"{_f(r, 'peak_mem_mb'):.0f}",
        ])
    header = ["config", "params", "dtype", "final acc", "ortho err", "it/s", "peak MB"]
    return _md_table(header, body)


def _parity_table(rows: list[dict]) -> str:
    """fp32-vs-bf16 trust gate (Stage 0c). Small acc_delta + same-order ortho => bf16 is safe."""
    if not rows:
        return "_(no parity CSVs found)_"
    body = []
    for r in rows:
        a32, a16 = _f(r, "final_acc_fp32"), _f(r, "final_acc_bf16")
        body.append([
            r.get("label", "?"),
            f"{int(_f(r, 'params')):,}" if _f(r, "params") == _f(r, "params") else "?",
            f"{a32 * 100:.2f}%" if a32 == a32 else "-",
            f"{a16 * 100:.2f}%" if a16 == a16 else "-",
            f"{_f(r, 'acc_delta') * 100:+.2f}%",
            f"{_f(r, 'final_ortho_fp32'):.1e}",
            f"{_f(r, 'final_ortho_bf16'):.1e}",
            f"{_f(r, 'steps_per_sec_fp32'):.1f}",
            f"{_f(r, 'steps_per_sec_bf16'):.1f}",
        ])
    header = ["config", "params", "acc fp32", "acc bf16", "Δacc", "ortho fp32",
              "ortho bf16", "it/s fp32", "it/s bf16"]
    note = ("\n\n_GATE 0c: trust bf16 if |Δacc| is small (target ≤ 1.0%) and ortho stays the same "
            "order of magnitude. Otherwise run the scaling suite in fp32._")
    return _md_table(header, body) + note


def _load_scaling(perf_dir: Path, dev: str) -> list[dict]:
    """Read every scaling_<tier>_cfg<idx>_<dev>.csv (scaling_suite output), ordered by index."""
    files = sorted(glob.glob(str(perf_dir / f"scaling_*_cfg*_{dev}.csv")), key=_cfg_idx)
    rows: list[dict] = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _load_lr_sweep(perf_dir: Path, dev: str) -> list[dict]:
    """Read every lr_sweep_<tier>_cfg<idx>_<dev>.csv (lr_sweep output), ordered by index."""
    files = sorted(glob.glob(str(perf_dir / f"lr_sweep_*_cfg*_{dev}.csv")), key=_cfg_idx)
    rows: list[dict] = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _lr_sweep_table(rows: list[dict]) -> str:
    """Per-method LR calibration (Stage 0.5). Marks the best-acc LR per method (★) so the Stage-1
    headline can pin each method to its own optimum -- LR is not a fair shared axis across geometries."""
    if not rows:
        return "_(no lr_sweep CSVs found)_"
    # group by method, preserving first-seen (config-index) order
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for r in rows:
        m = r.get("method") or r.get("label", "?")
        if m not in groups:
            groups[m] = []
            order.append(m)
        groups[m].append(r)

    body, summary = [], []
    for m in order:
        rs = sorted(groups[m], key=lambda r: _f(r, "lr_value"))
        best = max(rs, key=lambda r: _f(r, "final_acc"))
        for r in rs:
            acc = _f(r, "final_acc")
            body.append([
                m, r.get("lr_swept", "-"), f"{_f(r, 'lr_value'):.0e}",
                f"{acc * 100:.2f}%" if acc == acc else "-",
                f"{_f(r, 'final_ortho_err'):.1e}",
                f"{_f(r, 'steps_per_sec'):.1f}",
                "★" if r is best else "",
            ])
        summary.append(f"{m}: {best.get('lr_swept', '?')}={_f(best, 'lr_value'):.0e} "
                       f"({_f(best, 'final_acc') * 100:.2f}%)")
    header = ["method", "lr group", "lr", "final acc", "ortho err", "it/s", "best"]
    note = ("\n\n_Best LR per method: " + " · ".join(summary) +
            ". Pin each method to its own best LR for the Stage-1 headline (LR is per-geometry, not a "
            "shared axis); record the lr_riemann(wss) / lr_euclid(dense) ratio as a finding._")
    return _md_table(header, body) + note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PERF_DIR), help="dir holding the per-config CSVs")
    ap.add_argument("--device", default="auto", help="cuda|cpu|mps|auto (which CSV suffix to read)")
    args = ap.parse_args()
    perf_dir = Path(args.dir)

    devs = [args.device]
    if args.device == "auto":
        present = {Path(p).stem.rsplit("_", 1)[-1]
                   for p in glob.glob(str(perf_dir / "*_cfg*_*.csv"))}
        devs = [d for d in ("cuda", "mps", "cpu") if d in present] or ["cuda"]

    for dev in devs:
        prof = _load_group(perf_dir, "profile", dev)
        conv = _load_group(perf_dir, "convergence", dev)
        parity = _load_group(perf_dir, "parity", dev)
        scaling = _load_scaling(perf_dir, dev)
        lr_sweep = _load_lr_sweep(perf_dir, dev)
        if not (prof or conv or parity or scaling or lr_sweep):
            print(f"[{dev}] no CSVs in {perf_dir}")
            continue
        _merge_write(prof, perf_dir / f"profile_all_{dev}.csv")
        _merge_write(conv, perf_dir / f"convergence_all_{dev}.csv")
        _merge_write(parity, perf_dir / f"parity_all_{dev}.csv")
        _merge_write(scaling, perf_dir / f"scaling_all_{dev}.csv")
        _merge_write(lr_sweep, perf_dir / f"lr_sweep_all_{dev}.csv")
        print(f"\n## Throughput / phase attribution ({dev})\n")
        print(_profile_table(prof))
        print(f"\n## LR calibration — best LR per method (Stage 0.5) ({dev})\n")
        print(_lr_sweep_table(lr_sweep))
        print(f"\n## fp32-vs-bf16 parity — trust gate ({dev})\n")
        print(_parity_table(parity))
        print(f"\n## Convergence — param-matched accuracy ({dev})\n")
        print(_convergence_table(conv))
        print(f"\n## Scaling sweeps — depth↔width & J↔r ({dev})\n")
        print(_convergence_table(scaling))
        print()


if __name__ == "__main__":
    main()
