"""Profile WSS retraction methods for GPU throughput (perf branch).

Two modes:
  * --mode profile      throughput micro-benchmark on synthetic CIFAR-shaped batches. Attributes
                        wall-clock per phase (forward / diversity / backward / optimizer-step =
                        the retraction) via CUDA events, records peak memory and orthonormality
                        drift, and optionally dumps a torch.profiler kernel table + trace.
  * --mode convergence  short real CIFAR-10 training (reuses train.fit) to confirm the faster
                        retractions (qr / newton_schulz) match canonical accuracy & orthonormality
                        and to quantify the cost of the NON-FAITHFUL controls (none, retract_every>1).

The config MATRIX is indexable so a SLURM array runs one independent job per config
(``--config $SLURM_ARRAY_TASK_ID``); ``--mode list`` prints it. Sizes: a shrunk ~100K ViT
(primary) and the existing ~1M ViT (continuity). Layer types: dense (throughput ceiling),
single_rank_Jr, wss. Methods: canonical / qr / newton_schulz / none, plus newton_schulz with
retract_every in {2,4} for wss (the retraction-frequency tradeoff).

Usage (from repo root, inside the venv):
    python src/complex/experiments/profile_retraction.py --mode list
    python src/complex/experiments/profile_retraction.py --mode profile --config 0 --profiler
    python src/complex/experiments/profile_retraction.py --mode profile           # all configs
    python src/complex/experiments/profile_retraction.py --mode convergence --config 3 --epochs 8
"""

from __future__ import annotations

# MUST precede torch import so the MPS->CPU fallback for qr/solve/eigvalsh is active (harmless on CUDA).
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

from complex.config import TrainConfig, ViTConfig
from complex.device import get_device
from complex.seed import seed_everything
from complex.train import _orthonormality, build_optimizers, fit
from complex.vit import ViT

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "perf"

# ── size presets ────────────────────────────────────────────────────────────────────────────
# Each tier shares ONE architecture (dim/depth) across its dense-baseline + factorized configs;
# the factorized models come out ~half the params at r=4 (J*r = dim/4 ish), so we add a SECOND,
# shrunk dense ("dense_matched") whose param count ~matches the factorized models -- the
# equal-param dense control that disentangles "WSS helps" from "smaller models just regularize
# better". dense_matched_dim is tuned (head-divisible) to land within a few % of the wss count.
#   100k tier (r=4): dense(dim56)~110K, wss/single_rank~60K, dense_matched(dim40)~58K
#   1m   tier (r=16, kept for continuity with prior runs): dense~811K, wss~715K, dense_matched(dim120)~714K
SIZES = {
    "100k": dict(dim=56, depth=4, heads=4, J=4, r=4, mlp_ratio=2, dense_matched_dim=40),
    "1m": dict(dim=128, depth=6, heads=4, J=4, r=16, mlp_ratio=2, dense_matched_dim=120),
}


def build_configs() -> list[dict]:
    """The full (size x layer_type x retraction) matrix, in a stable index order.

    Per tier: dense baseline, dense_matched (equal-param control), single_rank_Jr (canonical +
    newton_schulz), and wss (canonical / qr / newton_schulz / none + newton_schulz at K=2,4).
    """
    configs: list[dict] = []
    for size_name, size in SIZES.items():
        arch = {k: v for k, v in size.items() if k != "dense_matched_dim"}
        # 1) dense baseline (large): throughput ceiling, no Stiefel params -> retraction irrelevant.
        configs.append(dict(label=f"{size_name}-dense", size_name=size_name, layer_type="dense",
                            retraction_method="auto", retract_every=1, **arch))
        # 2) dense_matched: dense shrunk to ~the factorized param count (equal-param control).
        matched = {**arch, "dim": size["dense_matched_dim"]}
        configs.append(dict(label=f"{size_name}-dense_matched", size_name=size_name, layer_type="dense",
                            retraction_method="auto", retract_every=1, **matched))
        # 3) single_rank_Jr (J=1, rank J*r): the honest factorized control -- canonical + NS only.
        for method in ("canonical", "newton_schulz"):
            configs.append(dict(label=f"{size_name}-single_rank_Jr-{method}", size_name=size_name,
                                layer_type="single_rank_Jr", retraction_method=method,
                                retract_every=1, **arch))
        # 4) wss: the method. Full retraction-speed sweep + the retraction-frequency tradeoff.
        for method in ("canonical", "qr", "newton_schulz", "none"):
            configs.append(dict(label=f"{size_name}-wss-{method}", size_name=size_name,
                                layer_type="wss", retraction_method=method, retract_every=1, **arch))
        for K in (2, 4):
            configs.append(dict(label=f"{size_name}-wss-ns_K{K}", size_name=size_name,
                                layer_type="wss", retraction_method="newton_schulz",
                                retract_every=K, **arch))
    return configs


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_model_and_tcfg(cfg: dict, args) -> tuple[ViT, TrainConfig]:
    vcfg = ViTConfig(
        layer_type=cfg["layer_type"], attn_type=("dense" if cfg["layer_type"] == "dense" else "wss_separate"),
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], mlp_ratio=cfg["mlp_ratio"],
        J=cfg["J"], r=cfg["r"], retraction_method=cfg["retraction_method"],
        retract_every=cfg["retract_every"], lambda_div=args.lambda_div,
    )
    model = ViT(vcfg)
    # 'none' control and lazy(K>1) must NOT be silently re-orthonormalized by geoopt's periodic
    # `stabilize` projx -- disable it so the control/tradeoff is pure. Faithful methods keep 50.
    stabilize = 10**9 if (cfg["retraction_method"] == "none" or cfg["retract_every"] > 1) else 50
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr_riemann=args.lr_riemann,
                       lr_euclid=args.lr_euclid, lambda_div=args.lambda_div, dataset="cifar10",
                       device=args.device, stabilize=stabilize, seed=args.seed)
    return model, tcfg


# ── timing helpers ────────────────────────────────────────────────────────────────────────────
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class PhaseTimer:
    """Accumulate per-phase wall-clock (ms). On CUDA uses cuda.Event; else perf_counter + sync.

    Per-phase timing inserts a device sync after each phase, so the *sum* slightly overstates a
    real (un-synced) step -- that's why throughput is measured separately, without these syncs.
    The phase *proportions* are what this is for (where does the time go?).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.totals: dict[str, float] = {}

    def time(self, name: str, fn):
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
        else:
            _sync(self.device)
            t0 = time.perf_counter()
            out = fn()
            _sync(self.device)
            ms = (time.perf_counter() - t0) * 1e3
        self.totals[name] = self.totals.get(name, 0.0) + ms
        return out


def profile_one(cfg: dict, args, device: torch.device) -> dict:
    seed_everything(args.seed)
    model, tcfg = make_model_and_tcfg(cfg, args)
    model = model.to(device)
    opts = build_optimizers(model, tcfg)
    criterion = nn.CrossEntropyLoss()
    n_params = count_params(model)

    # synthetic CIFAR-shaped batch (throughput doesn't need real data)
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    def step(timer: PhaseTimer | None = None):
        for o in opts:
            o.zero_grad()
        if timer is None:
            logits = model(x)
            div = model.diversity_loss()
            loss = criterion(logits, y) + tcfg.lambda_div * div
            loss.backward()
            for o in opts:
                o.step()
        else:
            logits = timer.time("forward", lambda: model(x))
            ce = criterion(logits, y)
            div = timer.time("diversity", lambda: model.diversity_loss())
            loss = ce + tcfg.lambda_div * div
            timer.time("backward", loss.backward)
            timer.time("optim_step(retraction)", lambda: [o.step() for o in opts])

    # warmup (also triggers cuDNN/autotune + lazy CUDA init)
    for _ in range(args.warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # throughput: no per-phase syncs
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    _sync(device)
    dt = time.perf_counter() - t0
    steps_per_sec = args.steps / dt

    # per-phase attribution (separate pass, with syncs)
    timer = PhaseTimer(device)
    for _ in range(args.steps):
        step(timer)
    phase_ms = {k: v / args.steps for k, v in timer.totals.items()}

    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    ortho = _orthonormality(model)   # post last step (post-projection for faithful; mid-drift for lazy/none)

    row = {
        "label": cfg["label"], "size": cfg["size_name"], "layer_type": cfg["layer_type"],
        "retraction_method": cfg["retraction_method"], "retract_every": cfg["retract_every"],
        "params": n_params, "steps_per_sec": round(steps_per_sec, 2),
        "ms_per_step": round(1e3 / steps_per_sec, 3), "peak_mem_mb": round(peak_mb, 2),
        "ortho_err": ortho,
        **{f"ms_{k}": round(v, 4) for k, v in phase_ms.items()},
    }
    if args.profiler:
        row["trace"] = _run_torch_profiler(step, device, cfg["label"], args)
    return row


def _run_torch_profiler(step, device: torch.device, label: str, args) -> str:
    """Dump a kernel table + chrome trace for one config (kernel-level attribution)."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(min(args.steps, 20)):
            step()
        _sync(device)
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
    txt = OUT_DIR / f"profiler_{label}.txt"
    txt.write_text(table)
    trace = OUT_DIR / f"trace_{label}.json"
    prof.export_chrome_trace(str(trace))
    print(f"  [profiler] {txt.name}, {trace.name}")
    return str(trace)


def convergence_one(cfg: dict, args, device: torch.device) -> dict:
    """Short real-CIFAR-10 training to check accuracy/orthonormality per method."""
    from complex.data import get_loaders
    seed_everything(args.seed)
    train_loader, test_loader = get_loaders("cifar10", args.batch_size, augment=not args.no_augment, seed=args.seed)
    model, tcfg = make_model_and_tcfg(cfg, args)
    n_params = count_params(model)
    print(f"\n=== {cfg['label']}  ({n_params:,} params, {args.epochs} epochs) ===")
    hist = fit(model, train_loader, test_loader, tcfg, device=device)
    return {
        "label": cfg["label"], "size": cfg["size_name"], "layer_type": cfg["layer_type"],
        "retraction_method": cfg["retraction_method"], "retract_every": cfg["retract_every"],
        "params": n_params, "final_acc": hist["final_acc"], "final_ortho_err": hist["ortho_err"][-1],
        "steps_per_sec": hist["steps_per_sec"][-1], "peak_mem_mb": hist.get("peak_mem_mb", float("nan")),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["profile", "convergence", "list"], default="profile")
    ap.add_argument("--config", type=int, default=-1, help="config index (-1 = all); see --mode list")
    ap.add_argument("--steps", type=int, default=100, help="timed steps (profile mode)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8, help="convergence mode")
    ap.add_argument("--lr_riemann", type=float, default=1e-3)
    ap.add_argument("--lr_euclid", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--profiler", action="store_true", help="also dump torch.profiler trace (profile mode)")
    ap.add_argument("--out", default=None, help="output CSV path (default: outputs/perf/<mode>_<tag>.csv)")
    args = ap.parse_args()

    configs = build_configs()
    if args.mode == "list":
        print(f"{len(configs)} configs (params built on CPU):")
        for i, c in enumerate(configs):
            try:
                model, _ = make_model_and_tcfg(c, args)
                pstr = f"{count_params(model):>8,}p"
            except Exception as e:  # surface a bad size combo immediately
                pstr = f"ERR: {e}"
            print(f"  [{i:2d}] {c['label']:<24} {pstr}  dim={c['dim']:<4} "
                  f"layer={c['layer_type']:<14} method={c['retraction_method']:<13} K={c['retract_every']}")
        return

    device = get_device(args.device)
    selected = configs if args.config < 0 else [configs[args.config]]
    print(f"device={device} | mode={args.mode} | {len(selected)} config(s) | "
          f"steps={args.steps} warmup={args.warmup} bs={args.batch_size}")

    runner = profile_one if args.mode == "profile" else convergence_one
    rows = []
    for c in selected:
        row = runner(c, args, device)
        rows.append(row)
        if args.mode == "profile":
            phases = " ".join(f"{k[3:]}={v:.2f}ms" for k, v in row.items() if k.startswith("ms_"))
            print(f"  {row['label']:<28} {row['params']:>8,}p  {row['steps_per_sec']:>7.1f} it/s  "
                  f"peak={row['peak_mem_mb']}MB  ortho={row['ortho_err']:.1e}  | {phases}")

    tag = (f"cfg{args.config}" if args.config >= 0 else "all") + f"_{device.type}"
    out = Path(args.out) if args.out else OUT_DIR / f"{args.mode}_{tag}.csv"
    _write_csv(rows, out)


if __name__ == "__main__":
    main()
