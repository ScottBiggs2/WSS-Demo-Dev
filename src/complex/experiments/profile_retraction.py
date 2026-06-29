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

import wandb

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

from complex.config import TrainConfig, ViTConfig
from complex.device import autocast_ctx, get_device
from complex.experiments.param_budget import dense_matched_dim, vit_param_count
from complex.seed import seed_everything
from complex.train import _orthonormality, build_optimizers, fit
from complex.vit import ViT

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "perf"

# The scaling flight trains CIFAR-100 (100 classes) at ALL tiers -- see PERF_NOTES / the plan.
NUM_CLASSES = 100
DATASET = "cifar100"
PATCH_SIZE = 4
IMG_SIZE = 32

# ── size presets (CIFAR-100) ──────────────────────────────────────────────────────────────────
# Each tier shares ONE human-chosen anchor architecture (dim/depth, sensible aspect) across its
# dense-baseline + factorized configs. A SECOND, shrunk dense ("dense_matched") whose param count
# ~matches the factorized (wss/single_rank) count is the equal-param control that disentangles "WSS
# helps" from "smaller models just regularize better" -- THE headline comparator. dense_matched_dim
# is DERIVED via param_budget (not hand-tuned), so it correctly accounts for CIFAR-100's larger head
# (dim*100). Verified factorized vs dense_matched counts (param_budget == live ViT count):
#   100k (Jr=24): wss/single 116,236 ; dense 184,780 ; matched dim56 -> 115,068 (-1.0%)
#   1m   (Jr=48): wss/single 830,308 ; dense 1,823,908 ; matched dim126 -> 797,806 (-3.9%)
#   10m  (Jr=192): wss/single 12,534,244 ; dense 14,289,892 ; matched dim360 -> 12,567,340 (+0.3%)
SIZES = {
    "100k": dict(dim=72,  depth=4,  heads=4, J=4, r=6,  mlp_ratio=2),
    "1m":   dict(dim=192, depth=6,  heads=6, J=4, r=12, mlp_ratio=2),
    "10m":  dict(dim=384, depth=12, heads=8, J=8, r=24, mlp_ratio=2),
}


def _matched_dim(size: dict) -> int:
    """Dense dim whose param count matches this tier's factorized (wss) count -- equal-param control."""
    wss = vit_param_count(num_classes=NUM_CLASSES, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
                          layer_type="wss", attn_type="wss_separate", **size)
    return dense_matched_dim(wss, depth=size["depth"], heads=size["heads"],
                             mlp_ratio=size["mlp_ratio"], num_classes=NUM_CLASSES,
                             patch_size=PATCH_SIZE, img_size=IMG_SIZE)


def build_configs() -> list[dict]:
    """The full (size x layer_type x retraction) matrix, in a stable index order.

    Per tier: dense baseline, dense_matched (equal-param control), single_rank_Jr (canonical +
    newton_schulz), and wss (canonical / qr / newton_schulz / none + newton_schulz at K=2,4).
    """
    configs: list[dict] = []
    for size_name, size in SIZES.items():
        arch = dict(size)
        # 1) dense baseline (large): throughput ceiling, no Stiefel params -> retraction irrelevant.
        configs.append(dict(label=f"{size_name}-dense", size_name=size_name, layer_type="dense",
                            retraction_method="auto", retract_every=1, **arch))
        # 2) dense_matched: dense shrunk to ~the factorized param count (equal-param control).
        matched = {**arch, "dim": _matched_dim(size)}
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
        num_classes=NUM_CLASSES, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], mlp_ratio=cfg["mlp_ratio"],
        J=cfg["J"], r=cfg["r"], retraction_method=cfg["retraction_method"],
        retract_every=cfg["retract_every"], lambda_div=args.lambda_div,
        attn_dropout=args.attn_dropout, mlp_dropout=args.mlp_dropout,
    )
    model = ViT(vcfg)
    # 'none' control and lazy(K>1) must NOT be silently re-orthonormalized by geoopt's periodic
    # `stabilize` projx -- disable it so the control/tradeoff is pure. Faithful methods keep 50.
    stabilize = 10**9 if (cfg["retraction_method"] == "none" or cfg["retract_every"] > 1) else 50
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr_riemann=args.lr_riemann,
                       lr_euclid=args.lr_euclid, lambda_div=args.lambda_div, dataset=DATASET,
                       device=args.device, stabilize=stabilize, seed=args.seed,
                       weight_decay=args.weight_decay, amp=args.amp, allow_tf32=args.allow_tf32)
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
    x = torch.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=device)

    # Mirror train_epoch's precision: bf16 autocast over forward+diversity when --amp; the
    # optimizer step (retraction) runs OUTSIDE on fp32 masters. nullcontext when amp is off.
    def _fwd():
        with autocast_ctx(device, tcfg.amp):
            return model(x)

    def _div():
        with autocast_ctx(device, tcfg.amp):
            return model.diversity_loss()

    def step(timer: PhaseTimer | None = None):
        for o in opts:
            o.zero_grad()
        if timer is None:
            logits = _fwd()
            div = _div()
            loss = criterion(logits, y) + tcfg.lambda_div * div
            loss.backward()
            for o in opts:
                o.step()
        else:
            logits = timer.time("forward", _fwd)
            ce = criterion(logits, y)
            div = timer.time("diversity", _div)
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
        "seed": args.seed, "dtype": ("bf16" if args.amp else "fp32"),
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
    """Short real-CIFAR-100 training to check accuracy/orthonormality per method."""
    from complex.data import get_loaders
    seed_everything(args.seed)
    train_loader, test_loader = get_loaders(DATASET, args.batch_size, augment=not args.no_augment, seed=args.seed)
    model, tcfg = make_model_and_tcfg(cfg, args)
    n_params = count_params(model)
    print(f"\n=== {cfg['label']}  ({n_params:,} params, {args.epochs} epochs, {('bf16' if args.amp else 'fp32')}) ===")
    hist = fit(model, train_loader, test_loader, tcfg, device=device)
    return {
        "label": cfg["label"], "size": cfg["size_name"], "layer_type": cfg["layer_type"],
        "retraction_method": cfg["retraction_method"], "retract_every": cfg["retract_every"],
        "seed": args.seed, "dtype": ("bf16" if args.amp else "fp32"),
        "params": n_params, "final_acc": hist["final_acc"], "final_ortho_err": hist["ortho_err"][-1],
        "steps_per_sec": hist["steps_per_sec"][-1], "peak_mem_mb": hist.get("peak_mem_mb", float("nan")),
    }


def parity_one(cfg: dict, args, device: torch.device) -> dict:
    """fp32-vs-bf16 convergence parity for ONE config (the bf16-trust gate, Stage 0c).

    Trains the same config twice -- fp32 then bf16 autocast -- under an identical seed (so init and
    data order match) and reports the accuracy / orthonormality deltas. If |acc_delta| is small and
    ortho stays the same order of magnitude, the whole bf16 scaling suite is trustworthy.
    """
    from complex.data import get_loaders
    res = {}
    n_params = 0
    for tag, amp in (("fp32", False), ("bf16", True)):
        seed_everything(args.seed)
        train_loader, test_loader = get_loaders(DATASET, args.batch_size,
                                                augment=not args.no_augment, seed=args.seed)
        model, tcfg = make_model_and_tcfg(cfg, args)
        tcfg.amp = amp
        tcfg.allow_tf32 = amp and args.allow_tf32   # tf32 only on the fast (bf16) path
        n_params = count_params(model)
        print(f"\n=== PARITY {cfg['label']} [{tag}]  ({n_params:,} params, {args.epochs} epochs) ===")
        res[tag] = fit(model, train_loader, test_loader, tcfg, device=device)
    return {
        "label": cfg["label"], "size": cfg["size_name"], "params": n_params,
        "final_acc_fp32": res["fp32"]["final_acc"], "final_acc_bf16": res["bf16"]["final_acc"],
        "acc_delta": res["bf16"]["final_acc"] - res["fp32"]["final_acc"],
        "final_ortho_fp32": res["fp32"]["ortho_err"][-1], "final_ortho_bf16": res["bf16"]["ortho_err"][-1],
        "ortho_delta": res["bf16"]["ortho_err"][-1] - res["fp32"]["ortho_err"][-1],
        "steps_per_sec_fp32": res["fp32"]["steps_per_sec"][-1],
        "steps_per_sec_bf16": res["bf16"]["steps_per_sec"][-1],
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
    ap.add_argument("--mode", choices=["profile", "convergence", "parity", "list"], default="profile")
    ap.add_argument("--config", type=int, default=-1, help="config index (-1 = all); see --mode list")
    ap.add_argument("--steps", type=int, default=100, help="timed steps (profile mode)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8, help="convergence/parity mode")
    ap.add_argument("--lr_riemann", type=float, default=1e-3)
    ap.add_argument("--lr_euclid", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--mlp_dropout", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast forward+loss (CUDA); linalg stays fp32")
    ap.add_argument("--allow_tf32", action="store_true", help="enable TF32 matmul/cudnn (CUDA)")
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
    if args.config < 0 and args.mode == "parity":
        # parity defaults to the 100k wss newton_schulz config (the cheapest faithful WSS point).
        default = next(i for i, c in enumerate(configs) if c["label"] == "100k-wss-newton_schulz")
        selected = [configs[default]]
    else:
        selected = configs if args.config < 0 else [configs[args.config]]
    print(f"device={device} | mode={args.mode} | {len(selected)} config(s) | "
          f"steps={args.steps} warmup={args.warmup} bs={args.batch_size} amp={args.amp} tf32={args.allow_tf32}")

    wandb.init(
        project="wss-perf",
        name=f"{args.mode}_{args.config if args.config >= 0 else 'all'}_{device.type}",
        config={
            "mode": args.mode, "config_idx": args.config, "device": str(device),
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr_riemann": args.lr_riemann, "lr_euclid": args.lr_euclid,
            "amp": args.amp, "allow_tf32": args.allow_tf32,
            "steps": args.steps, "warmup": args.warmup,
        }
    )

    runner = {"profile": profile_one, "convergence": convergence_one, "parity": parity_one}[args.mode]
    rows = []
    for c in selected:
        row = runner(c, args, device)
        rows.append(row)
        wandb.log(row)
        if args.mode == "profile":
            phases = " ".join(f"{k[3:]}={v:.2f}ms" for k, v in row.items() if k.startswith("ms_"))
            print(f"  {row['label']:<28} {row['params']:>8,}p  {row['steps_per_sec']:>7.1f} it/s  "
                  f"peak={row['peak_mem_mb']}MB  ortho={row['ortho_err']:.1e}  | {phases}")
        elif args.mode == "parity":
            print(f"  {row['label']:<28} fp32={row['final_acc_fp32']:.3%} bf16={row['final_acc_bf16']:.3%} "
                  f"d_acc={row['acc_delta']:+.3%}  ortho fp32={row['final_ortho_fp32']:.1e} "
                  f"bf16={row['final_ortho_bf16']:.1e}")

    tag = (f"cfg{args.config}" if args.config >= 0 else "all") + f"_{device.type}"
    out = Path(args.out) if args.out else OUT_DIR / f"{args.mode}_{tag}.csv"
    _write_csv(rows, out)
    wandb.finish()


if __name__ == "__main__":
    main()
