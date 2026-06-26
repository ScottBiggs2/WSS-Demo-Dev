"""Torchvision raw ViT experiment with WSS-style projection replacement.

This is intentionally different from headline_vit.py: the backbone is a raw torchvision ViT
(e.g. vit_b_16) with torchvision's default architecture. For factorized runs, we replace the
original transformer's attention projections (Q/K/V/O) and MLP Linear layers with the same
projection families used elsewhere in this repo:

  dense            -- raw torchvision model, no projection replacement
  single_rank_Jr   -- one rank-(J*r) factorization per replaced projection
  wss              -- J gated rank-r components with Stiefel U/V + spectrum
  wss_trung        -- J gated rank-r Euclidean L/R factors
  wss_trung_1      -- L/R initialized from SVD of the raw torchvision layer weight
  wss_trung_2      -- direct balanced L/R init so LR product has He/Kaiming variance
  wss_trung_3      -- SVD of raw layer weight, then rescale effective weight to He/Kaiming variance

The classification head is dense and set to num_classes for the dataset; patch embedding, norms,
positional embeddings, dropout, depth, hidden dim, heads, etc. stay at the torchvision model defaults.
Datasets: CIFAR-10, Flowers102, and Oxford-IIIT Pets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from complex.config import GateConfig, TrainConfig
from complex.device import get_device
from complex.memory import measure_breakdown
from complex.models import _min_principal_angle
from complex.seed import seed_everything, seed_worker
from complex.superposition import SuperpositionLinear, WssTrungLinear, make_proj
from complex.train import fit

OUT_DIR = Path(__file__).resolve().parent / "outputs"
_MEM_KEYS = ("mem_weight_mb", "mem_activation_mb", "mem_grad_mb", "mem_optim_mb")
_MODEL_FACTORIES = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
}
_DATASET_NUM_CLASSES = {"cifar10": 10, "flowers102": 102, "pets": 37}



def _infer_init_policy(weight: torch.Tensor) -> dict:
    """Infer the closest common init scale from a concrete dense weight tensor."""
    w = weight.detach().float()
    if w.ndim != 2:
        return {"kind": "observed", "target_var": float(w.var(unbiased=False).item()), "observed_var": float(w.var(unbiased=False).item())}
    fan_out, fan_in = w.shape
    observed = float(w.var(unbiased=False).item())
    candidates = {
        "xavier": 2.0 / (fan_in + fan_out),
        "kaiming_he": 2.0 / fan_in,
        "torch_linear": 1.0 / (3.0 * fan_in),
    }
    if observed < 1e-16:
        return {"kind": "zero", "target_var": candidates["xavier"], "observed_var": observed}
    kind, target = min(candidates.items(), key=lambda kv: abs(math.log(observed / kv[1])))
    rel = abs(observed - target) / max(target, 1e-12)
    if rel > 0.75:
        kind, target = "observed", observed
    return {"kind": kind, "target_var": float(target), "observed_var": observed}


def _record_policy(log: list[dict], name: str, weight: torch.Tensor, override: dict | None = None) -> float:
    info = _infer_init_policy(weight) if override is None else {**override, "observed_var": float(weight.detach().float().var(unbiased=False).item())}
    log.append({"name": name, **info})
    return info["target_var"]


def _print_init_policy_summary(log: list[dict], layer_type: str) -> None:
    if not log or layer_type not in ("wss_trung_2", "wss_trung_3"):
        return
    counts = {}
    for item in log:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  init policy for {layer_type}: auto per replaced layer ({summary})")
    for item in log[:8]:
        print(f"    {item['name']}: {item['kind']} target_var={item['target_var']:.3e} observed={item['observed_var']:.3e}")
    if len(log) > 8:
        print(f"    ... {len(log) - 8} more layers")


class ReplacementSelfAttention(nn.Module):
    """MultiheadAttention-compatible wrapper with replaceable Q/K/V/O projections."""

    def __init__(self, original: nn.MultiheadAttention, layer_type: str, *, J: int, r: int,
                 gate: GateConfig, policy_log: list[dict], prefix: str,
                 stiefel_canonical: bool = True):
        super().__init__()
        if not original.batch_first:
            raise ValueError("torchvision ViT is expected to use batch_first=True attention")
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.dropout = original.dropout
        self.batch_first = True
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        use_bias = original.in_proj_bias is not None
        in_proj_policy = _infer_init_policy(original.in_proj_weight.detach())
        q_w, k_w, v_w = original.in_proj_weight.detach().chunk(3, dim=0)
        if use_bias:
            q_b, k_b, v_b = original.in_proj_bias.detach().chunk(3, dim=0)
        else:
            q_b = k_b = v_b = None

        def mk(suffix, base_weight, base_bias):
            target_var = _record_policy(policy_log, f"{prefix}.{suffix}", base_weight, override=in_proj_policy)
            return make_proj(layer_type, self.embed_dim, self.embed_dim, J=J, r=r,
                             use_bias=use_bias, gate=gate, stiefel_canonical=stiefel_canonical,
                             base_weight=base_weight, base_bias=base_bias, init_target_var=target_var)

        self.q_proj = mk("q_proj", q_w, q_b)
        self.k_proj = mk("k_proj", k_w, k_b)
        self.v_proj = mk("v_proj", v_w, v_b)
        o_target_var = _record_policy(policy_log, f"{prefix}.o_proj", original.out_proj.weight)
        self.o_proj = make_proj(layer_type, self.embed_dim, self.embed_dim, J=J, r=r,
                                use_bias=original.out_proj.bias is not None, gate=gate,
                                stiefel_canonical=stiefel_canonical,
                                base_weight=original.out_proj.weight, base_bias=original.out_proj.bias,
                                init_target_var=o_target_var)
        self.attn_drop = nn.Dropout(original.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if key is not query or value is not query:
            raise NotImplementedError("ReplacementSelfAttention only supports self-attention")
        if key_padding_mask is not None or attn_mask is not None or is_causal:
            raise NotImplementedError("Masks are not used by torchvision ViT encoder blocks")

        # Reimplement SuperpositionMultiHeadAttn here so we can preserve attention dropout.
        B, N, _ = query.shape
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(query))
        v = self._split_heads(self.v_proj(query))
        weights = (q @ k.transpose(-2, -1)) * self.scale
        weights = weights.softmax(dim=-1)
        weights = self.attn_drop(weights)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.o_proj(out)

        if not need_weights:
            return out, None
        attn_weights = weights.mean(dim=1) if average_attn_weights else weights
        return out, attn_weights


class TorchvisionViTWithDiagnostics(nn.Module):
    def __init__(self, model_name: str, layer_type: str, *, num_classes: int, J: int, r: int,
                 gate_phi: str, stiefel_canonical: bool = True):
        super().__init__()
        if model_name not in _MODEL_FACTORIES:
            raise ValueError(f"unknown model {model_name!r}; expected one of {sorted(_MODEL_FACTORIES)}")
        self.model_name = model_name
        self.layer_type = layer_type
        self.J = J
        self.r = r
        self.gate = GateConfig(phi=gate_phi)
        self.init_policy_log: list[dict] = []
        self.model = _MODEL_FACTORIES[model_name](weights=None, num_classes=num_classes)
        if layer_type != "dense":
            self._replace_transformer_projections(stiefel_canonical=stiefel_canonical)

    @property
    def image_size(self) -> int:
        return int(self.model.image_size)

    def _replace_transformer_projections(self, *, stiefel_canonical: bool) -> None:
        for block_idx, block in enumerate(self.model.encoder.layers):
            prefix = f"encoder.layers.{block_idx}"
            block.self_attention = ReplacementSelfAttention(
                block.self_attention, self.layer_type, J=self.J, r=self.r, gate=self.gate,
                policy_log=self.init_policy_log, prefix=f"{prefix}.self_attention",
                stiefel_canonical=stiefel_canonical,
            )
            fc1, fc2 = block.mlp[0], block.mlp[3]
            fc1_target_var = _record_policy(self.init_policy_log, f"{prefix}.mlp.0", fc1.weight)
            fc2_target_var = _record_policy(self.init_policy_log, f"{prefix}.mlp.3", fc2.weight)
            block.mlp[0] = make_proj(
                self.layer_type, fc1.in_features, fc1.out_features,
                J=self.J, r=self.r, use_bias=fc1.bias is not None, gate=self.gate,
                stiefel_canonical=stiefel_canonical, base_weight=fc1.weight, base_bias=fc1.bias,
                init_target_var=fc1_target_var,
            )
            block.mlp[3] = make_proj(
                self.layer_type, fc2.in_features, fc2.out_features,
                J=self.J, r=self.r, use_bias=fc2.bias is not None, gate=self.gate,
                stiefel_canonical=stiefel_canonical, base_weight=fc2.weight, base_bias=fc2.bias,
                init_target_var=fc2_target_var,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _named_wss(self) -> list[tuple[str, nn.Module]]:
        return [(name, m) for name, m in self.named_modules()
                if isinstance(m, (SuperpositionLinear, WssTrungLinear)) and m.J > 1]

    def diversity_loss(self) -> torch.Tensor:
        from complex.diversity import summed_diversity
        layers = [m for _, m in self._named_wss()]
        if not layers:
            return torch.zeros((), device=next(self.parameters()).device)
        if all(isinstance(m, SuperpositionLinear) for m in layers):
            return summed_diversity([m.U for m in layers] + [m.V for m in layers])
        return torch.stack([m.diversity()["D"] for m in layers]).sum()

    @torch.no_grad()
    def diagnostics(self) -> dict:
        out = {}
        for name, m in self._named_wss():
            d = m.diversity()
            finite_params = all(torch.isfinite(p).all().item() for p in m.parameters())
            try:
                U = m.U.detach() if isinstance(m, SuperpositionLinear) else m.diversity_frames()[0].detach()
            except RuntimeError:
                U = torch.full((m.J, m.in_dim, m.r), float("nan"), device=next(m.parameters()).device)
            out[name] = {
                "ENC_L": d["ENC_L"].item(),
                "ENC_R": d["ENC_R"].item(),
                "min_principal_angle": _min_principal_angle(U),
                "finite_params": float(finite_params),
            }
        return out


def _image_transforms(image_size: int, *, augment: bool):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    test_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, test_tf


def _build_datasets(dataset: str, root: str, train_tf, test_tf):
    if dataset == "cifar10":
        return (
            datasets.CIFAR10(root, train=True, download=True, transform=train_tf),
            datasets.CIFAR10(root, train=False, download=True, transform=test_tf),
        )
    if dataset == "flowers102":
        train_set = torch.utils.data.ConcatDataset([
            datasets.Flowers102(root, split="train", download=True, transform=train_tf),
            datasets.Flowers102(root, split="val", download=True, transform=train_tf),
        ])
        test_set = datasets.Flowers102(root, split="test", download=True, transform=test_tf)
        return train_set, test_set
    if dataset == "pets":
        return (
            datasets.OxfordIIITPet(root, split="trainval", target_types="category",
                                   download=True, transform=train_tf),
            datasets.OxfordIIITPet(root, split="test", target_types="category",
                                   download=True, transform=test_tf),
        )
    raise ValueError(f"unknown dataset {dataset!r}; expected {sorted(_DATASET_NUM_CLASSES)}")


def _image_loaders(dataset: str, image_size: int, batch_size: int, *, root: str | None, augment: bool,
                   seed: int, num_workers: int = 0, train_subset: int | None = None,
                   test_subset: int | None = None):
    dataset = dataset.lower()
    root = root or str(Path(__file__).resolve().parents[3] / "data")
    train_tf, test_tf = _image_transforms(image_size, augment=augment)
    train_set, test_set = _build_datasets(dataset, root, train_tf, test_tf)
    if train_subset is not None:
        train_set = Subset(train_set, range(min(train_subset, len(train_set))))
    if test_subset is not None:
        test_set = Subset(test_set, range(min(test_subset, len(test_set))))
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=gen, worker_init_fn=seed_worker,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, test_loader


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def final_enc(history) -> dict:
    diag = history["diagnostics"][-1] if history["diagnostics"] else {}
    if not diag:
        return {}
    return {
        "ENC_L": sum(v["ENC_L"] for v in diag.values()) / len(diag),
        "ENC_R": sum(v["ENC_R"] for v in diag.values()) / len(diag),
        "min_principal_angle": min(v["min_principal_angle"] for v in diag.values()),
    }


def build_run(name: str, layer_type: str, args, tcfg_base: dict):
    model = TorchvisionViTWithDiagnostics(
        args.model, layer_type, num_classes=args.num_classes, J=args.J, r=args.r,
        gate_phi=args.gate_phi, stiefel_canonical=not args.euclidean,
    )
    _print_init_policy_summary(model.init_policy_log, layer_type)
    lambda_div = args.lambda_div if layer_type in ("wss", "wss_trung", "wss_trung_1", "wss_trung_2", "wss_trung_3") else 0.0
    tcfg = TrainConfig(**{**tcfg_base, "lambda_div": lambda_div, "retraction": True})
    return name, model, tcfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_b_16", choices=sorted(_MODEL_FACTORIES))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_euclid", type=float, default=1e-3)
    ap.add_argument("--lr_riemann", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-3)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--gate_phi", default="softmax")
    ap.add_argument("--euclidean", action="store_true", help="use QR Stiefel retraction for wss/single_rank")
    ap.add_argument("--runs", default="dense,single_rank_Jr,wss,wss_trung",
                    help="comma-separated subset of {dense,single_rank_Jr,wss,wss_trung,wss_trung_1,wss_trung_2,wss_trung_3} or 'all'")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dataset", default="cifar10", choices=sorted(_DATASET_NUM_CLASSES))
    ap.add_argument("--num_classes", type=int, default=None,
                    help="override dataset num_classes; defaults to 10/102/37")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--train_subset", type=int, default=None, help="optional train subset for smoke runs")
    ap.add_argument("--test_subset", type=int, default=None, help="optional test subset for smoke runs")
    ap.add_argument("--quick", action="store_true", help="1 epoch on a tiny subset for a fast sanity run")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 1
        args.train_subset = args.train_subset or 64
        args.test_subset = args.test_subset or 64

    all_runs = {
        "dense": "dense",
        "single_rank_Jr": "single_rank_Jr",
        "wss": "wss",
        "wss_trung": "wss_trung",
        "wss_trung_1": "wss_trung_1",
        "wss_trung_2": "wss_trung_2",
        "wss_trung_3": "wss_trung_3",
    }
    selected = list(all_runs) if args.runs == "all" else [s.strip() for s in args.runs.split(",")]
    unknown = [s for s in selected if s not in all_runs]
    if unknown:
        raise ValueError(f"unknown runs {unknown}; expected {list(all_runs)}")

    args.dataset = args.dataset.lower()
    if args.num_classes is None:
        args.num_classes = _DATASET_NUM_CLASSES[args.dataset]

    seed_everything(args.seed)
    device = get_device(args.device)
    probe = TorchvisionViTWithDiagnostics(args.model, "dense", num_classes=args.num_classes,
                                          J=args.J, r=args.r, gate_phi=args.gate_phi)
    image_size = probe.image_size
    del probe
    print(f"device={device} | model={args.model}(raw torchvision, weights=None) | dataset={args.dataset} "
          f"num_classes={args.num_classes} image_size={image_size} J={args.J} r={args.r} lambda_div={args.lambda_div} "
          f"epochs={args.epochs} gate={args.gate_phi} seed={args.seed}")

    tcfg_base = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr_riemann=args.lr_riemann,
        lr_euclid=args.lr_euclid, dataset=args.dataset, device=args.device,
        stabilize=50, seed=args.seed,
    )

    results, histories = [], {}
    for run_name in selected:
        seed_everything(args.seed)
        train_loader, test_loader = _image_loaders(
            args.dataset, image_size, args.batch_size, root=args.data_root, augment=not args.no_augment,
            seed=args.seed, train_subset=args.train_subset, test_subset=args.test_subset,
        )
        name, model, tcfg = build_run(run_name, all_runs[run_name], args, tcfg_base)
        n_params = count_params(model)
        print(f"\n=== {name} ({n_params:,} params) ===")
        hist = fit(model, train_loader, test_loader, tcfg, device=device)
        row = {
            "name": name,
            "model": args.model,
            "dataset": args.dataset,
            "params": n_params,
            "final_acc": hist["final_acc"],
            "final_ortho_err": hist["ortho_err"][-1],
            "steps_per_sec": hist["steps_per_sec"][-1],
            "peak_mem_mb": hist.get("peak_mem_mb", float("nan")),
            **final_enc(hist),
        }
        try:
            seed_everything(args.seed)
            mem_loader, _ = _image_loaders(
                args.dataset, image_size, args.batch_size, root=args.data_root, augment=not args.no_augment,
                seed=args.seed, train_subset=args.train_subset, test_subset=args.test_subset,
            )
            mem_batch = next(iter(mem_loader))
            mem = measure_breakdown(model, tcfg, mem_batch, device=device)
            row.update({k: mem[k] for k in _MEM_KEYS})
        except Exception as e:
            print(f"  [memory] breakdown failed: {e}")
            row.update({k: float("nan") for k in _MEM_KEYS})
        results.append(row)
        histories[name] = hist

    print("\n" + "=" * 96)
    print(f"  {'run':<18} {'params':>12} {'acc':>8} {'ortho_err':>11} {'ENC_L':>7} {'ENC_R':>7} {'it/s':>7}")
    print("-" * 96)
    for r in results:
        print(f"  {r['name']:<18} {r['params']:>12,} {r['final_acc']:>8.3%} "
              f"{r['final_ortho_err']:>11.2e} {r.get('ENC_L', float('nan')):>7.3f} "
              f"{r.get('ENC_R', float('nan')):>7.3f} {r['steps_per_sec']:>7.2f}")
    print("=" * 96)

    print("\n  Memory utilization (MB)")
    print(f"  {'run':<18} {'weight':>9} {'activation':>11} {'gradient':>9} {'optimizer':>10} {'total':>9}")
    print("-" * 72)
    for r in results:
        w, a = r.get("mem_weight_mb", float("nan")), r.get("mem_activation_mb", float("nan"))
        g, o = r.get("mem_grad_mb", float("nan")), r.get("mem_optim_mb", float("nan"))
        tot = sum(v for v in (w, a, g, o) if v == v)
        print(f"  {r['name']:<18} {w:>9.3f} {a:>11.3f} {g:>9.3f} {o:>10.3f} {tot:>9.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"torchvision_{args.model}_{args.dataset}_quick" if args.quick
           else f"torchvision_{args.model}_{args.dataset}_e{args.epochs}_J{args.J}_r{args.r}")
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
