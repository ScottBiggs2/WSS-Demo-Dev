"""Subspace interpretability probe (Phase 2.5 tooling).

Hypothesis (the two obvious low-rank settings):
  * J = 10, very low rank -> components specialize one-per-class ("one-hot"/early classification):
    the gate-weight heatmap should look like a (permuted) identity block -- class c lights subspace c.
  * J = 7,  very low rank -> components act like a 7-segment display: each class lights a
    characteristic SUBSET of the 7 subspaces (a combinatorial code), not a single one.

We test this by visualizing per-component GATE WEIGHTS as a function of input class.

Artifact (user-confirmed layout): a SINGLE aggregate heatmap.
  * y-axis: (network layer, subspace) -- one row-block per wss layer, one subrow per subspace j.
  * x-axis: the dataset classes 0..9.
  * cell:   mean gate weight of that subspace over all test samples of that (true) class.
  * top strip: per-class test accuracy (the correct/incorrect indication).

Gate extraction uses a forward HOOK on each wss layer that recomputes compute_gate() on the
captured layer input -- the numerics core (superposition.py / gate.py) is NOT modified.

Usage (from repo root, inside the venv):
    python src/complex/experiments/subspace_interp.py --J 10 --r 2 --epochs 8
    python src/complex/experiments/subspace_interp.py --J 7  --r 2 --epochs 8 --dataset fmnist
"""

from __future__ import annotations

# MUST precede torch import so the MPS->CPU fallback for qr/solve is active.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from complex.config import GateConfig, ModelConfig, TrainConfig
from complex.data import get_loaders
from complex.device import get_device
from complex.gate import compute_gate
from complex.models import MLP
from complex.superposition import SuperpositionLinear
from complex.train import fit

OUT_DIR = Path(__file__).resolve().parent / "outputs"

_FMNIST_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def _wss_layers_indexed(model):
    """[(layer_index, layer)] for the wss SuperpositionLinear layers (J>1), in order."""
    return [(i, l) for i, l in enumerate(model.layers)
            if isinstance(l, SuperpositionLinear) and l.J > 1]


@torch.no_grad()
def collect_gate_weights(model, loader, device, num_classes=10):
    """Mean gate weight per (wss layer, subspace, true class) over the loader.

    Returns:
        per_layer:  dict {layer_index: tensor (J, num_classes)} of mean gate weights
        class_acc:  tensor (num_classes,) per-class accuracy
        class_count: tensor (num_classes,) sample count per class
    """
    model.eval()
    wss = _wss_layers_indexed(model)
    if not wss:
        raise ValueError("model has no wss layers (J>1) -- nothing to probe")

    # Forward hooks recompute the gate on each wss layer's captured input (no core edit).
    cache: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(mod, inp, _out):
            X = inp[0]
            g, _c = compute_gate(X, mod.U, mod.cfg.gate, mod.gate_alpha, mod.gate_beta)  # (J, B)
            cache[idx] = g.detach()
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in wss]

    sums = {i: torch.zeros(l.J, num_classes, dtype=torch.float64) for i, l in wss}
    class_count = torch.zeros(num_classes, dtype=torch.float64)
    class_correct = torch.zeros(num_classes, dtype=torch.float64)

    try:
        for x, y in loader:
            x = x.to(device)
            logits = model(x)                       # triggers hooks -> fills cache
            preds = logits.argmax(1).cpu()
            yc = y.cpu()
            for cls in range(num_classes):
                mask = (yc == cls)
                n = int(mask.sum())
                if n == 0:
                    continue
                class_count[cls] += n
                class_correct[cls] += int((preds[mask] == cls).sum())
                for i, _l in wss:
                    g = cache[i].cpu().double()     # (J, B)
                    sums[i][:, cls] += g[:, mask].sum(dim=1)
    finally:
        for h in handles:
            h.remove()

    per_layer = {i: (sums[i] / class_count.clamp_min(1.0).unsqueeze(0)) for i, _l in wss}
    class_acc = class_correct / class_count.clamp_min(1.0)
    return per_layer, class_acc, class_count


def verify_hooks_are_readonly(model, x, device):
    """The hooks must not perturb the forward: hooked logits == clean logits."""
    model.eval()
    with torch.no_grad():
        clean = model(x.to(device))
    wss = _wss_layers_indexed(model)
    cache = {}
    handles = [l.register_forward_hook(
        (lambda idx: (lambda mod, inp, _o: cache.__setitem__(
            idx, compute_gate(inp[0], mod.U, mod.cfg.gate, mod.gate_alpha, mod.gate_beta)[0])))(i))
        for i, l in wss]
    try:
        with torch.no_grad():
            hooked = model(x.to(device))
    finally:
        for h in handles:
            h.remove()
    return torch.allclose(clean, hooked, atol=1e-5)


def _render(per_layer, class_acc, J, r, dataset, num_classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layer_ids = sorted(per_layer)
    # stack layers vertically: rows = concat over layers of (J rows each)
    blocks = [per_layer[i].numpy() for i in layer_ids]
    mat = np.concatenate(blocks, axis=0)                    # (sum_J, num_classes)
    row_labels = [f"L{i}.s{j}" for i in layer_ids for j in range(per_layer[i].shape[0])]
    layer_boundaries = np.cumsum([b.shape[0] for b in blocks])[:-1]

    # Deviation from the per-layer uniform baseline (1/J): the near-uniform softmax gate makes
    # raw weights ~1/J everywhere, so absolute color washes out. Centering on uniform reveals
    # which subspaces each class actually prefers (red) or avoids (blue) -- i.e. specialization.
    uniform = np.array([1.0 / per_layer[i].shape[0] for i in layer_ids
                        for _ in range(per_layer[i].shape[0])])[:, None]
    dev = mat - uniform
    vlim = max(float(np.abs(dev).max()), 1e-9)

    classes = _FMNIST_CLASSES if dataset == "fmnist" else [str(c) for c in range(num_classes)]

    fig, (ax_acc, ax) = plt.subplots(
        2, 1, figsize=(1.5 + 0.7 * num_classes, 1.4 + 0.32 * mat.shape[0]),
        gridspec_kw={"height_ratios": [1, max(6, mat.shape[0])], "hspace": 0.04}, sharex=True)

    # top: per-class accuracy strip (the correct/incorrect indication)
    acc = class_acc.numpy()[None, :]
    ax_acc.imshow(acc, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    for c in range(num_classes):
        ax_acc.text(c, 0, f"{acc[0, c]*100:.0f}", ha="center", va="center", fontsize=7)
    ax_acc.set_yticks([0]); ax_acc.set_yticklabels(["acc %"], fontsize=8)
    ax_acc.set_xticks([]); ax_acc.set_title(
        f"Subspace gate weight − uniform(1/J)  ({dataset}, J={J}, r={r}; red=preferred)", fontsize=11)

    # main heatmap: deviation from uniform, diverging + symmetric about 0
    im = ax.imshow(dev, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(classes, fontsize=8,
                                                          rotation=45 if dataset == "fmnist" else 0)
    ax.set_xlabel("class")
    for b in layer_boundaries:
        ax.axhline(b - 0.5, color="black", lw=1.4)
    fig.colorbar(im, ax=[ax_acc, ax], fraction=0.025, pad=0.02, label="gate weight − 1/J")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return mat, row_labels, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int, default=10, help="components (10 -> one-hot, 7 -> 7-segment)")
    ap.add_argument("--r", type=int, default=2, help="rank per component (keep very low)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_div", type=float, default=1e-2)
    ap.add_argument("--dataset", default="mnist", help="mnist | fmnist")
    ap.add_argument("--gate_phi", default="softmax", help="softmax gives normalized importances")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    dims = [784, 256, 128, 10]
    num_classes = 10
    print(f"device={device} | dataset={args.dataset} | J={args.J} r={args.r} "
          f"epochs={args.epochs} gate={args.gate_phi} seed={args.seed}")

    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, seed=args.seed)

    mcfg = ModelConfig(layer_type="wss", dims=dims, J=args.J, r=args.r,
                       gate=GateConfig(phi=args.gate_phi), lambda_div=args.lambda_div)
    model = MLP(mcfg)
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                       lr_riemann=args.lr, lr_euclid=args.lr, lambda_div=args.lambda_div,
                       dataset=args.dataset, device=args.device, stabilize=50, seed=args.seed)

    hist = fit(model, train_loader, test_loader, tcfg, device=device)
    print(f"  final test acc: {hist['final_acc']:.3%}")

    # sanity: the extraction hooks must be read-only
    xb, _ = next(iter(test_loader))
    ok = verify_hooks_are_readonly(model, xb, device)
    print(f"  hooks read-only (hooked logits == clean): {ok}")
    assert ok, "gate-extraction hooks perturbed the forward!"

    per_layer, class_acc, class_count = collect_gate_weights(
        model, test_loader, device, num_classes=num_classes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_J{args.J}_r{args.r}"
    png = OUT_DIR / f"subspace_interp_{tag}.png"
    mat, row_labels, classes = _render(per_layer, class_acc, args.J, args.r,
                                       args.dataset, num_classes, png)

    # persist the matrix + accuracy for downstream analysis
    with open(OUT_DIR / f"subspace_interp_{tag}.json", "w") as f:
        json.dump({
            "dataset": args.dataset, "J": args.J, "r": args.r,
            "final_acc": hist["final_acc"],
            "row_labels": row_labels, "classes": classes,
            "gate_weights": mat.tolist(),
            "per_class_acc": class_acc.tolist(),
            "per_class_count": class_count.tolist(),
        }, f, indent=2)

    # quantitative specialization read-out per wss layer.
    #   selectivity = mean over classes of the effective #subspaces the gate spreads over,
    #   = exp(entropy of the per-class gate distribution). ~1 => one-hot specialization,
    #   ~J => uniform (no specialization). This is the gate analogue of the ENC diagnostic.
    print("  gate selectivity (effective #subspaces/class; 1=one-hot, J=uniform):")
    for i in sorted(per_layer):
        M = per_layer[i]                                  # (J, num_classes)
        p = M / M.sum(0, keepdim=True).clamp_min(1e-12)
        ent = -(p * p.clamp_min(1e-12).log()).sum(0)      # (num_classes,)
        eff = ent.exp().mean().item()
        dom = M.argmax(0).tolist()
        print(f"    layer {i}: eff={eff:.2f}/{M.shape[0]}  "
              f"distinct dominant={len(set(dom))}/{M.shape[0]}  dom_per_class={dom}")
    print(f"\nWrote {png.name} + {png.stem}.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
