"""Memory-utilization breakdown for scaling analysis (Phase 2.5 tooling).

Four separate datapoints, the ones that matter for scaling decisions:
  * weight     -- model parameters (exact, from tensor shapes; wss split into U/V/spectrum/bias)
  * gradient   -- .grad buffers after a backward (exact; == trainable param bytes)
  * optimizer  -- optimizer state (exact, read from opt.state[p]). Both RiemannianAdam and Adam
                  keep exp_avg + exp_avg_sq per param, so this is ~2x weights.
  * activation -- intermediates retained for backward. EMPIRICAL via current_allocated_memory()
                  snapshots around the forward (MPS/CUDA), cross-checked against an ANALYTIC
                  per-layer model parametrized by batch size B (the real scaling signal).

This module is called ONCE per run (never in the hot training loop), so it has no impact on
training throughput. The numerics core is not touched: we reuse build_optimizers/partition_params
from train.py and read shapes off the existing modules.

Note (MPS): torch 2.3.1 MPS exposes only current_allocated_memory()/driver_allocated_memory()
-- there is no peak/reset API -- so the empirical activation number is a best-effort live-alloc
delta. The analytic estimate is the device-independent figure to extrapolate from.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from .superposition import SuperpositionLinear
from .train import build_optimizers

_MB = 1.0e6


def _bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _elem_size(model: nn.Module) -> int:
    for p in model.parameters():
        return p.element_size()
    return 4


# ── weights ──────────────────────────────────────────────────────────────────────
def param_breakdown(model: nn.Module) -> dict:
    """Exact parameter bytes (MB), split by role. wss frames split into U/V/spectrum/bias."""
    cats: dict[str, float] = defaultdict(float)
    for mod in model.modules():
        if isinstance(mod, SuperpositionLinear):
            cats["U"] += _bytes(mod.U)
            cats["V"] += _bytes(mod.V)
            cats["spectrum"] += _bytes(mod.spectrum.s)
            if mod.bias is not None:
                cats["bias"] += _bytes(mod.bias)
            if mod.gate_alpha is not None:
                cats["gate_scalars"] += _bytes(mod.gate_alpha) + _bytes(mod.gate_beta)
        elif isinstance(mod, nn.Linear):
            cats["dense_weight"] += _bytes(mod.weight)
            if mod.bias is not None:
                cats["bias"] += _bytes(mod.bias)
        elif isinstance(mod, nn.Conv2d):              # ViT patch embed (always dense)
            cats["conv_weight"] += _bytes(mod.weight)
            if mod.bias is not None:
                cats["bias"] += _bytes(mod.bias)
        elif isinstance(mod, nn.LayerNorm):           # ViT norms
            if mod.weight is not None:
                cats["norm"] += _bytes(mod.weight)
            if mod.bias is not None:
                cats["norm"] += _bytes(mod.bias)
    total = sum(_bytes(p) for p in model.parameters() if p.requires_grad)
    # Reconciliation bucket: anything not categorized above (e.g. ViT cls_token / pos_embed bare
    # nn.Parameters). Guarantees categories sum to total for any model (== 0 for the MLP).
    other = total - sum(cats.values())
    if other > 1e-9:
        cats["other"] = other
    out = {k: v / _MB for k, v in cats.items()}
    out["total"] = total / _MB
    return out


# ── gradients ────────────────────────────────────────────────────────────────────
def grad_bytes(model: nn.Module) -> float:
    """Bytes (MB) of populated .grad buffers (call AFTER a backward)."""
    return sum(_bytes(p.grad) for p in model.parameters() if p.grad is not None) / _MB


# ── optimizer state ────────────────────────────────────────────────────────────────
def optimizer_breakdown(opts) -> dict:
    """Exact optimizer-state bytes (MB), summed across optimizers and split by buffer name."""
    by_key: dict[str, float] = defaultdict(float)
    for opt in opts:
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    by_key[k] += _bytes(v)
    out = {k: v / _MB for k, v in by_key.items()}
    out["total"] = sum(out.values())
    return out


# ── activations (analytic scaling model) ───────────────────────────────────────────
def activation_elems_analytic(model: nn.Module, B: int) -> tuple[int, list[int]]:
    """Dominant retained-for-backward element count vs batch B (the scaling signal).

    Per layer:
      * SuperpositionLinear (n->m, J, r): H + HgS + Y + g + out = 2*J*B*r + J*B*m + J*B + B*m
      * nn.Linear (n->m):                 out = B*m
      * + ReLU output (B*width) after each hidden layer
      * + the (flattened) network input B*n0, counted once
    This sums the major intermediates; it is a scaling estimate, not an allocator-exact figure.
    """
    layers = list(model.layers)
    total = 0
    per_layer: list[int] = []
    first = layers[0]
    in0 = first.in_dim if isinstance(first, SuperpositionLinear) else first.in_features
    total += B * in0
    for i, l in enumerate(layers):
        if isinstance(l, SuperpositionLinear):
            J, r, m = l.J, l.r, l.out_dim
            e = 2 * J * B * r + J * B * m + J * B + B * m
            width = m
        elif isinstance(l, nn.Linear):
            m = l.out_features
            e = B * m
            width = m
        else:
            e = 0
            width = 0
        if i < len(layers) - 1:
            e += B * width  # relu output
        per_layer.append(e)
        total += e
    return total, per_layer


def activation_bytes_analytic(model: nn.Module, B: int, elem_size: int | None = None) -> float:
    # The analytic activation model is MLP-specific (walks model.layers). For other topologies
    # (e.g. the ViT, which has .blocks) we have no closed-form estimate yet -> return nan and rely
    # on the empirical live-alloc number. (A ViT-specific analytic model is deferred.)
    if not hasattr(model, "layers"):
        return float("nan")
    elem_size = elem_size or _elem_size(model)
    total, _ = activation_elems_analytic(model, B)
    return total * elem_size / _MB


# ── empirical helpers ──────────────────────────────────────────────────────────────
def _allocated(device: torch.device) -> float | None:
    """Live allocated bytes on the device, or None if no counter is available (CPU)."""
    try:
        if device.type == "mps":
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            return float(torch.mps.current_allocated_memory())
        if device.type == "cuda":
            torch.cuda.synchronize()
            return float(torch.cuda.memory_allocated())
    except Exception:
        return None
    return None


# ── orchestrator ───────────────────────────────────────────────────────────────────
def measure_breakdown(model: nn.Module, tcfg, batch, device=None, lambda_div: float | None = None) -> dict:
    """Full weight/activation/gradient/optimizer breakdown (MB) for one representative batch.

    Exact for weights/grads/optimizer (shapes + opt.state). Activation is empirical (live-alloc
    delta around the forward) on MPS/CUDA, with the analytic model as the value on CPU and always
    reported alongside for cross-check. Runs one fwd/bwd/step -- call after training/eval, since
    it leaves .grad populated and advances the optimizer by one step on the passed batch.
    """
    from .device import get_device
    device = device or get_device(tcfg.device)
    model = model.to(device)
    x, y = batch
    x, y = x.to(device), y.to(device)
    B = x.shape[0]
    lam = tcfg.lambda_div if lambda_div is None else lambda_div

    pb = param_breakdown(model)
    weight_mb = pb["total"]
    analytic_act = activation_bytes_analytic(model, B, _elem_size(model))

    opts = build_optimizers(model, tcfg)
    for o in opts:
        o.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()

    base = _allocated(device)
    logits = model(x)
    loss = criterion(logits, y) + lam * model.diversity_loss()
    after_fwd = _allocated(device)
    loss.backward()
    after_bwd = _allocated(device)

    grad_mb = grad_bytes(model)
    for o in opts:
        o.step()
    ob = optimizer_breakdown(opts)
    optim_mb = ob["total"]

    if base is not None and after_fwd is not None:
        empirical_act = max(0.0, (after_fwd - base) / _MB)
        act_mb = empirical_act
        source = f"empirical-{device.type}"
    else:
        empirical_act = float("nan")
        act_mb = analytic_act
        source = "analytic-cpu"

    return {
        "batch_size": B,
        "mem_weight_mb": weight_mb,
        "mem_activation_mb": act_mb,
        "mem_grad_mb": grad_mb,
        "mem_optim_mb": optim_mb,
        "mem_activation_analytic_mb": analytic_act,
        "mem_activation_empirical_mb": empirical_act,
        "mem_total_mb": weight_mb + act_mb + grad_mb + optim_mb,
        "activation_source": source,
        "param_breakdown_mb": pb,
        "optim_breakdown_mb": ob,
    }
