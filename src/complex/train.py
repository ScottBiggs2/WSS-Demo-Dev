"""Training loop and optimizer wiring (agent_guide §3.6, §4.4).

Two parameter groups:
  * Stiefel frames (U, V) -> geoopt.optim.RiemannianAdam  (retraction + first-moment transport)
  * Euclidean (s, bias, gate scalars, dense layers) -> torch.optim.Adam

Remark-8 control (tcfg.retraction=False): the Stiefel frames are updated by PLAIN torch SGD
on their raw .data (no projection, no retraction). This is expected to visibly degrade
orthonormality and accuracy -- the proof that the manifold step is load-bearing (the failure
mode of src/simple/).

NOTE (deferred seam, agent_guide §6): RiemannianAdam transports the first moment (exp_avg)
but NOT the second moment (exp_avg_sq). The principled fix is an LDAdam-style projection-aware
second-moment correction; intentionally NOT implemented in Phases 1-2.
"""

from __future__ import annotations

import time

import geoopt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import TrainConfig
from .manifold import orthonormality_error


def partition_params(model: nn.Module) -> tuple[list, list]:
    """Split into (stiefel ManifoldParameters, euclidean params)."""
    stiefel, euclid = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, geoopt.ManifoldParameter):
            stiefel.append(p)
        else:
            euclid.append(p)
    return stiefel, euclid


def build_optimizers(model: nn.Module, tcfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Return the optimizer(s) to step each batch. Honors the retraction toggle."""
    stiefel, euclid = partition_params(model)
    opts: list[torch.optim.Optimizer] = []
    if stiefel:
        if tcfg.retraction:
            # Need to consider if weight decay is helpful here, it probably is
            opts.append(geoopt.optim.RiemannianAdam(stiefel, lr=tcfg.lr_riemann, stabilize=tcfg.stabilize))
        else:
            # Remark-8 control: no retraction -- plain Euclidean SGD on the manifold params.
            opts.append(torch.optim.SGD(stiefel, lr=tcfg.lr_riemann))
    if euclid:
        # Note: This is not a fair comparison because the geopt Reimannian Adam loses V_t, while dense keeps it.
        opts.append(torch.optim.Adam(euclid, lr=tcfg.lr_euclid))
        
        # After review, Adam is the correct comparison point.
        # We just need to be aware that euclidean adam has much stronger preconditioning than Reimannian adam
        # opts.append(torch.optim.SGD(euclid, lr=tcfg.lr_euclid, momentum = 0.9))
        # opts.appaned(torch.optim.RMSProp(euclid, lr = tcfg.lr_euclid, momentum = 0.9))

    return opts


def _peak_memory(device: torch.device) -> float:
    """Peak allocated memory in MB (MPS), best-effort."""
    try:
        if device.type == "mps":
            return torch.mps.current_allocated_memory() / 1e6
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return float("nan")


def train_epoch(model, loader, opts, lambda_div, device, criterion, *, epoch: int | None = None, total_epochs: int | None = None, progress: bool = True) -> dict:
    model.train()
    total_loss = total_div = 0.0
    n_seen = 0
    t0 = time.perf_counter()
    n_batches = 0
    iterator = loader
    if progress:
        desc = f"epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "train"
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        for o in opts:
            o.zero_grad()
        logits = model(x)
        ce = criterion(logits, y)
        div = model.diversity_loss()
        loss = ce + lambda_div * div
        loss.backward()
        for o in opts:
            o.step()
        bs = x.size(0)
        total_loss += ce.item() * bs
        total_div += float(div) * bs
        n_seen += bs
        n_batches += 1
        if progress:
            iterator.set_postfix(loss=total_loss / n_seen, div=total_div / n_seen)
    dt = time.perf_counter() - t0
    return {
        "train_loss": total_loss / n_seen,
        "train_div": total_div / n_seen,
        "steps_per_sec": n_batches / dt if dt > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    correct = 0
    total_loss = 0.0
    n_seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n_seen += x.size(0)
    return {"test_acc": correct / n_seen, "test_loss": total_loss / n_seen}


def _orthonormality(model) -> float:
    stiefel, _ = partition_params(model)
    if not stiefel:
        return 0.0
    return orthonormality_error(*[p.detach() for p in stiefel])


def fit(model, train_loader, test_loader, tcfg: TrainConfig, device=None, verbose=True) -> dict:
    from .device import get_device
    device = device or get_device(tcfg.device)
    model = model.to(device)
    opts = build_optimizers(model, tcfg)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "test_acc": [], "test_loss": [],
               "ortho_err": [], "steps_per_sec": [], "diagnostics": []}
    if verbose:
        print(f"  {'epoch':>5} {'tr_loss':>8} {'test_acc':>9} {'ortho':>9} {'it/s':>7}")
    for epoch in range(1, tcfg.epochs + 1):
        tr = train_epoch(model, train_loader, opts, tcfg.lambda_div, device, criterion)
        ev = evaluate(model, test_loader, device, criterion)
        ortho = _orthonormality(model)
        diag = model.diagnostics() if hasattr(model, "diagnostics") else {}
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["train_loss"])
        history["test_acc"].append(ev["test_acc"])
        history["test_loss"].append(ev["test_loss"])
        history["ortho_err"].append(ortho)
        history["steps_per_sec"].append(tr["steps_per_sec"])
        history["diagnostics"].append(diag)
        if verbose:
            print(f"  {epoch:>5} {tr['train_loss']:>8.4f} {ev['test_acc']:>9.3%} "
                  f"{ortho:>9.2e} {tr['steps_per_sec']:>7.1f}")
    history["peak_mem_mb"] = _peak_memory(device)
    history["final_acc"] = history["test_acc"][-1]
    return history


def smoke_train_step(model, X, y, tcfg: TrainConfig, n_steps: int = 20, device=None) -> list[float]:
    """Run n_steps on a single fixed batch; return the loss trajectory (should decrease)."""
    from .device import get_device
    device = device or get_device(tcfg.device)
    model = model.to(device)
    X, y = X.to(device), y.to(device)
    opts = build_optimizers(model, tcfg)
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(n_steps):
        for o in opts:
            o.zero_grad()
        loss = criterion(model(X), y) + tcfg.lambda_div * model.diversity_loss()
        loss.backward()
        for o in opts:
            o.step()
        losses.append(loss.item())
    return losses
