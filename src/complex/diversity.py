"""Subspace-diversity penalty via density-matrix von Neumann entropy (agent_guide §2.5-2.6).

We never form the n x n density matrix. Instead we work with the small Jr x Jr Gram
G = (1/J) U_cols^T U_cols (U_cols = [U_1 ... U_J] in R^{n x Jr}), whose nonzero eigenvalues
match those of the n x n density matrix. eigvalsh runs on CPU (MPS gap, and its backward is
more robust there). Entropy bounds for Jr <= n: S in [log r, log(Jr)], so ENC = exp(S)/r in [1, J].

Numerical risk (§2.5): at init the Jr eigenvalues are all ~1/(Jr) -> near-degenerate, and
eigvalsh's backward has 1/(lambda_i - lambda_j) terms. Mitigation: the eps floor + CPU eig;
a closed-form fallback lives in grads_reference.diversity_closed_form.
"""

from __future__ import annotations

import torch

from .device import needs_cpu_linalg


def stack_frames(U: torch.Tensor) -> torch.Tensor:
    """(J, n, r) -> (n, J*r): columns are [U_1 | U_2 | ... | U_J]."""
    J, n, r = U.shape
    return U.permute(1, 0, 2).reshape(n, J * r)


def gram(U_cols: torch.Tensor, J: int) -> torch.Tensor:
    """(1/J) U_cols^T U_cols, shape (Jr, Jr). Trace = r for orthonormal frames.

    Forced fp32: this matmul feeds eigvalsh (no bf16 kernel + ill-conditioned backward at init).
    diversity_loss() is called inside the bf16 autocast region in train_epoch, which would otherwise
    downcast this `@`. autocast(enabled=False) re-enables fp32 and .float() guards bf16 inputs. For
    fp32 callers (every existing test/run) .float() is a no-op -> byte-identical; harmless off-CUDA.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        Uc = U_cols.float()
        return (Uc.transpose(-1, -2) @ Uc) / J


def von_neumann(U: torch.Tensor, J: int, r: int, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S, ENC) for one stacked frame. S is the von Neumann entropy, ENC = exp(S)/r.

    eigvalsh is computed on CPU (so its backward runs on CPU too); the scalar results are
    moved back to U's device.
    """
    n = U.shape[1]
    U_cols = stack_frames(U)                                # (n, Jr)
    G = gram(U_cols, J)                                     # (Jr, Jr)
    # MPS has no eigvalsh kernel -> CPU there (its backward is also more robust on CPU); native
    # on CUDA/CPU so we stay on-device and avoid a host round-trip + sync. See device.needs_cpu_linalg.
    work = G.cpu() if needs_cpu_linalg(G.device) else G
    lam = torch.linalg.eigvalsh(work)                       # (Jr,) ascending
    p = lam.clamp_min(eps)
    p = p / p.sum()                                         # unit trace
    S = -(p * p.log()).sum()                                # scalar (CPU)
    ENC = (S.exp() / r)
    return S.to(U.device), ENC.to(U.device)


def diversity_penalty(U: torch.Tensor, V: torch.Tensor, J: int, r: int, eps: float = 1e-12) -> dict:
    """Both-sides diversity. Returns {S_L, S_R, ENC_L, ENC_R, D} with D = -(S_L + S_R).

    Total training loss is L_pred + lambda_div * D; minimizing D maximizes entropy (spreads
    the subspaces apart). All values carry gradients (autograd through eigvalsh on CPU).

    This per-layer form is kept for diagnostics/tests (it also yields ENC). The TRAINING loss
    sums D over many layers via summed_diversity() below, which batches the eigvalsh.
    """
    S_L, ENC_L = von_neumann(U, J, r, eps)
    S_R, ENC_R = von_neumann(V, J, r, eps)
    D = -(S_L + S_R)
    return {"S_L": S_L, "S_R": S_R, "ENC_L": ENC_L, "ENC_R": ENC_R, "D": D}


def summed_diversity(frames: list[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Scalar D = -(sum of von Neumann entropies over `frames`), for the training loss.

    MATHEMATICALLY IDENTICAL to `-sum_i von_neumann(frame_i)` (the (1/J) Gram scaling cancels in
    the unit-trace normalization, so the per-frame entropy is unchanged). The only difference is a
    PERFORMANCE one for M1: instead of one CPU `eigvalsh` + one MPS->CPU `.cpu()` sync per frame
    (72 of each for a depth-6 ViT), we stack all the equally-sized (Jr x Jr) Grams and do ONE
    batched `eigvalsh` after ONE sync. Same numbers, far fewer fallback round-trips.

    `frames`: stacked frames (each (J, n_i, r)); n_i may differ but all must share J*r (true by
    construction -- a model uses one (J, r) for all its wss layers). A defensive per-frame path
    handles any mixed-size case without changing results.
    """
    if not frames:
        raise ValueError("summed_diversity() needs at least one frame")
    dev = frames[0].device
    grams = [gram(stack_frames(U), U.shape[0]) for U in frames]   # each (Jr, Jr); /J cancels below
    sizes = {g.shape[-1] for g in grams}
    # eigvalsh is CPU-only on MPS (no kernel; backward more robust); native on CUDA/CPU, so there
    # we keep it on-device -- removing the per-step GPU->CPU->GPU sync. See device.needs_cpu_linalg.
    cpu_linalg = needs_cpu_linalg(dev)
    if len(sizes) == 1:
        g_stack = torch.stack(grams, dim=0)                       # (N, Jr, Jr)
        work = g_stack.cpu() if cpu_linalg else g_stack
        lam = torch.linalg.eigvalsh(work)                         # (N, Jr) -- ONE batched eig
        p = lam.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True)                       # per-frame unit trace
        S = -(p * p.log()).sum(dim=-1).sum()                      # scalar
        return (-S).to(dev)
    # Mixed Jr (not produced by the current models): fall back to per-frame, same math.
    total = torch.zeros((), device="cpu" if cpu_linalg else dev)
    for g in grams:
        work = g.cpu() if cpu_linalg else g
        lam = torch.linalg.eigvalsh(work)
        p = lam.clamp_min(eps); p = p / p.sum()
        total = total - (p * p.log()).sum()
    return (-total).to(dev)
