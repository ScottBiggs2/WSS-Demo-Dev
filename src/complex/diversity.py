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


def stack_frames(U: torch.Tensor) -> torch.Tensor:
    """(J, n, r) -> (n, J*r): columns are [U_1 | U_2 | ... | U_J]."""
    J, n, r = U.shape
    return U.permute(1, 0, 2).reshape(n, J * r)


def gram(U_cols: torch.Tensor, J: int) -> torch.Tensor:
    """(1/J) U_cols^T U_cols, shape (Jr, Jr). Trace = r for orthonormal frames."""
    return (U_cols.transpose(-1, -2) @ U_cols) / J


def von_neumann(U: torch.Tensor, J: int, r: int, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S, ENC) for one stacked frame. S is the von Neumann entropy, ENC = exp(S)/r.

    eigvalsh is computed on CPU (so its backward runs on CPU too); the scalar results are
    moved back to U's device.
    """
    n = U.shape[1]
    U_cols = stack_frames(U)                                # (n, Jr)
    G = gram(U_cols, J)                                     # (Jr, Jr)
    G_cpu = G.cpu()
    lam = torch.linalg.eigvalsh(G_cpu)                      # (Jr,) ascending, CPU
    p = lam.clamp_min(eps)
    p = p / p.sum()                                         # unit trace
    S = -(p * p.log()).sum()                                # scalar (CPU)
    ENC = (S.exp() / r)
    return S.to(U.device), ENC.to(U.device)


def diversity_penalty(U: torch.Tensor, V: torch.Tensor, J: int, r: int, eps: float = 1e-12) -> dict:
    """Both-sides diversity. Returns {S_L, S_R, ENC_L, ENC_R, D} with D = -(S_L + S_R).

    Total training loss is L_pred + lambda_div * D; minimizing D maximizes entropy (spreads
    the subspaces apart). All values carry gradients (autograd through eigvalsh on CPU).
    """
    S_L, ENC_L = von_neumann(U, J, r, eps)
    S_R, ENC_R = von_neumann(V, J, r, eps)
    D = -(S_L + S_R)
    return {"S_L": S_L, "S_R": S_R, "ENC_L": ENC_L, "ENC_R": ENC_R, "D": D}
