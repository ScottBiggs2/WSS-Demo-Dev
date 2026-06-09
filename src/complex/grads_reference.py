"""Closed-form Euclidean gradient oracles -- TEST ORACLE ONLY (agent_guide §2.4, §2.5).

These are NOT the training path. RiemannianAdam consumes U.grad / V.grad from autograd and
does projection + transport + retraction itself. These closed forms exist so test_grads.py
can validate the forward's autograd against analytic gradients (gate detached), and so
diversity.py has a fallback when eigvalsh's backward is ill-conditioned.

Einsum strings verified to ~1e-13 against autograd in float64.
"""

from __future__ import annotations

import torch


def euclidean_grads(
    X: torch.Tensor,        # (B, n)
    U: torch.Tensor,        # (J, n, r)
    V: torch.Tensor,        # (J, m, r)
    sigma: torch.Tensor,    # (J, r)
    g: torch.Tensor,        # (J, B)  realized gate, DETACHED
    c: float,
    delta: torch.Tensor,    # (B, m)  dL/dA_tilde
) -> dict[str, torch.Tensor]:
    """Analytic gradients of the data-fit loss w.r.t. U, V, sigma, s (gate detached).

    Returns {"dU": (J,n,r), "dV": (J,m,r), "dsigma": (J,r), "ds": (J,r)}.
    ds is the gradient w.r.t. the log-spectrum s (chain rule through sigma = exp(s)).
    """
    H = torch.einsum("bn,jnr->jbr", X, U)                 # X U_j      (J, B, r)
    Hg = H * g.unsqueeze(-1)                              # H_tilde_j  (J, B, r)

    # dL/dV_j = c (delta^T H_tilde_j) S_j
    dV = c * torch.einsum("bm,jbr->jmr", delta, Hg) * sigma.unsqueeze(1)        # (J, m, r)

    # dL/dsigma_j = c diag(H_tilde_j^T delta V_j)
    dsigma = c * torch.einsum("jbr,bm,jmr->jr", Hg, delta, V)                   # (J, r)

    # dL/dU_j = c X^T (g_j ⊙ (delta V_j S_j))
    dVS = torch.einsum("bm,jmr->jbr", delta, V) * sigma.unsqueeze(1)            # delta V_j S_j (J,B,r)
    dU = c * torch.einsum("bn,jbr->jnr", X, dVS * g.unsqueeze(-1))              # (J, n, r)

    ds = dsigma * sigma                                                        # d/ds, sigma = exp(s)
    return {"dU": dU, "dV": dV, "dsigma": dsigma, "ds": ds}


def diversity_closed_form(
    U: torch.Tensor,        # (J, n, r)
    J: int,
    r: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Closed-form gradient of the von Neumann entropy S w.r.t. the stacked frame.

    dS/dU_j = -(2/(J r)) (I + log rho_bar) U_j, where rho_bar = (1/(J r)) sum_j U_j U_j^T
    is the (normalized, unit-trace) left density matrix. Returns dS/dU stacked as (J, n, r).

    This is the oracle / fallback for the autograd-through-eigvalsh path; eigvalsh's backward
    is ill-conditioned near eigenvalue degeneracy (true at init), so this gives a stable check.
    """
    n = U.shape[1]
    # rho = (1/(J r)) sum_j U_j U_j^T, unit trace (sum of J*r singular contributions / (J r)).
    rho = torch.einsum("jnr,jms->nm", U, U) / (J * r)              # (n, n), trace ~ 1
    rho = 0.5 * (rho + rho.transpose(-1, -2))                       # symmetrize
    evals, evecs = torch.linalg.eigh(rho.cpu())
    evals = evals.clamp_min(eps)
    log_rho = (evecs * evals.log()) @ evecs.transpose(-1, -2)       # (n, n) = log(rho)
    log_rho = log_rho.to(U.device).to(U.dtype)
    eye = torch.eye(n, device=U.device, dtype=U.dtype)
    M = -(2.0 / (J * r)) * (eye + log_rho)                          # (n, n)
    return torch.einsum("nm,jmr->jnr", M, U)                        # (J, n, r)
