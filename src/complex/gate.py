"""Content gate (agent_guide §2.2).

The gate reads the LEFT frame U (not V): the input is read by X @ U_j, and the gate
energy reuses the forward-pass quantity H_j = X U_j. Internal convention is stacked
(J, B); callers expose (B, J) at public boundaries if needed.

Normalization XOR (§0.5): softmax normalizes over j (prefactor c = 1); every other phi
is non-normalized (c = 1/J). c is derived from phi in config.LayerConfig.c, so the two
can never be combined.
"""

from __future__ import annotations

import torch

from .config import GateConfig


def gate_energy(
    X: torch.Tensor,
    U: torch.Tensor,
    granularity: str = "sample",
    *,
    H: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Gate energy u (agent_guide §2.2).

    X: (B, n), U: (J, n, r). Returns u: (J, B), each entry in [0, 1].
    If H = X U_j (J, B, r) is already computed in the forward, pass it to avoid recompute.

    sample mode: u_{j,b} = ||x_b U_j||^2 / ||x_b||^2
    batch  mode: u_{j}   = ||X U_j||_F^2 / ||X||_F^2   (broadcast over b)
    """
    if H is None:
        H = torch.einsum("bn,jnr->jbr", X, U)          # (J, B, r)
    if granularity == "sample":
        num = (H * H).sum(dim=2)                        # (J, B)
        den = (X * X).sum(dim=1).clamp_min(eps)         # (B,)
        return num / den.unsqueeze(0)                   # (J, B)
    elif granularity == "batch":
        num = (H * H).sum(dim=(1, 2))                   # (J,)
        den = (X * X).sum().clamp_min(eps)              # scalar
        u = (num / den).unsqueeze(1)                    # (J, 1)
        return u.expand(-1, X.shape[0])                 # (J, B)
    else:
        raise ValueError(f"bad granularity {granularity!r}")


def gate_phi(
    u: torch.Tensor,
    cfg: GateConfig,
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the nonlinearity phi to the energy u (J, B). Returns same shape."""
    kind = cfg.phi
    if kind == "linear":
        return u
    if kind == "exp":
        return u.exp()
    if kind == "pow":
        return u.clamp_min(0.0).pow(cfg.gamma)
    if kind == "sigmoid":
        assert alpha is not None and beta is not None, "sigmoid phi needs alpha, beta"
        return torch.sigmoid(alpha * u + beta)
    if kind == "softmax":
        return torch.softmax(u, dim=0)                  # normalize over components j
    raise ValueError(f"bad phi {kind!r}")


def gate_normalize(phi_u: torch.Tensor, cfg: GateConfig) -> tuple[torch.Tensor, float]:
    """Return (g, c). XOR rule: softmax -> c=1; otherwise c=1/J. No double-counting."""
    J = phi_u.shape[0]
    c = 1.0 if cfg.is_normalized else 1.0 / J
    # Defensive: c must agree with whether phi self-normalizes.
    assert (c == 1.0) == cfg.is_normalized, "normalization XOR violated"
    return phi_u, c


def compute_gate(
    X: torch.Tensor,
    U: torch.Tensor,
    cfg: GateConfig,
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    *,
    H: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Full gate pipeline. Returns (g, c) with g of shape (J, B).

    If cfg.disabled, the gate is forced to all-ones (f == 1) and c is still derived from
    the XOR rule (c = 1/J for the non-normalized default), matching the init-variance test.
    If cfg.detach, g is detached before being returned (removes the gate gradient path).
    """
    J = U.shape[0]
    B = X.shape[0]
    if cfg.disabled:
        g = torch.ones(J, B, device=X.device, dtype=X.dtype)
        c = 1.0 if cfg.is_normalized else 1.0 / J
        return g, c
    u = gate_energy(X, U, cfg.granularity, H=H)
    phi_u = gate_phi(u, cfg, alpha, beta)
    g, c = gate_normalize(phi_u, cfg)
    if cfg.detach:
        g = g.detach()
    return g, c
