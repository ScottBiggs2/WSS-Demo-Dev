"""SuperpositionLinear -- the WSS core layer (agent_guide §2.3, §3.4).

W(X) = c * sum_j (g_j ⊙ (X U_j)) S_j V_j^T + b, with:
  * U_j (n x r), V_j (m x r) on Stiefel manifolds (orthonormal columns),
  * S_j = diag(sigma_j), sigma_j = exp(s_j) > 0,
  * g_j the content gate reading U (gate.py),
  * c the normalization-XOR prefactor (1/J non-normalized, 1 for softmax).

The dense (n x m) weight is NEVER materialized in the forward/backward hot path; only
materialize_weight() builds it, for tests/diagnostics, guarded by an in-forward assertion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import LayerConfig
from .gate import compute_gate
from .manifold import make_stiefel_param
from .spectrum import Spectrum


class SuperpositionLinear(nn.Module):
    def __init__(self, cfg: LayerConfig, *, device=None, dtype=None, generator=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        n, m, J, r = cfg.in_dim, cfg.out_dim, cfg.J, cfg.r
        self.in_dim, self.out_dim, self.J, self.r = n, m, J, r

        # Stiefel frames, stacked (J, n, r) / (J, m, r). One ManifoldParameter each.
        self.U = make_stiefel_param(
            n, r, J, canonical=cfg.stiefel_canonical, device=device, dtype=dtype, generator=generator
        )
        self.V = make_stiefel_param(
            m, r, J, canonical=cfg.stiefel_canonical, device=device, dtype=dtype, generator=generator
        )

        # Spectrum: He fan-in init sigma0 = sqrt(2 J m / r) (agent_guide §2.7, §0.4).
        sigma0 = math.sqrt(2.0 * J * m / r)
        self.spectrum = Spectrum(J, r, sigma0)
        if dtype is not None:
            self.spectrum.s.data = self.spectrum.s.data.to(dtype)
        if device is not None:
            self.spectrum.s.data = self.spectrum.s.data.to(device)

        # Bias (Euclidean), shared across components.
        if cfg.use_bias:
            self.bias = nn.Parameter(torch.zeros(m, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Learnable gate scalars (sigmoid phi only).
        if cfg.gate.phi == "sigmoid":
            self.gate_alpha = nn.Parameter(torch.tensor(cfg.gate.alpha_init, device=device, dtype=dtype))
            self.gate_beta = nn.Parameter(torch.tensor(cfg.gate.beta_init, device=device, dtype=dtype))
        else:
            self.register_parameter("gate_alpha", None)
            self.register_parameter("gate_beta", None)

        self._in_forward = False

    # ── forward ────────────────────────────────────────────────────────────────
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._in_forward = True
        try:
            H = torch.einsum("bn,jnr->jbr", X, self.U)                  # X U_j           (J, B, r)
            g, c = compute_gate(X, self.U, self.cfg.gate, self.gate_alpha, self.gate_beta, H=H)  # (J, B)
            sigma = self.spectrum.sigma()                               # (J, r)
            HgS = (H * g.unsqueeze(-1)) * sigma.unsqueeze(1)            # gate ⊙ then S_j (J, B, r)
            Y = torch.einsum("jbr,jmr->jbm", HgS, self.V)               # ... V_j^T       (J, B, m)
            out = c * Y.sum(dim=0)                                       # (B, m)
            if self.bias is not None:
                out = out + self.bias
            return out
        finally:
            self._in_forward = False

    # ── diagnostics / tests only ─────────────────────────────────────────────────
    def materialize_weight(self, summed: bool = True) -> torch.Tensor:
        """Build the dense weight (TEST/DIAGNOSTIC ONLY -- never called in forward).

        Per-component W_j = U_j S_j V_j^T, shape (J, n, m). If summed, returns the effective
        dense map c * sum_j W_j (with the gate set to f == 1, i.e. ungated), shape (n, m).
        """
        assert not self._in_forward, "materialize_weight() must not be called in the forward hot path"
        sigma = self.spectrum.sigma()                                   # (J, r)
        W = torch.einsum("jnr,jr,jmr->jnm", self.U, sigma, self.V)      # (J, n, m)
        if summed:
            c = self.cfg.c
            return c * W.sum(dim=0)                                      # (n, m)
        return W

    def diversity(self) -> dict:
        """Per-layer diversity diagnostics (delegates to diversity.py to avoid import cycle)."""
        from .diversity import diversity_penalty
        return diversity_penalty(self.U, self.V, self.J, self.r)

    # ── parameter groups for the two-optimizer setup ─────────────────────────────
    def stiefel_params(self) -> list[nn.Parameter]:
        return [self.U, self.V]

    def euclidean_params(self) -> list[nn.Parameter]:
        params = [self.spectrum.s]
        if self.bias is not None:
            params.append(self.bias)
        if self.gate_alpha is not None:
            params += [self.gate_alpha, self.gate_beta]
        return params

    # ── deferred seam (agent_guide §6) -- do NOT implement now ───────────────────
    def maybe_refactor(self) -> None:
        """No-op stub for later global re-factorization.

        Trigger (deferred): when effective number of components drops, e.g.
            if self.diversity()["ENC_L"] < self.J / 2: <re-orthogonalize / merge frames>
        Left intentionally empty for Phases 1-2.
        """
        return None
