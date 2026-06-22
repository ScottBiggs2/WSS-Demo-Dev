"""MLP with three interchangeable layer types (agent_guide §4.1).

  * dense           -- plain nn.Linear (baseline).
  * single_rank_Jr  -- one rank-(J*r) factorization U S V^T, ungated (the J=1 control).
  * wss             -- SuperpositionLinear: J components of rank r, gated + diversity.

Design note: the readout layer (-> num_classes, e.g. 10) cannot carry rank Jr since rank
<= out_dim. We therefore keep the readout DENSE for ALL three model types and factorize
only the wide hidden layers (where Jr <= width). This makes wss and single_rank_Jr exactly
parameter-matched on the layers that matter, sharing an identical dense readout -- the
honest §4.2.1 comparison. (wss J components of rank r vs single_rank 1 component of rank
J*r -> identical frame/spectrum param counts.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig
from .superposition import SuperpositionLinear, make_proj


def make_layer(layer_type: str, in_dim: int, out_dim: int, cfg: ModelConfig) -> nn.Module:
    """Thin adapter from a ModelConfig to the shared make_proj factory (superposition.py).
    Behavior-preserving: dense -> nn.Linear, single_rank_Jr -> rank-J*r ungated, wss -> gated."""
    return make_proj(layer_type, in_dim, out_dim, J=cfg.J, r=cfg.r,
                     use_bias=cfg.use_bias, gate=cfg.gate)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        dims = cfg.dims
        layers: list[nn.Module] = []
        n_layers = len(dims) - 1
        for i in range(n_layers):
            is_readout = (i == n_layers - 1)
            # readout always dense; hidden layers use the configured type
            lt = "dense" if is_readout else cfg.layer_type
            layers.append(make_layer(lt, dims[i], dims[i + 1], cfg))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    # ── superposition-specific helpers ──────────────────────────────────────────
    def _wss_layers(self):
        return [l for l in self.layers if isinstance(l, SuperpositionLinear) and l.J > 1]

    def diversity_loss(self) -> torch.Tensor:
        """Sum of D = -(S_L + S_R) over wss layers (J>1). 0 if none (e.g. dense/single_rank).

        Uses summed_diversity (batched eigvalsh) -- mathematically identical to summing each
        layer's diversity()["D"], but one CPU sync instead of one-per-frame (M1 speedup)."""
        from .diversity import summed_diversity
        wss = self._wss_layers()
        if not wss:
            return torch.zeros((), device=next(self.parameters()).device)
        return summed_diversity([l.U for l in wss] + [l.V for l in wss])

    @torch.no_grad()
    def diagnostics(self) -> dict:
        """Per-wss-layer ENC_L/ENC_R and min principal angle between components (radians)."""
        out = {}
        for idx, l in enumerate(self.layers):
            if not isinstance(l, SuperpositionLinear) or l.J <= 1:
                continue
            d = l.diversity()
            out[f"layer{idx}"] = {
                "ENC_L": d["ENC_L"].item(),
                "ENC_R": d["ENC_R"].item(),
                "min_principal_angle": _min_principal_angle(l.U.detach()),
            }
        return out


def _min_principal_angle(U: torch.Tensor) -> float:
    """Smallest principal angle (radians) between any pair of the J subspaces in U (J,n,r).

    Principal angles between subspaces span(U_j), span(U_k) have cosines = singular values
    of U_j^T U_k. The smallest angle (most-aligned pair) is arccos of the largest such value.
    """
    J = U.shape[0]
    if J < 2:
        return float("nan")
    max_cos = 0.0
    for j in range(J):
        for k in range(j + 1, J):
            s = torch.linalg.svdvals((U[j].transpose(-1, -2) @ U[k]).cpu())
            max_cos = max(max_cos, s.max().item())
    max_cos = min(max_cos, 1.0)
    return math.acos(max_cos)


