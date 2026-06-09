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

from .config import GateConfig, LayerConfig, ModelConfig
from .superposition import SuperpositionLinear


def make_layer(layer_type: str, in_dim: int, out_dim: int, cfg: ModelConfig) -> nn.Module:
    if layer_type == "dense":
        return nn.Linear(in_dim, out_dim, bias=cfg.use_bias)

    if layer_type == "single_rank_Jr":
        # one component (J=1) of rank J*r, gate forced off (f == 1): the honest control.
        lcfg = LayerConfig(
            in_dim=in_dim, out_dim=out_dim, J=1, r=cfg.J * cfg.r,
            use_bias=cfg.use_bias,
            gate=GateConfig(phi="linear", disabled=True),
        )
        return SuperpositionLinear(lcfg)

    if layer_type == "wss":
        lcfg = LayerConfig(
            in_dim=in_dim, out_dim=out_dim, J=cfg.J, r=cfg.r,
            use_bias=cfg.use_bias,
            gate=cfg.gate,
        )
        return SuperpositionLinear(lcfg)

    raise ValueError(f"unknown layer_type {layer_type!r}")


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
        """Sum of D = -(S_L + S_R) over wss layers (J>1). 0 if none (e.g. dense/single_rank)."""
        wss = self._wss_layers()
        if not wss:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack([l.diversity()["D"] for l in wss]).sum()

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
