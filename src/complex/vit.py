"""Tiny pre-norm Vision Transformer whose projections are WSS layers (Phase 3).

Architecture mirrors a standard ViT (Conv2d patch embed, learnable cls + pos embed, pre-norm
blocks `x = x + attn(norm1 x); x = x + mlp(norm2 x)`, head on the cls token). The attention
Q/K/V/O and the MLP fc1/fc2 are built through `make_proj`, so a single `layer_type`
(dense / single_rank_Jr / wss) drives every factorized projection -- the same three-way matched
comparison as the MNIST MLP. The patch-embed Conv2d and the classification head stay dense.

`attn_type` selects the attention *module* independently of `layer_type`, so a WSS MLP can be
paired with a conventional dense attention block. Future attention variants (fused QKV, the
gate-folded "idea 1") are reserved as `attn_type` values + commented seams in superposition.py.

train.fit() consumes `model.diversity_loss()` / `model.diagnostics()` (provided by the mixin)
and `partition_params` already splits ManifoldParameters across all submodules -- so the existing
two-optimizer training loop works on this model with NO changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ViTConfig
from .models import _min_principal_angle
from .superposition import SuperpositionLinear, SuperpositionMultiHeadAttn, WssTrungLinear, make_proj


# ── shared WSS diagnostics (walks all submodules; ViT WSS layers are nested in blocks) ──────────
class _SuperpositionDiagnosticsMixin:
    def _named_wss(self) -> list[tuple[str, nn.Module]]:
        return [(name, m) for name, m in self.named_modules()
                if isinstance(m, (SuperpositionLinear, WssTrungLinear)) and m.J > 1]

    def diversity_loss(self) -> torch.Tensor:
        """Sum of D = -(S_L + S_R) over all wss layers (J>1). 0 if none (dense / single_rank).

        Uses summed_diversity (batched eigvalsh): one CPU sync for all frames instead of one per
        frame -- a big deal for a ViT with 4-6 wss layers/block. Math is identical to per-layer."""
        from .diversity import summed_diversity
        wss = [m for _, m in self._named_wss()]
        if not wss:
            return torch.zeros((), device=next(self.parameters()).device)
        if all(isinstance(m, SuperpositionLinear) for m in wss):
            return summed_diversity([m.U for m in wss] + [m.V for m in wss])
        return torch.stack([m.diversity()["D"] for m in wss]).sum()

    @torch.no_grad()
    def diagnostics(self) -> dict:
        """Per-wss-layer ENC_L/ENC_R + min principal angle, keyed by module path (e.g.
        'blocks.0.attn.q_proj'). Key names are arbitrary -- the headline plot/final_enc are
        key-agnostic (they iterate values())."""
        out = {}
        for name, m in self._named_wss():
            d = m.diversity()
            U = m.U.detach() if isinstance(m, SuperpositionLinear) else m.diversity_frames()[0].detach()
            out[name] = {
                "ENC_L": d["ENC_L"].item(),
                "ENC_R": d["ENC_R"].item(),
                "min_principal_angle": _min_principal_angle(U),
            }
        return out


def build_attention(cfg: ViTConfig) -> nn.Module:
    """Select the attention module from cfg.attn_type. Only the first two are built; the rest are
    reserved seams (see SuperpositionMultiHeadAttn) so future variants drop in here."""
    if cfg.attn_type == "wss_separate":
        return SuperpositionMultiHeadAttn(cfg.dim, cfg.heads, cfg.layer_type,
                                          J=cfg.J, r=cfg.r, use_bias=cfg.use_bias, gate=cfg.gate,
                                          stiefel_canonical=cfg.stiefel_canonical,
                                          retraction_method=cfg.retraction_method,
                                          retract_every=cfg.retract_every,
                                          attn_dropout=cfg.attn_dropout)
    if cfg.attn_type == "dense":
        # Conventional MHA: force dense projections regardless of the MLP's layer_type, so a WSS
        # MLP can be paired with standard attention (the compatibility check). Same attention math.
        return SuperpositionMultiHeadAttn(cfg.dim, cfg.heads, "dense",
                                          J=cfg.J, r=cfg.r, use_bias=cfg.use_bias, gate=cfg.gate,
                                          stiefel_canonical=cfg.stiefel_canonical,
                                          retraction_method=cfg.retraction_method,
                                          retract_every=cfg.retract_every,
                                          attn_dropout=cfg.attn_dropout)
    raise NotImplementedError(
        f"attn_type {cfg.attn_type!r} is a reserved seam (fused-QKV / gate-folded 'idea 1'); "
        "see the commented variants in superposition.SuperpositionMultiHeadAttn.")


class ViTBlock(nn.Module):
    """Pre-norm transformer block: x = x + attn(norm1 x); x = x + mlp(norm2 x)."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        d, hidden = cfg.dim, cfg.hidden_dim
        self.norm1 = nn.LayerNorm(d)
        self.attn = build_attention(cfg)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = make_proj(cfg.layer_type, d, hidden, J=cfg.J, r=cfg.r, use_bias=cfg.use_bias,
                             gate=cfg.gate, stiefel_canonical=cfg.stiefel_canonical,
                             retraction_method=cfg.retraction_method, retract_every=cfg.retract_every)
        self.fc2 = make_proj(cfg.layer_type, hidden, d, J=cfg.J, r=cfg.r, use_bias=cfg.use_bias,
                             gate=cfg.gate, stiefel_canonical=cfg.stiefel_canonical,
                             retraction_method=cfg.retraction_method, retract_every=cfg.retract_every)
        self.act = nn.GELU()
        # p=0.0 (default) => identity -> faithful no-op. Standard ViT places dropout after the activation.
        self.mlp_drop = nn.Dropout(cfg.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, N, d)
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.mlp_drop(self.act(self.fc1(self.norm2(x)))))
        return x


class ViT(_SuperpositionDiagnosticsMixin, nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.dim
        # Patch embed (DENSE Conv2d) -> (B, d, H/p, W/p); flatten to tokens.
        self.patch_embed = nn.Conv2d(cfg.in_chans, d, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.num_classes)              # DENSE always (out_dim < J*r)
        if cfg.init_scale != 1.0:
            self._apply_init_scale(cfg.init_scale)

    def _apply_init_scale(self, scale: float) -> None:
        """NON-FAITHFUL stabilization knob (default off / scale==1.0 -> never called).

        The WSS contract inits spectra at sigma0 = sqrt(2*J*m/r) (He fan-in for a ReLU MLP). In a
        pre-norm GELU residual transformer that can be hot. This uniformly rescales every factorized
        projection's effective gain by `scale` (WSS: sigma *= scale via s += log(scale); dense:
        weight *= scale). Use ONLY to probe stability -- it deviates from the faithful init and any
        result obtained with scale != 1.0 must be reported as such.
        """
        with torch.no_grad():
            log_s = math.log(scale)
            for m in self.modules():
                if isinstance(m, SuperpositionLinear):
                    m.spectrum.s.add_(log_s)
                elif isinstance(m, WssTrungLinear):
                    factor_scale = math.sqrt(scale)
                    m.L.mul_(factor_scale)
                    m.R.mul_(factor_scale)
                elif isinstance(m, nn.Linear) and m is not self.head:
                    m.weight.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # (B, n_patches, d)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)   # (B, seq, d)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                              # classify cls token -> (B, num_classes)
