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

from .config import GateConfig, LayerConfig
from .gate import compute_gate
from .manifold import haar_init, make_stiefel_param
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
            # Accept any leading shape (..., n). We flatten the leading dims to a single batch
            # axis, run the layer, and reshape back. This is MATHEMATICALLY IDENTICAL to applying
            # the 2D (B, n) layer independently to each row -- the gate energy ||x U_j||^2/||x||^2
            # is per-row, so for a ViT token stream (B, N, dim) the gate fires PER TOKEN. The 2D
            # case (lead == (B,)) round-trips bit-identically, so all MNIST call sites are unchanged.
            lead = X.shape[:-1]                                         # () / (B,) / (B, N) / ...
            n = X.shape[-1]
            Xf = X.reshape(-1, n)                                        # (Bflat, n); reshape (not view) for transposed inputs
            H = torch.einsum("bn,jnr->jbr", Xf, self.U)                 # X U_j           (J, Bflat, r)
            g, c = compute_gate(Xf, self.U, self.cfg.gate, self.gate_alpha, self.gate_beta, H=H)  # (J, Bflat)
            sigma = self.spectrum.sigma()                               # (J, r)
            HgS = (H * g.unsqueeze(-1)) * sigma.unsqueeze(1)            # gate ⊙ then S_j (J, Bflat, r)
            Y = torch.einsum("jbr,jmr->jbm", HgS, self.V)               # ... V_j^T       (J, Bflat, m)
            out = c * Y.sum(dim=0)                                       # (Bflat, m)
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*lead, self.out_dim)                     # (..., m)
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


class WssTrungLinear(nn.Module):
    """Two-factor Euclidean WSS variant.

    Initializes from the same Haar frames and spectrum as SuperpositionLinear, but stores
    L_j = U_j sqrt(S_j) and R_j = V_j sqrt(S_j) directly as unconstrained Euclidean
    parameters. The effective component is (X L_j) R_j^T, so no Stiefel retraction is used.
    """

    def __init__(self, cfg: LayerConfig, *, device=None, dtype=None, generator=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        n, m, J, r = cfg.in_dim, cfg.out_dim, cfg.J, cfg.r
        self.in_dim, self.out_dim, self.J, self.r = n, m, J, r

        U = haar_init(n, r, J, dtype=dtype, generator=generator)
        V = haar_init(m, r, J, dtype=dtype, generator=generator)
        sigma0 = math.sqrt(2.0 * J * m / r)
        sqrt_sigma0 = math.sqrt(sigma0)
        self.L = nn.Parameter((U * sqrt_sigma0).to(device=device, dtype=dtype))
        self.R = nn.Parameter((V * sqrt_sigma0).to(device=device, dtype=dtype))

        if cfg.use_bias:
            self.bias = nn.Parameter(torch.zeros(m, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        if cfg.gate.phi == "sigmoid":
            self.gate_alpha = nn.Parameter(torch.tensor(cfg.gate.alpha_init, device=device, dtype=dtype))
            self.gate_beta = nn.Parameter(torch.tensor(cfg.gate.beta_init, device=device, dtype=dtype))
        else:
            self.register_parameter("gate_alpha", None)
            self.register_parameter("gate_beta", None)

        self._in_forward = False

    def init_balanced_product(self, target_var: float | None = None, *, generator=None) -> None:
        """Initialize L/R so Var(sum_k L_ik R_jk) ~= target_var."""
        with torch.no_grad():
            total_rank = self.J * self.r
            target = 2.0 / self.in_dim if target_var is None else float(target_var)
            std = (target / total_rank) ** 0.25
            self.L.normal_(0.0, std, generator=generator)
            self.R.normal_(0.0, std, generator=generator)
            if self.bias is not None:
                self.bias.zero_()

    def rescale_effective_weight(self, target_var: float | None = None) -> None:
        """Rescale L/R so the current effective dense weight has the requested variance."""
        with torch.no_grad():
            W = self.materialize_weight()
            var = W.var(unbiased=False).clamp_min(1e-12)
            target = torch.tensor(2.0 / self.in_dim if target_var is None else float(target_var),
                                  device=W.device, dtype=W.dtype)
            factor = (target / var).sqrt().sqrt()
            self.L.mul_(factor)
            self.R.mul_(factor)

    def _orthonormalized(self, X: torch.Tensor) -> torch.Tensor:
        """Return QR frames spanning current factor columns; parameters stay unconstrained.

        LoRA-style variants can have an all-zero side at initialization. In that rank-degenerate
        case, use a fixed coordinate frame for diagnostics/diversity so eigvalsh stays well-defined.
        """
        q, rmat = torch.linalg.qr(X)
        sign = rmat.diagonal(dim1=-2, dim2=-1).sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        q = q * sign.unsqueeze(-2)
        tiny = X.norm(dim=(-2, -1)) < 1e-12
        if tiny.any():
            fallback = torch.zeros(X.shape[-2], X.shape[-1], device=X.device, dtype=X.dtype)
            fallback[:X.shape[-1], :] = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            q = q.clone()
            q[tiny] = fallback
        return q

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._in_forward = True
        try:
            lead = X.shape[:-1]
            n = X.shape[-1]
            Xf = X.reshape(-1, n)
            H = torch.einsum("bn,jnr->jbr", Xf, self.L)                 # X L_j
            g, c = compute_gate(Xf, self.L, self.cfg.gate, self.gate_alpha, self.gate_beta, H=H)
            Hg = H * g.unsqueeze(-1)
            Y = torch.einsum("jbr,jmr->jbm", Hg, self.R)                # ... R_j^T
            out = c * Y.sum(dim=0)
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*lead, self.out_dim)
        finally:
            self._in_forward = False

    def materialize_weight(self, summed: bool = True) -> torch.Tensor:
        assert not self._in_forward, "materialize_weight() must not be called in the forward hot path"
        W = torch.einsum("jnr,jmr->jnm", self.L, self.R)
        if summed:
            return self.cfg.c * W.sum(dim=0)
        return W

    def diversity_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._orthonormalized(self.L), self._orthonormalized(self.R)

    def diversity(self) -> dict:
        from .diversity import diversity_penalty
        Lq, Rq = self.diversity_frames()
        try:
            return diversity_penalty(Lq, Rq, self.J, self.r)
        except RuntimeError:
            dev = self.L.device
            dtype = self.L.dtype
            s = torch.log(torch.tensor(float(self.r), device=dev, dtype=dtype))
            enc = torch.ones((), device=dev, dtype=dtype)
            d = torch.zeros((), device=dev, dtype=dtype)
            return {"S_L": s, "S_R": s, "ENC_L": enc, "ENC_R": enc, "D": d}




def _synthetic_weight(in_dim: int, out_dim: int, target_var: float | None, *, device=None, dtype=None) -> torch.Tensor:
    target = 2.0 / in_dim if target_var is None else float(target_var)
    return torch.randn(in_dim, out_dim, device=device, dtype=dtype) * math.sqrt(target)


def _svd_factors_from_weight(base_weight: torch.Tensor, in_dim: int, out_dim: int, J: int, r: int,
                             *, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L/R chunks from truncated SVD of a dense linear weight."""
    w = base_weight.detach().to(device=device, dtype=dtype).clone()
    if w.shape == (out_dim, in_dim):
        w = w.transpose(0, 1).contiguous()
    if w.shape != (in_dim, out_dim):
        raise ValueError(f"base_weight must have shape {(out_dim, in_dim)} or {(in_dim, out_dim)}, got {tuple(base_weight.shape)}")
    k = J * r
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    take = min(k, S.numel())
    Lflat = torch.zeros(in_dim, k, device=w.device, dtype=w.dtype)
    Rflat = torch.zeros(out_dim, k, device=w.device, dtype=w.dtype)
    sqrt_s = S[:take].clamp_min(0).sqrt()
    Lflat[:, :take] = U[:, :take] * sqrt_s.unsqueeze(0)
    Rflat[:, :take] = Vh[:take, :].transpose(0, 1) * sqrt_s.unsqueeze(0)
    L = Lflat.reshape(in_dim, J, r).permute(1, 0, 2).contiguous()
    R = Rflat.reshape(out_dim, J, r).permute(1, 0, 2).contiguous()
    return L, R


def init_wss_trung_from_weight(layer: WssTrungLinear, base_weight: torch.Tensor,
                               base_bias: torch.Tensor | None = None) -> WssTrungLinear:
    """Initialize an existing WssTrungLinear from the truncated SVD of a dense weight."""
    L, R = _svd_factors_from_weight(base_weight, layer.in_dim, layer.out_dim, layer.J, layer.r,
                                    device=layer.L.device, dtype=layer.L.dtype)
    with torch.no_grad():
        scale = 1.0 / layer.cfg.c
        layer.L.copy_(L * scale)
        layer.R.copy_(R)
        if layer.bias is not None:
            if base_bias is None:
                layer.bias.zero_()
            else:
                layer.bias.copy_(base_bias.detach().to(device=layer.bias.device, dtype=layer.bias.dtype))
    return layer


def make_proj(layer_type: str, in_dim: int, out_dim: int, *, J: int, r: int,
              use_bias: bool = True, gate: GateConfig | None = None,
              stiefel_canonical: bool = True, device=None, dtype=None,
              base_weight: torch.Tensor | None = None, base_bias: torch.Tensor | None = None,
              init_target_var: float | None = None) -> nn.Module:
    """Build one projection of the requested family. Shared by models.MLP and the ViT so the
    three baselines (dense / single_rank_Jr / wss) are built from one code path:

      * dense          -> nn.Linear
      * single_rank_Jr -> SuperpositionLinear, J=1 at rank J*r, gate forced off (the honest control)
      * wss            -> SuperpositionLinear, J components of rank r, gated
      * wss_trung      -> WssTrungLinear, same init folded into unconstrained L/R factors
      * wss_trung_1    -> WssTrungLinear initialized from truncated SVD of base_weight if given
      * wss_trung_2    -> direct balanced L/R init so LR product has He fan-in variance
      * wss_trung_3    -> SVD init from base_weight, then rescale effective weight to He variance

    stiefel_canonical selects the Stiefel retraction (passed to LayerConfig): True = canonical
    (Cayley/solve, the agent_guide default), False = euclidean (QR). BOTH keep U^T U = I; they
    differ only in the manifold metric/trajectory. Euclidean (QR) is ~2.7x faster on MPS / ~4.5x
    on CPU here (the retraction is the M1 bottleneck), so it is a faithful speed option.
    """
    if layer_type == "dense":
        return nn.Linear(in_dim, out_dim, bias=use_bias, device=device, dtype=dtype)
    if layer_type == "single_rank_Jr":
        lcfg = LayerConfig(in_dim=in_dim, out_dim=out_dim, J=1, r=J * r, use_bias=use_bias,
                           stiefel_canonical=stiefel_canonical, gate=GateConfig(phi="linear", disabled=True))
        return SuperpositionLinear(lcfg, device=device, dtype=dtype)
    if layer_type == "wss":
        lcfg = LayerConfig(in_dim=in_dim, out_dim=out_dim, J=J, r=r, use_bias=use_bias,
                           stiefel_canonical=stiefel_canonical,
                           gate=gate if gate is not None else GateConfig(phi="softmax"))
        return SuperpositionLinear(lcfg, device=device, dtype=dtype)
    if layer_type in ("wss_trung", "wss_trung_1", "wss_trung_2", "wss_trung_3"):
        lcfg = LayerConfig(in_dim=in_dim, out_dim=out_dim, J=J, r=r, use_bias=use_bias,
                           stiefel_canonical=stiefel_canonical,
                           gate=gate if gate is not None else GateConfig(phi="softmax"))
        layer = WssTrungLinear(lcfg, device=device, dtype=dtype)
        if layer_type == "wss_trung_2":
            layer.init_balanced_product(init_target_var)
            return layer
        if layer_type in ("wss_trung_1", "wss_trung_3"):
            if base_weight is None:
                base_weight = _synthetic_weight(in_dim, out_dim, init_target_var, device=device, dtype=dtype)
                base_bias = None
            init_wss_trung_from_weight(layer, base_weight, base_bias)
            if layer_type == "wss_trung_3":
                layer.rescale_effective_weight(init_target_var)
        return layer
    raise ValueError(f"unknown layer_type {layer_type!r}")


class SuperpositionMultiHeadAttn(nn.Module):
    """WSS multi-head self-attention -- "idea 2": materialize Q,K,V,O as separate factorized
    projections, then standard scaled-dot-product attention. `layer_type` selects the projection
    family via make_proj; layer_type="dense" yields a faithful conventional MHA.

    Shapes: x (B,N,d) -> q,k,v each (B,h,N,dh) with dh=d//h
            -> attn = softmax(q kᵀ / sqrt(dh)) (B,h,N,N) -> (B,h,N,dh) -> (B,N,d) -> O proj.

    DEFERRED VARIANTS (reserved seams -- intentionally NOT built; preserve for future experiments):
      (fused QKV)  one make_proj(layer_type, d, 3*d) split into q,k,v -> Q,K,V would SHARE one set
                   of WSS frames/gate/spectrum (fewer params, different model). Selected via
                   ViTConfig.attn_type="wss_fused".
      (idea 1)     gate-folded attention: never materialize W_Q = (1/J) Σ_j φ_j(X) U_j S_j V_jᵀ;
                   fold the per-component gate directly into the QKᵀ score. Should win when J*r < dh.
                   ViTConfig.attn_type="wss_folded". See tc_paris_collab_sketch / the original stub.
    """

    def __init__(self, dim: int, heads: int, layer_type: str, *, J: int, r: int,
                 use_bias: bool = True, gate: GateConfig | None = None,
                 stiefel_canonical: bool = True, device=None, dtype=None):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} not divisible by heads={heads}"
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        def mk(i, o):
            return make_proj(layer_type, i, o, J=J, r=r, use_bias=use_bias, gate=gate,
                             stiefel_canonical=stiefel_canonical, device=device, dtype=dtype)

        # Separate Q/K/V/O (each its own frames/gate/spectrum when wss) -- user-chosen design.
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = mk(dim, dim), mk(dim, dim), mk(dim, dim), mk(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:   # (B,N,d) -> (B,h,N,dh)
        B, N, _ = x.shape
        return x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, N, d)
        B, N, _ = x.shape
        q = self._split_heads(self.q_proj(x))                  # (B, h, N, dh)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        # Faithful scaled-dot-product attention, written out explicitly (NOT F.scaled_dot_product
        # _attention): keeps the math inspectable and avoids MPS fused-kernel variance. Seq len is
        # tiny (~65 tokens) so there is no performance reason to fuse.
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                          # (B, h, N, dh)
        out = out.transpose(1, 2).reshape(B, N, self.dim)       # (B, N, d)
        return self.o_proj(out)
                