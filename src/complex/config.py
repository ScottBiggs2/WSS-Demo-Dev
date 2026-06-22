"""Configuration dataclasses for the WSS (Weight Subspace Superposition) prototype.

These are plain data holders. The only logic here is validation, which is also the
single enforcement site for the normalization-XOR rule (agent_guide §0.5): a gate is
either softmax-normalized (prefactor c=1) or non-normalized (c=1/J), never both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# phi variants and which of them are self-normalizing over the component axis j.
PHI_KINDS = ("linear", "exp", "sigmoid", "pow", "softmax")
NORMALIZED_PHI = frozenset({"softmax"})


@dataclass
class GateConfig:
    """Content gate that reads the LEFT frame U (agent_guide §0.1, §2.2)."""

    granularity: str = "sample"   # {"sample", "batch"}
    phi: str = "linear"           # one of PHI_KINDS
    detach: bool = False          # if True, gate is .detach()-ed before the forward
    alpha_init: float = 1.0       # sigmoid only (learnable scalar)
    beta_init: float = 0.0        # sigmoid only (learnable scalar)
    gamma: float = 1.0            # pow only (fixed exponent > 0)
    disabled: bool = False        # if True, gate is forced to f == 1 (ablation / single_rank_Jr)

    def validate(self) -> None:
        assert self.granularity in ("sample", "batch"), f"bad granularity {self.granularity!r}"
        assert self.phi in PHI_KINDS, f"bad phi {self.phi!r}, must be one of {PHI_KINDS}"
        assert self.gamma > 0, "pow exponent gamma must be > 0"

    @property
    def is_normalized(self) -> bool:
        return self.phi in NORMALIZED_PHI


@dataclass
class LayerConfig:
    """One SuperpositionLinear layer R^n -> R^m with J components of rank r."""

    in_dim: int
    out_dim: int
    J: int
    r: int
    use_bias: bool = True
    stiefel_canonical: bool = True   # canonical (Cayley/solve) vs euclidean (QR) retraction
    gate: GateConfig = field(default_factory=GateConfig)

    def validate(self) -> None:
        assert self.J >= 1, "J must be >= 1"
        assert self.r >= 1, "r must be >= 1"
        # Stiefel St(r, n) requires r <= n (orthonormal columns); same for m.
        assert self.r <= self.in_dim, f"r={self.r} must be <= in_dim={self.in_dim} (Stiefel)"
        assert self.r <= self.out_dim, f"r={self.r} must be <= out_dim={self.out_dim} (Stiefel)"
        self.gate.validate()

    @property
    def c(self) -> float:
        """Forward prefactor (XOR rule). softmax -> 1.0, otherwise 1/J. Never both."""
        return 1.0 if self.gate.is_normalized else 1.0 / self.J


@dataclass
class ModelConfig:
    """An MLP built from one of three interchangeable layer types."""

    layer_type: str = "wss"                                   # {"dense","single_rank_Jr","wss"}
    dims: list[int] = field(default_factory=lambda: [784, 256, 128, 10])
    J: int = 4
    r: int = 16
    use_bias: bool = True
    gate: GateConfig = field(default_factory=GateConfig)
    lambda_div: float = 1e-2                                   # diversity penalty weight

    def validate(self) -> None:
        assert self.layer_type in ("dense", "single_rank_Jr", "wss"), self.layer_type
        assert len(self.dims) >= 2, "need at least input and output dims"
        self.gate.validate()

# Attention implementations the ViT can select between. Only the first two are built now;
# the others are reserved seams (see SuperpositionMultiHeadAttn) for future experiments.
ATTN_KINDS = (
    "wss_separate",   # separate WSS Q/K/V/O projections (idea 2; built now)
    "dense",          # conventional MHA with nn.Linear Q/K/V/O (for WSS-MLP + dense-attn checks)
    # "wss_fused",    # FUTURE: one fused WSS qkv (dim->3*dim), shared frames/gate across Q,K,V
    # "wss_folded",   # FUTURE: "idea 1" -- gate folded into the attention score (no materialized W_Q)
)


@dataclass
class ViTConfig:
    """Tiny pre-norm ViT whose projections can be dense / single_rank_Jr / wss.

    The patch-embed Conv2d and the classification head are ALWAYS dense (head out_dim=num_classes
    < J*r would violate the Stiefel r<=out_dim rule -- the same reason the MLP readout stays dense).
    Factorized linears: attention Q/K/V/O and the MLP fc1 (dim->hidden) / fc2 (hidden->dim).

    `layer_type` selects the factorization family for every WSS projection (attention + MLP).
    `attn_type` selects the attention *module* independently, so e.g. a WSS MLP can be paired with
    a conventional dense attention block (attn_type="dense").
    """

    layer_type: str = "wss"                 # {"dense","single_rank_Jr","wss"} -- all WSS projections
    attn_type: str = "wss_separate"         # one of ATTN_KINDS
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10
    dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 2
    J: int = 4
    r: int = 16
    use_bias: bool = True
    # Stiefel retraction for all wss projections. True = canonical (Cayley/solve, agent_guide
    # default); False = euclidean (QR). Both keep U^T U = I; euclidean is markedly faster on M1
    # (the retraction is the bottleneck) and is a faithful alternative -- see make_proj.
    stiefel_canonical: bool = True
    gate: GateConfig = field(default_factory=lambda: GateConfig(phi="softmax"))
    lambda_div: float = 1e-3
    # Faithfulness knob: the WSS contract inits the spectrum at sigma0 = sqrt(2*J*m/r) (He fan-in).
    # init_scale multiplies sigma0; KEEP 1.0 for a faithful build. Any value != 1.0 is an
    # explicitly NON-FAITHFUL stabilization for residual-stream/attention-logit hotness -- a
    # finding to report, not a silent default.
    init_scale: float = 1.0

    @property
    def hidden_dim(self) -> int:
        return self.mlp_ratio * self.dim

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        return self.n_patches + 1            # + cls token

    def validate(self) -> None:
        assert self.layer_type in ("dense", "single_rank_Jr", "wss"), self.layer_type
        assert self.attn_type in ATTN_KINDS, f"bad attn_type {self.attn_type!r}, expected {ATTN_KINDS}"
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"
        assert self.dim % self.heads == 0, f"dim={self.dim} not divisible by heads={self.heads}"
        assert self.J >= 1 and self.r >= 1, "J, r must be >= 1"
        # single_rank_Jr factorizes the dim->dim projections at rank J*r -> Stiefel needs J*r <= dim.
        assert self.J * self.r <= self.dim, (
            f"J*r={self.J * self.r} must be <= dim={self.dim} (Stiefel r<=out_dim on dim->dim proj)")
        self.gate.validate()

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    test_batch_size: int = 512
    lr_riemann: float = 1e-3       # RiemannianAdam for Stiefel params (U, V)
    lr_euclid: float = 1e-3        # Adam for Euclidean params (s, bias, gate scalars)
    lambda_div: float = 1e-2
    dataset: str = "mnist"
    seed: int = 0
    device: str = "auto"
    retraction: bool = True        # False -> Remark-8 control (plain SGD on raw .data, no retraction)
    stabilize: int = 50            # RiemannianAdam re-projection cadence (steps)
    log_every: int = 1             # epochs between diagnostic logs
