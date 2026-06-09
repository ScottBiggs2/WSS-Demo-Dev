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
