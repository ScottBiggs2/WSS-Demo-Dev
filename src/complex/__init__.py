"""Weight Subspace Superposition (WSS) -- core prototype.

A linear layer whose weight is a gated superposition of J rank-r factorized components
W_j = U_j S_j V_j^T, with U_j/V_j on Stiefel manifolds and sigma_j > 0, trained with a
Riemannian optimizer. See agent_guide.md for the full contract.
"""

from .config import GateConfig, LayerConfig, ModelConfig, TrainConfig
from .device import enable_mps_fallback, get_device
from .diversity import diversity_penalty, von_neumann
from .gate import compute_gate, gate_energy
from .manifold import check_orthonormal, haar_init, make_stiefel_param, orthonormality_error
from .models import MLP, make_layer
from .spectrum import Spectrum
from .superposition import SuperpositionLinear
from .train import build_optimizers, evaluate, fit, partition_params, smoke_train_step, train_epoch

__all__ = [
    "GateConfig",
    "LayerConfig",
    "ModelConfig",
    "TrainConfig",
    "enable_mps_fallback",
    "get_device",
    "diversity_penalty",
    "von_neumann",
    "compute_gate",
    "gate_energy",
    "check_orthonormal",
    "haar_init",
    "make_stiefel_param",
    "orthonormality_error",
    "MLP",
    "make_layer",
    "Spectrum",
    "SuperpositionLinear",
    "build_optimizers",
    "evaluate",
    "fit",
    "partition_params",
    "smoke_train_step",
    "train_epoch",
]
