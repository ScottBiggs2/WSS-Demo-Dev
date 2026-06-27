"""Device selection and the CPU/MPS split for MPS op gaps.

Verified facts about this machine (M1, torch 2.3.1):
  * einsum / bmm / elementwise run natively on MPS.
  * linalg.qr / linalg.solve / eigvalsh are NOT implemented on MPS; they fall back
    to CPU when PYTORCH_ENABLE_MPS_FALLBACK=1 is set (a UserWarning is emitted, op runs).
  * float64 CANNOT be allocated on MPS at all -> the float64 gradcheck is CPU-only.

Entry scripts must set PYTORCH_ENABLE_MPS_FALLBACK=1 BEFORE importing torch. Call
``enable_mps_fallback()`` at the very top of a script (it also sets the env var, which
only takes effect if torch has not yet initialized the MPS backend).
"""

from __future__ import annotations

import contextlib
import os

import torch


def setup_backend(allow_tf32: bool = False) -> None:
    """Set process-wide matmul precision flags ONCE (call at fit() start). CUDA-only; a no-op on
    MPS/CPU (the attributes exist but only bite on CUDA, and we guard on is_available anyway).

    TF32 uses 19-bit-mantissa tensor-core GEMM -- a large A100/H200 throughput win at a small
    precision cost. OFF by default to keep fp32 numerics byte-faithful; the scaling suite turns it
    on. Reduced-precision GEMM accumulation, so it is logged as faithful-with-tolerance.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def autocast_ctx(device: torch.device | str, enabled: bool, dtype: torch.dtype = torch.bfloat16):
    """bf16 autocast context for the matmul-heavy forward+loss, CUDA only.

    Returns a real torch.autocast only when ``enabled`` AND the device is CUDA; otherwise a
    nullcontext (MPS/CPU have no useful bf16 autocast here, and disabled => faithful fp32). bf16
    needs no GradScaler. Linalg that must stay fp32 (the diversity Gram + eigvalsh) re-enables fp32
    at its own call site; the optimizer step / retraction run OUTSIDE this context on fp32 masters.
    """
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def enable_mps_fallback() -> None:
    """Set the MPS->CPU fallback env var. Safe to call repeatedly."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def needs_cpu_linalg(device: torch.device | str) -> bool:
    """True iff dense-linalg ops (eigvalsh/qr/svd/solve) must be routed to CPU on this device.

    These have NO MPS kernel in torch 2.3.1, so on MPS we run them on CPU explicitly. On CUDA
    and CPU they are native -- routing to CPU there would be a pointless device round-trip and a
    host sync every step. This predicate is the single switch the hot-path callers (diversity
    eigvalsh) use to keep MPS behavior byte-identical while going on-device on CUDA.
    """
    return torch.device(device).type == "mps"


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a torch.device. ``pref`` in {"auto","mps","cuda","cpu"}."""
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_cpu_op(fn, *tensors):
    """Run ``fn(*tensors)`` on CPU, restoring the result to the first tensor's device.

    Single chokepoint for ops with MPS gaps (eigvalsh, qr) where we want the op AND its
    backward to run on CPU explicitly rather than relying on implicit fallback. Returns
    the result moved back to the original device. If ``fn`` returns a tuple, every tensor
    element is moved back.
    """
    if not tensors:
        return fn()
    orig = tensors[0].device
    cpu_args = [t.cpu() if torch.is_tensor(t) else t for t in tensors]
    out = fn(*cpu_args)
    if isinstance(out, tuple):
        return tuple(o.to(orig) if torch.is_tensor(o) else o for o in out)
    return out.to(orig) if torch.is_tensor(out) else out
