"""pytest bootstrap: make ``complex`` importable as a package and enable MPS fallback.

This file runs before any test module is imported, so setting PYTORCH_ENABLE_MPS_FALLBACK
here takes effect before torch initializes the MPS backend.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# parents[2] == .../src  (tests -> complex -> src); putting it on the path makes
# `import complex` resolve to this package and its relative imports work.
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402


def pytest_report_header(config):
    return f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()}"


def available_devices():
    """Devices a CPU+MPS test should sweep."""
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs
