"""Log-parametrized positive diagonal spectrum (agent_guide §2.1, §2.7).

We store s in R^{J x r} and define sigma = exp(s), so sigma is strictly positive for any
value of the (unconstrained, Euclidean) parameter s. This is what keeps S_j = diag(sigma_j)
a valid positive diagonal under a plain Adam update -- the failure mode in src/simple/ where
unconstrained singular values could go negative.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Spectrum(nn.Module):
    """Positive diagonal singular values for J components, each of rank r.

    sigma0 is the initial value for every singular value; it is set by the caller to
    sqrt(2 * J * m / r) (He fan-in, agent_guide §2.7) since it depends on the output dim m.
    """

    def __init__(self, J: int, r: int, sigma0: float):
        super().__init__()
        assert sigma0 > 0, "sigma0 must be positive"
        self.J = J
        self.r = r
        self.s = nn.Parameter(torch.full((J, r), math.log(sigma0)))

    def sigma(self) -> torch.Tensor:
        """Return sigma = exp(s), shape (J, r), strictly positive."""
        return self.s.exp()

    @staticmethod
    def s_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
        """Inverse map log(sigma) for round-tripping / construction from given sigmas."""
        return sigma.log()
