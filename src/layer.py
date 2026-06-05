import torch
import torch.nn as nn


class SimpleSubspaceLayer(nn.Module):
    """
    Approximates a weight matrix as a gated mixture of J low-rank factors:

        W(X) = (1/J) * sum_{j=1}^{J}  f_j(X) * (X V_j diag(s_j) U_j^T + b_j)

    Shapes:  X in R^{B x d_in},  V_j in R^{d_in x r},  s_j in R^r,  U_j in R^{d_out x r}
    Output:  R^{B x d_out}

    --- Gating (MoE-style spectral alignment) ---

        f_j(X) = Softmax_j( ||V_j^T X^T||_F^2  /  ||X||_F^2 )

    The numerator measures how strongly the batch projects onto subspace j.
    Dividing by ||X||_F^2 normalises for batch scale.  Softmax across j=1..J
    ensures the gates are a probability simplex, so subspaces compete.

    --- Initialisation: Haar-uniform QR ---

    Draw A ~ N(0,I_{d x r}) and thin-QR factor it: A = QR.  Q is then
    Haar-uniform on the Stiefel manifold St(d, r) — the unique rotation-
    invariant distribution over orthonormal r-frames in R^d.  Starting U_j,
    V_j this way gives mutually "generic" (near-orthogonal) subspaces for
    free, at zero extra cost.  Singular values s_j start at 1.

    --- Gradient orthogonality (prevents subspace collapse) ---

    Constraint:  grad_{V_j} L  ⊥  col(V_k)   for all k ≠ j

    A PyTorch gradient hook enforces this after every backward pass.  For
    each j, we subtract from grad[j] its projections onto the column spaces
    of all other V_k:

        g_j  ←  g_j  -  sum_{k ≠ j} V_k (V_k^T g_j)

    Assumes V_k columns remain approximately orthonormal (true at init;
    degrades slowly during training — re-QR V explicitly if running very long).
    This sidesteps sequential per-subspace gradient adjustments: because
    Haar init already places subspaces in "generic position", a single
    projection pass is sufficient.

    --- Why no custom backward? ---

    Registering U, S, V, biases as nn.Parameter means PyTorch's autograd
    engine differentiates through forward() automatically.  The gradient hook
    on V is a thin post-backward modifier — it does not break the computation
    graph and is invisible to the optimiser.  No manual backward needed.
    Use any standard optimiser (SGD, Adam) directly on model.parameters().
    """

    def __init__(self, in_dim: int, out_dim: int, J: int = 4, r: int = 8, use_biases: bool = True):
        super().__init__()
        self.J = J
        # QR of an (n, r) matrix only yields r orthonormal columns when r <= n,
        # so clamp r to the smaller of the two dimensions.
        self.r = r = min(r, in_dim, out_dim)
        self.use_biases = use_biases

        # Haar-uniform init: thin QR of random Gaussians
        # U: (J, out_dim, r)  — per-subspace left singular vectors
        # S: (J, r)           — per-subspace singular values, start at 1
        # V: (J, in_dim,  r)  — per-subspace right singular vectors
        U_init = torch.stack([torch.linalg.qr(torch.randn(out_dim, r))[0] for _ in range(J)])
        V_init = torch.stack([torch.linalg.qr(torch.randn(in_dim,  r))[0] for _ in range(J)])

        self.U = nn.Parameter(U_init)
        self.S = nn.Parameter(torch.ones(J, r))
        self.V = nn.Parameter(V_init)

        if use_biases:
            self.biases = nn.Parameter(torch.zeros(J, out_dim))
        else:
            self.biases = None

        # Gradient hook: keep V subspaces mutually orthogonal
        self.V.register_hook(self._ortho_grad_hook)

    # ------------------------------------------------------------------

    def _ortho_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Post-backward projection:  g_j ← g_j - sum_{k≠j} V_k (V_k^T g_j)

        Vectorised: compute the full cross-subspace projection in two einsums,
        then subtract the self-projection (k==j) that was incorrectly included.

        grad: (J, in_dim, r)
        """
        V = self.V.data  # (J, in_dim, r)

        # coeffs[j,k] = V_k^T g_j  →  (J, J, r, r)
        coeffs = torch.einsum('kir,jis->jkrs', V, grad)
        # full sum over all k including k==j: sum_k V_k (V_k^T g_j)  →  (J, in_dim, r)
        proj_all = torch.einsum('kir,jkrs->jis', V, coeffs)
        # subtract the k==j self-term to get sum_{k≠j}
        self_coeffs = torch.einsum('jir,jis->jrs', V, grad)   # (J, r, r)
        self_proj   = torch.einsum('jir,jrs->jis', V, self_coeffs)  # (J, in_dim, r)

        return grad - (proj_all - self_proj)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        returns: (J,) softmax gate weights.

        f_j = Softmax_j( ||V_j^T X^T||_F^2 / ||X||_F^2 )
        """
        x_sq = (x * x).sum() + 1e-8
        proj = torch.einsum('jir,bi->jrb', self.V, x)     # (J, r, B)
        scores = (proj * proj).sum(dim=(1, 2)) / x_sq     # (J,)
        return torch.softmax(scores, dim=0)                # (J,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        returns: (B, out_dim)
        """
        gates = self._gate(x)                                      # (J,)

        xV  = torch.einsum('bi,jir->jbr', x, self.V)              # (J, B, r)
        xVS = xV * self.S.unsqueeze(1)                             # (J, B, r)
        h   = torch.einsum('jbr,jor->jbo', xVS, self.U)           # (J, B, out_dim)

        if self.use_biases:
            h = h + self.biases.unsqueeze(1)                       # (J, B, out_dim)

        out = torch.einsum('j,jbo->bo', gates, h)                  # (B, out_dim)
        return out / self.J
