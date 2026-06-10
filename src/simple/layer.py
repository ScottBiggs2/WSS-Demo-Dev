import torch
import torch.nn as nn


class SimpleSubspaceLayer(nn.Module):
    """
    Approximates a weight matrix as a gated mixture of J low-rank factors:

        W(X) = (1/J) * sum_{j=1}^{J}  f_j(X) * (X B_j A_j^T + b_j)

    Each subspace j is parametrised by two free matrices:
        A_j  in R^{d_out x r}   (output side)
        B_j  in R^{d_in  x r}   (input side)
    so the effective weight is  W_j = A_j B_j^T  in R^{d_out x d_in}.

    This is the LoRA-style low-rank product.  There is no SVD structure to
    maintain — A and B are just free parameters, updated by any optimiser
    without structural constraints.  The /J normalisation keeps the output
    scale independent of J.

    --- Gating (MoE-style spectral alignment) ---

        f_j(X) = Softmax_j( ||B_j^T X^T||_F^2 / ||X||_F^2 )

    Measures how strongly the batch aligns with each input subspace B_j.

    --- Gradient orthogonality (prevents subspace collapse) ---

    A gradient hook on B keeps the input subspaces from collapsing:

        g_j  ←  g_j  -  sum_{k ≠ j} B_k (B_k^T g_j)

    This removes from each B_j gradient the components that would push it
    into the span of other subspaces.

    --- Initialisation ---

    A_j ~ N(0, 1/r),  B_j ~ N(0, 1/r).  The product A_j B_j^T then has
    entries ~ O(1/r * r) = O(1), comparable to a kaiming-uniform linear layer.
    """

    def __init__(self, in_dim: int, out_dim: int, J: int = 4, r: int = 8, use_biases: bool = True):
        super().__init__()
        self.J = J
        self.r = r = min(r, in_dim, out_dim)
        self.use_biases = use_biases

        scale = r ** -0.5
        self.A = nn.Parameter(torch.randn(J, out_dim, r) * scale)  # (J, d_out, r)
        self.B = nn.Parameter(torch.randn(J, in_dim,  r) * scale)  # (J, d_in,  r)

        if use_biases:
            self.biases = nn.Parameter(torch.zeros(J, out_dim))
        else:
            self.biases = None

        self.B.register_hook(self._ortho_grad_hook)

    # ------------------------------------------------------------------

    def _ortho_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        g_j ← g_j - sum_{k≠j} B_k (B_k^T g_j)

        Vectorised: sum over all k then subtract the k==j self-term.
        grad: (J, in_dim, r)
        """
        B = self.B.data
        coeffs   = torch.einsum('kir,jis->jkrs', B, grad)          # (J, J, r, r)
        proj_all = torch.einsum('kir,jkrs->jis', B, coeffs)        # (J, in_dim, r)
        self_c   = torch.einsum('jir,jis->jrs', B, grad)           # (J, r, r)
        self_p   = torch.einsum('jir,jrs->jis', B, self_c)         # (J, in_dim, r)
        return grad - (proj_all - self_p)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim) → (J,) softmax weights
        f_j = Softmax_j( ||B_j^T X^T||_F^2 / ||X||_F^2 )
        """
        x_sq   = (x * x).sum() + 1e-8
        proj   = torch.einsum('jir,bi->jrb', self.B, x)            # (J, r, batch)
        scores = (proj * proj).sum(dim=(1, 2)) / x_sq              # (J,)
        return torch.softmax(scores, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_dim) → (batch, out_dim)
        h_j = x B_j A_j^T + b_j
        """
        gates = self._gate(x)                                       # (J,)
        xB    = torch.einsum('bi,jir->jbr', x, self.B)             # (J, batch, r)
        h     = torch.einsum('jbr,jor->jbo', xB, self.A)           # (J, batch, out_dim)
        if self.use_biases:
            h = h + self.biases.unsqueeze(1)
        out   = torch.einsum('j,jbo->bo', gates, h)                # (batch, out_dim)
        return out / self.J
