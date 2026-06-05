import torch
import torch.nn as nn
from layer import SimpleSubspaceLayer


class SimpleSubspaceClassifier(nn.Module):
    """
    MLP where every linear transform is a SimpleSubspaceLayer.
    Architecture: input_dim -> hidden_dims[0] -> ... -> output_classes
    ReLU between hidden layers; no activation on the final logit layer.
    """

    def __init__(self, input_dim: int, output_classes: int, hidden_dims: list,
                 J: int = 4, r: int = 16):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(SimpleSubspaceLayer(dims[i], dims[i + 1], J=J, r=r, use_biases=True))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class DenseClassifier(nn.Module):
    """
    Plain MLP with nn.Linear layers — same depth/width as SimpleSubspaceClassifier
    for an apples-to-apples speed and accuracy comparison.
    """

    def __init__(self, input_dim: int, output_classes: int, hidden_dims: list):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))
