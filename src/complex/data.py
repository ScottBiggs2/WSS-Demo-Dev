"""MNIST / Fashion-MNIST loaders (agent_guide §4; mirrors src/simple data setup).

Data is downloaded to <repo_root>/data so the location is independent of the cwd
(the experiment scripts live under src/complex/experiments). num_workers=0 -- MPS plus
DataLoader workers can be flaky on macOS.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# <repo_root>/data : data.py is at src/complex/data.py -> parents[2] == repo root
DEFAULT_ROOT = str(Path(__file__).resolve().parents[2] / "data")

_DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,   # present for parity; not exercised in Phase-2 goalpost
}


def get_loaders(
    dataset: str = "mnist",
    batch_size: int = 256,
    test_batch_size: int = 512,
    root: str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(_DATASETS)}")
    cls = _DATASETS[dataset]
    root = root or DEFAULT_ROOT
    transform = transforms.ToTensor()
    train_set = cls(root, train=True, download=True, transform=transform)
    test_set = cls(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
