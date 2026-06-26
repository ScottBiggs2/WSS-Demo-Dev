"""MNIST / Fashion-MNIST loaders (agent_guide §4; mirrors src/simple data setup).

Data is downloaded to <repo_root>/data so the location is independent of the cwd
(the experiment scripts live under src/complex/experiments). num_workers=0 -- MPS plus
DataLoader workers can be flaky on macOS.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .seed import seed_worker
from torchvision import datasets, transforms

# <repo_root>/data : data.py is at src/complex/data.py -> parents[2] == repo root
DEFAULT_ROOT = str(Path(__file__).resolve().parents[2] / "data")

_DATASETS = {
    "mnist": datasets.MNIST,
    "fmnist": datasets.FashionMNIST,   # present for parity; not exercised in Phase-2 goalpost
    "cifar10": datasets.CIFAR10,       # Phase-3 ViT task (32x32x3, 10 classes)
}

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_transform(dataset: str, train: bool, augment: bool):
    """Per-dataset, per-split transform. MNIST/FMNIST keep the original ToTensor() (unchanged).
    CIFAR-10 always normalizes; train-time augmentation (RandomCrop pad4 + HFlip) is gated by
    `augment`."""
    if dataset in ("mnist", "fmnist"):
        return transforms.ToTensor()
    if dataset == "cifar10":
        norm = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)
        if train and augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        return transforms.Compose([transforms.ToTensor(), norm])
    raise ValueError(f"no transform defined for dataset {dataset!r}")


def get_loaders(
    dataset: str = "mnist",
    batch_size: int = 256,
    test_batch_size: int = 512,
    root: str | None = None,
    num_workers: int = 0,
    augment: bool = True,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(_DATASETS)}")
    cls = _DATASETS[dataset]
    root = root or DEFAULT_ROOT
    train_set = cls(root, train=True, download=True,
                    transform=_build_transform(dataset, train=True, augment=augment))
    test_set = cls(root, train=False, download=True,
                   transform=_build_transform(dataset, train=False, augment=augment))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    worker_init_fn = seed_worker if seed is not None else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=generator, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                             worker_init_fn=worker_init_fn)
    return train_loader, test_loader
