"""
Train SimpleSubspaceClassifier vs DenseClassifier on MNIST.
Prints loss, test accuracy, and wall-clock time per epoch for both.

Run from the src/ directory:
    python mnist_demo.py
"""

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleSubspaceClassifier, DenseClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


# ── training loop ─────────────────────────────────────────────────────────────

def run(name, model, train_loader, test_loader, epochs, device, lr=1e-3):
    model = model.to(device)

    # neither of these work because they break steifel geometry
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"  {count_params(model):,} trainable parameters")
    print(f"{'─' * 55}")
    print(f"  {'epoch':>5}  {'loss':>8}  {'test acc':>9}  {'time':>6}")

    total_time = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc  = accuracy(model, test_loader, device)
        dt   = time.perf_counter() - t0
        total_time += dt
        print(f"  {epoch:>5}  {loss:>8.4f}  {acc:>9.3%}  {dt:>5.1f}s")

    print(f"  total: {total_time:.1f}s  final acc: {acc:.3%}")
    return acc, total_time


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    transform = transforms.ToTensor()
    root = "../data"
    train_set = datasets.MNIST(root, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=0)

    INPUT_DIM   = 784
    HIDDEN_DIMS = [128, 64]
    N_CLASSES   = 10
    EPOCHS      = 10
    J, R        = 4, 8

    subspace = SimpleSubspaceClassifier(INPUT_DIM, N_CLASSES, HIDDEN_DIMS, J=J, r=R)
    dense    = DenseClassifier(INPUT_DIM, N_CLASSES, HIDDEN_DIMS)

    acc_sub,  t_sub  = run(f"SubspaceClassifier  J={J}  r={R}",
                           subspace, train_loader, test_loader, EPOCHS, device)
    acc_dense, t_dense = run("DenseClassifier",
                             dense, train_loader, test_loader, EPOCHS, device)

    print(f"\n{'═' * 55}")
    print(f"  {'model':<30}  {'acc':>9}  {'time':>6}")
    print(f"  {'SubspaceClassifier':<30}  {acc_sub:>9.3%}  {t_sub:>5.1f}s")
    print(f"  {'DenseClassifier':<30}  {acc_dense:>9.3%}  {t_dense:>5.1f}s")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
