"""The standard PyTorch training loop, written in module 02b and reused everywhere after.

    history = fit(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, device=env.device)

It grows over the course (mixed precision in 05, schedulers and checkpoints, etc.).
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn


def _accuracy_count(logits: torch.Tensor, y: torch.Tensor) -> int:
    """Number of correct predictions for single-label classification (logits: (B, C), y: (B,))."""
    return int((logits.argmax(dim=1) == y).sum().item())


def train_one_epoch(model: nn.Module, loader, loss_fn, optimizer, device: str) -> tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()  # turns on dropout, batchnorm updates, ...
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # 1. forward
        loss = loss_fn(logits, y)  # 2. how wrong are we?
        optimizer.zero_grad()  # 3. clear old gradients (PyTorch accumulates them)
        loss.backward()  # 4. backprop: compute d(loss)/d(every weight)
        optimizer.step()  # 5. nudge every weight downhill
        total_loss += loss.item() * len(y)
        correct += _accuracy_count(logits, y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()  # no gradients needed: faster, less memory
def evaluate(model: nn.Module, loader, loss_fn, device: str) -> tuple[float, float]:
    """Loss and accuracy on a dataset, without training. Returns (mean loss, accuracy)."""
    model.eval()  # turns off dropout, uses running batchnorm stats
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += _accuracy_count(logits, y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model: nn.Module, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """All logits and labels for a dataset, on the CPU. Useful for confusion matrices and error galleries."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs: int,
    device: str,
    scheduler=None,
    patience: int | None = None,
    verbose: bool = True,
) -> dict:
    """Train for `epochs`, validating after each one.

    patience: stop early if validation loss hasn't improved for this many epochs,
              and restore the best weights (early stopping).
    Returns a history dict: train_loss, val_loss, train_acc, val_acc (one value per epoch).
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step()
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        if verbose:
            print(
                f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.perf_counter() - t0:.1f}s"
            )

        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                if verbose:
                    print(f"early stopping: no val improvement for {patience} epochs (best val loss {best_loss:.4f})")
                break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)  # keep the best model, not the last one
    return history
