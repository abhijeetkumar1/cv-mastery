"""The standard PyTorch training loop, written in module 02b and reused everywhere after.

    history = fit(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, device=env.device)

Grows with the course:
    02b  basic loop, early stopping
    05   mixed precision (amp=True), per-batch LR schedules (OneCycle, warmup), multi-label accuracy
"""

from __future__ import annotations

import contextlib
import copy
import time

import torch
from torch import nn


def _correct_count(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-label (y: (B,) class ids): number of correct argmax predictions.
    Multi-label (y: (B, C) of 0/1): number of correct label decisions at probability 0.5, divided by C."""
    if y.ndim == 1:
        return float((logits.argmax(dim=1) == y).sum().item())
    return float(((logits > 0) == (y > 0.5)).float().mean(dim=1).sum().item())


def autocast(device: str, enabled: bool):
    """Mixed precision context: float16 on GPUs (cuda/mps), bfloat16 on CPU. No-op when disabled."""
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    return torch.autocast(device_type=device, dtype=dtype)


def train_one_epoch(model: nn.Module, loader, loss_fn, optimizer, device: str,
                    amp: bool = False, scaler=None, batch_scheduler=None) -> tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy).

    amp: run the forward pass in float16 (see module 05b). On CUDA pass a torch.amp.GradScaler as `scaler`.
    batch_scheduler: an LR scheduler stepped after every batch (OneCycle, warmup); epoch schedulers go to fit().
    """
    model.train()  # turns on dropout, batchnorm updates, ...
    total_loss, correct, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device, amp):
            logits = model(x)  # 1. forward
            loss = loss_fn(logits, y)  # 2. how wrong are we?
        optimizer.zero_grad(set_to_none=True)  # 3. clear old gradients (PyTorch accumulates them)
        if scaler is not None:  # float16 gradients can underflow: scale the loss up, then back down
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()  # 4. backprop: compute d(loss)/d(every weight)
            optimizer.step()  # 5. nudge every weight downhill
        if batch_scheduler is not None:
            batch_scheduler.step()
        total_loss += loss.item() * len(y)
        correct += _correct_count(logits.float(), y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()  # no gradients needed: faster, less memory
def evaluate(model: nn.Module, loader, loss_fn, device: str, amp: bool = False) -> tuple[float, float]:
    """Loss and accuracy on a dataset, without training. Returns (mean loss, accuracy)."""
    model.eval()  # turns off dropout, uses running batchnorm stats
    total_loss, correct, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device, amp):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += _correct_count(logits, y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model: nn.Module, loader, device: str, amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """All logits and labels for a dataset, on the CPU. Useful for confusion matrices and error galleries."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        with autocast(device, amp):
            all_logits.append(model(x.to(device)).float().cpu())
        all_y.append(y.cpu() if torch.is_tensor(y) else torch.as_tensor(y))
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
    batch_scheduler=None,
    amp: bool = False,
    patience: int | None = None,
    verbose: bool = True,
) -> dict:
    """Train for `epochs`, validating after each one.

    scheduler:       LR scheduler stepped once per epoch (e.g. CosineAnnealingLR(T_max=epochs)).
    batch_scheduler: LR scheduler stepped once per batch (e.g. OneCycleLR, warmup).
    amp:             mixed precision (float16 autocast; a GradScaler is created automatically on CUDA).
    patience:        stop early if validation loss hasn't improved for this many epochs,
                     and restore the best weights (early stopping).
    Returns a history dict: train_loss, val_loss, train_acc, val_acc, lr (one value per epoch).
    """
    model.to(device)
    scaler = torch.amp.GradScaler("cuda") if (amp and device == "cuda") else None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, amp, scaler, batch_scheduler)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device, amp)
        if scheduler is not None:
            scheduler.step()
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc, lr_now)):
            history[k].append(v)
        if verbose:
            print(
                f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr_now:.2e} | {time.perf_counter() - t0:.1f}s"
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
