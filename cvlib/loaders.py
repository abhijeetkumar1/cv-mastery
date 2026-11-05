"""Fast loading for small image datasets that fit in memory (CIFAR, MNIST, ...).

The whole dataset is kept on the device as uint8 (CIFAR-10 train = 150 MB) and each batch is
converted, augmented and normalized on the device. Much faster than decoding images one by one.

    train_loader = GPULoader(X_train, y_train, batch_size=256, shuffle=True,
                             augment=random_crop_flip, mean=CIFAR_MEAN, std=CIFAR_STD, device=env.device)
    for xb, yb in train_loader: ...      # xb: float (B, 3, H, W), normalized, on the device
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class GPULoader:
    def __init__(self, x, y, batch_size=256, shuffle=False, augment=None, mean=None, std=None,
                 device="cpu", drop_last=False, seed=0):
        x = torch.as_tensor(np.asarray(x))
        if x.ndim == 3:  # (N, H, W) grayscale → (N, H, W, 1)
            x = x[..., None]
        self.x = x.permute(0, 3, 1, 2).contiguous().to(device)  # (N, C, H, W) uint8
        self.y = torch.as_tensor(np.asarray(y)).to(device)
        c = self.x.shape[1]
        self.mean = torch.tensor(mean if mean is not None else [0.0] * c, device=device).view(1, c, 1, 1)
        self.std = torch.tensor(std if std is not None else [1.0] * c, device=device).view(1, c, 1, 1)
        self.batch_size, self.shuffle, self.augment, self.drop_last = batch_size, shuffle, augment, drop_last
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        n = len(self.x) / self.batch_size
        return math.floor(n) if self.drop_last else math.ceil(n)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        order = order.to(self.device)
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            xb = self.x[idx].float().div_(255)
            if self.augment is not None:
                xb = self.augment(xb)
            yield (xb - self.mean) / self.std, self.y[idx]


def random_resized_crop_flip(out_size: int = 224, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)):
    """ImageNet-style augmentation on a batch, done on the GPU in one step.

    For each image: pick a random box covering `scale` of the area with a random aspect `ratio`,
    resize it to out_size × out_size, and flip left-right half the time.
    Implemented as one affine transform per image (affine_grid + grid_sample): crop, resize and flip at once.
    """
    log_r = (math.log(ratio[0]), math.log(ratio[1]))

    def augment(x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        dev = x.device
        area = torch.empty(B, device=dev).uniform_(*scale)
        r = torch.exp(torch.empty(B, device=dev).uniform_(*log_r))
        w = torch.sqrt(area * r).clamp(max=1.0)  # crop width as a fraction of the image width
        h = torch.sqrt(area / r).clamp(max=1.0)
        cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)  # crop center in [-1, 1] coordinates
        cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
        flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
        theta[:, 1, 1], theta[:, 1, 2] = h, cy
        grid = F.affine_grid(theta, (B, C, out_size, out_size), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    return augment


def center_resize(out_size: int = 224, crop_frac: float = 0.875):
    """Evaluation transform: take the central `crop_frac` of the image and resize to out_size."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        ch, cw = round(H * crop_frac), round(W * crop_frac)
        top, left = (H - ch) // 2, (W - cw) // 2
        x = x[..., top:top + ch, left:left + cw]
        return F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)

    return transform


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Standard CIFAR augmentation on a batch (B, C, H, W) in 0..1:
    pad by `pad` pixels (reflect), take a random crop of the original size, flip left-right half the time."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (dy[:, None] + torch.arange(H, device=x.device))[:, :, None]  # (B, H, 1)
    cols = (dx[:, None] + torch.arange(W, device=x.device))[:, None, :]  # (B, 1, W)
    out = xp.permute(0, 2, 3, 1)[torch.arange(B, device=x.device)[:, None, None], rows, cols]  # (B, H, W, C)
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()
