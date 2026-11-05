"""Small models built during the course, for reuse in later notebooks."""

from __future__ import annotations

import torch.nn as nn


def conv_bn(c_in: int, c_out: int, bn: bool = True) -> nn.Sequential:
    """3×3 conv → BatchNorm → ReLU (module 03)."""
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """Module 03's CIFAR-10 CNN: 3 stages of (conv-bn-relu ×2, maxpool), global average pool, linear head."""

    def __init__(self, widths=(32, 64, 128), bn: bool = True, n_classes: int = 10, in_ch: int = 3):
        super().__init__()
        stages, c = [], in_ch
        for w in widths:
            stages += [conv_bn(c, w, bn), conv_bn(w, w, bn), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
