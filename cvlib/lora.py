"""LoRA (Low-Rank Adaptation), written from scratch in module 05a and reused for ViTs (08) and VLMs (13).

Idea: keep a pretrained weight W frozen and learn a small low-rank update: W' = W + (alpha / r) · B·A
with A: (r × in) and B: (out × r), r ≪ in, out. B starts at zero, so training starts exactly from the pretrained model.

    add_lora(model, r=8, target=lambda name, m: name.startswith("layer4"))
    # → wraps matching Conv2d / Linear layers, freezes everything else
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero → no change at the start
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


class LoRAConv2d(nn.Module):
    """LoRA for convolutions: a k×k conv down to r channels (A), then a 1×1 conv back up (B)."""

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride, base.padding,
                           dilation=base.dilation, groups=1, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale


def add_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, target=None) -> nn.Module:
    """Freeze `model` and wrap every Conv2d/Linear for which target(name, module) is True with LoRA.

    Grouped/depthwise convs are skipped (LoRA on them saves nothing). Returns the model (modified in place).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    target = target or (lambda name, m: True)
    to_wrap = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and target(name, m):
            to_wrap.append((name, LoRALinear(m, r, alpha)))
        elif isinstance(m, nn.Conv2d) and m.groups == 1 and target(name, m):
            to_wrap.append((name, LoRAConv2d(m, r, alpha)))
    for name, new in to_wrap:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, new)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
