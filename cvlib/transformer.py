"""Attention and a Vision Transformer, written from scratch in module 08.

The module and parameter names match timm's VisionTransformer, so pretrained timm weights load straight into `ViT`
(see 08b). `MultiHeadAttention` does both self-attention and cross-attention (used again by DETR in module 09).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    """Scaled dot-product attention. q: (..., Nq, d), k and v: (..., Nk, d).
    mask: bool, broadcastable to (..., Nq, Nk), True = may attend. Returns (output, attention weights)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # (..., Nq, Nk): how well each query matches each key
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)                               # each row sums to 1
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with one fused q/k/v projection (the layout timm and nn.MultiheadAttention use).

    forward(x)            → self-attention: queries, keys and values all come from x
    forward(x, context)   → cross-attention: queries from x, keys and values from context
    keep_attn=True stores the last attention weights in .attn (for visualization).
    fast=True uses PyTorch's fused kernel (F.scaled_dot_product_attention) instead of `attention` above.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, fast: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.fast, self.keep_attn, self.attn = fast, False, None

    def _split(self, t):                       # (B, N, dim) → (B, heads, N, head_dim)
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context=None, mask=None):
        B, N, dim = x.shape
        if context is None:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        else:                                  # same weights, split: rows [0:dim] make queries, the rest keys and values
            w, b = self.qkv.weight, self.qkv.bias
            q = F.linear(x, w[:dim], None if b is None else b[:dim])
            k, v = F.linear(context, w[dim:], None if b is None else b[dim:]).chunk(2, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.fast and not self.keep_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out, w = attention(q, k, v, mask)
            self.attn = w.detach() if self.keep_attn else None
        return self.proj(out.transpose(1, 2).reshape(B, N, dim))   # concatenate heads, then mix them


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, fast: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads, fast=fast)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Image → sequence of patch tokens. A conv with kernel = stride = patch size is exactly
    "cut into non-overlapping patches, flatten each, apply the same linear layer"."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, dim: int = 768):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (B, 3, H, W) → (B, N, dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    """A plain Vision Transformer with a [CLS] token and learned position embeddings (ViT-B/16 by default)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, fast=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[Block(dim, num_heads, mlp_ratio, fast) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes) if num_classes else nn.Identity()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed
        return self.norm(self.blocks(x))       # (B, 1 + N, dim)

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])   # classify from the [CLS] token

    def keep_attention(self, on: bool = True):
        for b in self.blocks:
            b.attn.keep_attn = on
        return self

    def attention_maps(self):
        """Attention weights of every block from the last forward pass: list of (B, heads, 1+N, 1+N)."""
        return [b.attn.attn for b in self.blocks]
