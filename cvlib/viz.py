"""Plotting helpers for images and training curves."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _prep(img: np.ndarray, bgr: bool):
    """Return (image, imshow kwargs) ready for matplotlib."""
    img = np.asarray(img)
    if img.dtype == bool:
        return img.astype(np.uint8) * 255, dict(cmap="gray", vmin=0, vmax=255)
    if img.ndim == 2:
        kw = dict(cmap="gray")
        if img.dtype == np.uint8:
            kw.update(vmin=0, vmax=255)  # show true brightness, don't auto-stretch
        return img, kw
    if img.ndim == 3 and img.shape[2] == 1:
        return _prep(img[:, :, 0], bgr)
    if img.ndim == 3 and img.shape[2] in (3, 4) and bgr:
        code = cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGBA
        return cv2.cvtColor(img, code), {}
    return img, {}


def show(
    images,
    titles: Sequence[str] | str | None = None,
    cols: int | None = None,
    size: float = 4.0,
    bgr: bool = True,
    suptitle: str | None = None,
    **imshow_kwargs,
):
    """Show one image or a list of images in a grid.

    Color images are assumed to be BGR (OpenCV's order) and converted for display.
    Pass bgr=False for images that are already RGB (e.g. from PIL or scikit-image).
    Grayscale uint8 images are shown with their true brightness (0=black, 255=white).
    """
    if isinstance(images, np.ndarray) and (images.ndim == 2 or (images.ndim == 3 and images.shape[2] in (1, 3, 4))):
        images = [images]
    images = list(images)
    if isinstance(titles, str):
        titles = [titles]
    n = len(images)
    cols = cols or min(n, 4)
    rows = math.ceil(n / cols)
    h, w = np.asarray(images[0]).shape[:2]
    aspect = h / w if w else 1
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * aspect * rows + (0.4 if titles else 0)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes.flat[i]
        arr, kw = _prep(img, bgr)
        kw.update(imshow_kwargs)
        ax.imshow(arr, **kw)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_history(history: dict, keys: Sequence[str] | None = None):
    """Plot curves from a history dict like {"train_loss": [...], "val_loss": [...], "val_acc": [...]}.

    Losses and accuracies go on separate panels; train/val pairs share a panel.
    """
    keys = keys or [k for k, v in history.items() if isinstance(v, (list, tuple)) and len(v) > 0]
    groups: dict[str, list[str]] = {}
    for k in keys:
        metric = k.split("_", 1)[-1]  # "train_loss" -> "loss"
        groups.setdefault(metric, []).append(k)
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.5), squeeze=False)
    for ax, (metric, ks) in zip(axes.flat, groups.items()):
        for k in ks:
            ys = history[k]
            ax.plot(range(1, len(ys) + 1), ys, marker="o", label=k)
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def save_gif(frames: Sequence[np.ndarray], path, fps: int = 10, bgr: bool = True, max_width: int = 480):
    """Save a list of frames as an animated GIF (displayable inline with IPython.display.Image)."""
    from PIL import Image

    pil_frames = []
    for f in frames:
        if f.ndim == 3 and bgr:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if f.shape[1] > max_width:
            scale = max_width / f.shape[1]
            f = cv2.resize(f, (max_width, int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        pil_frames.append(Image.fromarray(f))
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)
    return path
