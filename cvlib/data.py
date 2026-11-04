"""Download helpers with caching: each file is downloaded once, then reused."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .env import setup

OPENCV_SAMPLES = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
OPENCV_STITCHING = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/"

_DATA_DIR: Path | None = None


def _data_dir() -> Path:
    global _DATA_DIR
    if _DATA_DIR is None:
        _DATA_DIR = setup(verbose=False).data_dir
    return _DATA_DIR


def download(url: str, subdir: str = "downloads", filename: str | None = None) -> Path:
    """Download `url` into data/<subdir>/ unless it is already there. Returns the local path."""
    dest_dir = _data_dir() / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (filename or url.rsplit("/", 1)[-1])
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "cv-mastery"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(dest)
    return dest


def opencv_sample(name: str) -> Path:
    """A file from OpenCV's samples/data folder, e.g. opencv_sample("sudoku.png")."""
    return download(OPENCV_SAMPLES + name, subdir="opencv_samples")


def stitching_sample(name: str) -> Path:
    """An image from OpenCV's panorama test set, e.g. stitching_sample("budapest1.jpg")."""
    return download(OPENCV_STITCHING + name, subdir="opencv_stitching")
