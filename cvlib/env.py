"""Environment detection, so the same notebook runs on your Mac, Kaggle or Colab."""

from __future__ import annotations

import os
import platform as _platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Hugging Face's "Xet" download transport can fail on some networks (unreachable cas-server.xethub.hf.co).
# Plain HTTP downloads work everywhere. This must be set before huggingface_hub / timm / transformers are imported,
# which is why it lives here (the standard first cell imports cvlib.env first). Set HF_HUB_DISABLE_XET=0 to opt back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Env:
    platform: str  # "local", "kaggle" or "colab"
    device: str  # "cuda", "mps" or "cpu"
    root: Path  # project root (where cvlib/ lives)
    data_dir: Path  # where datasets and downloads are stored (writable)
    out_dir: Path  # where notebook outputs (images, GIFs, models) are saved


def detect_platform() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def get_device() -> str:
    """Best available accelerator: NVIDIA GPU (cuda) > Apple GPU (mps) > cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = 42) -> None:
    """Make results repeatable: same random numbers on every run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)  # also seeds cuda and mps
    except ImportError:
        pass


def setup(seed: int = 42, verbose: bool = True) -> Env:
    plat = detect_platform()
    if plat == "kaggle":
        data_dir = Path("/kaggle/working/data")
    elif plat == "colab":
        data_dir = Path("/content/data")
    else:
        data_dir = ROOT / "data"
    out_dir = data_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    env = Env(platform=plat, device=get_device(), root=ROOT, data_dir=data_dir, out_dir=out_dir)

    if verbose:
        parts = [f"platform={env.platform}", f"device={env.device}", f"python={_platform.python_version()}"]
        for name in ("numpy", "cv2", "torch"):
            mod = sys.modules.get(name)
            if mod is None:
                try:
                    mod = __import__(name)
                except ImportError:
                    continue
            parts.append(f"{name}={mod.__version__}")
        print(" | ".join(parts))
        print(f"data_dir={env.data_dir}")
    return env
