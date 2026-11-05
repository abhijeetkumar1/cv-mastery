"""Dataset helpers: download once, cache in data/, return ready-to-use arrays or folders.

    X, y = cifar10("train")                    # uint8 (N, 32, 32, 3) RGB, int64 labels
    root = imagenette()                        # folder with train/ and val/, ImageFolder-ready
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np

from .data import _data_dir, download

# ── CIFAR-10 ─────────────────────────────────────────────────────────────────
# The official host (cs.toronto.edu) can be very slow, so we use the Hugging Face copy (Parquet files).
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
_CIFAR10_HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def cifar10(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 as NumPy arrays: images uint8 (N, 32, 32, 3) in RGB order, labels int64 (N,).

    split: "train" (50,000 images) or "test" (10,000 images). Cached as .npz after the first call.
    """
    cache = _data_dir() / "cifar10" / f"{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]

    import pyarrow.parquet as pq
    from PIL import Image

    pq_path = download(_CIFAR10_HF.format(split=split), subdir="cifar10", filename=f"{split}.parquet")
    table = pq.read_table(pq_path).to_pydict()
    img_col = next(k for k in table if "im" in k.lower())  # "img"
    x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table[img_col]])
    y = np.array(table["label"], dtype=np.int64)
    np.savez(cache, x=x, y=y)
    pq_path.unlink()  # keep only the compact .npz
    return x, y


# ── Imagenette: 10 easy ImageNet classes (fast.ai) ───────────────────────────
# Because they are real ImageNet classes, ImageNet-pretrained models can be scored on them with no training.
IMAGENETTE_CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church",
                      "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]  # index in the 1000 ImageNet classes


def imagenette(size: int = 160) -> Path:
    """Download and extract Imagenette (size 160 or 320 px on the short side). Returns the folder with train/ and val/."""
    name = f"imagenette2-{size}"
    root = _data_dir() / "imagenette" / name
    if not (root / "val").exists():
        tgz = download(f"https://s3.amazonaws.com/fast-ai-imageclas/{name}.tgz", subdir="imagenette")
        with tarfile.open(tgz) as tf:
            tf.extractall(root.parent, filter="data")
        tgz.unlink()
    return root


# ── Pre-resized image caches ─────────────────────────────────────────────────
def cache_images(name: str, items, size: int, labels=None) -> tuple[np.ndarray, np.ndarray]:
    """Decode + resize images once and cache them as a uint8 array (N, size, size, 3) on disk.

    items: a sequence of image paths, or a torchvision dataset returning (PIL image, label).
    Each image is resized so its short side = size, then center-cropped to size × size.
    The array is memory-mapped, so a large cache doesn't have to fit in RAM.
    Returns (images, labels).
    """
    from PIL import Image

    d = _data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    x_path, y_path = d / f"{name}_{size}_x.npy", d / f"{name}_{size}_y.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path, mmap_mode="r"), np.load(y_path)

    n = len(items)
    x = np.lib.format.open_memmap(x_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    ys = []
    for i in range(n):
        if isinstance(items[i], (str, Path)):
            img, lab = Image.open(items[i]), (labels[i] if labels is not None else -1)
        else:
            img, lab = items[i]
        img = img.convert("RGB")
        w, h = img.size
        s = size / min(w, h)
        img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        x[i] = np.asarray(img.crop((left, top, left + size, top + size)))
        ys.append(lab)
    x.flush()
    del x
    x_path.with_suffix(".tmp.npy").rename(x_path)
    np.save(y_path, np.asarray(ys))
    return np.load(x_path, mmap_mode="r"), np.load(y_path)


def stratified_split(y: np.ndarray, fractions, seed: int = 0) -> list[np.ndarray]:
    """Split indices into parts with the given fractions, keeping class proportions in every part."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fractions]
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        bounds = np.round(np.cumsum([0, *fractions]) * len(idx)).astype(int)
        for k in range(len(fractions)):
            parts[k].extend(idx[bounds[k]:bounds[k + 1]])
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


# ── PASCAL VOC 2007 as a multi-label classification dataset ──────────────────
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def voc2007_multilabel(image_set: str = "trainval") -> tuple[list[Path], np.ndarray]:
    """VOC 2007 turned into multi-label classification.

    Returns (image paths, labels) where labels is a float32 (N, 20) 0/1 matrix: 1 if at least one
    object of that class appears in the image. "Difficult" objects are ignored.
    """
    import xml.etree.ElementTree as ET

    from torchvision.datasets import VOCDetection

    root = _data_dir() / "voc"
    VOCDetection(str(root), year="2007", image_set=image_set, download=not (root / "VOCdevkit" / "VOC2007" / "Annotations").exists())
    base = root / "VOCdevkit" / "VOC2007"
    ids = (base / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split()
    paths, labels = [], np.zeros((len(ids), len(VOC_CLASSES)), np.float32)
    for i, img_id in enumerate(ids):
        tree = ET.parse(base / "Annotations" / f"{img_id}.xml")
        for obj in tree.findall("object"):
            if obj.findtext("difficult", "0") == "1":
                continue
            labels[i, VOC_CLASSES.index(obj.findtext("name"))] = 1
        paths.append(base / "JPEGImages" / f"{img_id}.jpg")
    return paths, labels


# ── Detection & segmentation datasets (module 06–07) ─────────────────────────
# VOC names → COCO names (COCO-pretrained detectors know all 20 VOC classes under these names)
VOC_TO_COCO = {"aeroplane": "airplane", "bicycle": "bicycle", "bird": "bird", "boat": "boat", "bottle": "bottle",
               "bus": "bus", "car": "car", "cat": "cat", "chair": "chair", "cow": "cow", "diningtable": "dining table",
               "dog": "dog", "horse": "horse", "motorbike": "motorcycle", "person": "person", "pottedplant": "potted plant",
               "sheep": "sheep", "sofa": "couch", "train": "train", "tvmonitor": "tv"}


def voc2007_detection(image_set: str = "val") -> list[dict]:
    """VOC 2007 boxes: one dict per image with path, boxes (N, 4) xyxy, labels (N,) VOC class ids, difficult (N,) bool."""
    import xml.etree.ElementTree as ET

    voc2007_multilabel("trainval")  # makes sure the dataset is downloaded
    base = _data_dir() / "voc" / "VOCdevkit" / "VOC2007"
    out = []
    for img_id in (base / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split():
        root = ET.parse(base / "Annotations" / f"{img_id}.xml").getroot()
        boxes, labels, diff = [], [], []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            boxes.append([float(bb.findtext(k)) - (1 if k in ("xmin", "ymin") else 0) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(VOC_CLASSES.index(obj.findtext("name")))
            diff.append(obj.findtext("difficult", "0") == "1")
        size = root.find("size")
        out.append(dict(id=img_id, path=base / "JPEGImages" / f"{img_id}.jpg", width=int(size.findtext("width")),
                        height=int(size.findtext("height")), boxes=np.array(boxes, np.float32).reshape(-1, 4),
                        labels=np.array(labels, np.int64), difficult=np.array(diff, bool)))
    return out


def voc2007_segmentation(image_set: str = "val") -> list[dict]:
    """VOC 2007 segmentation subset (422 images): image path + class mask path (PNG: 0 = background, 1–20 = VOC classes, 255 = boundary/ignore)."""
    voc2007_multilabel("trainval")
    base = _data_dir() / "voc" / "VOCdevkit" / "VOC2007"
    ids = (base / "ImageSets" / "Segmentation" / f"{image_set}.txt").read_text().split()
    return [dict(id=i, path=base / "JPEGImages" / f"{i}.jpg", mask=base / "SegmentationClass" / f"{i}.png",
                 instances=base / "SegmentationObject" / f"{i}.png") for i in ids]


def pennfudan() -> list[dict]:
    """Penn-Fudan pedestrians (170 images): image path, instance mask path (0 = background, 1..N = person id), boxes (N, 4) xyxy."""
    import zipfile

    from PIL import Image

    root = _data_dir() / "pennfudan" / "PennFudanPed"
    if not root.exists():
        z = download("https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip", subdir="pennfudan")
        zipfile.ZipFile(z).extractall(z.parent)
    out = []
    for img in sorted((root / "PNGImages").glob("*.png")):
        mpath = root / "PedMasks" / f"{img.stem}_mask.png"
        m = np.array(Image.open(mpath))
        ids = [i for i in np.unique(m) if i != 0]
        boxes = []
        for i in ids:
            ys, xs = np.nonzero(m == i)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        out.append(dict(id=img.stem, path=img, mask=mpath, boxes=np.array(boxes, np.float32).reshape(-1, 4), instance_ids=np.array(ids)))
    return out


def cache_pairs(name: str, items, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache (image, mask) pairs: images bicubic-resized, masks NEAREST-resized (labels must stay whole numbers),
    both with the same short-side resize + center crop. items: sequence of (PIL image, PIL mask) or dicts with
    "path" and "mask" keys. Returns uint8 arrays (N, size, size, 3) and (N, size, size), memory-mapped."""
    from PIL import Image

    d = _data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    x_path, m_path = d / f"{name}_{size}_x.npy", d / f"{name}_{size}_m.npy"
    if x_path.exists() and m_path.exists():
        return np.load(x_path, mmap_mode="r"), np.load(m_path, mmap_mode="r")
    n = len(items)
    X = np.lib.format.open_memmap(x_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    M = np.lib.format.open_memmap(m_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, size, size))
    for i in range(n):
        it = items[i]
        img, mask = (Image.open(it["path"]), Image.open(it["mask"])) if isinstance(it, dict) else it
        img = img.convert("RGB")
        w, h = img.size
        s = size / min(w, h)
        new = (max(size, round(w * s)), max(size, round(h * s)))
        img, mask = img.resize(new, Image.BICUBIC), mask.resize(new, Image.NEAREST)
        left, top = (new[0] - size) // 2, (new[1] - size) // 2
        box = (left, top, left + size, top + size)
        X[i], M[i] = np.asarray(img.crop(box)), np.asarray(mask.crop(box))
    X.flush(); M.flush()
    del X, M
    x_path.with_suffix(".tmp.npy").rename(x_path)
    m_path.with_suffix(".tmp.npy").rename(m_path)
    return np.load(x_path, mmap_mode="r"), np.load(m_path, mmap_mode="r")
