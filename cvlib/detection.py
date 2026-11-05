"""Object-detection utilities built in module 06a: box formats, IoU, NMS and COCO-style mAP.

Boxes are float arrays of shape (N, 4) in absolute pixels, "xyxy" = (x1, y1, x2, y2) unless stated otherwise.

    res = evaluate_detections(gts, preds)       # gts/preds: one dict per image
    res["mAP"], res["AP50"], res["AP_small"], res["per_class_AP50"]
"""

from __future__ import annotations

import numpy as np


# ── Box formats ──────────────────────────────────────────────────────────────
def box_convert(boxes, src: str, dst: str) -> np.ndarray:
    """Convert between "xyxy" (corners), "xywh" (top-left + size, COCO) and "cxcywh" (center + size, YOLO)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if src == dst:
        return b.copy()
    if src == "xywh":
        b = np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]]
    elif src == "cxcywh":
        b = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
    if dst == "xywh":
        return np.c_[b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
    if dst == "cxcywh":
        return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
    return b


def box_area(b) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def box_iou(a, b) -> np.ndarray:
    """IoU between every box in a (N, 4) and every box in b (M, 4) → (N, M)."""
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


# ── Non-maximum suppression ──────────────────────────────────────────────────
def nms(boxes, scores, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy NMS: keep the highest-scoring box, drop boxes overlapping it by more than the threshold, repeat.
    Returns the kept indices, highest score first."""
    boxes, scores = np.asarray(boxes, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def batched_nms(boxes, scores, labels, iou_threshold: float = 0.5) -> np.ndarray:
    """NMS separately per class (boxes of different classes never suppress each other).
    Trick: shift each class's boxes far apart so they can't overlap, then run one NMS."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)
    offset = np.asarray(labels, dtype=np.float64)[:, None] * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_threshold)


# ── COCO-style evaluation ────────────────────────────────────────────────────
AREA_RANGES = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def _match_class(gts_c, preds_c, iou_thr, area_rng, max_dets):
    """Match detections of one class to ground truth at one IoU threshold, COCO rules.
    Returns (scores, tp, ignored) for all detections and the number of non-ignored GT boxes."""
    scores_all, tp_all, ign_all, n_gt = [], [], [], 0
    for (g_boxes, g_ignore), (d_boxes, d_scores) in zip(gts_c, preds_c):
        g_area = box_area(g_boxes)
        g_ig = g_ignore | (g_area < area_rng[0]) | (g_area > area_rng[1])
        g_order = np.argsort(g_ig, kind="stable")  # non-ignored GT first
        g_boxes, g_ig = g_boxes[g_order], g_ig[g_order]
        n_gt += int((~g_ig).sum())
        d_order = np.argsort(-d_scores, kind="stable")[:max_dets]
        d_boxes, d_scores = d_boxes[d_order], d_scores[d_order]
        ious = box_iou(d_boxes, g_boxes) if len(g_boxes) and len(d_boxes) else np.zeros((len(d_boxes), len(g_boxes)))
        g_taken = np.zeros(len(g_boxes), bool)
        tp, d_ig = np.zeros(len(d_boxes), bool), np.zeros(len(d_boxes), bool)
        for di in range(len(d_boxes)):
            best, best_iou = -1, min(iou_thr, 1 - 1e-10)
            for gi in range(len(g_boxes)):
                if g_taken[gi]:
                    continue
                if best > -1 and not g_ig[best] and g_ig[gi]:  # already matched a real GT: don't switch to an ignored one
                    break
                if ious[di, gi] < best_iou:
                    continue
                best, best_iou = gi, ious[di, gi]
            if best == -1:
                continue
            g_taken[best] = True
            d_ig[di] = g_ig[best]
            tp[di] = not g_ig[best]
        # unmatched detections outside the area range are ignored, not counted as false positives
        d_area = box_area(d_boxes)
        d_ig |= (~tp) & (~d_ig) & ((d_area < area_rng[0]) | (d_area > area_rng[1]))
        scores_all.append(d_scores); tp_all.append(tp); ign_all.append(d_ig)
    if not scores_all:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), n_gt
    return np.concatenate(scores_all), np.concatenate(tp_all), np.concatenate(ign_all), n_gt


def _ap_101(scores, tp, ignored, n_gt):
    """COCO AP: precision envelope sampled at 101 recall points. Also returns the raw PR curve."""
    if n_gt == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    keep = ~ignored
    scores, tp = scores[keep], tp[keep]
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    tps, fps = np.cumsum(tp), np.cumsum(~tp)
    recall = tps / n_gt
    precision = tps / np.maximum(tps + fps, np.spacing(1))
    env = precision.copy()
    for i in range(len(env) - 1, 0, -1):  # make precision non-increasing
        env[i - 1] = max(env[i - 1], env[i])
    rec_thr = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rec_thr, side="left")
    q = np.array([env[i] if i < len(env) else 0.0 for i in idx])
    return float(q.mean()), precision, recall


def evaluate_detections(gts, preds, n_classes: int | None = None, iou_thresholds=None, max_dets: int = 100) -> dict:
    """COCO-style detection evaluation (matches pycocotools' bbox evaluation).

    gts:   list (one per image) of dicts with "boxes" (N, 4) xyxy, "labels" (N,), optional "ignore" (N,) bool
           (e.g. VOC "difficult" or COCO "crowd" boxes: matching them is neither rewarded nor punished)
    preds: list (one per image) of dicts with "boxes" (M, 4) xyxy, "scores" (M,), "labels" (M,)
    Returns mAP (IoU .50:.95), AP50, AP75, AP_small/medium/large, per-class AP and AP50, and PR curves at IoU 0.5.
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10) if iou_thresholds is None else np.asarray(iou_thresholds)
    labels_all = np.concatenate([np.asarray(g["labels"]).ravel() for g in gts] + [np.zeros(0)])
    classes = np.arange(n_classes) if n_classes is not None else np.unique(labels_all).astype(int)

    def per_image(c):
        gc, pc = [], []
        for g, p in zip(gts, preds):
            gl = np.asarray(g["labels"]).ravel()
            gb = np.asarray(g["boxes"], dtype=np.float64).reshape(-1, 4)[gl == c]
            gi = np.asarray(g.get("ignore", np.zeros(len(gl), bool))).ravel().astype(bool)[gl == c]
            pl = np.asarray(p["labels"]).ravel()
            pb = np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4)[pl == c]
            ps = np.asarray(p["scores"], dtype=np.float64).ravel()[pl == c]
            gc.append((gb, gi)); pc.append((pb, ps))
        return gc, pc

    ap = {rng: np.full((len(classes), len(iou_thresholds)), np.nan) for rng in AREA_RANGES}
    pr_curves = {}
    for ci, c in enumerate(classes):
        gc, pc = per_image(c)
        for rng, area_rng in AREA_RANGES.items():
            for ti, t in enumerate(iou_thresholds):
                s, tp, ig, n_gt = _match_class(gc, pc, t, area_rng, max_dets)
                ap[rng][ci, ti], prec, rec = _ap_101(s, tp, ig, n_gt)
                if rng == "all" and np.isclose(t, 0.5):
                    pr_curves[int(c)] = (prec, rec)

    def mean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    i50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    i75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))
    return dict(
        mAP=mean(ap["all"]), AP50=mean(ap["all"][:, i50]), AP75=mean(ap["all"][:, i75]),
        AP_small=mean(ap["small"]), AP_medium=mean(ap["medium"]), AP_large=mean(ap["large"]),
        per_class_AP={int(c): float(np.nanmean(ap["all"][i])) if np.isfinite(ap["all"][i]).any() else float("nan") for i, c in enumerate(classes)},
        per_class_AP50={int(c): float(ap["all"][i, i50]) for i, c in enumerate(classes)},
        pr_curves=pr_curves, classes=[int(c) for c in classes],
    )


def to_coco(gts, preds, n_classes: int):
    """Build pycocotools-style ground-truth and detection lists from our per-image dicts (for cross-checking)."""
    images, annotations, dets, ann_id = [], [], [], 1
    for img_id, (g, p) in enumerate(zip(gts, preds), start=1):
        images.append({"id": img_id, "width": 10000, "height": 10000})
        ign = np.asarray(g.get("ignore", np.zeros(len(g["labels"]), bool))).astype(bool)
        for b, lab, ig in zip(box_convert(g["boxes"], "xyxy", "xywh"), np.asarray(g["labels"]).ravel(), ign):
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": int(lab) + 1, "bbox": [float(v) for v in b],
                                "area": float(b[2] * b[3]), "iscrowd": 0, "ignore": int(ig)})
            ann_id += 1
        for b, s, lab in zip(box_convert(p["boxes"], "xyxy", "xywh"), np.asarray(p["scores"]).ravel(), np.asarray(p["labels"]).ravel()):
            dets.append({"image_id": img_id, "category_id": int(lab) + 1, "bbox": [float(v) for v in b], "score": float(s)})
    gt = {"images": images, "annotations": annotations, "categories": [{"id": i + 1, "name": str(i)} for i in range(n_classes)]}
    return gt, dets


def error_breakdown(gts, preds, score_thr: float = 0.3, iou_thr: float = 0.5) -> dict:
    """Count true positives and classify every error (module 06a, section 5.3).

    False positives: localization (right class, 0.1 ≤ IoU < iou_thr), duplicate (object already detected),
    class confusion (good overlap with an object of another class), background (nothing there).
    Plus missed objects. Ground-truth boxes with ignore=True are skipped.
    """
    counts = {"TP": 0, "localization": 0, "duplicate": 0, "class confusion": 0, "background": 0, "missed": 0}
    for g, p in zip(gts, preds):
        keep = ~np.asarray(g.get("ignore", np.zeros(len(g["labels"]), bool))).astype(bool)
        gb, gl = np.asarray(g["boxes"]).reshape(-1, 4)[keep], np.asarray(g["labels"])[keep]
        sel = np.asarray(p["scores"]) >= score_thr
        pb, ps, pl = np.asarray(p["boxes"]).reshape(-1, 4)[sel], np.asarray(p["scores"])[sel], np.asarray(p["labels"])[sel]
        matched = np.zeros(len(gb), bool)
        ious = box_iou(pb, gb) if len(gb) and len(pb) else np.zeros((len(pb), len(gb)))
        for i in np.argsort(-ps):
            same = gl == pl[i]
            iou_same = np.where(same, ious[i], -1)
            j = int(iou_same.argmax()) if len(gb) else -1
            if j >= 0 and iou_same[j] >= iou_thr and not matched[j]:
                counts["TP"] += 1
                matched[j] = True
            elif j >= 0 and iou_same[j] >= iou_thr:
                counts["duplicate"] += 1
            elif len(gb) and np.where(~same, ious[i], -1).max() >= iou_thr:
                counts["class confusion"] += 1
            elif j >= 0 and iou_same[j] >= 0.1:
                counts["localization"] += 1
            else:
                counts["background"] += 1
        counts["missed"] += int((~matched).sum())
    return counts


def mask_iou(a, b) -> np.ndarray:
    """IoU between every mask in a (N, H, W) and every mask in b (M, H, W) → (N, M)."""
    a = np.asarray(a, dtype=bool).reshape(len(a), -1).astype(np.float32)
    b = np.asarray(b, dtype=bool).reshape(len(b), -1).astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def coco_mask_ap(gt_masks, pred_masks, pred_scores) -> dict:
    """Instance-segmentation AP with pycocotools ("segm" IoU = mask IoU), single class.

    gt_masks: list (per image) of (N, H, W) bool arrays; pred_masks: list of (M, H, W) bool; pred_scores: list of (M,).
    Returns mAP, AP50, AP75 (plus size breakdown), computed on masks instead of boxes."""
    import contextlib
    import io

    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    def rle(m):
        r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        r["counts"] = r["counts"].decode("ascii")
        return r

    images, anns, dets, aid = [], [], [], 1
    for img_id, (g, p, s) in enumerate(zip(gt_masks, pred_masks, pred_scores), start=1):
        H, W = (g.shape[1:] if len(g) else p.shape[1:]) if (len(g) or len(p)) else (1, 1)
        images.append({"id": img_id, "height": int(H), "width": int(W)})
        for m in g:
            r = rle(m)
            anns.append({"id": aid, "image_id": img_id, "category_id": 1, "segmentation": r, "area": float(mask_utils.area(r)),
                         "bbox": [float(v) for v in mask_utils.toBbox(r)], "iscrowd": 0})
            aid += 1
        for m, sc in zip(p, s):
            dets.append({"image_id": img_id, "category_id": 1, "segmentation": rle(m), "score": float(sc)})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "object"}]}; gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        E = COCOeval(gt, dt, "segm"); E.evaluate(); E.accumulate(); E.summarize()
    return dict(zip(["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"], [float(v) for v in E.stats[:6]]))
