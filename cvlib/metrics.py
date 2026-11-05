"""Evaluation metrics built from scratch during the course (05b, 05c), reused later (06 detection mAP, 14 evaluation)."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_metrics(y_true, y_pred, n_classes: int) -> dict:
    """Single-label classification: accuracy, macro/micro F1, and per-class precision/recall/F1."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = cm.diagonal().astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    acc = tp.sum() / cm.sum()
    return dict(accuracy=acc, macro_f1=f1.mean(), micro_f1=acc, precision=precision, recall=recall, f1=f1, cm=cm)


def pr_curve(scores, positives):
    """Precision and recall at every threshold, highest score first."""
    scores, positives = np.asarray(scores), np.asarray(positives).astype(bool)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(positives[order])
    fp = np.cumsum(~positives[order])
    precision = tp / (tp + fp)
    recall = tp / max(positives.sum(), 1)
    return precision, recall


def average_precision(scores, positives) -> float:
    """Area under the (interpolated) precision-recall curve. Threshold-free ranking quality for one class."""
    positives = np.asarray(positives).astype(bool)
    if positives.sum() == 0:
        return float("nan")
    precision, recall = pr_curve(scores, positives)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # interpolate: precision never increases with recall
    r_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - r_prev) * precision))


def multilabel_metrics(probs, targets, thresholds=0.5) -> dict:
    """Multi-label metrics. probs, targets: (N, C); thresholds: a scalar or one per class.

    Returns per-class AP and mAP (threshold-free), per-class/macro/micro F1 at the thresholds,
    exact-match ratio (all labels of an image right) and Hamming loss (fraction of wrong label decisions).
    """
    probs, targets = np.asarray(probs), np.asarray(targets).astype(bool)
    pred = probs >= np.broadcast_to(np.asarray(thresholds, dtype=float), (probs.shape[1],))
    ap = np.array([average_precision(probs[:, c], targets[:, c]) for c in range(probs.shape[1])])
    tp = (pred & targets).sum(0); fp = (pred & ~targets).sum(0); fn = (~pred & targets).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    micro_f1 = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return dict(ap=ap, mAP=float(np.nanmean(ap)), f1=f1, macro_f1=float(f1.mean()), micro_f1=float(micro_f1),
                exact_match=float((pred == targets).all(1).mean()), hamming_loss=float((pred != targets).mean()))


def best_thresholds(probs, targets, grid=np.linspace(0.05, 0.95, 19)) -> np.ndarray:
    """Per-class threshold that maximizes F1 (choose on validation data, never on test)."""
    probs, targets = np.asarray(probs), np.asarray(targets).astype(bool)
    out = np.empty(probs.shape[1])
    for c in range(probs.shape[1]):
        f1s = []
        for t in grid:
            p = probs[:, c] >= t
            tp, fp, fn = (p & targets[:, c]).sum(), (p & ~targets[:, c]).sum(), (~p & targets[:, c]).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        out[c] = grid[int(np.argmax(f1s))]
    return out
