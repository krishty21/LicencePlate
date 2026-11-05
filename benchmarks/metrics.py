"""Metrics — single source of truth for evaluation.

Every metric is a function with a defined signature. No notebooks, no magic.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------- detection metrics ----------

def iou(box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float]) -> float:
    """IoU for (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    return inter / max(a + b - inter, 1e-9)


def average_precision(preds: list[tuple[tuple, float]],
                      gts: list[tuple],
                      iou_thresh: float = 0.5) -> float:
    """AP @ single IoU threshold. preds = [(box, score), ...]; gts = [box, ...]."""
    if not gts:
        return 0.0
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    gt_matched = [False] * len(gts)
    for i, (pbox, _) in enumerate(preds_sorted):
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gts):
            if gt_matched[j]:
                continue
            cur = iou(pbox, gbox)
            if cur > best_iou:
                best_iou = cur
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            gt_matched[best_j] = True
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / len(gts)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0.0) / 11.0
    return float(ap)


def mean_ap_at_iou(preds_per_image: list[list[tuple]],
                   gts_per_image: list[list[tuple]],
                   iou_thresh: float = 0.5) -> float:
    """mAP at single IoU threshold across images."""
    aps = []
    for preds, gts in zip(preds_per_image, gts_per_image):
        aps.append(average_precision(preds, gts, iou_thresh))
    return float(np.mean(aps)) if aps else 0.0


# ---------- corner error ----------

def corner_error_px(pred_corners: np.ndarray,
                    gt_corners: np.ndarray) -> float:
    """Mean L2 distance between predicted and GT corners (in pixels)."""
    if pred_corners.shape != gt_corners.shape:
        return float("inf")
    diff = pred_corners - gt_corners
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))


# ---------- string metrics ----------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if len(a) < len(b):
        return edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein / max(len) — bounded in [0, 1]."""
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))


def cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit_distance / len(gt)."""
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


def char_accuracy(pred: str, gt: str) -> float:
    """1 - CER."""
    return 1.0 - cer(pred, gt)


def exact_match(pred: str, gt: str) -> int:
    """1 if strings match exactly, else 0."""
    return int(pred == gt)


# ---------- aggregation ----------

@dataclass
class PlateMetrics:
    """Aggregated metrics across a test set."""
    n: int = 0
    plate_accuracy: float = 0.0          # exact match rate
    char_accuracy: float = 0.0           # 1 - CER
    normalized_edit_distance: float = 0.0
    grammar_valid_rate: float = 0.0      # fraction of reads that pass grammar
    corrected_rate: float = 0.0          # fraction of reads with state-code correction
    enhanced_rate: float = 0.0           # fraction of reads that triggered ESRGAN
    parseq_rate: float = 0.0             # fraction of reads that used PARSeq
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


def aggregate(reads: list[tuple[str, str, dict]]) -> PlateMetrics:
    """Aggregate per-plate (pred, gt, meta) tuples into PlateMetrics.

    meta keys: grammar_valid, corrected, enhanced, recognizer, latency_ms
    """
    n = len(reads)
    if n == 0:
        return PlateMetrics()
    em = sum(exact_match(p, g) for p, g, _ in reads)
    cas = [char_accuracy(p, g) for p, g, _ in reads]
    neds = [normalized_edit_distance(p, g) for p, g, _ in reads]
    gv = sum(1 for _, _, m in reads if m.get("grammar_valid"))
    cr = sum(1 for _, _, m in reads if m.get("corrected"))
    en = sum(1 for _, _, m in reads if m.get("enhanced"))
    pq = sum(1 for _, _, m in reads if m.get("recognizer") == "parseq")
    lats = sorted(m.get("latency_ms", 0.0) for _, _, m in reads)
    p50 = lats[len(lats) // 2]
    p95 = lats[int(len(lats) * 0.95)] if len(lats) >= 20 else lats[-1]
    return PlateMetrics(
        n=n,
        plate_accuracy=em / n,
        char_accuracy=sum(cas) / n,
        normalized_edit_distance=sum(neds) / n,
        grammar_valid_rate=gv / n,
        corrected_rate=cr / n,
        enhanced_rate=en / n,
        parseq_rate=pq / n,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
    )
