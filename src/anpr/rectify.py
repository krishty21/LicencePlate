"""Perspective rectification — homography from 4 corners → class-aware target shape.

Implements v1.1 spec §2.2 + §4.4 (class-aware shapes for two-wheelers/cars/trucks).
The v1.0 spec used a fixed 192×64 (3:1) target — wrong for ~70% of Indian plates.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .types import Corner, PlateClass, PlateColor, PlateDetection

logger = logging.getLogger(__name__)

# v1.1 class-aware rectification targets — REVIEW_AND_RECOMMENDATIONS.md §4.4.
# (W, H) tuples. Recognizer pads to max envelope internally if needed.
PLATE_SHAPES: dict[PlateClass, tuple[int, int]] = {
    PlateClass.CAR: (200, 48),               # ~4.17:1
    PlateClass.TWO_WHEELER_REAR: (128, 64),  # 2:1
    PlateClass.TWO_WHEELER_FRONT: (192, 30), # 6.4:1
    PlateClass.TRUCK: (192, 112),            # 1.7:1
    PlateClass.THREE_WHEELER: (128, 64),     # 2:1
    PlateClass.UNKNOWN: (192, 64),           # v1.0 fallback
}

# Maximum envelope for the recognizer input.
MAX_ENVELOPE: tuple[int, int] = (200, 112)   # (W, H)


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as [TL, TR, BR, BL] clockwise from top-left.

    Robust to any input ordering. Algorithm:
      1. Sum and diff of (x, y) — TL has min sum, BR has max sum,
         TR has max diff (x - y), BL has min diff.
    """
    if corners.shape != (4, 2):
        raise ValueError(f"expected (4, 2) corners, got {corners.shape}")
    s = corners[:, 0] + corners[:, 1]
    d = corners[:, 0] - corners[:, 1]
    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmin(d)]   # x - y smallest = x small, y large? No:
    # actually tr = max(x - y) when y is small. Let me re-derive.
    # For a plate image with origin at top-left, x grows right, y grows down:
    #   TL: small x, small y → small (x+y), small (x-y)
    #   TR: large x, small y → small (x+y) approximately, large (x-y)
    #   BR: large x, large y → large (x+y), small (x-y) approximately
    #   BL: small x, large y → large (x+y) approximately, large (x-y)?? No.
    # Let's use the canonical approach: min sum = TL, max sum = BR,
    # then of the remaining two, smaller x = BL, larger x = TR.
    tl_idx = int(np.argmin(s))
    br_idx = int(np.argmax(s))
    remaining = [i for i in range(4) if i not in (tl_idx, br_idx)]
    remaining_pts = corners[remaining]
    # The one with smaller x is BL, larger x is TR.
    if remaining_pts[0, 0] < remaining_pts[1, 0]:
        bl = remaining_pts[0]
        tr = remaining_pts[1]
    else:
        bl = remaining_pts[1]
        tr = remaining_pts[0]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def is_quad_valid(corners: np.ndarray, min_area_px: float = 100.0,
                  max_aspect: float = 8.0) -> bool:
    """Sanity gate: reject degenerate quads (false positives)."""
    if corners.shape != (4, 2):
        return False
    ordered = order_corners(corners)
    # Use OpenCV's contourArea — handles self-intersection correctly.
    area = cv2.contourArea(ordered)
    if area < min_area_px:
        return False
    # Aspect ratio from min/max bounding rect of the ordered corners.
    w = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
    h = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))
    if h < 1e-6:
        return False
    aspect = w / h
    if aspect > max_aspect or aspect < 1.0 / max_aspect:
        return False
    return True


def classify_plate_shape(corners: np.ndarray) -> PlateClass:
    """Infer plate class from the corner aspect ratio.

    This is a heuristic — the detector can also be trained to output a class
    directly. Used as a fallback when the detector doesn't classify.
    """
    ordered = order_corners(corners)
    w = float(np.max(ordered[:, 0]) - np.min(ordered[:, 0]))
    h = float(np.max(ordered[:, 1]) - np.min(ordered[:, 1]))
    if h < 1e-6:
        return PlateClass.UNKNOWN
    aspect = w / h
    # Map aspect ratio → class. Tolerant because pre-rectification aspect
    # is distorted by perspective.
    if aspect > 5.5:
        return PlateClass.TWO_WHEELER_FRONT
    if aspect > 3.5:
        return PlateClass.CAR
    if aspect > 2.7:
        return PlateClass.UNKNOWN  # ambiguous — use fallback shape
    if aspect > 1.5:
        return PlateClass.TWO_WHEELER_REAR
    return PlateClass.TRUCK


def rectify(image: np.ndarray, detection: PlateDetection,
            pad_to_envelope: bool = True) -> tuple[np.ndarray, PlateClass]:
    """Rectify one plate region to its class-appropriate target shape.

    Returns:
        (plate_img, plate_class) where plate_img is (H, W, 3) BGR uint8.

    Failure mode: if homography is ill-conditioned, log a warning and return
    an axis-aligned bbox crop (un-rectified) at the target shape — the OCR
    may still recover the text from the un-rectified crop.
    """
    t0 = time.perf_counter()
    corners = detection.corner_array
    if not is_quad_valid(corners):
        logger.warning("degenerate quad, falling back to bbox crop")
        x1, y1, x2, y2 = detection.bbox
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        plate_class = detection.plate_class
        target_w, target_h = PLATE_SHAPES.get(plate_class, PLATE_SHAPES[PlateClass.UNKNOWN])
        if crop.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8), plate_class
        return cv2.resize(crop, (target_w, target_h)), plate_class

    ordered = order_corners(corners)
    plate_class = detection.plate_class
    if plate_class == PlateClass.UNKNOWN:
        plate_class = classify_plate_shape(corners)
    target_w, target_h = PLATE_SHAPES.get(plate_class, PLATE_SHAPES[PlateClass.UNKNOWN])

    dst = np.array(
        [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]],
        dtype=np.float32,
    )
    try:
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h),
                                      flags=cv2.INTER_LINEAR)
    except cv2.error as e:
        logger.warning("homography ill-conditioned: %s, falling back to bbox", e)
        x1, y1, x2, y2 = detection.bbox
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8), plate_class
        warped = cv2.resize(crop, (target_w, target_h))

    if pad_to_envelope and (target_w, target_h) != MAX_ENVELOPE:
        env_w, env_h = MAX_ENVELOPE
        padded = np.zeros((env_h, env_w, 3), dtype=np.uint8)
        padded[:target_h, :target_w] = warped
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("rectify: %.2fms shape=%s class=%s", elapsed, (target_w, target_h), plate_class)
        return padded, plate_class

    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug("rectify: %.2fms shape=%s class=%s", elapsed, (target_w, target_h), plate_class)
    return warped, plate_class


def classify_color(plate_img: np.ndarray) -> PlateColor:
    """Classify plate color from HSV histogram peaks.

    Cheap (sub-ms). Used to choose per-color contrast normalization.
    See PlateColor enum for the 9 official Indian combinations.
    """
    if plate_img.size == 0:
        return PlateColor.UNKNOWN
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    # Hue histogram (180 bins, OpenCV convention).
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    peak_hue = int(np.argmax(hist_h))

    # Green dominant → EV plate
    if 35 <= peak_hue <= 85 and mean_s > 50:
        # Yellow-on-green (commercial EV) is the lowest-contrast case
        yellow_amount = float(np.sum(hist_h[20:35])) / np.sum(hist_h)
        if yellow_amount > 0.15:
            return PlateColor.COMMERCIAL_EV
        return PlateColor.PRIVATE_EV
    # Yellow dominant → commercial
    if 20 <= peak_hue <= 35 and mean_s > 50:
        return PlateColor.COMMERCIAL
    # Blue dominant → diplomatic
    if 100 <= peak_hue <= 130 and mean_s > 50:
        return PlateColor.DIPLOMATIC
    # Red dominant → temporary or trade
    if (peak_hue <= 10 or peak_hue >= 170) and mean_s > 50:
        if mean_v < 100:
            return PlateColor.TRADE      # white on red
        return PlateColor.TEMPORARY      # red on yellow
    # Black on white (high V, low S) — private
    if mean_v > 150 and mean_s < 40:
        return PlateColor.PRIVATE
    # White on black (low V) — defense
    if mean_v < 80 and mean_s < 40:
        return PlateColor.DEFENSE
    return PlateColor.UNKNOWN
