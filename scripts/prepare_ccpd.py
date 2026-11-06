"""Convert CCPD dataset to YOLO pose format with 4 corner keypoints.

CCPD filenames encode annotations in a base36-ish format:
    025-95_113-154&383_386&473-386&473_177&413_154&383_367&402-0_0_22_27_27_33_16-37-15.jpg

Format: area-tilt&bbox&-corners&-text-fields.jpg
  - The 4 corners are: TL-BR-TR-BL (NOT the order we use; we re-order)
  - Plate text is the 7 chars after the corners

Reference: github.com/detectRecog/CCPD

Usage:
  python -m scripts.prepare_ccpd --src /kaggle/input/ccpd --out data/ccpd_yolo
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_ccpd_filename(fname: str) -> Optional[dict]:
    """Parse a CCPD filename into corners + plate text + bbox.

    Returns None if the filename doesn't match the CCPD convention.
    """
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        # parts[1]: tilt&bbox like "113-154&383_386&473" — first number is tilt,
        # then bbox as two (x, y) pairs.
        # parts[2]: 4 corners as (x&y)_(x&y)_(x&y)_(x&y) — order: BR, TL, BL, TR
        # (varies by CCPD version; we re-order by our order_corners function)
        # parts[4]: plate text in CCPD's Chinese-plate encoding — skip for OCR.
        # 4 corners as list of (x, y)
        corner_strs = parts[2].split("_")
        if len(corner_strs) != 4:
            return None
        corners = []
        for cs in corner_strs:
            xs, ys = cs.split("&")
            corners.append((int(xs), int(ys)))
        # Compute bbox from corners
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        return {
            "corners": corners,   # 4 (x, y) tuples in CCPD order
            "bbox": bbox,
            "text": "",   # CCPD plate text is Chinese — we don't use it for OCR
        }
    except (ValueError, IndexError) as e:
        logger.debug("failed to parse %s: %s", fname, e)
        return None


def corners_to_yolo_pose(corners: list[tuple[int, int]],
                          img_w: int, img_h: int) -> str:
    """Convert 4 corners to YOLO pose label format.

    YOLO pose label: class x1 y1 v1 x2 y2 v2 ... (normalized, visibility flag)
    For our use: visibility = 1 (always visible — corners are guaranteed).
    """
    # Order: TL, TR, BR, BL clockwise (per our types.py)
    # CCPD corner order is approximately: BR, TL, BL, TR (depending on version).
    # We re-order using sum/diff like rectify.order_corners.
    import numpy as np
    arr = np.array(corners, dtype=np.float64)
    s = arr[:, 0] + arr[:, 1]
    tl_idx = int(np.argmin(s))
    br_idx = int(np.argmax(s))
    remaining = [i for i in range(4) if i not in (tl_idx, br_idx)]
    if arr[remaining[0], 0] < arr[remaining[1], 0]:
        bl_idx, tr_idx = remaining[0], remaining[1]
    else:
        bl_idx, tr_idx = remaining[1], remaining[0]
    ordered = [corners[i] for i in (tl_idx, tr_idx, br_idx, bl_idx)]
    parts = ["0"]   # class 0 = plate
    for x, y in ordered:
        parts.append(f"{x / img_w:.6f}")
        parts.append(f"{y / img_h:.6f}")
        parts.append("2")   # visibility=2 (visible, not occluded)
    return " ".join(parts)


def bbox_to_yolo(bbox: tuple[int, int, int, int],
                 img_w: int, img_h: int) -> str:
    """Convert bbox to YOLO format (class x_center y_center w h)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process(src: Path, out: Path, sample_n: Optional[int] = None):
    """Convert CCPD src directory to YOLO pose format at out.

    Args:
        src: CCPD root (e.g. /kaggle/input/ccpd2019/ccpd_base)
        out: output directory (will create images/ and labels/ subdirs)
        sample_n: if set, randomly sample this many images
    """
    import cv2
    import random
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    # CCPD images are typically in subdirectories like ccpd_base, ccpd_blur, etc.
    all_images = list(src.rglob("*.jpg"))
    if sample_n and len(all_images) > sample_n:
        random.seed(42)
        all_images = random.sample(all_images, sample_n)
        logger.info("sampled %d images from %d", sample_n, len(list(src.rglob("*.jpg"))))
    n_ok, n_fail = 0, 0
    for img_path in all_images:
        try:
            parsed = parse_ccpd_filename(img_path.name)
            if parsed is None:
                n_fail += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                n_fail += 1
                continue
            h, w = img.shape[:2]
            # Write image (just copy)
            out_img = out / "images" / f"{img_path.stem}.jpg"
            cv2.imwrite(str(out_img), img)
            # Write label
            label_path = out / "labels" / f"{img_path.stem}.txt"
            yolo_line = corners_to_yolo_pose(parsed["corners"], w, h)
            label_path.write_text(yolo_line + "\n")
            n_ok += 1
            if (n_ok + n_fail) % 5000 == 0:
                logger.info("processed %d ok / %d fail / %d total",
                            n_ok, n_fail, n_ok + n_fail)
        except Exception as e:
            logger.debug("failed %s: %s", img_path, e)
            n_fail += 1
    logger.info("done: %d ok, %d fail", n_ok, n_fail)
    # Write data.yaml
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['plate']\n"
        f"\n"
        f"kpt_shape: [4, 3]\n"   # 4 keypoints, each (x, y, visibility)
        f"flip_idx: [0, 1, 2, 3]\n"
    )
    logger.info("wrote data.yaml at %s", yaml_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert CCPD to YOLO pose")
    p.add_argument("--src", type=Path, required=True,
                   help="CCPD root (e.g. /kaggle/input/ccpd2019/ccpd_base)")
    p.add_argument("--out", type=Path, default=Path("data/ccpd_yolo"))
    p.add_argument("--sample", type=int, default=None,
                   help="random sample N images")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    process(args.src, args.out, sample_n=args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
