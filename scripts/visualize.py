"""Visualization tools — draw detections + corner keypoints + read text on images.

Useful for:
  - Sanity-checking the detector during training
  - Building demo GIFs for HF Spaces
  - Debugging pipeline failures

CLI:
  python -m scripts.visualize <image> [--out viz.jpg]
  python -m scripts.visualize <dir> --batch [--out viz/]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colors for the 4 corner keypoints (TL, TR, BR, BL clockwise)
CORNER_COLORS = [
    (0, 0, 255),    # TL = red
    (0, 255, 255),  # TR = yellow
    (255, 0, 0),    # BR = blue
    (0, 255, 0),    # BL = green
]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def draw_detection(image: np.ndarray, plate_result) -> np.ndarray:
    """Draw one plate result on the image (in place copy).

    plate_result is a PlateResult (from anpr.pipeline) or a dict with the
    same shape.
    """
    out = image.copy()
    if hasattr(plate_result, "bbox"):
        bbox = plate_result.bbox
        corners = plate_result.corners
        text = plate_result.plate
        conf = plate_result.confidence
        plate_class = plate_result.plate_class.value if hasattr(plate_result.plate_class, "value") else str(plate_result.plate_class)
        plate_color = plate_result.plate_color.value if hasattr(plate_result.plate_color, "value") else str(plate_result.plate_color)
        enhanced = plate_result.enhanced
    else:
        bbox = plate_result["bbox"]
        corners = plate_result.get("corners", [])
        text = plate_result.get("plate", "")
        conf = plate_result.get("confidence", 0)
        plate_class = plate_result.get("plate_class", "unknown")
        plate_color = plate_result.get("plate_color", "unknown")
        enhanced = plate_result.get("enhanced", False)

    # Bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 255, 0) if not enhanced else (0, 165, 255)   # green or orange
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # Corners (only if present)
    if corners:
        for i, c in enumerate(corners):
            if hasattr(c, "x"):
                cx, cy = int(c.x), int(c.y)
            else:
                cx, cy = int(c[0]), int(c[1])
            cv2.circle(out, (cx, cy), 5, CORNER_COLORS[i], -1)
            cv2.putText(out, CORNER_NAMES[i], (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CORNER_COLORS[i], 1)
        # Connect corners with thin lines to show the quad
        pts = []
        for c in corners:
            if hasattr(c, "x"):
                pts.append([int(c.x), int(c.y)])
            else:
                pts.append([int(c[0]), int(c[1])])
        cv2.polylines(out, [np.array(pts, dtype=np.int32)], True,
                      (255, 255, 255), 1)

    # Label box (above the bbox)
    label = f"{text} {conf:.2f}"
    if enhanced:
        label += " [ENH]"
    label_y = max(y1 - 10, 20)
    cv2.rectangle(out, (x1, label_y - 18), (x1 + len(label) * 9, label_y + 2),
                  color, -1)
    cv2.putText(out, label, (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Plate class + color (small text below bbox)
    info = f"{plate_class} | {plate_color}"
    cv2.putText(out, info, (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return out


def draw_many(image: np.ndarray, plate_results: list) -> np.ndarray:
    """Draw all plate results on the image."""
    out = image.copy()
    for pr in plate_results:
        out = draw_detection(out, pr)
    return out


def visualize_image(image_path: Path, config_path: str = "configs/serve.yaml",
                     out_path: Optional[Path] = None) -> Path:
    """Run pipeline on one image and save a visualization."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import yaml
    from anpr.pipeline import Pipeline
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pipe = Pipeline.from_config(cfg)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    results = pipe.process_image(img)
    viz = draw_many(img, results)
    if out_path is None:
        out_path = image_path.parent / f"{image_path.stem}_viz.jpg"
    cv2.imwrite(str(out_path), viz)
    logger.info("wrote visualization → %s (plates: %d)", out_path, len(results))
    return out_path


def visualize_dir(in_dir: Path, out_dir: Path,
                  config_path: str = "configs/serve.yaml"):
    """Visualize all images in a directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(in_dir.glob("*.jpg")):
        try:
            visualize_image(img_path, config_path, out_dir / f"{img_path.stem}_viz.jpg")
        except Exception as e:
            logger.warning("failed %s: %s", img_path, e)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Visualize ANPR detections")
    p.add_argument("input", type=Path, help="image file or directory")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", default="configs/serve.yaml")
    p.add_argument("--batch", action="store_true",
                   help="process all images in a directory")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    if args.input.is_dir() or args.batch:
        out = args.out or args.input.parent / "viz"
        visualize_dir(args.input, out, args.config)
    else:
        visualize_image(args.input, args.config, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
