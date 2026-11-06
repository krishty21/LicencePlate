"""Generate the degraded test set — controlled corruptions for robustness eval.

Applies 5 severity levels of motion blur, low-light, Gaussian noise, JPEG q30
to clean test crops. Used to:
  1. Tune the gating threshold τ (IMPLEMENTATION_SPEC.md §3.4)
  2. Plot the "robustness curve" (IMPLEMENTATION_SPEC.md §5)

Usage:
  python -m scripts.make_degraded_set --src data/test_crops --out data/test_crops_degraded
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SEVERITIES = [1, 2, 3, 4, 5]   # 5 levels per corruption type
CORRUPTIONS = ["motion_blur", "low_light", "noise", "jpeg"]


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ksize = severity * 2 + 1   # 3, 5, 7, 9, 11
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel)


def low_light(img: np.ndarray, severity: int) -> np.ndarray:
    # gamma 0.7 → 0.4 (severity 1 → 5)
    gamma = 0.7 - severity * 0.075
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = severity * 5   # 5, 10, 15, 20, 25
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def jpeg_compress(img: np.ndarray, severity: int) -> np.ndarray:
    q = max(10, 100 - severity * 18)   # 82, 64, 46, 28, 10
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


CORRUPTION_FNS = {
    "motion_blur": motion_blur,
    "low_light": low_light,
    "noise": gaussian_noise,
    "jpeg": jpeg_compress,
}


def process(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    import csv
    images_dir = src / "images"
    if not images_dir.exists():
        logger.error("expected %s/images/ directory", src)
        return
    out_images = out / "images"
    out_images.mkdir(exist_ok=True)
    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text", "kind", "corruption", "severity"])
        np.random.seed(42)
        for img_path in sorted(images_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            base = img_path.stem
            # Original
            cv2.imwrite(str(out_images / f"{base}_orig.jpg"), img)
            w.writerow([f"{base}_orig.jpg", "", "", "none", 0])
            # Each corruption × each severity
            for cname in CORRUPTIONS:
                fn = CORRUPTION_FNS[cname]
                for sev in SEVERITIES:
                    out_img = fn(img, sev)
                    fname = f"{base}_{cname}_{sev}.jpg"
                    cv2.imwrite(str(out_images / fname), out_img)
                    w.writerow([fname, "", "", cname, sev])
    logger.info("wrote degraded set to %s", out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Make degraded test set")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    process(args.src, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
