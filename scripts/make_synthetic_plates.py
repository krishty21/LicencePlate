"""Synthetic Indian plate generator — recognizer pretraining data.

Generates random plate strings per Indian grammar, renders them in a BIS-like
font, applies random perspective/blur/gamma to simulate real conditions.

100k strips cost ~1 GPU-hour, zero labeling. This is the highest-leverage
training asset for the recognizer (per DATASETS_AND_ARCHITECTURE.md §A.2).

Usage:
  python -m scripts.make_synthetic_plates --n 150000 --out data/synthetic_indian_plates
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Charset — A-Z minus {I, O, Q}
SERIES_LETTERS = list("ABCDEFGHJKLMNPRSTUVWXYZ")
DIGITS = list("0123456789")
STATE_CODES = ["AN","AP","AR","AS","BR","CH","CG","DD","DL","GA","GJ","HR",
               "HP","JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ",
               "NL","OD","PY","PB","RJ","SK","TN","TS","TR","UP","UK","WB"]

# Bh series year prefix (2021 onward)
BH_YEARS = ["21", "22", "23", "24", "25", "26"]


def random_plate(rng: random.Random) -> tuple[str, str]:
    """Return (plate_text, format_kind)."""
    kind = rng.choices(
        ["private", "private", "private", "private", "bh", "va"],
        weights=[40, 30, 15, 10, 3, 2],
    )[0]
    if kind == "private":
        state = rng.choice(STATE_CODES)
        rto = rng.choice([f"{n:02d}" for n in range(1, 100)])
        n_letters = rng.choices([1, 2, 3], weights=[20, 60, 20])[0]
        letters = "".join(rng.choices(SERIES_LETTERS, k=n_letters))
        serial = f"{rng.randint(1, 9999):04d}"
        return f"{state}{rto}{letters}{serial}", "private"
    if kind == "bh":
        year = rng.choice(BH_YEARS)
        serial = f"{rng.randint(1, 9999):04d}"
        n_letters = rng.choices([1, 2], weights=[60, 40])[0]
        letters = "".join(rng.choices(SERIES_LETTERS, k=n_letters))
        return f"{year}BH{serial}{letters}", "bh"
    if kind == "va":
        state = rng.choice(STATE_CODES)
        letters = "".join(rng.choices(SERIES_LETTERS, k=2))
        serial = f"{rng.randint(1, 9999):04d}"
        return f"{state}VA{letters}{serial}", "va"
    raise AssertionError(kind)


def render_plate(text: str, w: int = 200, h: int = 48) -> np.ndarray:
    """Render a plate image. Uses PIL with a plate-like font if available.

    Falls back to a simple OpenCV-rendered version if PIL/Pillow isn't installed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Try to load a plate-like font; fall back to default.
        font_path = None
        for candidate in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if Path(candidate).exists():
                font_path = candidate
                break
        font_size = max(20, int(h * 0.7))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        # Center text
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        x = (w - tw) // 2
        y = (h - th) // 2 - 2
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        return np.array(img)[:, :, ::-1]   # RGB → BGR
    except ImportError:
        import cv2
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                    font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return img


def apply_random_augmentation(img: np.ndarray,
                              rng: random.Random) -> np.ndarray:
    """Random perspective warp, blur, gamma, JPEG."""
    import cv2
    h, w = img.shape[:2]
    # Perspective warp
    if rng.random() < 0.6:
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        max_shift = 0.15
        dst = src + np.array([
            [rng.uniform(-max_shift, max_shift) * w, rng.uniform(-max_shift, max_shift) * h],
            [rng.uniform(-max_shift, max_shift) * w, rng.uniform(-max_shift, max_shift) * h],
            [rng.uniform(-max_shift, max_shift) * w, rng.uniform(-max_shift, max_shift) * h],
            [rng.uniform(-max_shift, max_shift) * w, rng.uniform(-max_shift, max_shift) * h],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h))
    # Motion blur
    if rng.random() < 0.4:
        ksize = rng.choice([3, 5, 7])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        img = cv2.filter2D(img, -1, kernel)
    # Gamma
    if rng.random() < 0.5:
        gamma = rng.uniform(0.4, 1.6)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)]).astype(np.uint8)
        img = cv2.LUT(img, table)
    # JPEG compression
    if rng.random() < 0.5:
        q = rng.randint(30, 70)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def generate(n: int, out_dir: Path, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    labels_csv = out_dir / "labels.csv"
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    with labels_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text", "kind"])
        for i in range(n):
            text, kind = random_plate(rng)
            img = render_plate(text)
            img = apply_random_augmentation(img, rng)
            # Downscale to simulate tiny plates, then upsample
            if rng.random() < 0.3:
                h, w = img.shape[:2]
                small = cv2.resize(img, (max(8, w // 4), max(4, h // 4)))
                img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            fname = f"{i:06d}.jpg"
            import cv2
            cv2.imwrite(str(images_dir / fname), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            writer.writerow([fname, text, kind])
            if (i + 1) % 10000 == 0:
                logger.info("generated %d/%d", i + 1, n)
    logger.info("done. %d plates in %s", n, out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic Indian plates")
    p.add_argument("--n", type=int, default=150000)
    p.add_argument("--out", type=Path, default=Path("data/synthetic_indian_plates"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    generate(args.n, args.out, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
