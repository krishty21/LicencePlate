"""Download pretrained weights with SHA256 verification.

Sources:
  - YOLO26s-pose base: Ultralytics HF Hub
  - Roboflow license-plate-new v3 trained weights: HF PrasannaBAImodel/license-plate-keypoint-detection
  - Real-ESRGAN realesr-general-x4v3: HF qualcomm/Real-ESRGAN-General-x4v3
  - PARSeq base: HF baudm/parseq
  - PaddleOCR PP-OCRv4 mobile: paddleocr package auto-download

Usage:
  python -m scripts.download_weights --out weights/
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# All weights + their SHA256 — extend as you train your own.
# SHA256 verification prevents silent corruption (Kaggle datasets occasionally
# have transfer issues). Run `sha256sum <file>` after download to verify.
WEIGHTS: list[dict] = [
    {
        "name": "yolo26s-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v826/yolo26s-pose.pt",
        "sha256": None,   # verify after first download
        "license": "AGPL-3.0",
        "note": "Ultralytics YOLO26s-pose base; you will fine-tune this on plate data.",
    },
    {
        "name": "license_plate_keypoint_yolov8m.pt",
        "url": "https://huggingface.co/PrasannaBAImodel/license-plate-keypoint-detection/resolve/main/license_plate_keypoint.pt",
        "sha256": None,
        "license": "Apache-2.0 (weights) + AGPL-3.0 (Ultralytics base)",
        "note": "Existing pretrained plate keypoint detector. YOLOv8m, ~101 MB. Use as a baseline / starting checkpoint, OR a source of training data via the Roboflow dataset.",
    },
    {
        "name": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "sha256": None,
        "license": "Apache-2.0",
        "note": "Real-ESRGAN compact general SR model (~16 MB).",
    },
    {
        "name": "parseq_base.pth",
        "url": "https://huggingface.co/baudm/parseq/resolve/main/parseq-bb57a2a9.pt",
        "sha256": None,
        "license": "MIT",
        "note": "PARSeq base recognizer — SOTA on STR benchmarks (95.95% combined).",
    },
]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path) -> bool:
    """Download with curl (Kaggle has curl preinstalled)."""
    import subprocess
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--silent", "--show-error",
           "-o", str(out), url]
    logger.info("downloading %s → %s", url, out)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("download failed: %s", result.stderr)
        return False
    return True


def download_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for w in WEIGHTS:
        out_path = out_dir / w["name"]
        if out_path.exists():
            logger.info("exists, skipping: %s", out_path)
            continue
        ok = download(w["url"], out_path)
        if not ok:
            logger.error("failed: %s", w["name"])
            continue
        actual_sha = sha256sum(out_path)
        if w["sha256"] is None:
            logger.info("first-time download — record this sha256 in scripts/download_weights.py:")
            logger.info("  %s = %s", w["name"], actual_sha)
        elif actual_sha != w["sha256"]:
            logger.error("SHA256 mismatch for %s — got %s, expected %s",
                         w["name"], actual_sha, w["sha256"])
            out_path.unlink(missing_ok=True)
        # Write a sidecar with license + note.
        sidecar = out_path.with_suffix(out_path.suffix + ".LICENSE.txt")
        sidecar.write_text(
            f"name: {w['name']}\n"
            f"url: {w['url']}\n"
            f"license: {w['license']}\n"
            f"sha256: {actual_sha}\n"
            f"note: {w['note']}\n"
        )
    logger.info("done. Weights in %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download pretrained weights")
    p.add_argument("--out", type=Path, default=Path("weights"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    download_all(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
