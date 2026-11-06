"""Download Roboflow datasets and convert to a unified YOLO pose format.

Targets the datasets listed in REVIEW_AND_RECOMMENDATIONS.md §3:
  - license-plate-new v3 (24k images, 4 corner keypoints) — primary pretraining
  - license-plate-4-corner-detection-v2 (ALIELSAKA, 2k)
  - Indian aggregate (Rama + Bharani + Prasiddh + jishu + Jeremiah)

Requires ROBOFLOW_API_KEY env var. Get one at https://roboflow.com → Settings.

Usage:
  export ROBOFLOW_API_KEY=...
  python -m scripts.prepare_roboflow --out data/roboflow_combined
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Datasets to download. workspace/project/version format.
# Verified October-November 2026 via universe.roboflow.com listings.
ROBOFLOW_DATASETS: list[dict] = [
    {
        "name": "license_plate_new_v3",
        "workspace": "prasannabaimodel",
        "project": "license-plate-new",
        "version": 3,
        "format": "yolov8-pose",
        "license": "CC BY 4.0",
        "role": "detector_pretrain_with_corners",
        "has_corners": True,
    },
    {
        "name": "license_plate_4_corner_v2",
        "workspace": "alielsaka",
        "project": "license-plate-4-corner-detection-v2",
        "version": 2,
        "format": "yolov8-pose",
        "license": "open",
        "role": "detector_pretrain_with_corners",
        "has_corners": True,
    },
    {
        "name": "indian_license_plate_rama",
        "workspace": "rama-shirodkar-y6et6",
        "project": "indian-license-plate-5n0vn-lgdye",
        "version": 1,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "role": "indian_finetune",
        "has_corners": False,
    },
    {
        "name": "indian_number_plate_bharani",
        "workspace": "bharani-ejlrn",
        "project": "indian-number-plate-detection",
        "version": 1,
        "format": "yolov8",
        "license": "open",
        "role": "indian_finetune",
        "has_corners": False,
    },
    {
        "name": "indian_number_plate_prasiddh",
        "workspace": "prasiddh-mori",
        "project": "indian-number-plate-id3jg",
        "version": 1,
        "format": "yolov8",
        "license": "CC BY 4.0",
        "role": "indian_finetune",
        "has_corners": False,
    },
]


def download_one(spec: dict, out_dir: Path) -> bool:
    """Download one Roboflow dataset using the roboflow Python SDK."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error("ROBOFLOW_API_KEY env var not set — get one at "
                     "https://roboflow.com → Settings → API")
        return False
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow SDK not installed. Run: pip install roboflow")
        return False
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(spec["workspace"])
    project = workspace.project(spec["project"])
    version = project.version(spec["version"])
    ds = version.download(spec["format"], location=str(out_dir / spec["name"]))
    logger.info("downloaded %s → %s (n_images: %s)",
                spec["name"], out_dir / spec["name"],
                getattr(ds, "n_images", "?"))
    return True


def combine_into_unified(out_dir: Path, datasets: list[dict]) -> None:
    """Combine multiple YOLO datasets into one with train/val/test splits."""
    combined = out_dir / "combined"
    for split in ("train", "valid", "test"):
        (combined / split / "images").mkdir(parents=True, exist_ok=True)
        (combined / split / "labels").mkdir(parents=True, exist_ok=True)
    n_total = 0
    for spec in datasets:
        ds_dir = out_dir / spec["name"]
        if not ds_dir.exists():
            logger.warning("dataset not downloaded: %s", spec["name"])
            continue
        # Roboflow datasets have data.yaml + train/valid/test/{images,labels}
        for split in ("train", "valid", "test"):
            src_imgs = ds_dir / split / "images"
            src_labels = ds_dir / split / "labels"
            if not src_imgs.exists():
                continue
            for img in src_imgs.iterdir():
                if not img.is_file():
                    continue
                # Prefix with dataset name to avoid collisions
                target_img = combined / split / "images" / f"{spec['name']}_{img.name}"
                target_img.symlink_to(img.resolve())
                # Label (same stem, .txt)
                label = src_labels / (img.stem + ".txt")
                if label.exists():
                    target_label = combined / split / "labels" / f"{spec['name']}_{img.stem}.txt"
                    target_label.symlink_to(label.resolve())
                n_total += 1
    logger.info("combined %d images into %s", n_total, combined)
    # Write data.yaml
    yaml_path = combined / "data.yaml"
    has_corners = any(s["has_corners"] for s in datasets)
    yaml_text = (
        f"path: {combined}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['plate']\n"
    )
    if has_corners:
        yaml_text += f"\nkpt_shape: [4, 3]\nflip_idx: [0, 1, 2, 3]\n"
    yaml_path.write_text(yaml_text)
    logger.info("wrote %s (has_corners=%s)", yaml_path, has_corners)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download + combine Roboflow datasets")
    p.add_argument("--out", type=Path, default=Path("data/roboflow"))
    p.add_argument("--skip-combine", action="store_true",
                   help="only download, don't combine")
    p.add_argument("--only", default=None,
                   help="comma-separated dataset names to download (default: all)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    selected = ROBOFLOW_DATASETS
    if args.only:
        names = set(args.only.split(","))
        selected = [s for s in ROBOFLOW_DATASETS if s["name"] in names]
    # Download
    for spec in selected:
        if (args.out / spec["name"]).exists():
            logger.info("exists, skipping: %s", spec["name"])
            continue
        ok = download_one(spec, args.out)
        if not ok:
            logger.error("failed: %s", spec["name"])
    if not args.skip_combine:
        combine_into_unified(args.out, selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
