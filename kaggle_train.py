"""Kaggle training entry point — paste this into a Kaggle notebook cell.

Usage on Kaggle:
  1. Create a new notebook.
  2. Add this project as a Kaggle Dataset (upload the anpr-india-kaggle.zip).
  3. Set accelerator to GPU T4 x2.
  4. In the first cell:
       import sys; sys.path.insert(0, '/kaggle/input/anpr-india-kaggle/src')
       exec(open('/kaggle/input/anpr-india-kaggle/kaggle_train.py').read())
  5. Run all cells.

Or run directly:
  python kaggle_train.py --stage all
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Kaggle paths — defaults that work when this project is uploaded as a Dataset.
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
PROJECT_DIR = KAGGLE_INPUT / "anpr-india-kaggle"
WEIGHTS_DIR = Path("/kaggle/working/weights")
DATA_DIR = Path("/kaggle/working/data")
OUTPUT_DIR = Path("/kaggle/working/output")


def setup_paths():
    """Ensure src is importable, weights/data dirs exist."""
    src = PROJECT_DIR / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = PROJECT_DIR / "configs"
    if configs.exists():
        os.environ.setdefault("ANPR_CONFIG", str(configs / "serve.yaml"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Kaggle paths: project=%s weights=%s data=%s",
                PROJECT_DIR, WEIGHTS_DIR, DATA_DIR)


def install_deps():
    """Install runtime deps in the Kaggle environment."""
    import subprocess
    # Kaggle already has: torch, numpy, opencv-python, pyyaml, pandas, pillow
    # We need: ultralytics (YOLO26), onnxruntime, supervision, paddleocr (optional)
    deps = [
        "ultralytics>=8.3.0",       # YOLO11/YOLO26
        "onnxruntime-gpu>=1.18",
        "supervision>=0.20",
        "huggingface_hub>=0.20",
        "pydantic>=2",
        "fastapi>=0.110",
        "ruff", "mypy", "pytest",
    ]
    for d in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", d], check=False)
    # Optional heavier deps — install only if you need them
    if os.environ.get("ANPR_INSTALL_PADDLE"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "paddlepaddle-gpu", "paddleocr"], check=False)
    if os.environ.get("ANPR_INSTALL_PARSEQ"):
        # PARSeq is a git install — clone to /kaggle/working/parseq
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/baudm/parseq.git",
                        "/kaggle/working/parseq"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                        "/kaggle/working/parseq"], check=False)


def download_weights():
    """Run scripts/download_weights.py to fetch pretrained weights."""
    from scripts.download_weights import download_all
    download_all(WEIGHTS_DIR)


def make_synthetic(n: int = 150_000):
    """Generate synthetic Indian plate strips for recognizer pretraining."""
    from scripts.make_synthetic_plates import generate
    generate(n, DATA_DIR / "synthetic_indian_plates", seed=42)


def train_detector():
    """Train YOLO26s-pose on plate data — 2-stage curriculum.

    Stage 1: pretrain on CCPD subset + Roboflow license-plate-new v3
    Stage 2: fine-tune on Indian Roboflow aggregate
    """
    from ultralytics import YOLO
    base = str(WEIGHTS_DIR / "yolo26s-pose.pt")
    if not Path(base).exists():
        # Fall back to YOLO11 if YOLO26 not available
        alt = str(WEIGHTS_DIR / "yolo11s-pose.pt")
        if Path(alt).exists():
            base = alt
            logger.warning("YOLO26 weights not found, using YOLO11s")
        else:
            raise RuntimeError(f"no pose weights in {WEIGHTS_DIR}")
    model = YOLO(base, task="pose")
    # Stage 1: pretrain
    cfg = PROJECT_DIR / "configs" / "detect.yaml"
    import yaml
    with open(cfg) as f:
        dcfg = yaml.safe_load(f)
    train_cfg = dcfg.get("train", {})
    data_yaml = _write_data_yaml(
        train_cfg["pretrain_sources"][1],   # Roboflow license-plate-new v3
        out_path=DATA_DIR / "data_pretrain.yaml",
    )
    logger.info("Stage 1: pretraining on %s", data_yaml)
    model.train(
        data=str(data_yaml),
        epochs=train_cfg.get("epochs", 80),
        batch=train_cfg.get("batch_size", 32),
        imgsz=dcfg.get("imgsz", 640),
        device=dcfg.get("device", "0"),
        project=str(OUTPUT_DIR / "runs"),
        name="detector_pretrain",
        patience=train_cfg.get("patience", 15),
        # YOLO26 hyperparams — these are mostly ignored on YOLO11 but accepted
        optimizer=train_cfg.get("optimizer", "SGD"),
        lr0=train_cfg.get("lr0", 0.01),
        lrf=train_cfg.get("lrf", 0.01),
        momentum=train_cfg.get("momentum", 0.937),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        cos_lr=True,
        fliplr=0.0, flipud=0.0,   # plates are directional
    )
    # Stage 2: fine-tune
    finetune_data = _write_data_yaml(
        train_cfg["finetune_sources"][0],
        out_path=DATA_DIR / "data_finetune.yaml",
    )
    logger.info("Stage 2: fine-tuning on %s", finetune_data)
    model.train(
        data=str(finetune_data),
        epochs=20,
        batch=train_cfg.get("batch_size", 32),
        imgsz=dcfg.get("imgsz", 640),
        device=dcfg.get("device", "0"),
        project=str(OUTPUT_DIR / "runs"),
        name="detector_finetune",
        patience=10,
        cos_lr=True,
        lr0=0.001,    # lower for fine-tune
        fliplr=0.0, flipud=0.0,
    )
    # Save best
    best = OUTPUT_DIR / "runs" / "detector_finetune" / "weights" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy(best, WEIGHTS_DIR / "yolo26s-pose-plate.pt")
        logger.info("saved best detector → %s", WEIGHTS_DIR / "yolo26s-pose-plate.pt")


def _write_data_yaml(source: dict, out_path: Path) -> Path:
    """Write a YOLO-format data.yaml pointing at a Roboflow-style directory."""
    path = source.get("path", "/kaggle/input/dataset")
    # Expect subdirs: path/{train,val,test}/{images,labels}
    content = (
        f"path: {path}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['plate']\n"
        f"\n"
        f"# Keypoint shape: 4 corners, each (x, y, visibility)\n"
        f"kpt_shape: [4, 3]\n"
        f"flip_idx: [0, 1, 2, 3]\n"
    )
    out_path.write_text(content)
    return out_path


def train_lprnet():
    """Train LPRNet v2 on synthetic + real Indian crops.

    v1.1: the LPRNet model class is now implemented in src/anpr/models/lprnet.py.
    This function runs the full pretrain → fine-tune recipe.
    """
    from anpr.models.lprnet_trainer import train, TrainConfig
    finetune_path = DATA_DIR / "indian_plate_crops_real"
    cfg = TrainConfig(
        pretrain_data=DATA_DIR / "synthetic_indian_plates",
        finetune_data=finetune_path if finetune_path.exists() else None,
        pretrain_epochs=10,
        finetune_epochs=5 if finetune_path.exists() else 0,
        pretrain_batch_size=256,
        finetune_batch_size=64,
        device="cuda",
        output_dir=OUTPUT_DIR / "lprnet",
        num_workers=4,
        seed=42,
    )
    best_path = train(cfg)
    logger.info("LPRNet training done. Best weights: %s", best_path)
    # Copy to weights/ for downstream ONNX export
    if best_path.exists():
        import shutil
        target = WEIGHTS_DIR / "lprnet_indian_v2.pt"
        shutil.copy(best_path, target)
        logger.info("copied best LPRNet → %s", target)


def export_lprnet_onnx():
    """Export the trained LPRNet to ONNX (fp16-ready, dynamic batch)."""
    import torch
    from anpr.models.lprnet import LPRNet, INPUT_H, INPUT_W
    weights = WEIGHTS_DIR / "lprnet_indian_v2.pt"
    if not weights.exists():
        logger.warning("LPRNet weights not found at %s — skipping ONNX export", weights)
        return
    model = LPRNet()
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    onnx_path = WEIGHTS_DIR / "lprnet_indian_v2.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, INPUT_H, INPUT_W),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info("exported LPRNet ONNX → %s", onnx_path)


def train_parseq():
    """Fine-tune PARSeq on Indian plate crops."""
    parseq_dir = Path("/kaggle/working/parseq")
    if not parseq_dir.exists():
        logger.warning("PARSeq not cloned; skipping. Run with ANPR_INSTALL_PARSEQ=1.")
        return
    # PARSeq has its own training script:
    #   python -m parseq.train \
    #     --trainer.default_root_dir /kaggle/working/parseq_runs \
    #     --data.root_dir /kaggle/working/data/indian_plate_crops_real \
    #     --model.parseq ...
    # See github.com/baudm/parseq for full options.
    logger.info("PARSeq fine-tune: see /kaggle/working/parseq/train.py")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ANPR-India Kaggle training")
    p.add_argument("--stage", choices=["all", "setup", "weights", "synthetic",
                                        "detector", "lprnet", "parseq",
                                        "export-onnx"],
                   default="all")
    p.add_argument("--n-synthetic", type=int, default=150_000)
    args = p.parse_args(argv)

    setup_paths()

    if args.stage in ("all", "setup"):
        install_deps()
    if args.stage in ("all", "weights"):
        download_weights()
    if args.stage in ("all", "synthetic"):
        make_synthetic(args.n_synthetic)
    if args.stage in ("all", "detector"):
        train_detector()
    if args.stage in ("all", "lprnet"):
        train_lprnet()
    if args.stage in ("all", "parseq"):
        train_parseq()
    if args.stage in ("all", "export-onnx"):
        export_lprnet_onnx()
    return 0


if __name__ == "__main__":
    sys.exit(main())
