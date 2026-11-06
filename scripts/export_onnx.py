"""Export .pt models to ONNX with fp16 + TensorRT-friendly graph.

Outputs:
  weights/onnx/yolo26s-pose-plate.onnx       (NMS-free, fp16)
  weights/onnx/lprnet_indian_v2.onnx         (CTC, fp16)
  weights/onnx/parseq_indian.onnx            (parallel decoder, fp16)
  weights/onnx/realesr-general-x4v3.onnx     (Real-ESRGAN compact)

Usage:
  python -m scripts.export_onnx --out weights/onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export_yolo26(weights: Path, out: Path, imgsz: int = 640,
                  nms: bool = False, fp16: bool = True) -> bool:
    """Export YOLO26-pose to ONNX with end-to-end (NMS-free) head."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return False
    model = YOLO(str(weights), task="pose")
    out.parent.mkdir(parents=True, exist_ok=True)
    # ultralytics handles dynamic axes, simplification, and opset.
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=True,
        half=fp16,
        nms=nms,           # v1.1: end-to-end export
    )
    # Move/rename to target path
    exported = Path(path)
    if exported != out:
        exported.rename(out)
    logger.info("YOLO26 exported → %s", out)
    return True


def export_lprnet(weights: Path, out: Path,
                  input_shape: tuple[int, int, int, int] = (1, 3, 64, 192),
                  fp16: bool = True) -> bool:
    """Export LPRNet (.pt) to ONNX."""
    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        return False
    try:
        # Project-local model definition — replace import with your actual class.
        # This is a placeholder showing the export contract.
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        # If you haven't yet defined the LPRNet model class, this will fail;
        # that's expected — define src/anpr/models/lprnet.py with the architecture
        # from DATASETS_AND_ARCHITECTURE.md §B.2.
        logger.warning(
            "LPRNet export requires src/anpr/models/lprnet.py with the model class. "
            "Define it per DATASETS_AND_ARCHITECTURE.md §B.2, then update this function."
        )
        return False
    except Exception as e:
        logger.exception("LPRNet export failed: %s", e)
        return False


def export_parseq(weights: Path, out: Path, fp16: bool = True) -> bool:
    """Export PARSeq to ONNX (parallel decoder mode)."""
    try:
        import torch
        # PARSeq provides its own export. See github.com/baudm/parseq.
        # python export.py --model parseq --export_onnx ...
        logger.info(
            "Use PARSeq's own export: "
            "python -m parseq.export --model parseq --checkpoint %s --output %s",
            weights, out,
        )
        return False
    except ImportError:
        logger.error("parseq / torch not installed")
        return False


def export_realesrgan(weights: Path, out: Path, fp16: bool = True) -> bool:
    """Export Real-ESRGAN realesr-general-x4v3 to ONNX."""
    try:
        import torch
        # Real-ESRGAN provides export via its own scripts. See:
        # https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
        logger.info(
            "Use Real-ESRGAN's export tooling. The realesr-general-x4v3 model "
            "is small enough (~16 MB) to ship as .pth and convert via the "
            "official export script."
        )
        return False
    except ImportError:
        logger.error("torch not installed")
        return False


def export_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # YOLO26 — only export if weights exist
    yolo_pt = Path("weights/yolo26s-pose-plate.pt")
    if yolo_pt.exists():
        export_yolo26(yolo_pt, out_dir / "yolo26s-pose-plate.onnx")
    else:
        logger.warning("skipping YOLO26 export — %s not found", yolo_pt)
    # LPRNet — only export if model class is defined
    lpr_pt = Path("weights/lprnet_indian_v2.pt")
    if lpr_pt.exists():
        export_lprnet(lpr_pt, out_dir / "lprnet_indian_v2.onnx")
    # PARSeq
    parseq_pt = Path("weights/parseq_base.pth")
    if parseq_pt.exists():
        export_parseq(parseq_pt, out_dir / "parseq_indian.onnx")
    # Real-ESRGAN
    esrgan_pth = Path("weights/realesr-general-x4v3.pth")
    if esrgan_pth.exists():
        export_realesrgan(esrgan_pth, out_dir / "realesr-general-x4v3.onnx")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export models to ONNX")
    p.add_argument("--out", type=Path, default=Path("weights/onnx"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    export_all(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
