"""CLI entry — `python -m anpr.cli <command>`.

Commands:
  recognize <image>          Run pipeline on one image, print JSON.
  video <video> [--emit N]   Run pipeline on a video, stream JSON-lines.
  benchmark <dir>            Run eval harness on a directory of images.
  export-onnx                Export all .pt models to ONNX.
  download-weights           Fetch pretrained weights from HF Hub.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anpr", description="ANPR-India CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", default="configs/serve.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("recognize", help="process one image")
    p_rec.add_argument("image", type=Path)
    p_rec.add_argument("--log", action="store_true",
                       help="emit JSON-lines log per plate")

    p_vid = sub.add_parser("video", help="process a video file")
    p_vid.add_argument("video", type=Path)
    p_vid.add_argument("--emit", type=int, default=5)

    p_bench = sub.add_parser("benchmark", help="run eval on a directory")
    p_bench.add_argument("dir", type=Path)
    p_bench.add_argument("--out", type=Path, default=Path("benchmarks_out.json"))

    p_exp = sub.add_parser("export-onnx", help="export .pt models to ONNX")
    p_exp.add_argument("--out", type=Path, default=Path("weights/onnx"))

    p_dl = sub.add_parser("download-weights", help="fetch pretrained weights")
    p_dl.add_argument("--out", type=Path, default=Path("weights"))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "recognize":
        return _cmd_recognize(args)
    if args.cmd == "video":
        return _cmd_video(args)
    if args.cmd == "benchmark":
        return _cmd_benchmark(args)
    if args.cmd == "export-onnx":
        return _cmd_export_onnx(args)
    if args.cmd == "download-weights":
        return _cmd_download_weights(args)
    parser.print_help()
    return 1


def _load_pipeline(config_path: str):
    import yaml
    from .pipeline import Pipeline
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return Pipeline.from_config(cfg)


def _cmd_recognize(args) -> int:
    import cv2
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2
    pipe = _load_pipeline(args.config)
    results = pipe.process_image(img)
    for r in results:
        if args.log:
            print(pipe.emit_log(r, source=str(args.image)))
        else:
            print(json.dumps({
                "plate": r.plate, "confidence": r.confidence,
                "bbox": list(r.bbox),
                "grammar_valid": r.grammar_valid,
                "grammar_kind": r.grammar_kind,
                "enhanced": r.enhanced,
                "recognizer": r.recognizer,
                "plate_class": r.plate_class.value,
                "plate_color": r.plate_color.value,
                "error": r.error,
            }, indent=2))
    return 0


def _cmd_video(args) -> int:
    pipe = _load_pipeline(args.config)
    for frame_result in pipe.process_video(args.video, emit_every=args.emit):
        print(json.dumps(frame_result))
    return 0


def _cmd_benchmark(args) -> int:
    from benchmarks.run_all import run_benchmark
    results = run_benchmark(args.dir, config_path=args.config)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote benchmark results to {args.out}")
    return 0


def _cmd_export_onnx(args) -> int:
    from scripts.export_onnx import export_all
    export_all(out_dir=args.out)
    return 0


def _cmd_download_weights(args) -> int:
    from scripts.download_weights import download_all
    download_all(out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
