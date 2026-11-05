"""Run all ablation experiments A1-A10.

Pre-registered matrix from REVIEW_AND_RECOMMENDATIONS.md §6.
Same frozen test set, seeds ×3 where stochastic, auto-generated markdown tables.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

ABLATIONS: list[dict] = [
    {"id": "A1", "name": "OCR only, no rectification",
     "measures": "value of corners/homography",
     "config_override": {"rectify.enabled": False}},
    {"id": "A2", "name": "never vs always vs gated enhancement",
     "measures": "accuracy/latency frontier",
     "config_override": {"enhance.policy": ["never", "always", "gated"]},
     "headline": True},
    {"id": "A3", "name": "synthetic-pretrained vs from-scratch recognizer",
     "measures": "value of synthetic data",
     "config_override": {"ocr.train_lprnet.pretrain.enabled": [True, False]}},
    {"id": "A4", "name": "grammar post-processing on/off",
     "measures": "string error rate",
     "config_override": {"grammar.enabled": [True, False]}},
    {"id": "A5", "name": "temporal voting on/off (UFPR video)",
     "measures": "plate accuracy on video",
     "config_override": {"track.voting.enabled": [True, False]},
     "dataset": "ufpr_alpr"},
    {"id": "A6", "name": "YOLO26n-pose vs YOLO26s-pose",
     "measures": "accuracy vs latency frontier (detector scale)",
     "config_override": {"detect.weights": [
         "weights/yolo26n-pose-plate.pt",
         "weights/yolo26s-pose-plate.pt"]}},
    {"id": "A7", "name": "LPRNet+ESRGAN-gated (v1.0) vs PARSeq-only (no ESRGAN)",
     "measures": "architecture simplification — can one heavy recognizer replace the gated branch?",
     "config_override": {"ocr.mode": ["hybrid_lprnet_gated", "parseq_only"]}},
    {"id": "A8", "name": "4-way enhancement: Real-ESRGAN vs Plate-SR vs TTPSR vs CLAHE",
     "measures": "best SR for plate OCR",
     "config_override": {"enhance.backend": [
         "realesrgan", "plate_sr", "ttpsr", "clahe"]}},
    {"id": "A9", "name": "YOLO11s-pose vs YOLO26s-pose vs YOLO26n-pose",
     "measures": "detector architecture & scale frontier (Nov 2026)",
     "config_override": {"detect.weights": [
         "weights/yolo11s-pose-plate.pt",
         "weights/yolo26s-pose-plate.pt",
         "weights/yolo26n-pose-plate.pt"]}},
    {"id": "A10", "name": "rectify-3:1-fixed vs class-aware-rectify",
     "measures": "value of two-wheeler / car / truck shape handling",
     "config_override": {"rectify.mode": ["fixed_192x64", "class_aware"]}},
]


def run_benchmark(test_dir: Path, config_path: str = "configs/serve.yaml") -> dict:
    """Run baseline (no ablation) on a directory of test images + labels.

    Expected layout:
      test_dir/images/*.jpg
      test_dir/labels.csv (filename, plate_text)
    """
    import yaml
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from anpr.pipeline import Pipeline
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pipe = Pipeline.from_config(cfg)
    images_dir = test_dir / "images"
    labels = {}
    labels_path = test_dir / "labels.csv"
    if labels_path.exists():
        import csv
        with labels_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row.get("text", "")
    results: list[tuple[str, str, dict]] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        plate_results = pipe.process_image(img)
        gt = labels.get(img_path.name, "")
        for r in plate_results:
            results.append((
                r.plate, gt,
                {"grammar_valid": r.grammar_valid,
                 "corrected": r.corrected,
                 "enhanced": r.enhanced,
                 "recognizer": r.recognizer,
                 "latency_ms": r.timings.total_ms},
            ))
            break   # one plate per image for the benchmark
    from .metrics import aggregate
    metrics = aggregate(results)
    return {"n": metrics.n, **asdict(metrics)}


def run_ablation(ablation_id: str, test_dir: Path,
                 config_path: str = "configs/serve.yaml",
                 seeds: tuple[int, ...] = (42, 7, 123)) -> dict:
    """Run one ablation row across N seeds. Returns averaged metrics."""
    spec = next((a for a in ABLATIONS if a["id"] == ablation_id), None)
    if spec is None:
        raise ValueError(f"unknown ablation: {ablation_id}")
    # NOTE: this is a runner scaffold — to actually run, you need to:
    #   1. Apply config_override to a copy of the base config
    #   2. Re-construct the Pipeline with the modified config
    #   3. Run run_benchmark() with the modified config_path
    #   4. Average metrics across seeds
    # The scaffolding is here; the per-override logic is implemented in
    # benchmarks/_ablation_impls.py (not included — write per-ablation as needed).
    logger.warning("ablation runner is a scaffold — implement per-ablation "
                    "config-override logic for %s in benchmarks/_ablation_impls.py",
                    ablation_id)
    return {"ablation": ablation_id, "spec": spec, "seeds": list(seeds),
            "status": "scaffold"}


def render_markdown_table(rows: list[dict]) -> str:
    """Render a markdown table from a list of dicts."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run ANPR benchmarks")
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/serve.yaml")
    p.add_argument("--ablation", default=None,
                   help="run one ablation by id (A1..A10); omit to run baseline only")
    p.add_argument("--out", type=Path, default=Path("benchmarks_results.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    if args.ablation:
        result = run_ablation(args.ablation, args.test_dir, args.config)
    else:
        result = run_benchmark(args.test_dir, args.config)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
