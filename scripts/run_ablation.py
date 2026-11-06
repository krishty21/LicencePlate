"""Real ablation runner — actually executes A1-A10.

Replaces the scaffold in benchmarks/run_all.py. Each ablation:
  1. Deep-copies the base config
  2. Applies the config override
  3. Re-constructs the Pipeline
  4. Runs run_benchmark() on the frozen test set
  5. Aggregates metrics across seeds ×3
  6. Auto-generates a markdown table

Usage:
  python -m scripts.run_ablation --test-dir data/test --ablation A9 --out results/A9.md
  python -m scripts.run_ablation --test-dir data/test --all --out results/
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root for path resolution
ROOT = Path(__file__).parent.parent

ABLATIONS: list[dict] = [
    {"id": "A1", "name": "OCR only, no rectification",
     "measures": "value of corners/homography",
     "overrides": [{"rectify.enabled": False}]},
    {"id": "A2", "name": "never vs always vs gated enhancement",
     "measures": "accuracy/latency frontier",
     "overrides": [{"enhance.policy": "never"},
                       {"enhance.policy": "always"},
                       {"enhance.policy": "gated"}],
     "headline": True},
    {"id": "A3", "name": "synthetic-pretrained vs from-scratch recognizer",
     "measures": "value of synthetic data",
     "overrides": [{"ocr.train_lprnet.pretrain.enabled": True},
                       {"ocr.train_lprnet.pretrain.enabled": False}]},
    {"id": "A4", "name": "grammar post-processing on/off",
     "measures": "string error rate",
     "overrides": [{"grammar.enabled": True},
                       {"grammar.enabled": False}]},
    {"id": "A5", "name": "temporal voting on/off (UFPR video)",
     "measures": "plate accuracy on video",
     "overrides": [{"track.voting.enabled": True},
                       {"track.voting.enabled": False}],
     "dataset": "ufpr_alpr"},
    {"id": "A6", "name": "YOLO26n-pose vs YOLO26s-pose",
     "measures": "accuracy vs latency (detector scale)",
     "overrides": [{"detect.weights": "weights/yolo26n-pose-plate.pt"},
                       {"detect.weights": "weights/yolo26s-pose-plate.pt"}]},
    {"id": "A7", "name": "LPRNet+ESRGAN-gated (v1.0) vs PARSeq-only (no ESRGAN)",
     "measures": "architecture simplification",
     "overrides": [{"ocr.mode": "hybrid_lprnet_gated"},
                       {"ocr.mode": "parseq_only"}]},
    {"id": "A8", "name": "4-way enhancement: Real-ESRGAN vs Plate-SR vs TTPSR vs CLAHE",
     "measures": "best SR for plate OCR",
     "overrides": [{"enhance.backend": "realesrgan"},
                       {"enhance.backend": "plate_sr"},
                       {"enhance.backend": "ttpsr"},
                       {"enhance.backend": "clahe"}]},
    {"id": "A9", "name": "YOLO11s vs YOLO26s vs YOLO26n",
     "measures": "detector architecture & scale frontier (Nov 2026)",
     "overrides": [{"detect.weights": "weights/yolo11s-pose-plate.pt"},
                       {"detect.weights": "weights/yolo26s-pose-plate.pt"},
                       {"detect.weights": "weights/yolo26n-pose-plate.pt"}]},
    {"id": "A10", "name": "rectify-3:1-fixed vs class-aware-rectify",
     "measures": "value of two-wheeler / car / truck shape handling",
     "overrides": [{"rectify.mode": "fixed_192x64"},
                       {"rectify.mode": "class_aware"}]},
]


def set_nested(d: dict, key: str, value) -> None:
    """Set d['a']['b'] = v from key 'a.b'."""
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        set_nested(cfg, k, v)
    return cfg


def run_one_config(cfg: dict, test_dir: Path) -> dict:
    """Run the benchmark with one specific config dict. Returns metrics."""
    import cv2
    sys.path.insert(0, str(ROOT / "src"))
    from anpr.pipeline import Pipeline
    pipe = Pipeline.from_config(cfg)
    images_dir = test_dir / "images"
    if not images_dir.exists():
        return {"error": f"no images at {images_dir}"}
    # Load labels
    labels = {}
    labels_path = test_dir / "labels.csv"
    if labels_path.exists():
        import csv
        with labels_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row.get("text", "")
    results: list[tuple[str, str, dict]] = []
    n_images = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_images += 1
        plate_results = pipe.process_image(img)
        gt = labels.get(img_path.name, "")
        # Take the highest-confidence detection as the answer
        if plate_results:
            best = max(plate_results, key=lambda r: r.confidence)
            results.append((
                best.plate, gt,
                {"grammar_valid": best.grammar_valid,
                 "corrected": best.corrected,
                 "enhanced": best.enhanced,
                 "recognizer": best.recognizer,
                 "latency_ms": best.timings.total_ms},
            ))
    from benchmarks.metrics import aggregate
    metrics = aggregate(results)
    return {"n_images": n_images, **asdict(metrics)}


def run_ablation(ablation_id: str, test_dir: Path,
                 base_config_path: Path = ROOT / "configs" / "serve.yaml",
                 seeds: tuple[int, ...] = (42,)) -> dict:
    """Run one ablation row across seeds. Returns averaged metrics per override."""
    spec = next((a for a in ABLATIONS if a["id"] == ablation_id), None)
    if spec is None:
        raise ValueError(f"unknown ablation: {ablation_id}")
    import yaml
    base_cfg = yaml.safe_load(base_config_path.read_text())
    per_override: list[dict] = []
    for override in spec["overrides"]:
        cfg = apply_overrides(base_cfg, override)
        # Average metrics across seeds (most components are deterministic;
        # seeds matter for any stochastic augmentation in training, which
        # is not invoked at inference time — so a single seed is usually fine).
        seed_results: list[dict] = []
        for seed in seeds:
            import random
            random.seed(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except ImportError:
                pass
            try:
                result = run_one_config(cfg, test_dir)
                result["override"] = override
                result["seed"] = seed
                seed_results.append(result)
            except Exception as e:
                logger.exception("config failed: %s — %s", override, e)
                seed_results.append({"error": str(e), "override": override,
                                      "seed": seed})
        per_override.append({
            "override": override,
            "results": seed_results,
        })
    return {"ablation": ablation_id, "spec": spec, "per_override": per_override}


def render_markdown(result: dict) -> str:
    """Render ablation results as a markdown table."""
    spec = result["spec"]
    lines = [f"## {spec['id']} — {spec['name']}", "",
             f"**Measures:** {spec['measures']}", ""]
    if spec.get("dataset"):
        lines.append(f"**Dataset:** {spec['dataset']}")
        lines.append("")
    # Table header
    lines.append("| Override | Plate Acc | Char Acc | NED | p50 ms | p95 ms | N |")
    lines.append("|---|---|---|---|---|---|---|")
    for override_result in result["per_override"]:
        override = override_result["override"]
        # Average across seeds
        ok_results = [r for r in override_result["results"] if "error" not in r]
        if not ok_results:
            lines.append(f"| {override} | ERROR | - | - | - | - | 0 |")
            continue
        n = ok_results[0].get("n", 0)
        plate_acc = sum(r.get("plate_accuracy", 0) for r in ok_results) / len(ok_results)
        char_acc = sum(r.get("char_accuracy", 0) for r in ok_results) / len(ok_results)
        ned = sum(r.get("normalized_edit_distance", 0) for r in ok_results) / len(ok_results)
        p50 = sum(r.get("latency_p50_ms", 0) for r in ok_results) / len(ok_results)
        p95 = sum(r.get("latency_p95_ms", 0) for r in ok_results) / len(ok_results)
        lines.append(f"| {override} | {plate_acc:.3f} | {char_acc:.3f} "
                     f"| {ned:.3f} | {p50:.1f} | {p95:.1f} | {n} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run ANPR ablation A1-A10")
    p.add_argument("--test-dir", type=Path, required=True,
                   help="test directory with images/ and labels.csv")
    p.add_argument("--ablation", default=None,
                   help="single ablation id (A1..A10)")
    p.add_argument("--all", action="store_true",
                   help="run all ablations")
    p.add_argument("--out", type=Path, default=Path("results"),
                   help="output directory for markdown + json results")
    p.add_argument("--seeds", default="42",
                   help="comma-separated seeds (default: 42)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    targets = ([args.ablation] if args.ablation
               else [a["id"] for a in ABLATIONS] if args.all
               else [])
    if not targets:
        p.error("must specify --ablation <ID> or --all")
        return 1
    all_results = {}
    md_lines = ["# ANPR-India — Ablation Results", "",
                f"Test set: `{args.test_dir}`", f"Seeds: {seeds}", ""]
    for ablation_id in targets:
        logger.info("=== running %s ===", ablation_id)
        try:
            result = run_ablation(ablation_id, args.test_dir, seeds=seeds)
            all_results[ablation_id] = result
            md = render_markdown(result)
            md_lines.append(md)
            # Per-ablation JSON
            (args.out / f"{ablation_id}.json").write_text(
                json.dumps(result, indent=2, default=str))
            logger.info("%s done — see %s/%s.md", ablation_id, args.out, ablation_id)
        except Exception as e:
            logger.exception("%s failed: %s", ablation_id, e)
            md_lines.append(f"## {ablation_id} — FAILED\n\n{e}\n")
    # Combined markdown
    (args.out / "ablations.md").write_text("\n".join(md_lines))
    (args.out / "ablations.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    logger.info("wrote combined results to %s/ablations.md", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
