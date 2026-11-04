"""Pipeline orchestrator — owns stage sequencing, timers, structured logs.

Guarantees: never raises on bad input media. Errors are returned per-image
in the result payload, structured as PipelineError.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .detect import Detector
from .enhance import Enhancer
from .ocr import LPRNetRecognizer, PARSeqRecognizer, PaddleRecognizer, pick_better
from .rectify import rectify, classify_color
from .track import Tracker
from .types import (PlateResult, PlateDetection, ReadResult, StageTiming,
                    PipelineError, PlateClass, PlateColor)

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end image/video ANPR pipeline.

    Construct once with config; call process_image() per image or
    process_video() per video file.
    """

    def __init__(self, detector: Detector,
                 fast_recognizer: LPRNetRecognizer | PaddleRecognizer,
                 heavy_recognizer: Optional[PARSeqRecognizer] = None,
                 enhancer: Optional[Enhancer] = None,
                 tau: float = 0.75,
                 tracker: Optional[Tracker] = None):
        self.detector = detector
        self.fast_recognizer = fast_recognizer
        self.heavy_recognizer = heavy_recognizer
        self.enhancer = enhancer
        self.tau = tau
        self.tracker = tracker

    @classmethod
    def from_config(cls, cfg: dict) -> "Pipeline":
        """Construct a Pipeline from a parsed YAML config dict."""
        from .detect import Detector
        from .ocr import LPRNetRecognizer, PARSeqRecognizer, PaddleRecognizer
        from .enhance import Enhancer
        from .track import Tracker
        d = cfg["detect"]
        detector = Detector(
            weights=d["weights"], conf_thres=d.get("conf_thres", 0.25),
            iou_thres=d.get("iou_thres", 0.45), imgsz=d.get("imgsz", 640),
            max_det=d.get("max_det", 10), device=d.get("device", "cuda:0"),
            nms=d.get("nms", False),
        )
        o = cfg["ocr"]
        if o.get("fast", {}).get("kind") == "paddle":
            fast = PaddleRecognizer(lang=o["fast"].get("lang", "en"),
                                     use_gpu=o["fast"].get("use_gpu", True))
        else:
            fast = LPRNetRecognizer(
                weights=o["fast"]["weights"],
                device=o["fast"].get("device", "cuda"),
            )
        heavy = None
        if o.get("heavy"):
            heavy = PARSeqRecognizer(
                weights=o["heavy"]["weights"],
                device=o["heavy"].get("device", "cuda"),
                charset=o["heavy"].get("charset",
                                       "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"),
            )
        enhancer = None
        if cfg.get("enhance", {}).get("enabled", False):
            e_cfg = cfg["enhance"]
            enhancer = Enhancer(
                weights=e_cfg.get("weights"),
                backend=e_cfg.get("backend", "realesrgan"),
                tile=e_cfg.get("tile", 256),
                fp16=e_cfg.get("fp16", True),
                device=e_cfg.get("device", "cuda"),
                scale=e_cfg.get("scale", 4),
            )
        tracker = Tracker() if cfg.get("track", {}).get("enabled", False) else None
        return cls(
            detector=detector, fast_recognizer=fast,
            heavy_recognizer=heavy, enhancer=enhancer,
            tau=cfg.get("ocr", {}).get("tau", 0.75),
            tracker=tracker,
        )

    # ---------- image path ----------

    def process_image(self, image: np.ndarray) -> list[PlateResult]:
        """Process one BGR image. Returns 0+ plate results."""
        t_start = time.perf_counter()
        errors: list[PipelineError] = []
        try:
            detections = self.detector.detect(image)
        except Exception as e:
            logger.exception("detect failed")
            return [PlateResult(
                plate="", confidence=0.0, bbox=(0, 0, 0, 0),
                corners=tuple(), grammar_valid=False, grammar_kind=None,
                corrected=False, enhanced=False, recognizer="none",
                plate_class=PlateClass.UNKNOWN,
                plate_color=PlateColor.UNKNOWN,
                timings=StageTiming(),
                error=f"detect: {e}",
            )]
        results: list[PlateResult] = []
        for det in detections:
            res = self._process_one(image, det, errors)
            results.append(res)
        if errors:
            logger.warning("image had %d errors: %s", len(errors), errors)
        # Total time
        if results:
            t_total = (time.perf_counter() - t_start) * 1000
            results[-1].timings.total_ms = t_total
        return results

    def _process_one(self, image: np.ndarray, det: PlateDetection,
                     errors: list[PipelineError]) -> PlateResult:
        """Process one detection through rectify → read → maybe enhance."""
        timings = StageTiming()
        # Rectify
        t0 = time.perf_counter()
        try:
            plate_img, plate_class = rectify(image, det)
            plate_color = classify_color(plate_img)
            det = PlateDetection(
                bbox=det.bbox, corners=det.corners,
                det_conf=det.det_conf,
                plate_class=plate_class, plate_color=plate_color,
            )
        except Exception as e:
            errors.append(PipelineError(stage="rectify", message=str(e)))
            return PlateResult(
                plate="", confidence=0.0, bbox=det.bbox, corners=det.corners,
                grammar_valid=False, grammar_kind=None, corrected=False,
                enhanced=False, recognizer="none",
                plate_class=PlateClass.UNKNOWN,
                plate_color=PlateColor.UNKNOWN, timings=timings,
                error=f"rectify: {e}",
            )
        timings.rectify_ms = (time.perf_counter() - t0) * 1000
        # Fast read
        t1 = time.perf_counter()
        try:
            read_fast = self.fast_recognizer.read(plate_img)
        except Exception as e:
            errors.append(PipelineError(stage="ocr_fast", message=str(e)))
            read_fast = ReadResult(text="", char_confs=[], mean_conf=0.0,
                                    grammar_valid=False)
        timings.ocr_ms = (time.perf_counter() - t1) * 1000
        # Gated enhancement
        enhanced = False
        recognizer_used = read_fast.recognizer
        final_read = read_fast
        if (read_fast.mean_conf < self.tau
                and self.enhancer is not None
                and self.heavy_recognizer is not None):
            t2 = time.perf_counter()
            try:
                enhanced_img = self.enhancer.enhance(plate_img)
                read_heavy = self.heavy_recognizer.read(enhanced_img)
                final_read = pick_better(read_fast, read_heavy)
                recognizer_used = final_read.recognizer
                enhanced = True
            except Exception as e:
                errors.append(PipelineError(stage="enhance", message=str(e)))
            timings.enhance_ms = (time.perf_counter() - t2) * 1000
        # Post-process already applied inside recognizer (apply_grammar).
        timings.postprocess_ms = 0.0
        return PlateResult(
            plate=final_read.text,
            confidence=final_read.mean_conf,
            bbox=det.bbox, corners=det.corners,
            grammar_valid=final_read.grammar_valid,
            grammar_kind=final_read.grammar_kind,
            corrected=final_read.corrected,
            enhanced=enhanced, recognizer=recognizer_used,
            plate_class=det.plate_class, plate_color=det.plate_color,
            timings=timings,
        )

    # ---------- video path ----------

    def process_video(self, video_path: str | Path,
                      emit_every: int = 5) -> Iterator[dict]:
        """Process a video file. Yields a JSON-serializable dict per frame.

        Emits a `final` read per track when ≥ MIN_VOTES votes agree, plus
        per-frame `detections` for visualization.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        if self.tracker is None:
            raise RuntimeError("tracker not configured; set track.enabled=true")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                detections = self.detector.detect(frame)
                # Rectify + read for each detection.
                plate_imgs: list[np.ndarray] = []
                reads: list[ReadResult] = []
                for det in detections:
                    plate_img, _ = rectify(frame, det)
                    plate_imgs.append(plate_img)
                    reads.append(self.fast_recognizer.read(plate_img))
                # Track + vote.
                track_reads = self.tracker.update(frame, detections, reads, frame_idx)
                if frame_idx % emit_every == 0 or track_reads:
                    yield {
                        "frame": frame_idx,
                        "n_detections": len(detections),
                        "final_reads": [
                            {
                                "track_id": tr.track_id,
                                "plate": tr.plate,
                                "confidence": tr.confidence,
                                "votes": tr.votes,
                                "bbox": list(tr.bbox),
                            } for tr in track_reads
                        ],
                    }
        finally:
            cap.release()

    def emit_log(self, result: PlateResult, source: str = "") -> str:
        """Emit one JSON-lines log entry for offline analysis."""
        return json.dumps({
            "source": source,
            "plate": result.plate,
            "confidence": result.confidence,
            "bbox": list(result.bbox),
            "grammar_valid": result.grammar_valid,
            "grammar_kind": result.grammar_kind,
            "corrected": result.corrected,
            "enhanced": result.enhanced,
            "recognizer": result.recognizer,
            "plate_class": result.plate_class.value,
            "plate_color": result.plate_color.value,
            "timings": {
                "detect_ms": result.timings.detect_ms,
                "rectify_ms": result.timings.rectify_ms,
                "ocr_ms": result.timings.ocr_ms,
                "enhance_ms": result.timings.enhance_ms,
                "postprocess_ms": result.timings.postprocess_ms,
                "total_ms": result.timings.total_ms,
            },
            "error": result.error,
        })
