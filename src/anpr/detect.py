"""Detector — YOLO26s-pose wrapper for plate bbox + 4 corner keypoints.

Contract: np.ndarray (BGR) -> list[PlateDetection].

Uses Ultralytics YOLO26-pose. Falls back to YOLO11-pose if YOLO26 weights
not available (config-driven).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .types import PlateDetection, Corner, PlateClass

logger = logging.getLogger(__name__)


class Detector:
    """YOLO26-pose plate detector.

    Loads weights on init. ONNX path is preferred for inference; .pt is used
    for training export scripts.
    """

    def __init__(self, weights: str | Path, conf_thres: float = 0.25,
                 iou_thres: float = 0.45, imgsz: int = 640, max_det: int = 10,
                 device: str = "cuda:0", nms: bool = False):
        self.weights = str(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.nms = nms
        self._model = None   # lazy load

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e
        logger.info("loading detector weights: %s", self.weights)
        self._model = YOLO(self.weights, task="pose")

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Detect plates in a BGR image. Returns 0+ detections."""
        t0 = time.perf_counter()
        self._load()
        results = self._model(
            image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        detections: list[PlateDetection] = []
        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()       # (N, 4)
            confs = r.boxes.conf.cpu().numpy()        # (N,)
            kpts_xy = r.keypoints.xy.cpu().numpy()    # (N, K, 2)
            kpts_vis = r.keypoints.conf.cpu().numpy()  # (N, K) — visibility/conf
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                corners = []
                for k in range(4):
                    corners.append(Corner(
                        x=float(kpts_xy[i, k, 0]),
                        y=float(kpts_xy[i, k, 1]),
                        visibility=float(kpts_vis[i, k]),
                    ))
                detections.append(PlateDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    corners=tuple(corners),   # type: ignore
                    det_conf=float(confs[i]),
                    plate_class=PlateClass.UNKNOWN,   # rectify.py infers this
                ))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("detect: %.2fms found=%d", elapsed, len(detections))
        return detections

    def __call__(self, image: np.ndarray) -> list[PlateDetection]:
        return self.detect(image)


def collate_for_training(batch: list[dict]) -> dict:
    """Custom collate for YOLO26-pose training with corner supervision.

    Used by scripts/export_onnx.py and the Kaggle training notebook.
    """
    # Ultralytics handles its own collation; this is here as documentation
    # of the data format expected from a custom dataset loader.
    return {"batch": batch}
