"""Video tracking + temporal voting — ByteTrack + confidence-weighted voting.

Contract: (frame, detections, frame_idx) -> list[TrackRead].
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from .types import PlateDetection, ReadResult, TrackRead

logger = logging.getLogger(__name__)

GAMMA = 0.98   # recency decay per frame
MIN_VOTES = 3  # minimum votes before emitting a final plate read
MAX_DEQUE = 30 # frames of memory per track


class TrackVoter:
    """Per-track plate voting.

    Each track maintains a deque of (text, conf, frame_idx). The final plate
    is argmax of confidence-weighted votes, weight = conf * gamma^(age).
    """

    def __init__(self, gamma: float = GAMMA, min_votes: int = MIN_VOTES,
                 max_deque: int = MAX_DEQUE):
        self.gamma = gamma
        self.min_votes = min_votes
        self.max_deque = max_deque
        self._votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_deque))
        self._current_frame: int = 0

    def add_vote(self, track_id: int, text: str, conf: float, frame_idx: int):
        if text:   # skip empty reads
            self._votes[track_id].append((text, conf, frame_idx))

    def get_plate(self, track_id: int) -> Optional[TrackRead]:
        """Return the best plate read for a track, or None if not enough votes."""
        votes = self._votes.get(track_id)
        if not votes or len(votes) < self.min_votes:
            return None
        # Aggregate weighted votes per text string.
        weights: dict[str, float] = defaultdict(float)
        last_frame = 0
        last_bbox = None
        for text, conf, fidx in votes:
            age = self._current_frame - fidx
            w = conf * (self.gamma ** age)
            weights[text] += w
            if fidx > last_frame:
                last_frame = fidx
        best_text = max(weights, key=weights.get)
        best_conf = weights[best_text] / max(1, len(votes))
        return TrackRead(
            track_id=track_id,
            plate=best_text,
            confidence=best_conf,
            votes=len(votes),
            last_seen_frame=last_frame,
            bbox=(0, 0, 0, 0),   # filled in by caller from latest detection
        )

    def step(self, frame_idx: int):
        self._current_frame = frame_idx

    def reset(self):
        self._votes.clear()


class Tracker:
    """ByteTrack via supervision — wraps the per-frame detection+tracking loop.

    Tracks plates by bbox IoU. The plate itself is the re-id signal — no
    separate re-id embedding network is needed (saves a model).
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.7,
                 track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self._byte_tracker = None
        self._voter = TrackVoter()

    def _load(self):
        if self._byte_tracker is not None:
            return
        try:
            from supervision.tracker.byte_tracker.core import ByteTrack
        except ImportError:
            try:
                # supervision >= 0.20 renamed module path
                from supervision.tracker import ByteTrack   # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "supervision not installed. Run: pip install supervision"
                ) from e
        self._byte_tracker = ByteTrack(
            track_activation_thresh=self.track_thresh,
            minimum_match_threshold=self.match_thresh,
            minimum_consecutive_frames=self.track_buffer // 4,
        )

    def update(self, frame: np.ndarray,
               detections: list[PlateDetection],
               reads: list[ReadResult],
               frame_idx: int) -> list[TrackRead]:
        """Process one video frame: track plates, register reads as votes.

        Returns the current best-read for each active track (or None for
        tracks that don't have enough votes yet — caller filters).
        """
        t0 = time.perf_counter()
        self._load()
        import supervision as sv
        self._voter.step(frame_idx)
        if not detections:
            self._byte_tracker.update_with_detections(
                sv.Detections.empty()
            )
            return []
        # Build sv.Detections from PlateDetection list.
        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        conf = np.array([d.det_conf for d in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)
        sv_dets = sv.Detections(
            xyxy=xyxy, confidence=conf, class_id=class_id,
            data={"idx": np.arange(len(detections))}
        )
        tracked = self._byte_tracker.update_with_detections(sv_dets)
        # Map tracked indices back to original detections.
        idx_map = tracked.data.get("idx", np.arange(len(tracked)))
        tracker_ids = tracked.tracker_id
        results: list[TrackRead] = []
        for i, tid in enumerate(tracker_ids):
            orig_idx = int(idx_map[i])
            if orig_idx >= len(reads):
                continue
            read = reads[orig_idx]
            det = detections[orig_idx]
            self._voter.add_vote(int(tid), read.text, read.mean_conf, frame_idx)
            best = self._voter.get_plate(int(tid))
            if best is not None:
                best.bbox = det.bbox
                results.append(best)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("track: %.2fms active=%d emitted=%d",
                     elapsed, len(tracked), len(results))
        return results
