"""Core data types — I/O contracts for every module.

Every module in anpr.* consumes/produces these types. Do not pass naked dicts
or tuples between stages — the type system is the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class PlateClass(str, Enum):
    """Plate shape class — drives rectification target dimensions."""

    CAR = "car"                  # LMV: ~50×12 cm, aspect ~4.17:1
    TWO_WHEELER_REAR = "two_wheeler_rear"   # ~20×10 cm, aspect 2:1
    TWO_WHEELER_FRONT = "two_wheeler_front"  # ~28.5×4.5 cm, aspect 6.33:1
    TRUCK = "truck"              # MCV/HCV: ~34×20 cm, aspect 1.7:1
    THREE_WHEELER = "three_wheeler"  # ~20×10 cm, aspect 2:1
    UNKNOWN = "unknown"


class PlateColor(str, Enum):
    """Indian plate color coding — drives preprocessing.

    See: https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_India
    """

    PRIVATE = "private_black_on_white"
    PRIVATE_EV = "private_ev_white_on_green"
    COMMERCIAL = "commercial_black_on_yellow"
    COMMERCIAL_EV = "commercial_ev_yellow_on_green"   # lowest contrast
    RENTAL = "rental_yellow_on_black"
    DEFENSE = "defense_white_on_black"
    DIPLOMATIC = "diplomatic_white_on_blue"
    TEMPORARY = "temporary_red_on_yellow"
    TRADE = "trade_white_on_red"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Corner:
    """A single plate corner keypoint."""

    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class PlateDetection:
    """Detector output for one plate instance.

    corners is ordered [TL, TR, BR, BL] clockwise from top-left.
    """

    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in image px
    corners: tuple[Corner, Corner, Corner, Corner]
    det_conf: float
    plate_class: PlateClass = PlateClass.UNKNOWN
    plate_color: PlateColor = PlateColor.UNKNOWN

    @property
    def corner_array(self) -> np.ndarray:
        """Return corners as (4, 2) float32 array for OpenCV."""
        return np.array(
            [[c.x, c.y] for c in self.corners], dtype=np.float32
        )


@dataclass
class ReadResult:
    """Recognizer output for one plate image."""

    text: str
    char_confs: list[float]
    mean_conf: float
    grammar_valid: bool
    grammar_kind: Optional[str] = None   # "private", "bh", "va", "diplomatic", ...
    corrected: bool = False              # state-code correction applied?
    recognizer: str = "lprnet"          # "lprnet" | "parseq" | "paddle"

    def sort_key(self) -> tuple[bool, float]:
        """Total order used by pick_better — no ties possible."""
        return (self.grammar_valid, round(self.mean_conf, 3))


@dataclass
class StageTiming:
    """Per-stage latency in milliseconds."""

    detect_ms: float = 0.0
    rectify_ms: float = 0.0
    ocr_ms: float = 0.0
    enhance_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PlateResult:
    """End-to-end pipeline output for one image."""

    plate: str
    confidence: float
    bbox: tuple[float, float, float, float]
    corners: tuple[Corner, Corner, Corner, Corner]
    grammar_valid: bool
    grammar_kind: Optional[str]
    corrected: bool
    enhanced: bool
    recognizer: str
    plate_class: PlateClass
    plate_color: PlateColor
    timings: StageTiming
    error: Optional[str] = None


@dataclass
class TrackRead:
    """Video tracking output for one tracked vehicle."""

    track_id: int
    plate: str
    confidence: float
    votes: int                          # number of frames that contributed
    last_seen_frame: int
    bbox: tuple[float, float, float, float]


@dataclass
class PipelineError:
    """Structured error — pipeline never raises on bad input."""

    stage: str
    message: str
    recoverable: bool = True
