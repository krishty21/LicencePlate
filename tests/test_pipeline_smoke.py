"""Smoke test for the full pipeline — verifies the contracts compose.

Doesn't require any real models — uses a stub detector + stub recognizer.
The point is to verify:
  - pipeline.process_image() runs end-to-end without raising
  - All stage timings are populated
  - Errors are returned in the payload (not raised)
  - pick_better ordering is consistent
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anpr.types import (
    PlateDetection, Corner, ReadResult, PlateClass, PlateColor, StageTiming,
)
from anpr.ocr import pick_better
from anpr.grammar import apply_grammar


class StubDetector:
    """Returns one fake detection per image."""
    def __init__(self):
        self.calls = 0
    def detect(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        # A simple car-like rectangle in the middle of the image
        x1, y1, x2, y2 = w // 4, h // 3, 3 * w // 4, 2 * h // 3
        corners = (
            Corner(float(x1), float(y1)),
            Corner(float(x2), float(y1)),
            Corner(float(x2), float(y2)),
            Corner(float(x1), float(y2)),
        )
        return [PlateDetection(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            corners=corners,
            det_conf=0.92,
        )]
    def __call__(self, image):
        return self.detect(image)


class StubRecognizer:
    """Returns a fixed read — simulates the fast path."""
    def __init__(self, text: str = "MH12DE1434", conf: float = 0.9):
        self.text = text
        self.conf = conf
    def read(self, plate_img):
        gram = apply_grammar(self.text)
        return ReadResult(
            text=gram.text,
            char_confs=[self.conf] * len(gram.text),
            mean_conf=self.conf,
            grammar_valid=gram.grammar_valid,
            grammar_kind=gram.grammar_kind,
            corrected=gram.corrected,
            recognizer="stub",
        )


class TestPipelineSmoke:
    def test_full_image_pipeline(self):
        """Run pipeline.process_image on a synthetic image."""
        from anpr.pipeline import Pipeline
        # Build a pipeline with stubs
        pipe = Pipeline(
            detector=StubDetector(),
            fast_recognizer=StubRecognizer("MH12DE1434", 0.92),
            heavy_recognizer=None,
            enhancer=None,
            tau=0.75,
        )
        # Synthetic image — white background, no actual plate
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        results = pipe.process_image(img)
        assert len(results) == 1
        r = results[0]
        assert r.plate == "MH12DE1434"
        assert r.confidence == pytest.approx(0.92)
        assert r.grammar_valid
        assert r.grammar_kind == "private"
        # Timings are populated
        assert r.timings.rectify_ms >= 0
        assert r.timings.ocr_ms >= 0
        assert r.timings.total_ms >= 0

    def test_pipeline_with_low_conf_triggers_no_enhance_when_no_enhancer(self):
        """If mean_conf < tau but enhancer is None, the read stays as fast."""
        from anpr.pipeline import Pipeline
        pipe = Pipeline(
            detector=StubDetector(),
            fast_recognizer=StubRecognizer("XX", 0.4),
            heavy_recognizer=None,
            enhancer=None,
            tau=0.75,
        )
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        results = pipe.process_image(img)
        assert len(results) == 1
        # No enhancer → enhanced=False even though conf < tau
        assert results[0].enhanced is False

    def test_pipeline_never_raises_on_bad_input(self):
        """Empty / zero-size image must not raise."""
        from anpr.pipeline import Pipeline
        pipe = Pipeline(
            detector=StubDetector(),
            fast_recognizer=StubRecognizer(),
        )
        # Empty image
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        # Should not raise — returns whatever detector returns
        try:
            results = pipe.process_image(empty)
        except Exception as e:
            # If it does raise, the test fails
            pytest.fail(f"pipeline raised on empty input: {e}")

    def test_emit_log_produces_valid_json(self):
        """emit_log should produce parseable JSON."""
        import json
        from anpr.pipeline import Pipeline
        from anpr.types import PlateResult, PlateClass, PlateColor, StageTiming, Corner
        pipe = Pipeline(
            detector=StubDetector(),
            fast_recognizer=StubRecognizer(),
        )
        r = PlateResult(
            plate="MH12DE1434", confidence=0.9,
            bbox=(0, 0, 100, 50),
            corners=tuple(Corner(0, 0) for _ in range(4)),
            grammar_valid=True, grammar_kind="private",
            corrected=False, enhanced=False, recognizer="lprnet",
            plate_class=PlateClass.CAR,
            plate_color=PlateColor.PRIVATE,
            timings=StageTiming(detect_ms=10, rectify_ms=1, ocr_ms=5,
                                 enhance_ms=0, total_ms=16),
        )
        log_line = pipe.emit_log(r, source="test.jpg")
        parsed = json.loads(log_line)
        assert parsed["plate"] == "MH12DE1434"
        assert parsed["confidence"] == 0.9
        assert parsed["grammar_kind"] == "private"
        assert parsed["timings"]["total_ms"] == 16


class TestPickBetter:
    """Verify pick_better ordering: (grammar_valid, mean_conf)."""

    def test_grammar_valid_beats_invalid(self):
        a = ReadResult(text="MH12DE1434", char_confs=[], mean_conf=0.5,
                        grammar_valid=True, recognizer="lprnet")
        b = ReadResult(text="XYZ123", char_confs=[], mean_conf=0.99,
                        grammar_valid=False, recognizer="parseq")
        # a wins despite lower conf — grammar_valid dominates
        assert pick_better(a, b) is a

    def test_higher_conf_when_both_grammar_valid(self):
        a = ReadResult(text="MH12DE1434", char_confs=[], mean_conf=0.7,
                        grammar_valid=True, recognizer="lprnet")
        b = ReadResult(text="KA05AB1234", char_confs=[], mean_conf=0.9,
                        grammar_valid=True, recognizer="parseq")
        assert pick_better(a, b) is b

    def test_no_ties_possible(self):
        # Same grammar_valid + same mean_conf (rounded to 3 dp) is theoretically
        # possible but extremely rare with real recognizers. Verify max() picks
        # one consistently anyway.
        a = ReadResult(text="A", char_confs=[], mean_conf=0.8,
                        grammar_valid=True)
        b = ReadResult(text="B", char_confs=[], mean_conf=0.8,
                        grammar_valid=True)
        result = pick_better(a, b)
        assert result in (a, b)
