"""Tests for benchmarks/metrics.py — verify all metric computations."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metrics import (
    iou, average_precision, mean_ap_at_iou, corner_error_px,
    edit_distance, normalized_edit_distance, cer, char_accuracy,
    exact_match, aggregate, PlateMetrics,
)


class TestIoU:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_non_overlapping(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Boxes (0,0,10,10) and (5,5,15,15):
        #   intersection = (5,5)-(10,10) = 5x5 = 25
        #   union = 100 + 100 - 25 = 175
        #   IoU = 25/175 ≈ 0.143
        result = iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < result < 0.2


class TestEditDistance:
    def test_identical(self):
        assert edit_distance("MH12DE1434", "MH12DE1434") == 0

    def test_one_substitution(self):
        assert edit_distance("MH12DE1434", "MH12DE1435") == 1

    def test_one_insertion(self):
        assert edit_distance("MH12", "MH12X") == 1

    def test_one_deletion(self):
        assert edit_distance("MH12X", "MH12") == 1

    def test_completely_different(self):
        assert edit_distance("ABCD", "WXYZ") == 4

    def test_empty_first(self):
        assert edit_distance("", "AB") == 2

    def test_empty_second(self):
        assert edit_distance("AB", "") == 2


class TestNormalizedEditDistance:
    def test_identical(self):
        assert normalized_edit_distance("MH12", "MH12") == 0.0

    def test_one_sub_in_four(self):
        # 1 edit / 4 = 0.25
        assert normalized_edit_distance("MH12", "MH13") == pytest.approx(0.25)

    def test_completely_different(self):
        # 4 edits / 4 = 1.0
        assert normalized_edit_distance("ABCD", "WXYZ") == pytest.approx(1.0)


class TestCER:
    def test_perfect_match(self):
        assert cer("MH12DE1434", "MH12DE1434") == 0.0

    def test_one_error_in_ten(self):
        # 1 / 10 = 0.1
        assert cer("MH12DE1434", "MH12DE1435") == pytest.approx(0.1)

    def test_all_wrong(self):
        # 10 / 10 = 1.0
        assert cer("MH12DE1434", "XXXXXXXXXX") == pytest.approx(1.0)

    def test_empty_gt_with_empty_pred(self):
        # No GT, no pred → 0 error
        assert cer("", "") == 0.0

    def test_empty_gt_with_nonempty_pred(self):
        # No GT but pred exists → 100% error (we shouldn't have predicted)
        assert cer("anything", "") == 1.0

    def test_nonempty_gt_with_empty_pred(self):
        # GT exists but no pred → 100% error
        assert cer("", "anything") == 1.0


class TestCharAccuracy:
    def test_perfect(self):
        assert char_accuracy("MH12", "MH12") == 1.0

    def test_one_error(self):
        # 1 - 0.25 = 0.75
        assert char_accuracy("MH12", "MH13") == pytest.approx(0.75)


class TestExactMatch:
    def test_match(self):
        assert exact_match("MH12", "MH12") == 1

    def test_no_match(self):
        assert exact_match("MH12", "MH13") == 0


class TestCornerError:
    def test_identical(self):
        import numpy as np
        c = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        assert corner_error_px(c, c) == 0.0

    def test_one_pixel_off(self):
        import numpy as np
        pred = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        gt = np.array([[1, 1], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        # Only TL moved by (1,1) — L2 = sqrt(2)
        result = corner_error_px(pred, gt)
        assert result == pytest.approx(2 ** 0.5 / 4)   # mean of 4 corners

    def test_shape_mismatch(self):
        import numpy as np
        a = np.zeros((4, 2))
        b = np.zeros((3, 2))
        assert corner_error_px(a, b) == float("inf")


class TestAggregate:
    def test_empty(self):
        result = aggregate([])
        assert isinstance(result, PlateMetrics)
        assert result.n == 0

    def test_perfect_reads(self):
        reads = [
            ("MH12DE1434", "MH12DE1434",
             {"grammar_valid": True, "corrected": False, "enhanced": False,
              "recognizer": "lprnet", "latency_ms": 20.0}),
            ("KA05AB1234", "KA05AB1234",
             {"grammar_valid": True, "corrected": False, "enhanced": False,
              "recognizer": "lprnet", "latency_ms": 22.0}),
        ]
        result = aggregate(reads)
        assert result.n == 2
        assert result.plate_accuracy == 1.0
        assert result.char_accuracy == 1.0
        assert result.normalized_edit_distance == 0.0
        assert result.grammar_valid_rate == 1.0

    def test_mixed_reads(self):
        reads = [
            ("MH12DE1434", "MH12DE1434",
             {"grammar_valid": True, "corrected": False, "enhanced": False,
              "recognizer": "lprnet", "latency_ms": 20.0}),
            ("MH12DE1435", "MH12DE1434",   # one wrong char
             {"grammar_valid": True, "corrected": False, "enhanced": True,  # enhanced path
              "recognizer": "parseq", "latency_ms": 200.0}),
            ("KA05AB1234", "KA05AB1234",
             {"grammar_valid": True, "corrected": False, "enhanced": False,
              "recognizer": "lprnet", "latency_ms": 25.0}),
        ]
        result = aggregate(reads)
        assert result.n == 3
        assert result.plate_accuracy == pytest.approx(2 / 3)  # 2 of 3 exact
        assert result.enhanced_rate == pytest.approx(1 / 3)
        assert result.parseq_rate == pytest.approx(1 / 3)
        # p50 of [20, 25, 200] sorted, len=3, index=1, value=25
        assert result.latency_p50_ms == 25.0


class TestAveragePrecision:
    def test_perfect_detection(self):
        # Single perfect prediction, single GT
        preds = [((0, 0, 10, 10), 0.9)]
        gts = [(0, 0, 10, 10)]
        ap = average_precision(preds, gts, iou_thresh=0.5)
        assert ap == pytest.approx(1.0)

    def test_no_predictions(self):
        ap = average_precision([], [(0, 0, 10, 10)])
        assert ap == 0.0

    def test_no_ground_truth(self):
        ap = average_precision([((0, 0, 10, 10), 0.9)], [])
        assert ap == 0.0
