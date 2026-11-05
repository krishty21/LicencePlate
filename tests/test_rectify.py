"""Tests for rectify.py — corner ordering, sanity gates, class-aware shapes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anpr.rectify import (
    order_corners, is_quad_valid, classify_plate_shape, PLATE_SHAPES,
)
from anpr.types import PlateDetection, Corner, PlateClass


class TestOrderCorners:
    def test_already_ordered(self):
        corners = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
        ordered = order_corners(corners)
        # TL, TR, BR, BL clockwise
        assert tuple(ordered[0]) == (0, 0)
        assert tuple(ordered[1]) == (10, 0)
        assert tuple(ordered[2]) == (10, 5)
        assert tuple(ordered[3]) == (0, 5)

    def test_shuffled_input(self):
        corners = np.array([[10, 5], [10, 0], [0, 5], [0, 0]], dtype=np.float32)
        ordered = order_corners(corners)
        assert tuple(ordered[0]) == (0, 0)   # TL
        assert tuple(ordered[1]) == (10, 0)  # TR
        assert tuple(ordered[2]) == (10, 5)  # BR
        assert tuple(ordered[3]) == (0, 5)   # BL

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            order_corners(np.zeros((3, 2)))


class TestIsQuadValid:
    def test_valid_quad(self):
        corners = np.array([[0, 0], [50, 0], [50, 20], [0, 20]], dtype=np.float32)
        assert is_quad_valid(corners)

    def test_tiny_quad_rejected(self):
        corners = np.array([[0, 0], [5, 0], [5, 2], [0, 2]], dtype=np.float32)
        assert not is_quad_valid(corners)   # area = 10 < 100

    def test_extreme_aspect_rejected(self):
        corners = np.array([[0, 0], [800, 0], [800, 5], [0, 5]], dtype=np.float32)
        assert not is_quad_valid(corners)   # aspect 160 > 8


class TestClassifyPlateShape:
    def test_car_aspect(self):
        # 4:1 aspect → car
        corners = np.array([[0, 0], [200, 0], [200, 50], [0, 50]], dtype=np.float32)
        assert classify_plate_shape(corners) == PlateClass.CAR

    def test_two_wheeler_front_aspect(self):
        # 6.5:1 aspect → two_wheeler_front
        corners = np.array([[0, 0], [195, 0], [195, 30], [0, 30]], dtype=np.float32)
        assert classify_plate_shape(corners) == PlateClass.TWO_WHEELER_FRONT

    def test_two_wheeler_rear_aspect(self):
        # 2:1 aspect → two_wheeler_rear
        corners = np.array([[0, 0], [128, 0], [128, 64], [0, 64]], dtype=np.float32)
        assert classify_plate_shape(corners) == PlateClass.TWO_WHEELER_REAR

    def test_truck_aspect(self):
        # Trucks (MCV/HCV) are 34x20cm = 1.7:1, but this overlaps with
        # two-wheeler rear (20x10cm = 2:1). The function uses aspect <= 1.5
        # as the truck threshold; 1.5-2.7 falls in two-wheeler-rear range and
        # requires the DETECTOR to classify (not aspect ratio alone).
        # 1.4:1 aspect is unambiguously truck.
        corners = np.array([[0, 0], [140, 0], [140, 100], [0, 100]], dtype=np.float32)
        assert classify_plate_shape(corners) == PlateClass.TRUCK

    def test_truck_overlaps_two_wheeler_rear(self):
        # Real truck aspect (1.7:1) overlaps with two-wheeler rear (2:1).
        # classify_plate_shape returns TWO_WHEELER_REAR — the detector must
        # output plate_class directly to distinguish them in production.
        corners = np.array([[0, 0], [170, 0], [170, 100], [0, 100]], dtype=np.float32)
        assert classify_plate_shape(corners) == PlateClass.TWO_WHEELER_REAR


class TestPlateShapes:
    """Verify v1.1 class-aware targets match REVIEW_AND_RECOMMENDATIONS.md §4.4."""

    def test_car_shape(self):
        assert PLATE_SHAPES[PlateClass.CAR] == (200, 48)

    def test_two_wheeler_rear_shape(self):
        assert PLATE_SHAPES[PlateClass.TWO_WHEELER_REAR] == (128, 64)

    def test_two_wheeler_front_shape(self):
        assert PLATE_SHAPES[PlateClass.TWO_WHEELER_FRONT] == (192, 30)

    def test_truck_shape(self):
        assert PLATE_SHAPES[PlateClass.TRUCK] == (192, 112)

    def test_unknown_falls_back_to_v1_0(self):
        assert PLATE_SHAPES[PlateClass.UNKNOWN] == (192, 64)
