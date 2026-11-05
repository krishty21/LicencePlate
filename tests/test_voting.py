"""Tests for track.py — temporal voting math."""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from anpr.track import TrackVoter, GAMMA


class TestTrackVoter:
    def test_no_votes_returns_none(self):
        v = TrackVoter()
        assert v.get_plate(1) is None

    def test_too_few_votes_returns_none(self):
        v = TrackVoter(min_votes=3)
        v.add_vote(1, "MH12DE1434", 0.9, 0)
        v.add_vote(1, "MH12DE1434", 0.85, 1)
        v.step(2)
        assert v.get_plate(1) is None

    def test_majority_wins(self):
        v = TrackVoter(min_votes=3, gamma=1.0)
        v.add_vote(1, "MH12DE1434", 0.9, 0)
        v.add_vote(1, "MH12DE1434", 0.85, 1)
        v.add_vote(1, "MH12DE9999", 0.5, 2)
        v.step(2)
        r = v.get_plate(1)
        assert r is not None
        assert r.plate == "MH12DE1434"
        assert r.votes == 3

    def test_confidence_weighted(self):
        # One high-conf vote should outweigh two low-conf votes
        v = TrackVoter(min_votes=3, gamma=1.0)
        v.add_vote(1, "MH12AB1234", 0.95, 0)   # weight 0.95
        v.add_vote(1, "MH12CD5678", 0.4, 1)    # weight 0.4
        v.add_vote(1, "MH12CD5678", 0.4, 2)    # weight 0.4 — total 0.8
        v.step(2)
        r = v.get_plate(1)
        assert r is not None
        assert r.plate == "MH12AB1234"

    def test_recency_decay(self):
        # gamma=0.98: older votes decay
        v = TrackVoter(min_votes=2, gamma=0.5)   # extreme decay for test
        v.add_vote(1, "OLDPLATE", 0.9, 0)        # weight 0.9 * 0.5^(current-0)
        v.add_vote(1, "NEWPLATE", 0.5, 10)       # weight 0.5 * 0.5^(0) = 0.5
        v.step(10)
        # OLD weight = 0.9 * 0.5^10 ≈ 0.0009
        # NEW weight = 0.5 * 0.5^0 = 0.5
        # NEW wins despite lower per-vote conf
        r = v.get_plate(1)
        assert r is not None
        assert r.plate == "NEWPLATE"

    def test_empty_text_skipped(self):
        v = TrackVoter(min_votes=1)
        v.add_vote(1, "", 0.9, 0)   # empty — should not register
        v.step(0)
        assert v.get_plate(1) is None

    def test_max_deque_drops_old(self):
        v = TrackVoter(min_votes=1, max_deque=3)
        for i in range(5):
            v.add_vote(1, f"PLATE{i}", 0.9, i)
        v.step(5)
        # Only last 3 votes kept
        assert len(v._votes[1]) == 3

    def test_reset(self):
        v = TrackVoter()
        v.add_vote(1, "MH12DE1434", 0.9, 0)
        v.reset()
        assert v.get_plate(1) is None
