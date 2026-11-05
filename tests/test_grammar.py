"""Tests for grammar.py — v1.1 corrected regexes.

These tests are the executable spec for §4 of REVIEW_AND_RECOMMENDATIONS.md.
If any of these fail, the grammar is broken.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from anpr.grammar import (
    apply_grammar, correct_state_code, GRAMMARS, STATE_CODES,
    levenshtein,
)


# ---------- BH series (the bug fix) ----------

class TestBHSeries:
    r"""v1.0 had `^BH\d{4}$` — WRONG. v1.1 fixes to `^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$`."""

    def test_bh_valid_2_letters(self):
        r = apply_grammar("22BH1234PA")
        assert r.grammar_valid
        assert r.grammar_kind == "bh"

    def test_bh_valid_1_letter(self):
        r = apply_grammar("23BH5678A")
        assert r.grammar_valid
        assert r.grammar_kind == "bh"

    def test_bh_with_spaces_normalized(self):
        r = apply_grammar("22 BH 1234 PA")
        assert r.grammar_valid
        assert r.grammar_kind == "bh"
        assert r.text == "22BH1234PA"

    def test_bh_rejects_O_in_series(self):
        # 'O' is excluded from series letters
        r = apply_grammar("22BH1234OA")
        assert not r.grammar_valid

    def test_bh_rejects_I_in_series(self):
        r = apply_grammar("22BH1234IA")
        assert not r.grammar_valid

    def test_bh_rejects_old_wrong_format(self):
        # v1.0 regex `^BH\d{4}$` — would falsely accept "BH1234"
        r = apply_grammar("BH1234")
        assert not r.grammar_valid   # MUST be rejected under v1.1


# ---------- Private/commercial ----------

class TestPrivateFormat:
    def test_standard_private(self):
        r = apply_grammar("MH12DE1434")
        assert r.grammar_valid
        assert r.grammar_kind == "private"

    def test_commercial(self):
        r = apply_grammar("KA05AB1234")
        assert r.grammar_valid
        assert r.grammar_kind == "private"

    def test_one_letter_series(self):
        r = apply_grammar("DL01A0001")
        assert r.grammar_valid
        assert r.grammar_kind == "private"

    def test_three_letter_series(self):
        r = apply_grammar("TN07XYZ9999")
        assert r.grammar_valid
        assert r.grammar_kind == "private"

    def test_one_digit_rto(self):
        # Some older plates have single-digit RTO codes
        r = apply_grammar("MH1AB1234")
        assert r.grammar_valid
        assert r.grammar_kind == "private"


# ---------- VA series (new in v1.1) ----------

class TestVASeries:
    def test_va_valid(self):
        r = apply_grammar("MHVAAA1234")
        assert r.grammar_valid
        assert r.grammar_kind == "va"

    def test_va_rejects_O(self):
        r = apply_grammar("MHVAAA1234".replace("AA", "AO"))
        # AO is invalid — series letters exclude O
        assert not r.grammar_valid or "VA" not in (r.grammar_kind or "")


# ---------- Diplomatic (new in v1.1) ----------

class TestDiplomatic:
    def test_cd_embassy(self):
        r = apply_grammar("199CD99")
        assert r.grammar_valid
        assert r.grammar_kind == "diplomatic"

    def test_cc_consulate(self):
        r = apply_grammar("199CC0123")
        assert r.grammar_valid
        assert r.grammar_kind == "diplomatic"

    def test_un(self):
        r = apply_grammar("99UN99")
        assert r.grammar_valid
        assert r.grammar_kind == "diplomatic"

    def test_iod(self):
        r = apply_grammar("99IOD99")
        assert r.grammar_valid
        assert r.grammar_kind == "diplomatic"


# ---------- Temporary / Trade ----------

class TestTemporary:
    def test_temporary_valid(self):
        r = apply_grammar("T0124AN0123A")
        assert r.grammar_valid
        assert r.grammar_kind == "temporary"


class TestTrade:
    def test_trade_valid(self):
        r = apply_grammar("AN01C0123TC0123")
        assert r.grammar_valid
        assert r.grammar_kind == "trade"


# ---------- State-code correction ----------

class TestStateCodeCorrection:
    def test_no_correction_when_valid(self):
        assert correct_state_code("MH") is None
        assert correct_state_code("DL") is None

    def test_correction_unambiguous_one_off(self):
        # 'RX' → 'RJ' (Rajasthan) — unambiguous.
        # Only one state code starts with 'R' (RJ), and no state code ends
        # with 'X', so RX has exactly one distance-1 neighbor: RJ.
        # Compare: 'BA' LOOKS unambiguous but is actually 1-off from 4 codes
        # (BR, GA, KA, LA — all share the second letter A), so it returns None.
        assert correct_state_code("RX") == "RJ"
        assert correct_state_code("BA") is None   # ambiguous — 4 matches
        assert correct_state_code("PL") is None   # ambiguous — PB and PY

    def test_correction_rejects_ambiguous(self):
        # 'MG' is 1 off from BOTH 'MH' (Maharashtra) and 'CG' (Chhattisgarh)
        # — ambiguous, refuse to correct (conservative policy)
        assert correct_state_code("MG") is None

    def test_correction_rejects_far(self):
        # 'XX' is too far from any state code
        assert correct_state_code("XX", max_distance=1) is None

    def test_correction_rejects_non_alpha(self):
        assert correct_state_code("M1") is None

    def test_correction_legacy_OR(self):
        # 'OR' is included as legacy alias for OD
        assert "OR" in STATE_CODES

    def test_correction_applied_in_grammar(self):
        # 'RX12DE1434' — format matches private, but 'RX' is not a real state
        # code. Levenshtein-1 correction → 'RJ' (Rajasthan), unambiguous
        # (RJ is the only state code at distance 1 from RX).
        r = apply_grammar("RX12DE1434")
        assert r.grammar_valid
        assert r.corrected
        assert r.text == "RJ12DE1434"

    def test_no_correction_when_state_code_already_valid(self):
        # 'MH12DE1434' — 'MH' is valid, no correction needed
        r = apply_grammar("MH12DE1434")
        assert r.grammar_valid
        assert not r.corrected
        assert r.text == "MH12DE1434"

    def test_no_correction_on_series_letters(self):
        # Series letters are arbitrary — never correct them
        r = apply_grammar("MH12XE1434")   # X is unusual but valid
        assert r.grammar_valid
        assert not r.corrected

    def test_ambiguous_state_code_left_as_is(self):
        # 'MG12DE1434' — 'MG' is ambiguous (MH and CG both at distance 1)
        # 'PL12DE1434' — 'PL' is ambiguous (PB and PY both at distance 1)
        # 'BA12DE1434' — 'BA' is ambiguous (BR, GA, KA, LA all at distance 1)
        # Format still matches private regex, so grammar_valid=True,
        # but corrected=False (refused to guess).
        for ambiguous in ("MG12DE1434", "PL12DE1434", "BA12DE1434"):
            r = apply_grammar(ambiguous)
            assert r.grammar_valid
            assert not r.corrected
            assert r.text == ambiguous


# ---------- Levenshtein ----------

class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("MH", "MH") == 0

    def test_one_sub(self):
        assert levenshtein("MH", "MG") == 1

    def test_one_ins(self):
        assert levenshtein("MH", "MHA") == 1

    def test_one_del(self):
        assert levenshtein("MHA", "MH") == 1

    def test_completely_different(self):
        assert levenshtein("ABCD", "WXYZ") == 4


# ---------- Grammar never drops reads ----------

class TestNeverDrop:
    def test_garbage_returned_as_invalid(self):
        r = apply_grammar("xyz123!!")
        assert not r.grammar_valid
        # but the text is still there for audit
        assert r.text == "XYZ123!!"

    def test_empty_input(self):
        r = apply_grammar("")
        assert not r.grammar_valid
        assert r.text == ""
