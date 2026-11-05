"""Tests for LPRNet architecture — verify forward pass, decode, params.

These tests run on CPU only (no torch GPU needed). They verify:
  - Model constructs
  - Forward pass produces correct output shape
  - Greedy CTC decode collapses repeats + drops blanks
  - Param count is in the expected range (~1-3M)
  - Label encoding round-trips

Skip if torch is not installed (e.g. in CI without torch).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Skip the entire module if torch isn't available.
torch = pytest.importorskip("torch")

from anpr.models.lprnet import (
    LPRNet, build_lprnet_for_training, encode_label,
    INPUT_H, INPUT_W, VOCAB_SIZE, BLANK_IDX, CHARSET, IDX_TO_CHAR,
)


class TestLPRNetConstruction:
    def test_default_construction(self):
        m = LPRNet()
        assert m.vocab_size == VOCAB_SIZE
        assert m.input_h == INPUT_H
        assert m.input_w == INPUT_W

    def test_param_count_in_expected_range(self):
        m = LPRNet()
        n_params = sum(p.numel() for p in m.parameters())
        # Spec says ~1-3M params (target ~2 MB ONNX).
        # Be permissive — the exact count depends on conv choices.
        assert 500_000 < n_params < 5_000_000, f"params: {n_params:,}"

    def test_trainable_mode(self):
        m = LPRNet()
        m.train()
        assert m.training


class TestLPRNetForward:
    def test_forward_shape(self):
        m = LPRNet()
        m.eval()
        x = torch.randn(2, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            y = m(x)
        # Expected: (B, T, V) where T = W/4 = 48, V = vocab_size
        assert y.shape == (2, 48, VOCAB_SIZE), f"got {y.shape}"

    def test_forward_single_batch(self):
        m = LPRNet()
        m.eval()
        x = torch.randn(1, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 48, VOCAB_SIZE)

    def test_forward_batch_of_8(self):
        m = LPRNet()
        m.eval()
        x = torch.randn(8, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (8, 48, VOCAB_SIZE)


class TestLPRNetDecode:
    def test_decode_returns_strings(self):
        m = LPRNet()
        m.eval()
        x = torch.randn(2, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            logits = m(x)
            texts, confs = m.decode(logits)
        assert len(texts) == 2
        assert len(confs) == 2
        for t, c in zip(texts, confs):
            assert isinstance(t, str)
            assert isinstance(c, list)
            # All chars in decoded text must be in charset
            for ch in t:
                assert ch in CHARSET

    def test_decode_collapses_repeats(self):
        """CTC greedy decode must collapse repeated tokens."""
        m = LPRNet()
        # Construct logits where the same token is repeated 5 times.
        # Token index 1 = '0' in our charset.
        T, V = 48, VOCAB_SIZE
        logits = torch.full((1, T, V), -10.0)
        logits[0, :, 1] = 10.0   # token 1 everywhere
        texts, _ = m.decode(logits)
        # 5+ repeats of token 1 must collapse to a single '0'.
        assert texts[0] == "0", f"expected '0', got {texts[0]!r}"

    def test_decode_drops_blanks(self):
        """CTC blank (index 0) must never appear in decoded text."""
        m = LPRNet()
        T, V = 48, VOCAB_SIZE
        logits = torch.full((1, T, V), -10.0)
        # Alternate between blank and token 1 ('0')
        for t in range(T):
            logits[0, t, 0 if t % 2 == 0 else 1] = 10.0
        texts, _ = m.decode(logits)
        # Result should be a string of '0's — blanks dropped, repeats collapsed.
        assert texts[0] == "0" * (T // 2)


class TestEncodeLabel:
    def test_round_trip(self):
        text = "MH12DE1434"
        encoded = encode_label(text)
        assert encoded.dtype == torch.long
        decoded = "".join(IDX_TO_CHAR[int(i)] for i in encoded)
        assert decoded == text

    def test_charset_excludes_IOQ(self):
        # I, O, Q are not in the charset
        assert "I" not in CHARSET
        assert "O" not in CHARSET
        assert "Q" not in CHARSET

    def test_uppercase_normalization(self):
        # encode_label uppercases input
        e_lower = encode_label("mh12de1434")
        e_upper = encode_label("MH12DE1434")
        assert torch.equal(e_lower, e_upper)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            encode_label("")

    def test_rejects_invalid_chars(self):
        # 'I' is not in charset; encode_label silently drops it,
        # so the result will be shorter. If all chars are invalid, raises.
        with pytest.raises(ValueError):
            encode_label("III")


class TestBuildForTraining:
    def test_build_returns_model_and_loss(self):
        device = "cpu"
        model, ctc_loss = build_lprnet_for_training(device=device)
        assert isinstance(model, LPRNet)
        assert hasattr(ctc_loss, "forward")
        # Verify CTCLoss blank index
        assert ctc_loss.blank == BLANK_IDX
