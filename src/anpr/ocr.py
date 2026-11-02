"""Recognizer — LPRNet (fast path) + PARSeq (heavy path) hybrid.

Contract: plate_img -> ReadResult.

v1.1 architecture (REVIEW_AND_RECOMMENDATIONS.md §2.3):
  - Fast path: LPRNet-style CTC, ~2 MB, ~6 ms, runs always
  - Heavy path: PARSeq, ~24 MB, ~15 ms, runs only when fast-path mean_conf < τ
    AND gated enhancement has been applied

PaddleOCR PP-OCRv4 mobile is the v1 baseline; available here as PaddleRecognizer.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .grammar import apply_grammar
from .types import ReadResult

logger = logging.getLogger(__name__)

# Charset: 0-9 + A-Z minus {I, O, Q} + <pad>. CTC blank is index 0.
CHARSET = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"
VOCAB = [BLANK_TOKEN] + list(CHARSET) + [PAD_TOKEN]
VOCAB_SIZE = len(VOCAB)   # 1 + 33 + 1 = 35


class LPRNetRecognizer:
    """LPRNet-style CTC recognizer (our v2 model, ~2 MB ONNX).

    Architecture follows the LPRNet paper (Zherzong et al.) adapted for Indian
    plates. See DATASETS_AND_ARCHITECTURE.md §B.2 for the spec.

    ONNX input: (1, 3, H, W) float32, normalized to [0, 1].
    ONNX output: (T, 1, V) softmax log-probs, T = sequence length ~48.
    """

    def __init__(self, weights: str | Path, device: str = "cuda"):
        self.weights = str(weights)
        self.device = device
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        providers = (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.weights, providers=providers)
        logger.info("LPRNet ONNX loaded: %s", self.weights)

    def read(self, plate_img: np.ndarray) -> ReadResult:
        """Recognize one plate strip. Returns ReadResult with char_confs."""
        t0 = time.perf_counter()
        self._load()
        # Preprocess: resize to model input, normalize.
        # Model expects (1, 3, 64, 192) — strip class-aware shapes are padded
        # to (200, 112) by rectify.py; we resize preserving aspect.
        if plate_img.size == 0:
            return ReadResult(text="", char_confs=[], mean_conf=0.0,
                              grammar_valid=False, recognizer="lprnet")
        img = _preprocess_for_lprnet(plate_img)
        out = self._session.run(None, {self._session.get_inputs()[0].name: img})[0]
        # out shape: (T, 1, V). Greedy decode.
        text, char_confs = _ctc_greedy_decode(out[0])   # type: ignore
        mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
        gram = apply_grammar(text)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("lprnet: %.2fms text=%s conf=%.3f grammar=%s",
                     elapsed, gram.text, mean_conf, gram.grammar_kind)
        return ReadResult(
            text=gram.text,
            char_confs=char_confs,
            mean_conf=mean_conf,
            grammar_valid=gram.grammar_valid,
            grammar_kind=gram.grammar_kind,
            corrected=gram.corrected,
            recognizer="lprnet",
        )

    def __call__(self, plate_img: np.ndarray) -> ReadResult:
        return self.read(plate_img)


class PARSeqRecognizer:
    """PARSeq recognizer (SOTA on STR benchmarks, ~24 MB).

    Used only on the heavy path (post-ESRGAN). MIT licensed.

    ONNX input: (1, 3, 32, W) float32 normalized per ImageNet stats.
    ONNX output: greedy token ids via argmax over the parallel-decoder logits.
    """

    def __init__(self, weights: str | Path, device: str = "cuda",
                 charset: str = CHARSET):
        self.weights = str(weights)
        self.device = device
        self.charset = charset
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        providers = (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.weights, providers=providers)
        logger.info("PARSeq ONNX loaded: %s", self.weights)

    def read(self, plate_img: np.ndarray) -> ReadResult:
        t0 = time.perf_counter()
        self._load()
        if plate_img.size == 0:
            return ReadResult(text="", char_confs=[], mean_conf=0.0,
                              grammar_valid=False, recognizer="parseq")
        img = _preprocess_for_parseq(plate_img)
        # PARSeq ONNX export: input (1, 3, 32, W), output (1, T, V) logits.
        out = self._session.run(None, {self._session.get_inputs()[0].name: img})[0]
        # Parallel mode: greedy argmax over the per-position logits.
        token_ids = out[0].argmax(axis=-1)   # type: ignore
        # PARSeq tokenizer: ids 0=sos, 1=eos, 2..V=charset. Strip sos/eos.
        text_chars = []
        char_confs = []
        probs = _softmax(out[0], axis=-1)   # type: ignore
        for tid in token_ids:
            tid = int(tid)
            if tid == 1:   # eos
                break
            if tid < 2:
                continue
            idx = tid - 2
            if idx < len(self.charset):
                text_chars.append(self.charset[idx])
                char_confs.append(float(probs[tid]))
        text = "".join(text_chars)
        mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
        gram = apply_grammar(text)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("parseq: %.2fms text=%s conf=%.3f grammar=%s",
                     elapsed, gram.text, mean_conf, gram.grammar_kind)
        return ReadResult(
            text=gram.text,
            char_confs=char_confs,
            mean_conf=mean_conf,
            grammar_valid=gram.grammar_valid,
            grammar_kind=gram.grammar_kind,
            corrected=gram.corrected,
            recognizer="parseq",
        )


class PaddleRecognizer:
    """PaddleOCR PP-OCRv4 mobile baseline — used in Wk1-2 only.

    Replaced by LPRNetRecognizer in Wk3. Kept for the ablation matrix.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def _load(self):
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_angle_cls=False,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )

    def read(self, plate_img: np.ndarray) -> ReadResult:
        t0 = time.perf_counter()
        self._load()
        if plate_img.size == 0:
            return ReadResult(text="", char_confs=[], mean_conf=0.0,
                              grammar_valid=False, recognizer="paddle")
        # PaddleOCR expects RGB.
        rgb = plate_img[..., ::-1] if plate_img.ndim == 3 else plate_img
        result = self._ocr.ocr(rgb, cls=False)
        text = ""
        char_confs: list[float] = []
        if result and result[0]:
            for line in result[0]:
                box, (txt, conf) = line
                text += txt
                char_confs.append(float(conf))
        mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
        gram = apply_grammar(text)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("paddle: %.2fms text=%s conf=%.3f", elapsed, gram.text, mean_conf)
        return ReadResult(
            text=gram.text,
            char_confs=char_confs,
            mean_conf=mean_conf,
            grammar_valid=gram.grammar_valid,
            grammar_kind=gram.grammar_kind,
            corrected=gram.corrected,
            recognizer="paddle",
        )


# ---------- helpers ----------

def _preprocess_for_lprnet(img: np.ndarray, target_h: int = 64,
                            target_w: int = 192) -> np.ndarray:
    """BGR uint8 (H, W, 3) → (1, 3, 64, 192) float32 normalized."""
    import cv2
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0   # [0, 1]
    chw = np.transpose(normalized, (2, 0, 1))      # (3, H, W)
    return np.expand_dims(chw, 0)                  # (1, 3, H, W)


def _preprocess_for_parseq(img: np.ndarray, target_h: int = 32,
                            target_w: int = 128) -> np.ndarray:
    """BGR uint8 → (1, 3, 32, 128) ImageNet-normalized."""
    import cv2
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    # ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, 0)


def _ctc_greedy_decode(log_probs: np.ndarray) -> tuple[str, list[float]]:
    """CTC greedy decode: collapse repeated tokens, drop blanks."""
    # log_probs: (T, V) where index 0 is CTC blank.
    ids = log_probs.argmax(axis=-1)
    probs = _softmax(log_probs, axis=-1)
    text_chars: list[str] = []
    char_confs: list[float] = []
    prev = -1
    for t in range(len(ids)):
        idx = int(ids[t])
        if idx == 0 or idx == prev:   # blank or repeat
            prev = idx
            continue
        if idx - 1 < len(CHARSET):
            text_chars.append(CHARSET[idx - 1])
            char_confs.append(float(probs[t, idx]))
        prev = idx
    return "".join(text_chars), char_confs


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def pick_better(a: ReadResult, b: ReadResult) -> ReadResult:
    """Total order on ReadResult — used by the gated-enhancement branch.

    v1.0 spec §3.4: key = (grammar_valid, round(mean_conf, 3)).
    No ties possible (mean_conf is continuous).
    """
    return max(a, b, key=lambda r: r.sort_key())
