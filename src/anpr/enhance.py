"""Gated enhancement — Real-ESRGAN compact, only on low-confidence reads.

v1.1 spec §2.4: Real-ESRGAN `realesr-general-x4v3` (compact), tile=256, fp16.
Gating: only invoke if fast-path mean_conf < τ (default 0.75).

Also adds the ablation alternatives: Plate-SR (Nascimento 2023), TTPSR (Chen 2025),
classical CLAHE+unsharp. Controlled by config.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Enhancer:
    """SR wrapper — Real-ESRGAN compact by default."""

    def __init__(self, weights: str | Path | None = None,
                 backend: str = "realesrgan",
                 tile: int = 256, fp16: bool = True, device: str = "cuda",
                 scale: int = 4):
        self.weights = str(weights) if weights else None
        self.backend = backend
        self.tile = tile
        self.fp16 = fp16
        self.device = device
        self.scale = scale
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return
        if self.backend == "realesrgan":
            self._impl = _RealESRGANImpl(self.weights, self.tile, self.fp16,
                                          self.device, self.scale)
        elif self.backend == "clahe":
            self._impl = _CLAHEImpl()
        elif self.backend == "plate_sr":
            self._impl = _PlateSRImpl(self.weights, self.device, self.scale)
        elif self.backend == "ttpsr":
            self._impl = _TTPSRImpl(self.weights, self.device, self.scale)
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def enhance(self, plate_img: np.ndarray) -> np.ndarray:
        """Apply SR to a plate crop. Returns upscaled image."""
        t0 = time.perf_counter()
        if plate_img.size == 0:
            return plate_img
        self._load()
        out = self._impl.run(plate_img)   # type: ignore
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("enhance[%s]: %.2fms out_shape=%s",
                     self.backend, elapsed, out.shape)
        return out

    def __call__(self, plate_img: np.ndarray) -> np.ndarray:
        return self.enhance(plate_img)


class _RealESRGANImpl:
    """Real-ESRGAN realesr-general-x4v3 (compact, ~16 MB)."""

    def __init__(self, weights: str | None, tile: int, fp16: bool,
                 device: str, scale: int):
        self.weights = weights
        self.tile = tile
        self.fp16 = fp16
        self.device = device
        self.scale = scale
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        if not self.weights:
            raise RuntimeError("realesrgan backend requires `weights` path")
        import onnxruntime as ort
        providers = (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(self.weights, sess_opts=sess_opts,
                                              providers=providers)

    def run(self, img: np.ndarray) -> np.ndarray:
        self._load()
        # Real-ESRGAN ONNX takes (1, H, W, 3) float32 [0,1] RGB.
        import cv2
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)
        # Tile to limit memory — for plate crops (~200x112) this is usually
        # a single tile; the logic handles larger images too.
        h, w = arr.shape[1], arr.shape[2]
        if max(h, w) <= self.tile:
            out = self._session.run(None, {self._session.get_inputs()[0].name: arr})[0]
        else:
            # Tiled inference for larger inputs (rare for plate crops).
            out = _tiled_inference(self._session, arr, self.tile)
        out = np.clip(out[0] * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class _CLAHEImpl:
    """Classical CLAHE + unsharp mask — free ablation baseline (sub-ms).

    No weights needed. Surprisingly competitive on moderately-contrasty plates.
    """

    def __init__(self):
        pass

    def run(self, img: np.ndarray) -> np.ndarray:
        import cv2
        # Upscale 2x with Lanczos first, then CLAHE on L channel of LAB,
        # then unsharp mask. We deliberately do NOT upscale 4x — the point
        # of this baseline is to bound the SR contribution from above.
        h, w = img.shape[:2]
        up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Unsharp mask
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)


class _PlateSRImpl:
    """Plate-SR (Nascimento et al. 2023) — license-plate-specific SR.

    Placeholder for the ablation matrix; load the official ONNX from the
    paper's GitHub when running A8.
    """

    def __init__(self, weights: str | None, device: str, scale: int):
        self.weights = weights
        self.device = device
        self.scale = scale
        self._session = None

    def _load(self):
        if not self.weights:
            raise RuntimeError("plate_sr backend requires `weights`")
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self.device.startswith("cuda") else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(self.weights, providers=providers)

    def run(self, img: np.ndarray) -> np.ndarray:
        self._load()
        import cv2
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        out = self._session.run(None, {self._session.get_inputs()[0].name: arr})[0]
        out = np.clip(np.transpose(out[0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class _TTPSRImpl:
    """TTPSR (Chen et al. 2025) — text-prior SR. Ablation A8 row."""

    def __init__(self, weights: str | None, device: str, scale: int):
        self.weights = weights
        self.device = device
        self.scale = scale
        self._session = None

    def _load(self):
        if not self.weights:
            raise RuntimeError("ttpsr backend requires `weights`")
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self.device.startswith("cuda") else ["CPUExecutionProvider"])
        self._session = ort.InferenceSession(self.weights, providers=providers)

    def run(self, img: np.ndarray) -> np.ndarray:
        self._load()
        import cv2
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)), 0)
        out = self._session.run(None, {self._session.get_inputs()[0].name: arr})[0]
        out = np.clip(np.transpose(out[0], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _tiled_inference(session, arr: np.ndarray, tile: int) -> np.ndarray:
    """Tiled inference for Real-ESRGAN (handle large images)."""
    import cv2
    h, w = arr.shape[1], arr.shape[2]
    scale = 4
    out_h, out_w = h * scale, w * scale
    out = np.zeros((1, out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((1, out_h, out_w, 1), dtype=np.float32)
    # Simple non-overlapping tiles with feathered edges.
    step = tile
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, y2 = min(y, h), min(y + tile, h)
            x1, x2 = min(x, w), min(x + tile, w)
            tile_arr = arr[:, y1:y2, x1:x2, :]
            tile_out = session.run(None,
                {session.get_inputs()[0].name: tile_arr})[0]
            oy1, oy2 = y1 * scale, y2 * scale
            ox1, ox2 = x1 * scale, x2 * scale
            out[:, oy1:oy2, ox1:ox2, :] = tile_out
            weight[:, oy1:oy2, ox1:ox2, :] = 1.0
    return out / np.maximum(weight, 1e-8)
