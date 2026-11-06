"""FastAPI service — /v1/recognize, /v1/recognize/video, /v1/jobs/{id}.

Versioned under /v1 from day one. Pydantic v2 strict schemas. ONNX Runtime
backend. Simple token-bucket rate limiter for burst protection.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------- schemas ----------

class CornerOut(BaseModel):
    x: float
    y: float
    visibility: float = 1.0


class TimingOut(BaseModel):
    detect_ms: float = 0.0
    rectify_ms: float = 0.0
    ocr_ms: float = 0.0
    enhance_ms: float = 0.0
    total_ms: float = 0.0


class PlateOut(BaseModel):
    plate: str
    confidence: float
    bbox: list[float]
    corners: list[CornerOut]
    grammar_valid: bool
    grammar_kind: Optional[str] = None
    corrected: bool = False
    enhanced: bool = False
    recognizer: str
    plate_class: str
    plate_color: str
    timings: TimingOut
    error: Optional[str] = None


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    device: str
    models_loaded: bool


class JobOut(BaseModel):
    job_id: str
    status: str   # queued | running | done | failed
    plates: list[PlateOut] = Field(default_factory=list)
    error: Optional[str] = None


# ---------- rate limiter ----------

class TokenBucket:
    """Simple token bucket — burst protection."""

    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# ---------- app factory ----------

def create_app(config_path: str = "configs/serve.yaml"):
    """Build the FastAPI app. Config path is resolved at startup."""
    import yaml
    from fastapi import FastAPI, File, HTTPException, UploadFile, Request
    from fastapi.responses import JSONResponse
    import cv2

    with open(config_path) as f:
        cfg_full = yaml.safe_load(f)
    serve_cfg = cfg_full.get("serve", cfg_full)
    # Load the pipeline config (sibling yaml).
    pipeline_cfg_path = serve_cfg.get("pipeline_config", "configs/serve.yaml")
    with open(pipeline_cfg_path) as f:
        pipeline_cfg = yaml.safe_load(f)

    from .pipeline import Pipeline
    pipeline = Pipeline.from_config(pipeline_cfg)

    bucket = TokenBucket(rate=serve_cfg.get("rate_limit_rps", 10),
                          capacity=serve_cfg.get("rate_limit_burst", 20))
    jobs: dict[str, dict] = {}

    app = FastAPI(title="ANPR-India", version="1.1.0")
    MAX_IMAGE_BYTES = serve_cfg.get("max_image_mb", 10) * 1024 * 1024
    MAX_VIDEO_BYTES = serve_cfg.get("max_video_mb", 200) * 1024 * 1024

    @app.get("/health", response_model=HealthOut)
    def health():
        return HealthOut(
            version="1.1.0",
            device=serve_cfg.get("device", "cpu"),
            models_loaded=True,
        )

    @app.post("/v1/recognize", response_model=list[PlateOut])
    async def recognize(request: Request, file: UploadFile = File(...)):
        if not bucket.acquire():
            raise HTTPException(429, "rate limit exceeded")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"image too large (>{MAX_IMAGE_BYTES//1024//1024}MB)")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        results = pipeline.process_image(img)
        return [
            PlateOut(
                plate=r.plate, confidence=r.confidence,
                bbox=list(r.bbox),
                corners=[CornerOut(x=c.x, y=c.y, visibility=c.visibility)
                          for c in r.corners],
                grammar_valid=r.grammar_valid, grammar_kind=r.grammar_kind,
                corrected=r.corrected, enhanced=r.enhanced,
                recognizer=r.recognizer,
                plate_class=r.plate_class.value,
                plate_color=r.plate_color.value,
                timings=TimingOut(
                    detect_ms=r.timings.detect_ms,
                    rectify_ms=r.timings.rectify_ms,
                    ocr_ms=r.timings.ocr_ms,
                    enhance_ms=r.timings.enhance_ms,
                    total_ms=r.timings.total_ms,
                ),
                error=r.error,
            ) for r in results
        ]

    @app.post("/v1/recognize/video", response_model=JobOut)
    async def recognize_video(request: Request, file: UploadFile = File(...)):
        if not bucket.acquire():
            raise HTTPException(429, "rate limit exceeded")
        data = await file.read()
        if len(data) > MAX_VIDEO_BYTES:
            raise HTTPException(413, f"video too large (>{MAX_VIDEO_BYTES//1024//1024}MB)")
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued", "plates": [], "error": None}
        # In production: push to a task queue (Celery / RQ). For demo: run inline.
        import tempfile, asyncio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(data); tmp.close()
        async def run_job():
            jobs[job_id]["status"] = "running"
            try:
                for frame_result in pipeline.process_video(tmp.name):
                    jobs[job_id]["plates"].extend(frame_result.get("final_reads", []))
                jobs[job_id]["status"] = "done"
            except Exception as e:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
        asyncio.create_task(run_job())
        return JobOut(job_id=job_id, status="queued")

    @app.get("/v1/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "job not found")
        j = jobs[job_id]
        return JobOut(job_id=job_id, status=j["status"],
                       plates=j["plates"], error=j["error"])

    return app


# For `uvicorn anpr.api:app` — load default config.
def _default_app():
    cfg = os.environ.get("ANPR_CONFIG", "configs/serve.yaml")
    return create_app(cfg)


# Lazy global so uvicorn can import it.
_app = None
def get_app():
    global _app
    if _app is None:
        _app = _default_app()
    return _app
