# ANPR-India — Implementation Specification

**Project:** End-to-end Automatic Number Plate Recognition for Indian plates (images & video)
**Status:** v1.0 — implementation contract. Every module below has a defined I/O contract,
config surface, and failure mode. Code that violates this spec is a bug.

---

## 1. System Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │                  INPUT SOURCE                   │
                        │        image file │ video stream │ API upload   │
                        └───────────────────────┬─────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 1      │  DETECT + LOCALIZE (unified)                        │
                     │  YOLO11s-pose → plate bbox + 4 corner keypoints      │
                     │  (TL, TR, BR, BL)                                    │
                     └──────────────────────────┬──────────────────────────┘
                                                │ corners
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 2      │  RECTIFY                                             │
                     │  homography from 4 corners → 192×64 RGB plate strip  │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 3      │  READ v1                                             │
                     │  CTC recognizer (LPRNet-style) → text + confidence   │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                   conf ≥ τ ? ──yes──▶ accept
                                                │no
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 4      │  GATED ENHANCEMENT (only hard cases)                 │
                     │  Real-ESRGAN compact (x4, tile) → re-rectify → READ  │
                     │  keep better read by (grammar_valid, confidence)     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 5      │  POST-PROCESS                                        │
                     │  grammar constraints, state-code correction          │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        VIDEO ONLY   │  TRACK + TEMPORAL VOTE                                │
                     │  ByteTrack (supervision) → per-track plate voting    │
                     │  (confidence-weighted, recency-decayed)              │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
                     │  OUTPUT: JSON (plate, conf, bbox, track, timing)     │
                     └─────────────────────────────────────────────────────┘
```

**Design principle:** enhancement is the *exception path*, not the default path.
p95 latency is protected by paying the enhancement cost only on low-confidence reads.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | typing, ecosystem |
| Detection/keypoints | Ultralytics YOLO11s-pose | one model = bbox + 4 corners; small, fast, ONNX-exportable |
| Rectification | OpenCV `findHomography`/`getPerspectiveTransform` | deterministic, sub-ms |
| Recognizer v1 (baseline) | PaddleOCR PP-OCRv4 mobile | zero-training baseline, strong general OCR |
| Recognizer v2 (ours) | LPRNet-style CNN + CTC, trained on synthetic+real Indian crops | full control of charset, grammar-friendly, ~2 MB |
| Enhancement | Real-ESRGAN `realesr-general-x4v3` (compact) | pretrained, runs on T4, tileable |
| Tracking | `supervision` (ByteTrack) | maintained, ONNX-friendly |
| Serving | FastAPI + Uvicorn + ONNX Runtime | async, typed, standard |
| Config | YAML + pydantic-settings | typed config, no magic constants |
| Experiment tracking | Weights & Biases (free tier) | runs from Kaggle notebooks |
| Packaging | Docker, GitHub Actions CI, ruff + mypy + pytest | senior-SWE table stakes |

---

## 3. Module Specifications

### 3.1 `anpr.detect` — Detector
- **Contract:** `np.ndarray (BGR) -> list[PlateDetection(bbox, corners[4,2], det_conf)]`
- **Model:** YOLO11s-pose, 1 class `plate`, `kpt_shape: [4, 3]` (x, y, visibility).
  Keypoint order is fixed: `[TL, TR, BR, BL]` clockwise from top-left.
- **Runtime:** ONNX Runtime, batch=1 interactive / batch=N offline.
- **Config keys:** `weights, conf_thres=0.25, iou_thres=0.45, imgsz=640, max_det=10`
- **Failure modes:** no plate → empty list (caller must handle); degenerate corners
  (self-intersecting quad) → fall back to axis-aligned bbox crop.

### 3.2 `anpr.rectify` — Perspective Correction
- **Contract:** `(image, corners[4,2]) -> np.ndarray (64,192,3)`
- **Algorithm:** order corners → `getPerspectiveTransform(corners → [(0,0),(W,0),(W,H),(0,H)])` → warp.
- **Sanity gate:** reject quads with area < 100 px² or aspect ratio > 8 (false positives).
- **Failure mode:** homography ill-conditioned → log warning, return bbox crop un-rectified.

### 3.3 `anpr.ocr` — Recognizer + Grammar
- **Contract:** `(plate_img) -> ReadResult(text, char_confs, mean_conf, grammar_valid)`
- **Charset (v1):** digits `0-9` + letters `A-Z` minus `{I, O, Q}` + `<pad>` + CTC `<blank>`.
  Rationale: Indian series letters avoid I/O/Q (1/0 confusion); excluded chars are
  mapped at post-process (O→0, I→1, Q→0) when context demands.
- **Grammar:** `^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$` (private vehicles), `^BH\d{4}$` (Bharat series),
  plus optional chars-with-dots variants. Grammar match is a *feature*, not a filter —
  never hard-drop invalid reads; flag them.
- **State-code correction:** Levenshtein distance ≤ 1 against the 40+ official RTO state
  codes → correct + record `corrected: true`.
- **v1 vs v2:** PaddleOCR is the week-1/2 baseline; the custom CTC net replaces it in week 3.

### 3.4 `anpr.enhance` — Gated Enhancement
- **Contract:** `(plate_img, gate) -> np.ndarray` (same resolution class, x4 upscale)
- **Model:** Real-ESRGAN `realesr-general-x4v3`, tile=256, fp16.
- **Gating policy (core algorithm):**

```python
read = ocr(plate_img)
if read.mean_conf >= cfg.tau:          # tau = 0.75, tuned on val set
    return read, enhanced=False        # fast path ~0 extra ms
enhanced_img = esrgan(plate_img)
read2 = ocr(enhanced_img)
return pick_better(read, read2), enhanced=True

def pick_better(a, b):                 # total order, no ties possible
    key = lambda r: (r.grammar_valid, round(r.mean_conf, 3))
    return max(a, b, key=key)
```

- **Why not always-enhance:** +150–300 ms p95 on T4 for ~zero gain on clean plates
  (ablation in §5 will prove/disprove this with numbers).

### 3.5 `anpr.track` — Video Tracking + Temporal Voting
- **Contract:** `(frame, detections, t) -> list[TrackRead(plate, conf, track_id)]`
- **Tracker:** ByteTrack (via `supervision`), matched on bbox IoU.
- **Voting:** per track, maintain deque (maxlen=30) of `(text, conf, frame_idx)`.
  Final plate = argmax of confidence-weighted votes, weight `= conf · γ^(age)`,
  `γ=0.98`. Emit final read once (or on query) when ≥3 votes agree.
- **Failure mode:** ID switches → votes restart; acceptable for v1, log rate.

### 3.6 `anpr.pipeline` — Orchestrator
- Owns stage sequencing, per-stage timers (`time.perf_counter`), structured logs
  (JSON lines: `stage, latency_ms, conf, grammar_valid`), and result assembly.
- **Guarantees:** never raises on bad input media; returns per-image errors in payload.

---

## 4. Service API (FastAPI)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/v1/recognize` | POST | image file (or multipart list) | `[{plate, conf, bbox, corners, grammar_valid, enhanced, timings}]` |
| `/v1/recognize/video` | POST | video file | `job_id` (async) |
| `/v1/jobs/{id}` | GET | — | job status + per-track plate results |
| `/health` | GET | — | model loaded, version, device |
| `/metrics` | GET | — | latency histograms, request counters |

- **Schemas:** pydantic v2, strict types. Versioned under `/v1` from day one.
- **Inference:** ONNX Runtime, `intra_op=4`. Batch=1 path for interactive; micro-batch (≤8) for video.
- **Limits:** 10 MB/image, 2 min/video; 429 on burst (simple token bucket).

---

## 5. Evaluation Harness (`benchmarks/`)

**Metrics (single source of truth — a script, not a notebook):**

| Metric | Definition |
|---|---|
| Det mAP@0.5 / @0.5:0.95 | standard VOC/COCO protocol |
| Corner error (px) | mean L2 of predicted vs GT corners @ imgsz |
| Plate accuracy | exact string match rate (end-to-end) |
| Char accuracy | 1 − CER over matched plates |
| Edit distance | normalized Levenshtein |
| Latency | p50/p95 per stage + E2E, on T4, batch=1 |
| Uplift table | end-to-end accuracy: never vs always vs **gated** enhancement |

**Ablation protocol (the crown jewel):** same weights, same test set, three
enhancement policies → markdown table auto-generated into README. Split columns by
condition subset: `clean / night / blur / small / tilted`.

**Degradation test set:** apply severity-controlled corruptions (motion blur kernel
σ ∈ {1,2,3,4}, gamma 0.4–0.7 low-light, Gaussian noise, JPEG q30) to clean test
crops → 5 severity levels → "robustness curve" plots.

---

## 6. Testing & CI

- **Unit:** geometry (corner ordering, homography), grammar/correction logic, voting
  math, pick_better ordering, config parsing. Target ≥75% coverage on `src/`.
- **Integration:** fixture image + tiny test weights → assert pipeline returns ≥1 plate
  with valid JSON schema; golden-output test (frozen expected values).
- **API tests:** httpx TestClient — happy path, malformed file, oversized file, 404 job.
- **CI (GitHub Actions):** ruff → mypy → pytest → docker build. Runs on every PR.
- **Weights:** never in git. HF Hub / Kaggle dataset + `scripts/download_weights.py`
  with SHA256 verification.

---

## 7. Repo Layout & Milestone Mapping

```
anpr/
├── src/anpr/{detect,rectify,ocr,enhance,track,pipeline,api,cli}.py
├── configs/{detect,ocr,enhance,serve}.yaml
├── tests/          benchmarks/          docker/
├── scripts/{download_weights,make_degraded_set,export_onnx}.py
└── .github/workflows/ci.yaml
```

| Week | Ships |
|---|---|
| 1 | Repo skeleton, CI green, baseline pipeline (PaddleOCR), first numbers |
| 2 | Keypoint detector fine-tuned, eval harness v1, baseline metrics table |
| 3 | Custom CTC recognizer, gated enhancement + ablation, grammar post-processor |
| 4 | Video tracking+voting, ONNX export, FastAPI+Docker, HF Spaces demo, README final |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kaggle session limit kills training | every run checkpoints to `/kaggle/working`, resumable |
| Indian dataset quality poor | curation checklist (Datasets doc §3) + synthetic fallback |
| Enhancement hurts some cases | gating keeps it off the happy path; pick_better guards |
| Two-line / Devanagari plates | out of scope v1 — documented limitation, single-line Latin only |
| Privacy | public datasets only; no identifying plates in demo assets; README notice |
