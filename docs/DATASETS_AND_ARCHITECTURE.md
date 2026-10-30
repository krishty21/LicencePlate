# ANPR-India — Datasets & Model Architecture (In-Depth)

**Companion to:** `IMPLEMENTATION_SPEC.md`. This doc justifies *what we train on*
and *why this architecture* over the alternatives.

---

## Part A — Data Strategy

### A.1 Dataset Catalog

| # | Dataset | Source | Size | Annotations | Role |
|---|---|---|---|---|---|
| 1 | **CCPD** | github.com/detectRecog/CCPD | ~250k+ images | bbox + 4 corners + text | Detector **pretraining** (Chinese plates, but geometry is universal) |
| 2 | **Kaggle Indian plate sets** (candidates: "Indian Vehicle Number Plate Detection", "Vehicle Number Plate Recognition (YOLO)", "Indian License Plate OCR") | Kaggle (verify in Wk1) | 100s–10k imgs | bbox (+text on some) | **Fine-tuning** + real Indian test distribution |
| 3 | **Roboflow Universe** Indian ANPR collections | roboflow.com/universe (search "indian license plate", filter YOLO export) | variable | bbox/YOLO format | Augment training pool; instant YOLO-format export |
| 4 | **UFPR-ALPR** | github.com/raysonlaroca/ufpr-alpr-dataset | video + tracks | bboxes + strings per frame | **Tracking/voting evaluation** (real surveillance footage) |
| 5 | **Synthetic Indian plates** (we generate) | our generator | 50–200k strips | exact text | **Recognizer pretraining**; covers rare state codes |
| 6 | **Degraded test crops** (we generate) | corruptions of #2's test split | configurable | text | Robustness benchmark + gating threshold tuning |

**Verification rule (Week 1, day 1):** every candidate Kaggle set gets a 30-minute
audit before entering the pool — 20 random images eyeballed, class balance checked,
license checked (see A.5). Kaggle Indian plate sets vary wildly in quality; assume
30–50% curation waste and budget for it.

### A.2 Why this composition (analysis)

- **The distribution gap is the whole problem.** Pretrained models (CCPD lineage,
  PaddleOCR) are biased toward Chinese/EU plate fonts. Indian plates: specific tall
  condensed font, `XX NN LL NNNN` layout, frequent two-wheelers (tiny plates),
  dust/lighting extremes. Fix = pretrain on CCPD geometry → fine-tune on Indian data →
  recognize with a model trained on *synthetic Indian typography*.
- **Synthetic data is the highest-leverage asset.** We control font, spacing, state
  codes (real RTO list), colors (white/yellow/green EV plates), backgrounds, and
  corruptions. 100k synthetic strips cost ~1 GPU-hour and zero labeling. This is also
  the artifact interviewers find most interesting.
- **CCPD is detection-only, never OCR training.** Its plate grammar is entirely
  different; mixing it into OCR training teaches the wrong distribution.

### A.3 Label Schema (single unified format)

We convert everything to one schema — keypoints make bbox+corners lossless:

```yaml
# one plate instance
bbox:      [x1, y1, x2, y2]
corners:   [[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]   # clockwise from TL
text:      "MH12DE1434"        # optional for detection-only sets
condition: clean|night|blur|tilt|small       # tag from source or corruption
source:    ccpd|kaggle_x|synthetic|ufpr
```

Datasets with only bboxes (most Kaggle sets): corners = bbox corners initially;
optionally auto-refine corners in Week 2 with a corner-refinement pass (stretch goal).

### A.4 Splits & Curation Checklist

**Split (by source, then random 70/15/15):**
- Train: CCPD subset (~20k sampled) + all Indian train + all synthetic
- Val: Indian val + synthetic val (for recognizer early stopping)
- Test: **Indian real images only, never touched until Week 4** (frozen test discipline)
- UFPR-ALPR test videos are a separate tracking benchmark, never mixed into image metrics.

**Curation checklist (reject an image if):**
1. Plate < 24 px wide (below recognizer's effective floor)
2. Unreadable by a human at 3× zoom
3. Wrong/mislabeled box or truncated plate > 30%
4. Two-line plates → crop the Latin line only, tag `condition=two_line_excluded`
5. Duplicate vehicles (hash the plate region — Kaggle sets have dupes)

### A.5 Licensing & Ethics

- Kaggle/Roboflow sets: record license per dataset in `DATA_PROVENANCE.md`. CC-BY/CC0/MIT
  preferred; research-only licenses documented, no commercial claims.
- CCPD: published for academic use — cite the paper in README.
- Never scrape personal vehicles; no identifiable people in demo GIFs; README carries a
  responsible-use note (ANPR = surveillance-adjacent tech).

---

## Part B — Architecture Deep-Dive

### B.1 Decision Matrix (what we chose and what we rejected)

| Component | ✅ Chosen | ❌ Considered & rejected | Why |
|---|---|---|---|
| Plate localization | **YOLO11s-pose, 4 corner keypoints** | (a) plain YOLO bbox + deskew; (b) YOLOv8-obb rotated box; (c) two-stage corner-refine net | (a) fails on strong perspective (India's top failure mode: parked/oblique views); (b) fixes rotation but not shear; (c) = 2 models, 2× latency. Keypoints give a full homography in ONE pass — validated in practice (YOLOv8-pose plate-corner pipelines on GitHub) |
| Detector backbone | s-scale (~9M params) | n-scale (too weak on small plates), m/l (Kaggle session budget) | n fails <32px plates; s trains in <2h on T4 |
| Rectification | OpenCV homography → 192×64 | affine, STN inside recognizer | homography is exact for planar plates, sub-ms, debuggable |
| Recognizer | **LPRNet-style CNN + CTC** (ours, ~2 MB) | PaddleOCR (kept as baseline); CRNN+BiLSTM (2–3× slower); TrOCR (300M+, absurd for 8-char strings) | plates = fixed-ish grammar, tiny charset → CTC CNN is ideal; LPRNet lineage proven for plates; ONNX ~2 MB serves in <10 ms |
| Enhancement | Real-ESRGAN `general-x4v3` compact, **gated** | always-on; DeblurGAN-v2 (heavier, motion-blur specialist); Restormer (way too heavy); classical CLAHE+unsharp (kept as free ablation row) | compact variant ≈ 1/4 latency; gating caps cost at ~10–20% of frames |
| Tracking | ByteTrack (supervision) | DeepSORT (needs re-id embedding — extra model), BoT-SORT | plates+vehicle bbox are distinctive enough; ByteTrack needs zero extra network |
| Post-processing | grammar regex + Levenshtein state-code fix | hard filtering; beam-search constrained decoding | never drop reads (auditability); correction is logged, reversible |

### B.2 Recognizer Architecture (the model we train)

```
Input (3×64×192 strip)
│
├─ Conv 3×3, 64, BN, ReLU                 ┐
├─ MaxPool 2×2                            │
├─ Conv 3×3, 128, BN, ReLU                │  LPRNet-style "backbone"
├─ MaxPool 2×2                            │  small+kopt: ~6 conv blocks,
├─ Conv 3×3, 256, BN, ReLU ×2 + skip      │  width halved twice →
├─ MaxPool (2,1)  keep H compress         │  sequence length ~48
├─ Conv 3×3, 256, BN, ReLU                │
├─ Conv (H→1 collapse) 1×(H)×W → 256×48   ┘
├─ 1×5 linear context block (cheap, no RNN)
├─ Linear → (V+1) classes                 V = 33 chars (10 digits + 23 letters + pad), +1 CTC blank
└─ CTC loss, input len 48, label len ≤ 10
```

**Training recipe:**
1. Pretrain: synthetic set (150k strips, AdamW lr 1e-3, cosine, 10 epochs, bs 256)
2. Fine-tune: real Indian crops (×5 oversampled hard conditions), lr 1e-4, 5 epochs
3. Augment: perspective warp ±15°, motion blur, gamma 0.4–1.6, JPEG q40, downscale to
   24px then upsample (the "tiny plate" simulator), random state-code resampling
4. Class imbalance: 4× oversample of night/blur conditions

**Charset note:** `A-Z` minus `{I,O,Q}` per Indian registration practice —
verify against your curated data in Week 2; config-driven, not hardcoded.

### B.3 Detector Training Recipe

- Start: COCO-pretrained YOLO11s-pose → Stage 1 on CCPD-20k (corner supervision free)
  → Stage 2 on Indian train (freeze backbone 10 epochs, then full).
- Augmentation: mosaic, HSV jitter, **strong perspective warp** (teaches keypoints),
  copy-paste of small plates, night-style gamma downsampling.
- Hyperparams: imgsz 640, 60–100 epochs total, bs 32, cosine LR, patience 15.
- Success bar: mAP@0.5 ≥ 0.95 on Indian val; corner error ≤ 2 px on val.

### B.4 Latency Budget (T4, batch=1, fp16 ONNX)

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Detect+keypoints (640) | ~12 ms | ~18 ms | |
| Rectify | <1 ms | <1 ms | |
| OCR (CTC) | ~6 ms | ~9 ms | |
| Enhancement (gated) | 0 ms (75–90% frames) | ~180 ms | only low-conf frames |
| Track+vote (video) | ~2 ms | ~4 ms | amortized |
| **E2E image** | **~20 ms** | **~200 ms** | p95 = gated tail |
| Throughput | 40–50 FPS single stream | | video mode |

### B.5 Ablation Matrix (pre-registered — no cherry-picking)

| # | Experiment | Measures |
|---|---|---|
| A1 | OCR only, no rectification | value of corners/homography |
| A2 | never vs always vs **gated** enhancement | accuracy/latency frontier ← headline |
| A3 | synthetic-pretrained vs from-scratch recognizer | value of synthetic data |
| A4 | grammar post-processing on/off | string error rate |
| A5 | temporal voting on/off (UFPR video) | plate accuracy on video |
| A6 | model scale n vs s (detector) | accuracy vs latency |

Every row: same frozen test set, seeds ×3 where stochastic, auto-generated markdown
tables into README via `benchmarks/run_all.py`.
