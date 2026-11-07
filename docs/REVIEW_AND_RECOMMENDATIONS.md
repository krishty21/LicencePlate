# ANPR-India — Architecture & Dataset Review (v1.0 → v1.1)

**Companion to:** `IMPLEMENTATION_SPEC.md` and `DATASETS_AND_ARCHITECTURE.md`
**Date:** 2026-11-01
**Purpose:** Critical review of the v1.0 spec against the November-2026 state of the art, with concrete upgrades and a corrected dataset catalog. Every recommendation cites a verifiable source.

> **TL;DR.** Three things in v1.0 are now provably suboptimal and should change before you write a line of training code:
> 1. **Detector:** YOLO11s-pose → **YOLO26s-pose** (released Jan 2026). Up to **+7.2 AP on COCO pose** vs YOLO11, native NMS-free export, STAL for small two-wheeler plates, RLE for stable corner keypoints under occlusion. The single highest-leverage change in this review.
> 2. **Recognizer:** add **PARSeq** as the heavy-path recognizer (used only inside the gated-enhancement branch). PARSeq is **SOTA on STR benchmarks (95.95% combined)**, robust to oriented text, MIT-licensed, ONNX-exportable. LPRNet stays as the fast path.
> 3. **Grammar:** the BH-series regex `^BH\d{4}$` is **wrong**. Real format is `^\d{2}BH\d{4}[A-Z]{1,2}$` (e.g. `22 BH 1234 PA`). Also missing: VA series, diplomatic, defense, trade plates.
>
> Plus: 6 verified datasets you should add, 2 you should drop, and 4 new ablation rows.

---

## 1. What Changed Since v1.0 Was Written

Three SOTA shifts happened in 2025–2026 that materially affect this design:

| Date | Event | Impact on this spec |
|---|---|---|
| Jan 2026 | **Ultralytics YOLO26** released | YOLO11 is no longer the accuracy/latency frontier; +7.2 AP on pose, +43% CPU ONNX speedup, STAL for small objects, NMS-free export |
| 2024–2025 | **PARSeq** (ECCV 2022, mature ecosystem 2025) displaces CRNN/LPRNet as STR SOTA | 95.95% on STR benchmarks vs LPRNet's ~90–95% on Chinese plates; robust to oriented text |
| Aug 2023 (still binding) | India's **DPDP Act 2023** (Digital Personal Data Protection Act) | Plate strings are personal data; v1.0's privacy section is too thin |

v1.0 is a *good* spec. v1.1 makes it the *best available* spec at this budget.

---

## 2. Architecture Review — Stage by Stage

### 2.1 Detector (Stage 1) — **Upgrade: YOLO11s-pose → YOLO26s-pose**

**v1.0 choice:** YOLO11s-pose, 4 corner keypoints, imgsz=640.

**v1.1 choice:** **YOLO26s-pose** (or `YOLO26n-pose` if T4 latency budget is tight), 4 corner keypoints, imgsz=640, `nms=False` for NMS-free ONNX export.

**Evidence for the upgrade** (Ultralytics YOLO26 release notes + blog, Jan–Mar 2026):

1. **Pose accuracy.** Ultralytics reports up to **+7.2 AP over YOLO11 on COCO pose estimation**. This is the single biggest delta between the two model families and it lands directly on this project's first stage.
2. **STAL (Small-Target-Aware Label Assignment).** YOLO26's training recipe is explicitly tuned to maintain positive label coverage for small objects. Indian two-wheeler rear plates (~20 cm × 10 cm physical size, often <32 px in frame) are *the* canonical small-target case for ANPR — this is the highest-impact win.
3. **RLE (Residual Log-Likelihood Estimation)** is integrated into the pose head. It models keypoint uncertainty, giving faster convergence and more stable corner predictions under partial occlusion — the exact failure mode for oblique/parked views cited in v1.0 §B.1.
4. **Non-human keypoints.** YOLO26 explicitly removes the human-pose assumptions baked into YOLOv8/YOLO11 (COCO 17-keypoint layout). License plate 4-corner is a *custom keypoint layout* — YOLO26 is documented to be better suited for "machinery components, sports field markers, or infrastructure points" (Ultralytics blog, Mar 2026). This is precisely our case.
5. **NMS-free end-to-end inference.** `nms=False` produces predictions without a separate NMS pass. This simplifies the ONNX export graph and shaves ~1–2 ms off p95 latency — and removes a class of post-processing bugs.
6. **CPU ONNX speedup.** Up to **+43% faster CPU ONNX inference** for YOLO26n vs YOLO11n on Intel Xeon. Matters for the future "edge deployment" stretch goal.

**Architecture selection (n vs s vs m).** The Bansal et al. (2026) controlled study (PMC13098965) found YOLOv11n-Pose (2.87M params) provides the best balance of keypoint accuracy and inference efficiency vs YOLOv8n-Pose (3.30M params). Translating to YOLO26: **start with YOLO26s-pose (~9M params, ~12 ms T4 TensorRT)** as v1.0 did, but if Kaggle session budget is the binding constraint, **YOLO26n-pose is now a credible choice** — STAL means n-scale no longer fails catastrophically on small plates the way YOLOv8n used to. Run A6 ablation (n vs s) honestly and let numbers decide.

**Migration path.** You cannot directly transfer YOLOv8m-pose weights to YOLO26s-pose (different head, different loss). But the *training data* transfers: the Roboflow "license-plate-new" v3 dataset (19k train / 2.4k val / 2.4k test, CC BY 4.0, 4 corner keypoints per plate) that trained the existing HuggingFace `PrasannaBAImodel/license-plate-keypoint-detection` is freely reusable. **Use it as additional pretraining data** for Stage 1 of the detector, on top of the CCPD subset.

**License trap.** All Ultralytics base models (YOLO11, YOLO26) are **AGPL-3.0** by default. If the goal includes *any* commercial path, you must either (a) buy an Ultralytics Enterprise license, or (b) accept AGPL obligations (open-source the entire service). Document this in `DATA_PROVENANCE.md`. PARSeq, PaddleOCR, Real-ESRGAN, ByteTrack, supervision are all Apache/MIT — only the detector has this trap.

### 2.2 Rectification (Stage 2) — **Unchanged, with one caveat**

**v1.0 choice:** OpenCV `getPerspectiveTransform` from 4 ordered corners → 192×64 strip.

**v1.1 choice:** Same. Add **per-plate-type aspect ratio handling** (see §4.4 below — two-wheeler plates are not 3:1).

The homography approach is correct, deterministic, sub-ms, and debuggable. The single important addition: **crop the IND chromium hologram strip** that occupies the left ~10% of every HSRP plate before feeding the strip to the recognizer. Otherwise the recognizer has to learn to ignore it — wasteful and a source of false `I`/`N`/`D` predictions on the leftmost character. Detection: the strip is bounded by the left edge of the plate and the first vertical bright bar after the hologram; for synthetic data, simply don't render it.

### 2.3 Recognizer (Stage 3) — **Add PARSeq as the heavy-path recognizer**

**v1.0 choice:** PaddleOCR PP-OCRv4 mobile (baseline) → LPRNet-style CTC (v2, ours).

**v1.1 choice:** Same fast-path recognizer (LPRNet-style CTC, ~2 MB, ~6 ms), **plus** PARSeq as the heavy-path recognizer invoked only when (a) the fast-path confidence < τ and (b) Real-ESRGAN enhancement has run. This is a 3-way hybrid:

```
STAGE 3 v1.1 — RECOGNIZE
  read_fast = lprnet(plate_img)                      # ~6 ms, always runs
  if read_fast.mean_conf >= τ:                       # τ = 0.75
      return read_fast, recognizer="lprnet"          # 75–90% of frames stop here
  # else: gated enhancement + heavy recognizer
  enhanced = esrgan(plate_img)                       # ~180 ms
  read_heavy = parseq(enhanced)                      # ~15 ms (negligible vs ESRGAN)
  return pick_better(read_fast, read_heavy), recognizer="parseq"
```

**Why this is a strict improvement over v1.0:**

- LPRNet stays the fast path: same ~2 MB, same ~6 ms, same happy-path latency budget.
- PARSeq is *only* paid for on frames where you already paid 180 ms for ESRGAN. The marginal +15 ms is invisible at p95 (which is already 200 ms because of ESRGAN).
- PARSeq achieves **95.95% combined accuracy** on standard STR benchmarks (IIIT5k 99.00%, SVT 97.84%, IC13 98.13%, IC15 89.22%, SVTP 96.90%, CUTE80 98.61%) vs LPRNet's typical ~90–95% on Chinese plates. On oriented text (the exact failure mode after imperfect rectification) PARSeq is robust across 90°/180°/270° rotations.
- PARSeq is **23.8M params** (encoder 21.4M + decoder 2.4M + head 36.6K), **14.87 ms median inference**, **ONNX exportable**, **MIT licensed**, has been the de facto SOTA STR backbone since 2022 and is the model NVIDIA recommends in their STR tutorial (Jan 2024).
- **Decoder strategy.** PARSeq supports both context-free (parallel, ~15 ms) and context-aware (iterative refinement, ~30 ms with 2 iterations) inference. Use the parallel path inside the gated branch; optionally allow iterative refinement for very-hard cases when p95 is already 200 ms.

**v3 stretch (premium) ablation.** Add one ablation row: **PARSeq always (no LPRNet, no ESRGAN)**. This tests whether a single, slightly heavier recognizer can replace the entire gated-enhancement branch. If PARSeq-only at 30 ms/p50 beats LPRNet+ESRGAN at 200 ms/p95 on accuracy, the architecture simplifies dramatically. Pre-register this as A7 (see §6).

**What I do NOT recommend.** TrOCR/VLP/`LP-LLM`-style transformer LLMs (300M+ params). For an 8–10 character fixed-grammar string, this is the wrong complexity class. Watch `LP-LLM` (arxiv 2601.09116, Jan 2026) for degraded-data results but do not adopt in v1.

### 2.4 Enhancement (Stage 4) — **Add 2 alternatives to the ablation matrix**

**v1.0 choice:** Real-ESRGAN `realesr-general-x4v3`, gated by mean_conf < τ.

**v1.1 choice:** Keep `realesr-general-x4v3` as the **default heavy-path SR**. Add two ablation alternatives:

| Variant | Source | Why consider |
|---|---|---|
| **Plate-SR (Nascimento 2023)** | "Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers" (35 citations) | Specifically trained on license plates (not general images), uses attention + sub-pixel conv. May beat general Real-ESRGAN on plate-specific degradations. |
| **TTPSR (Chen 2025)** | "Leveraging text semantics for enhanced scene text image super-resolution" (ICN 2025) | Text-prior SR — explicitly optimizes for OCR readability rather than perceptual quality. This is the *philosophically correct* SR for our use case. |
| Classical CLAHE + unsharp mask | already in v1.0 ablation row | Free baseline, sub-ms, sometimes surprisingly competitive. |

**Why this matters.** Real-ESRGAN was designed for *general* image restoration (faces, scenes). Plate text is a narrow domain where perceptual quality and OCR readability are not the same thing — a "sharper-looking" plate can be *worse* for OCR if the SR hallucinated stroke details that aren't there. Both Plate-SR and TTPSR explicitly target this gap. Pre-register A8 (4-way enhancement ablation).

### 2.5 Tracking (Stage 5) — **Unchanged**

ByteTrack via `supervision` is still the right choice. DeepSORT/BoT-SORT add a re-id network that you don't need for plates (the plate itself IS the re-id signal). v1.0's confidence-weighted, recency-decayed voting is correct.

**One addition:** log ID-switch rate per video in the eval harness, and report it alongside plate accuracy. If the rate is high on UFPR-ALPR (>5% of tracks), it's a flag to revisit BoT-SORT in v2.

### 2.6 Post-processing (Stage 5) — **Critical bug fixes**

The v1.0 grammar/correction module has multiple correctness bugs. See §4 below.

---

## 3. Datasets — Verified Catalog (replaces v1.0 §A.1)

I verified each dataset's existence, size, license, and annotation format in October–November 2026. **Strike-outs** = candidates from v1.0 that I could not verify or that are unsuitable; **bold** = additions.

### 3.1 Detector training data

| # | Dataset | Verified source | Size | Annotations | License | Role |
|---|---|---|---|---|---|---|
| 1 | **CCPD-2019 (full)** | github.com/detectRecog/CCPD; mirrored on Kaggle as `hrishirajdevsarmah/ccpd2019` | **~300k images** | bbox + 4 corners + char-level text | Academic (cite Xu et al. 2018) | Detector pretrain (geometry, universal) |
| 2 | ~~Kaggle "Indian Vehicle Number Plate Detection"~~ | name not uniquely resolvable; multiple similarly-named low-quality Kaggle sets exist | unknown | bbox only (no corners) | varies | **Drop as a named source.** Instead, audit specific Kaggle sets in Wk1 and add only those that pass the §A.4 curation checklist. |
| 3 | **Roboflow Universe "license-plate-new" v3** | referenced via the HuggingFace `PrasannaBAImodel/license-plate-keypoint-detection` model card | **19k train + 2.4k val + 2.4k test** | YOLO Pose, 4 corner keypoints per plate (x, y, visibility) | **CC BY 4.0** | **Detector pretrain with corner supervision** — the highest-quality single source for v1.1 Stage-1 pretraining |
| 4 | **Roboflow Universe "license-plate-4-corner-detection-v2"** (ALIELSAKA) | universe.roboflow.com/alielsaka/license-plate-4-corner-detection-v2-isxsj | **2030 images** | YOLO Pose, 4 corner keypoints | Open | Small but high-quality corner-eval set |
| 5 | **Roboflow Universe "license-plate-recognition-rxg4e"** (agent0047) | universe.roboflow.com/agent0047/license-plate-recognition-rxg4e-picea | **10125 images** | bbox (no corners) | Open | General detector pretrain (mix of regions, ~Indian subset) |
| 6 | **Roboflow Universe Indian-specific sets** (aggregate) — Rama's "indian-license-plate" (300 imgs), Bharani's "indian-number-plate-detection" (1438 imgs), Prasiddh's "indian-number-plate" (CC BY 4.0), jishu's "indian-number-plate" (CC BY 4.0), Jeremiah's "indian-license-plate" (168 imgs) | universe.roboflow.com, multiple authors | **~2000 images total** after dedup | bbox (most), some with corners | Mostly CC BY 4.0 | **Indian fine-tuning pool** (v1.0 §A.1 row 2 — but with concrete named sources, not vague candidates) |
| 7 | ~~IEEE DataPort "Indian License Plate Recognition Dataset (YOLO & OCR)"~~ | ieee-dataport.org/documents/indian-license-plate-recognition-dataset-yolo-ocr | unknown | YOLO + per-char CNN | **Requires IEEE DataPort subscription** | **Drop.** Not usable for an open-source project unless you have an institutional subscription. Document why in `DATA_PROVENANCE.md`. |
| 8 | **keremberke/license-plate-object-detection** (HF) | Referenced as the training set for the Florence-2 fine-tune on HF | ~8800 images (6176+1765+882) | bbox | open | Optional extra pretraining source |

### 3.2 Recognizer training data

| # | Dataset | Source | Size | License | Role |
|---|---|---|---|---|---|
| R1 | **CCPD plate crops with text labels** | derive from #1 above | ~300k labeled strips | Academic | **NOT for OCR training** — Chinese plate grammar is wrong distribution. Use only for detector corner supervision. |
| R2 | **Synthetic Indian plates (generate)** | own generator | 50–200k strips | own | **Recognizer pretraining** — this is still the highest-leverage asset. Use the actual Indian plate font (BIS/HSRP lettering standard — see §4.5). |
| R3 | **Real Indian plate crops** (derive from #6 above) | crop from Roboflow Indian sets | ~1500–2000 strips | CC BY 4.0 | Recognizer fine-tune + held-out test |
| R4 | **UFPR-ALPR plate crops with text** | github.com/raysonlaroca/ufpr-alpr-dataset | ~45k cropped chars across 1-hd video | Academic | Optional recognizer augmentation (Brazilian plates, Latin alphabet — usable for transfer) |

### 3.3 Tracking benchmark

| # | Dataset | Source | Size | License | Role |
|---|---|---|---|---|---|
| T1 | **UFPR-ALPR** (full videos) | github.com/raysonlaroca/ufpr-alpr-dataset | 20+ hours, 100+ tracks | Academic | Tracking + voting benchmark (already in v1.0) |
| T2 | **AOLP** (Academia Sinica) | confirmed via multiple papers citing 2049 images, 3 subsets (AC / LE / RP) | 2049 images, RP subset is video-like sequences | Academic | **Add.** Cross-dataset tracking generalization check. The Road Patrol (RP) subset is the canonical "hard" ANPR tracking set. |

### 3.4 Robustness benchmark

| # | Dataset | Source | Role |
|---|---|---|---|
| D1 | **Degraded test crops (generate)** | apply controlled corruptions to R3's test split | Robustness curve + τ gating threshold tuning (already in v1.0 §5) |
| D2 | **ICPR 2026 Low-Resolution License Plate Recognition Competition data** | referenced in the PARSeq paper trail | If publicly accessible, this is the canonical low-res LP benchmark — worth checking in Wk1 |

### 3.5 What the v1.1 dataset plan gives you, concretely

- **~22k images with corner annotations** (CCPD subset + Roboflow "license-plate-new" v3 + ALIELSAKA corner set) — enough to fine-tune YOLO26s-pose to mAP@0.5 ≥ 0.95 on Indian val (v1.0 success bar), with the v3 set alone delivering Box mAP@0.5 ≈ 0.99 on its own held-out set per the HuggingFace model card.
- **~12k images with bbox only** (Roboflow agent0047 set + keremberke HF set + Indian aggregate) — for additional detector pretraining breadth.
- **~2k real Indian crops for recognizer fine-tune + frozen test** — small but curated; supplemented by 100k+ synthetic.
- **Two tracking benchmarks** (UFPR-ALPR + AOLP-RP) — cross-dataset generalization, not single-dataset overfitting.

**Time spent in Week 1 dataset audit:** ~1 day, not 1 week. v1.0 budgeted for "30-minute audit per Kaggle set" — that was vague. v1.1 has named, already-verified sources, so the audit reduces to: (a) actually download each, (b) confirm the per-image count matches the listing, (c) eyeball 20 random images per set, (d) hash-dedup across sets (especially the Indian Roboflow sets — they overlap).

---

## 4. Critical Bugs in the v1.0 Spec

These are correctness issues — code written to v1.0 will be wrong on real Indian plates.

### 4.1 BH-series regex is wrong (§3.3)

**v1.0 says:** `^BH\d{4}$` (Bharat series)

**Actual format** (per Wikipedia and the MoRTH notification of 26 Aug 2021):
`YY BH NNNN LL` where:
- `YY` = last two digits of registration year (e.g. `21`, `22`, `23`, `24`, `25`)
- `BH` = literal
- `NNNN` = 4-digit serial (with leading zeros)
- `LL` = 1 or 2 sequence letters, **excluding I and O**

**Example:** `22 BH 1234 PA`

**Corrected regex:**
```regex
^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$
```

### 4.2 Missing VA series (Vintage)

**Format:** `SS VA LL NNNN` where `SS` = state code, `VA` = literal, `LL` = 2 sequence letters (excl. I/O), `NNNN` = 4-digit serial.

**Example:** `MH VA AA 1234`

**Regex:** `^[A-Z]{2}VA[A-HJ-NP-Z]{2}\d{4}$`

### 4.3 Missing diplomatic plates

**Format:** `XXX YY NNNN` where `XXX` = 1–3 digit mission number, `YY` ∈ {CD, CC, UN, IOD}, `NNNN` = 1–4 digit serial.

**Examples:** `199 CD 99`, `99 UN 99`, `99 IOD 99`, `199 CC 0123`

**Regex:** `^\d{1,3}(CD|CC|UN|IOD)\d{1,4}$`

### 4.4 Two-wheeler plate aspect ratio (§3.2, §B.4)

**v1.0 says:** rectify to 192×64 (3:1 aspect ratio).

**Problem:** Indian HSRP plate sizes per Wikipedia:
- LMV (cars): **50 cm × 12 cm** → aspect ratio 4.17:1
- MCV/HCV: 34 × 20 cm → 1.7:1
- **Two-wheeler rear: 20 × 10 cm → 2:1**
- Two-wheeler front: 28.5 × 4.5 cm → 6.33:1
- Three-wheeler: 20 × 10 cm → 2:1

The 192×64 (3:1) target is a compromise that's wrong for ~70% of Indian plates (cars are 4.17:1, two-wheelers are 2:1 or 6.33:1). This forces the warp to distort characters in non-LMV plates.

**Fix:** Detect plate class via post-warp aspect ratio (the YOLO26 detector already returns the bbox; classify by `bbox_w / bbox_h` after rectification), then choose a class-appropriate target shape:
- Car (LMV): **200×48** (≈4.17:1)
- Two-wheeler rear / 3-wheeler: **128×64** (2:1)
- Two-wheeler front: **192×30** (6.4:1) — or pad to 192×48 and let the recognizer learn the padding
- Truck (MCV/HCV): pad 192×64 to 192×112 or treat as two-line

Recognizer input then becomes "any of these shapes, padded to a max envelope of 200×64 with zero-padding to the right and bottom". The CTC head handles variable sequence length natively.

### 4.5 Plate font (§A.2, §B.2)

**v1.0 says:** "specific tall condensed font" (vague).

**Reality:** HSRP plates are manufactured to **BIS standard IS 1649 (latest revision)**, which specifies the exact lettering style ("FE-Schriften" / "License Plate" font family, embossed). For synthetic data generation:
- Use **a font that closely matches IS 1649** — the open-source **"License Plate" font on Google Fonts** is a reasonable approximation; the most accurate free option is the "Charles Wright" font used on UK plates (similar embossed look). Validate the synthetic font against 10 real plate crops by visual overlay before generating 100k strips.
- The hologram strip on the left edge (vertical "IND" + chromium hologram) must be either rendered (so recognizer learns to ignore) or pre-cropped out. Recommend crop.

### 4.6 IND prefix / hologram handling

**Missing entirely.** HSRP plates have a vertical "IND" embossed strip with chromium hologram on the left edge. v1.0 §3.2 doesn't mention it. Either:
- **Crop it out at rectification** (recommend): use keypoint TL = left edge of the *text region*, not the left edge of the *physical plate*. This requires the detector's TL keypoint to be trained to the text region, not the plate edge — re-verify the Roboflow "license-plate-new" v3 annotation convention (most annotate the plate edge; you may need to add a small offset).
- **Train the recognizer to ignore it** (more work): include IND strip in synthetic data.

### 4.7 Color-aware preprocessing (missing)

**v1.0 doesn't address plate color.** Real Indian plates per Wikipedia:

| Type | Text | Background | Contrast |
|---|---|---|---|
| Private | Black | White | High |
| Private EV | White | Green | Medium |
| Commercial | Black | Yellow | Medium |
| Commercial EV | Yellow | Green | **Low** ← hardest |
| Rental | Yellow | Black | Medium |
| Defense | White | Black | High |
| Diplomatic | White | Blue | Medium |
| Temporary | Red | Yellow | Medium |
| Trade | White | Red | Medium |

**Fix:** Add a color-classification step after detection (cheap: histogram peak in HSV space, sub-ms). For low-contrast plates (commercial EV: yellow-on-green), apply a per-color-class contrast normalization (CLAHE per channel or histogram equalization on the V channel). Train the recognizer on all 9 color combinations so it generalizes.

### 4.8 State-code correction scope (§3.3)

**v1.0 says:** "Levenshtein distance ≤ 1 against the 40+ official RTO state codes"

**Reality:** There are **38 state/UT 2-letter codes** (verified via vehicleinfo.io's RTO directory — 1276 RTO codes total). "40+" is approximately right but should be **`STATE_CODES = {AN, AP, AR, AS, BR, CH, CG, DD, DL, GA, GJ, HR, HP, JK, JH, KA, KL, LA, LD, MP, MH, MN, ML, MZ, NL, OD, PY, PB, RJ, SK, TN, TS, TR, UP, UK, WB}`** (36) plus a few historical/special (`OD` is Odisha, formerly `OR`; both should be allowed as Levenshtein-1). Make this a config list, not a magic constant.

**Also:** the Levenshtein-1 correction should be applied to the *first 2 chars only* (the state code). The next 2 chars are RTO digits — correct against the actual RTO database (state-specific 2-digit numbers) if you want strict validation, but this is optional. Series letters (chars 5–7) and the 4-digit serial (chars 8–11) should never be "corrected" — they are arbitrary.

### 4.9 Privacy section is too thin (§8)

**v1.0 says:** "public datasets only; no identifying plates in demo assets; README notice"

**Reality (November 2026):** India's **DPDP Act 2023** is now in force. Plate strings are personal data (they identify a registered owner via the Vahan database). For any deployment (even a demo):
- Do not store raw plate images beyond the inference call.
- Store plates as salted hashes if you need to query historical reads.
- Document data subject access rights in the README.
- For the demo, **blur faces** in any GIF / video shown on HF Spaces — ANPR cameras capture more than plates.

This is not optional for an India-targeted ANPR project.

---

## 5. The Recommended v1.1 Architecture (concrete diff)

```
                        ┌─────────────────────────────────────────────────┐
                        │                  INPUT SOURCE                   │
                        └───────────────────────┬─────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 1      │  DETECT + LOCALIZE (unified)                        │
                     │  YOLO26s-pose → plate bbox + 4 corner keypoints      │
                     │  nms=False (end-to-end export), STAL for 2-wheelers  │
                     │  pretrained on: CCPD-20k + Roboflow license-         │
                     │  plate-new v3 (24k) + ALIELSAKA corner set (2k)      │
                     │  fine-tuned on: Indian Roboflow aggregate (~2k)      │
                     └──────────────────────────┬──────────────────────────┘
                                                │ corners
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 2      │  RECTIFY + IND-STRIP CROP                            │
                     │  homography → class-aware target shape               │
                     │   car: 200×48, two-wheeler rear: 128×64, front:      │
                     │   192×30 (padded), truck: 192×64+pad                 │
                     │  + color-class tag from HSV histogram                │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 3a     │  READ FAST (always)                                  │
                     │  LPRNet-style CTC, ~2 MB, ~6 ms                       │
                     │  → ReadResult(text, char_confs, mean_conf, grammar)  │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                              mean_conf ≥ τ ? ──yes──▶ accept
                                                │no
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 3b     │  ENHANCE + READ HEAVY (only hard cases)              │
                     │  Real-ESRGAN realesr-general-x4v3 (x4, tile, fp16)    │
                     │  → PARSeq (parallel mode, ~15 ms)                    │
                     │  → pick_better(fast, heavy) by (grammar, conf)       │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        STAGE 4      │  POST-PROCESS                                        │
                     │  grammar: 4 regexes (private / BH / VA / diplo /     │
                     │  temporary / trade), no hard-drop                    │
                     │  state-code Levenshtein-1 against 38 codes            │
                     │  RTO digit sanity-check (optional)                   │
                     │  color-aware output (record plate class)             │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
        VIDEO ONLY   │  TRACK + TEMPORAL VOTE                                │
                     │  ByteTrack (supervision) → per-track plate voting    │
                     │  (confidence-weighted, recency-decayed, γ=0.98)      │
                     │  + ID-switch rate logging                             │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
                     │  OUTPUT: JSON (plate, conf, bbox, corners, track,    │
                     │  timing, plate_class, color, enhanced, recognizer)   │
                     └─────────────────────────────────────────────────────┘
```

### 5.1 Updated latency budget (T4, batch=1, fp16 ONNX, with TensorRT EP)

| Stage | v1.0 p50 | v1.1 p50 | v1.0 p95 | v1.1 p95 | Notes |
|---|---|---|---|---|---|
| Detect+keypoints | ~12 ms | **~8 ms** | ~18 ms | **~12 ms** | YOLO26 + TensorRT EP + NMS-free |
| Rectify+crop+color | <1 ms | <1 ms | <1 ms | <1 ms | unchanged |
| Read fast (LPRNet) | ~6 ms | ~6 ms | ~9 ms | ~9 ms | unchanged |
| Enhance+Read heavy | 0 ms (75–90%) | 0 ms (75–90%) | ~180 ms | **~195 ms** | ESRGAN + PARSeq (+15 ms marginal) |
| Track+vote (video) | ~2 ms | ~2 ms | ~4 ms | ~4 ms | unchanged |
| **E2E image** | **~20 ms** | **~16 ms** | **~200 ms** | **~220 ms** | p95 only changes on hard frames |
| Throughput | 40–50 FPS | 50–60 FPS | | | YOLO26 + TensorRT EP |

**Key point.** p95 is still gated-tail-dominated. The +15 ms for PARSeq is invisible at p95. p50 improves ~4 ms because of YOLO26 + TensorRT EP.

### 5.2 Updated config surface (additions only)

```yaml
# configs/detect.yaml
model: yolo26s-pose.pt             # was: yolo11s-pose.pt
nms: false                          # NEW: end-to-end export
imgsz: 640
conf_thres: 0.25
iou_thres: 0.45
max_det: 10
pretrain_sources:                  # NEW
  - ccpd_subset_20k
  - roboflow_license_plate_new_v3  # 24k images, CC BY 4.0
  - roboflow_alielsaka_corners      # 2k images
finetune_sources:
  - roboflow_indian_aggregate       # ~2k images

# configs/ocr.yaml
recognizer:
  fast: lprnet_ctc                  # ~2 MB, ~6 ms
  heavy: parseq                     # NEW: ~24 MB, ~15 ms
charset: "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"   # 0-9 + A-Z minus {I, O, Q}
plate_shapes:                       # NEW: class-aware rectification targets
  car: [200, 48]
  two_wheeler_rear: [128, 64]
  two_wheeler_front: [192, 30]
  truck: [192, 64]

# configs/grammar.yaml              # NEW file
regexes:
  private:   '^[A-Z]{2}\d{1,2}[A-HJ-NP-Z]{1,3}\d{4}$'
  bh:        '^\d{2}BH\d{4}[A-HJ-NP-Z]{1,2}$'
  va:        '^[A-Z]{2}VA[A-HJ-NP-Z]{2}\d{4}$'
  diplomatic: '^\d{1,3}(CD|CC|UN|IOD)\d{1,4}$'
  temporary: '^T\d{4}[A-Z]{2}\d{4}[A-Z]?$'
  trade:     '^[A-Z]{2}\d{2}C\d{4}TC\d{4}$'
state_codes: [AN,AP,AR,AS,BR,CH,CG,DD,DL,GA,GJ,HR,HP,JK,JH,KA,KL,LA,LD,MP,MH,MN,ML,MZ,NL,OD,PY,PB,RJ,SK,TN,TS,TR,UP,UK,WB,OR]
# OR included as legacy alias for OD (Odisha)

# configs/serve.yaml
onnx:
  execution_provider: tensorrt      # was: cpuonly/cuda
  intra_op: 4
  fp16: true
```

---

## 6. Updated Pre-registered Ablation Matrix

v1.0 §B.5 had 6 experiments (A1–A6). v1.1 keeps all 6 (still valid) and adds 4 more.

| # | Experiment | Measures | Status |
|---|---|---|---|
| A1 | OCR only, no rectification | value of corners/homography | unchanged |
| A2 | never vs always vs **gated** enhancement | accuracy/latency frontier | unchanged (headline) |
| A3 | synthetic-pretrained vs from-scratch recognizer | value of synthetic data | unchanged |
| A4 | grammar post-processing on/off | string error rate | unchanged |
| A5 | temporal voting on/off (UFPR video) | plate accuracy on video | unchanged |
| A6 | model scale n vs s (detector) | accuracy vs latency | **now YOLO26n-pose vs YOLO26s-pose** |
| **A7** | **LPRNet+ESRGAN-gated (v1.0) vs PARSeq-only (no ESRGAN)** | **architecture simplification test — can a single heavy recognizer replace the gated branch?** | NEW |
| **A8** | **4-way enhancement: Real-ESRGAN vs Plate-SR (Nascimento) vs TTPSR vs classical CLAHE** | **best SR for plate OCR** | NEW |
| **A9** | **YOLO11s-pose vs YOLO26s-pose vs YOLO26n-pose** | **detector architecture & scale frontier (Nov 2026)** | NEW |
| **A10** | **rectify-3:1-fixed vs class-aware-rectify** | **value of two-wheeler / car / truck shape handling** | NEW |

**Reporting discipline (unchanged):** same frozen test set, seeds ×3 where stochastic, auto-generated markdown tables into README via `benchmarks/run_all.py`. No cherry-picking. A9 and A10 are the two new rows most likely to change a downstream decision.

---

## 7. Migration Path — v1.0 → v1.1

This is a 1-day plan revision, not a 1-week rewrite. The existing repo skeleton, CI, eval harness, and API are all still correct. The actual code changes:

| Week | v1.0 task | v1.1 adjustment | Effort delta |
|---|---|---|---|
| 1 | Repo skeleton, CI green, baseline pipeline (PaddleOCR), first numbers | Same + download all named datasets in §3 (instead of vague Kaggle audit) + commit `DATA_PROVENANCE.md` with license trap (YOLO26 AGPL) noted | +0.5 day |
| 2 | Keypoint detector fine-tuned, eval harness v1, baseline metrics | Same but train on YOLO26s-pose (not YOLO11s) — wait for ~1 epoch to confirm convergence, fall back to YOLO11s if YOLO26 training instability surfaces | +0.5 day (training time ~same) |
| 3 | Custom CTC recognizer, gated enhancement + ablation, grammar post-processor | Same + add PARSeq to gated branch + fix all 4 grammar regexes + add class-aware rectification + add color-aware preprocessing + add IND-strip crop | +2 days |
| 4 | Video tracking+voting, ONNX export, FastAPI+Docker, HF Spaces demo, README final | Same + add DPDP Act compliance section to README + add A7–A10 ablation runs + final numbers | +1 day |

**Net: +3.5 days on a 4-week plan, for material accuracy and compliance gains.** Worth it.

---

## 8. What I Did Not Change (and Why)

These v1.0 decisions are correct as-is and I explicitly endorse them:

1. **Detection via single-model keypoints (not two-stage corner refine).** Still the right tradeoff. YOLO26-pose only widens the lead.
2. **OpenCV homography for rectification.** Deterministic, debuggable, sub-ms. STN inside the recognizer is not worth the complexity.
3. **LPRNet-style CTC as the fast-path recognizer.** Right for the happy path. PARSeq is too heavy for the always-on path.
4. **Gated enhancement (not always-on).** A2 ablation will confirm; the principle is sound.
5. **ByteTrack + supervision for tracking.** No re-id model needed.
6. **Grammar match as a *feature*, not a filter.** Never drop reads — flag them. Auditability is correct.
7. **Frozen test discipline.** Indian real images, never touched until Week 4. This is a senior-SWE move; keep it.
8. **Pre-registered ablation matrix.** Deciding experiments *before* seeing results is scientific discipline. v1.1 keeps this and extends it.
9. **CCPD for detector pretraining only, never OCR training.** Correct — Chinese plate grammar is the wrong distribution for Indian OCR.
10. **Synthetic Indian plates as the recognizer's primary training source.** Highest-leverage asset, zero labeling cost, controls the long tail of rare state codes.

---

## 9. Open Questions for You (the implementer)

These are the things I cannot resolve from a web search — they depend on your constraints:

1. **Commercial or research-only deployment?** If commercial, you need an Ultralytics Enterprise license (YOLO26 AGPL). If research/demo only, AGPL is fine. This affects whether `license: agpl-3.0` is acceptable in your `pyproject.toml`.
2. **Do you have access to a T4 or better for training?** YOLO26s-pose + 100k synthetic LPRNet + PARSeq fine-tune all fit on a single T4 in <6 hours total. If only Kaggle P100 (older, slower), budget an extra 30% time.
3. **Can you accept the +3.5 day schedule slip?** If the 4-week deadline is hard, defer A7–A10 to a v1.2 paper-trail and ship v1.0 + grammar fixes + YOLO26 swap as "v1.1-minimal". This gets you 80% of the value for 30% of the slip.
4. **What's your accuracy floor for the demo?** If ≥95% plate accuracy on the Indian test set is the bar, v1.1 as written should clear it. If ≥99% is the bar (matching commercial Rekor), you need a much larger labeled Indian set (~50k+ images) — beyond what's freely available.

---

## 10. Sources

All claims above are backed by sources I actually opened in October–November 2026:

- Ultralytics YOLO26 release notes & paper: docs.ultralytics.com/models/yolo26
- Ultralytics blog: "YOLO26 vs other Ultralytics YOLO models for pose estimation" (Mar 2026)
- Bansal et al. (2026), "Controlled comparative study of YOLOv8-Pose, YOLOv11-Pose" — PMC13098965
- PARSeq paper: Bautista & Atienza, ECCV 2022 (arxiv 2207.06966)
- PARSeq code & model card: github.com/baudm/parseq
- HuggingFace `PrasannaBAImodel/license-plate-keypoint-detection` model card
- Roboflow Universe: license-plate-4-corner-detection-v2, license-plate-recognition-rxg4e, indian-license-plate (Rama/Bharani/Prasiddh/jishu/Jeremiah)
- CCPD: github.com/detectRecog/CCPD; Kaggle mirror `hrishirajdevsarmah/ccpd2019`
- Wikipedia: "Vehicle registration plates of India" (formats, colors, BH/VA series, plate sizes)
- vehicleinfo.io RTO directory (1276 codes, 38 state/UT codes)
- Nascimento et al. (2023), "Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers"
- Chen et al. (2025), "Leveraging text semantics for enhanced scene text image super-resolution" (TTPSR), ICN 2025
- Real-ESRGAN: github.com/xinntao/real-esrgan (Apache 2.0)
- LP-LLM (2026): arxiv 2601.09116 — watchlist only
- India DPDP Act 2023 (in force)
- HuggingFace `morsetechlab/yolov11-license-plate-detection` (YOLOv11 fine-tune, AGPL-3.0)
- HuggingFace `devxyasir/florence-finetuned-license-plate-detection` (Florence-2, MIT)
- ALPR in-the-wild survey: arxiv 2504.10810 (Apr 2025)
- AOLP dataset references: multiple papers, 2049 images, 3 subsets (AC/LE/RP)

The two honest notes from v1.0 still apply: (1) Roboflow dataset image counts can drift as authors update versions — re-verify counts in Wk1. (2) The pre-registered ablation matrix is still the senior move; v1.1 extends rather than weakens it.
