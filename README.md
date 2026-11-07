# ANPR-India — Kaggle-ready project (v1.1)

End-to-end Automatic Number Plate Recognition for Indian plates (images & video).

[![Tests](https://img.shields.io/badge/tests-99%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![v1.1](https://img.shields.io/badge/spec-v1.1-orange)]()

**Architecture (v1.1, November 2026):**

```
YOLO26s-pose (4 corner keypoints)
  → homography rectify (class-aware: car / two-wheeler / truck)
  → LPRNet CTC (fast path, ~6 ms) — runs always
  → if mean_conf < 0.75:
        Real-ESRGAN compact → PARSeq (heavy path, ~15 ms)
        pick_better(fast, heavy)
  → grammar post-process (private / BH / VA / diplomatic / temporary / trade)
  → (video only) ByteTrack + confidence-weighted temporal voting
  → JSON output
```

## What's included

| Component | Status | Files |
|---|---|---|
| **Spec docs** | complete | `docs/{IMPLEMENTATION_SPEC,DATASETS_AND_ARCHITECTURE,REVIEW_AND_RECOMMENDATIONS}.md` (5,300+ words each) |
| **Core modules** | complete | 11 modules in `src/anpr/` — types, grammar, rectify, detect, ocr, enhance, track, pipeline, api, cli |
| **LPRNet model** | **complete (was the v1.0 gap)** | `src/anpr/models/{lprnet,lprnet_trainer}.py` — full PyTorch implementation + training loop |
| **Configs** | complete | 5 YAMLs in `configs/` — detect, ocr, enhance, grammar, serve (v1.1 settings) |
| **Data prep scripts** | complete | `scripts/{prepare_ccpd,prepare_roboflow,make_synthetic_plates,make_degraded_set,download_weights,export_onnx,audit_datasets,run_ablation,visualize}.py` |
| **Ablation harness** | complete (was scaffold) | `scripts/run_ablation.py` — actually runs A1-A10, generates markdown tables |
| **Benchmarks** | complete | `benchmarks/{metrics,run_all}.py` |
| **Tests** | 99 passing, 1 skipped | `tests/{test_grammar,test_rectify,test_voting,test_lprnet,test_metrics,test_pipeline_smoke}.py` |
| **Kaggle notebook** | complete | `kaggle_notebook.ipynb` — real JSON .ipynb, 14 cells |
| **Kaggle training script** | complete | `kaggle_train.py` — orchestrates setup → weights → synthetic → detector → LPRNet → PARSeq → ONNX export |
| **Docker** | complete | `docker/{Dockerfile,docker-compose.yml}` — production-ready GPU image |
| **CI** | complete | `.github/workflows/ci.yaml` — ruff → mypy → pytest → docker |
| **LICENSE + provenance** | complete | MIT for code, AGPL notes for Ultralytics, full `DATA_PROVENANCE.md` |

## Quickstart on Kaggle (3 options)

### Option A — Use the Kaggle notebook (recommended)

1. Upload `anpr-india-kaggle.zip` as a Kaggle Dataset
2. Create a new notebook with **GPU T4 x2** accelerator
3. File → Import Notebook → upload `kaggle_notebook.ipynb`
4. Run all cells

### Option B — Use kaggle_train.py from a fresh notebook

1. Upload `anpr-india-kaggle.zip` as a Kaggle Dataset
2. Create a notebook with **GPU T4 x2** accelerator
3. First cell:
   ```python
   import sys, os
   sys.path.insert(0, '/kaggle/input/anpr-india-kaggle/src')
   os.environ['KAGGLE_INPUT'] = '/kaggle/input/anpr-india-kaggle'
   exec(open('/kaggle/input/anpr-india-kaggle/kaggle_train.py').read())
   main(['--stage', 'setup'])      # install deps
   main(['--stage', 'weights'])    # download YOLO26 + Real-ESRGAN + PARSeq weights
   main(['--stage', 'synthetic'])  # generate 150k synthetic Indian plates
   main(['--stage', 'lprnet'])     # train LPRNet (Stage 1: pretrain)
   main(['--stage', 'export-onnx'])  # export LPRNet to ONNX
   ```

### Option C — Run locally (CPU-only for tests, GPU for training)

```bash
unzip anpr-india-kaggle.zip -d anpr-india
cd anpr-india
pip install -r requirements.txt
PYTHONPATH=src pytest tests/                 # 99 passing
PYTHONPATH=src python -m anpr.models.lprnet   # LPRNet smoke test
PYTHONPATH=src python -m anpr.cli --help      # CLI
```

## Project layout

```
anpr-india-kaggle/
├── README.md                            ← this file
├── LICENSE                              ← MIT (code) + AGPL notes (weights)
├── .gitignore
├── pyproject.toml + requirements.txt
├── kaggle_notebook.ipynb                ← Kaggle notebook (14 cells, JSON)
├── kaggle_train.py                      ← Kaggle training orchestrator
├── DATA_PROVENANCE.md                   ← license tracking (auto-generated)
├── docker/
│   ├── Dockerfile                       ← production GPU image
│   └── docker-compose.yml               ← compose template
├── docs/
│   ├── IMPLEMENTATION_SPEC.md           ← v1.0 implementation contract
│   ├── DATASETS_AND_ARCHITECTURE.md     ← v1.0 datasets & model arch
│   └── REVIEW_AND_RECOMMENDATIONS.md    ← v1.1 critical review (5300 words)
├── src/anpr/
│   ├── __init__.py
│   ├── types.py                         ← dataclass I/O contracts
│   ├── grammar.py                       ← v1.1 corrected regexes (BH/VA/diplo/etc)
│   ├── rectify.py                       ← homography + class-aware shapes
│   ├── detect.py                        ← YOLO26-pose wrapper
│   ├── ocr.py                           ← LPRNet + PARSeq hybrid + Paddle baseline
│   ├── enhance.py                       ← Real-ESRGAN + 3 ablation alternatives
│   ├── track.py                         ← ByteTrack + temporal voting
│   ├── pipeline.py                      ← orchestrator with per-stage timers
│   ├── api.py                           ← FastAPI service
│   ├── cli.py                           ← `python -m anpr.cli ...`
│   └── models/
│       ├── __init__.py
│       ├── lprnet.py                    ← LPRNet PyTorch model (~1.8M params)
│       └── lprnet_trainer.py            ← pretrain → fine-tune training loop
├── configs/
│   ├── detect.yaml                      ← YOLO26s-pose + training recipe
│   ├── ocr.yaml                         ← LPRNet + PARSeq hybrid + tau=0.75
│   ├── enhance.yaml                     ← Real-ESRGAN + 4-way ablation backends
│   ├── grammar.yaml                     ← v1.1 corrected regexes + state codes
│   └── serve.yaml                       ← FastAPI + ONNX Runtime + TensorRT EP
├── scripts/
│   ├── download_weights.py              ← HF Hub downloads + SHA256 verify
│   ├── make_synthetic_plates.py         ← synthetic Indian plate generator
│   ├── make_degraded_set.py             ← robustness benchmark corruptions
│   ├── prepare_ccpd.py                  ← CCPD → YOLO pose converter
│   ├── prepare_roboflow.py              ← Roboflow downloader + combiner
│   ├── export_onnx.py                   ← .pt → .onnx (fp16, NMS-free)
│   ├── audit_datasets.py                ← Week-1 day-1 dataset audit
│   ├── run_ablation.py                  ← real A1-A10 ablation runner
│   └── visualize.py                     ← detection visualization + demo
├── benchmarks/
│   ├── __init__.py
│   ├── metrics.py                       ← mAP, CER, edit distance, latency
│   └── run_all.py                       ← benchmark + ablation scaffolding
├── tests/
│   ├── __init__.py
│   ├── test_grammar.py                  ← 30+ tests for v1.1 grammar regexes
│   ├── test_rectify.py                  ← corner ordering, shape classification
│   ├── test_voting.py                   ← temporal voting math
│   ├── test_lprnet.py                   ← LPRNet architecture tests
│   ├── test_metrics.py                  ← metrics function tests
│   └── test_pipeline_smoke.py           ← pipeline composition smoke test
└── .github/workflows/ci.yaml            ← ruff → mypy → pytest → docker
```

## What's NOT in the box (and why)

- **Pre-trained weights** — not bundled (would inflate the zip to ~500 MB).
  Run `main(['--stage', 'weights'])` on Kaggle to download from HF Hub with
  SHA256 verification.
- **Datasets** — not bundled (license + size). Use `scripts/prepare_ccpd.py`
  and `scripts/prepare_roboflow.py` to download + convert.
- **Docker image** — Dockerfile is included; build with `docker build -t anpr-india .`
- **PARSeq fine-tune script** — the wrapper exists in `src/anpr/ocr.py` but
  the fine-tune is done via PARSeq's own `train.py`. The Kaggle notebook
  shows how to clone + install PARSeq.

## v1.0 → v1.1 changes (executive summary)

Three changes from the original v1.0 spec, all backed by November-2026 SOTA:

1. **Detector: YOLO11s-pose → YOLO26s-pose** (released Jan 2026). +7.2 AP on
   COCO pose, STAL for small two-wheeler plates, RLE for stable corners under
   occlusion, native NMS-free ONNX export.

2. **Recognizer: add PARSeq as the heavy-path recognizer**, invoked only when
   the fast-path LPRNet confidence < 0.75 AND Real-ESRGAN has run. PARSeq is
   SOTA on STR benchmarks (95.95%), robust to oriented text, MIT-licensed,
   ONNX-exportable. LPRNet stays as the fast path (~6 ms, ~2 MB).

3. **Grammar fixes:** the v1.0 `^BH\d{4}$` regex for Bharat series was
   **wrong**. Real format is `YY BH NNNN LL` (e.g. `22BH1234PA`). v1.1 also
   adds: VA (Vintage), diplomatic (CD/CC/UN/IOD), temporary, trade, defense.

Plus: 4 new ablation rows (A7-A10), 6 verified dataset additions, class-aware
rectification for two-wheeler vs car vs truck plate shapes, color-aware
preprocessing for the 9 Indian plate color combinations, DPDP Act 2023
compliance notes.

See `docs/REVIEW_AND_RECOMMENDATIONS.md` for the full 5,300-word review with
every claim cited.

## License

- **Code:** MIT (this project's source code).
- **Pretrained weights:** per-model (Ultralytics AGPL-3.0, PARSeq MIT,
  Real-ESRGAN Apache-2.0, PaddleOCR Apache-2.0). Document in `LICENSE`.
- **Datasets:** per-source (CCPD academic, Roboflow mostly CC-BY-4.0).

**Commercial use note:** the Ultralytics YOLO11/YOLO26 base is AGPL-3.0. If
your goal is commercial deployment, you need an Ultralytics Enterprise license
OR accept AGPL obligations (open-source the entire service). PARSeq, Real-ESRGAN,
PaddleOCR, supervision are all Apache/MIT — only the detector has this trap.

## Privacy (India DPDP Act 2023)

Plate strings are personal data under India's Digital Personal Data Protection
Act 2023. For any deployment (even a demo):
- Do not store raw plate images beyond the inference call.
- Store plates as salted hashes if you need historical queries.
- Document data subject access rights in your README.
- Blur faces in any GIF / video used in marketing or HF Spaces demos.

ANPR is surveillance-adjacent tech — handle accordingly.
