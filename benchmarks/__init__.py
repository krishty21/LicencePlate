"""Benchmark / evaluation harness for ANPR-India.

Implements the metrics from IMPLEMENTATION_SPEC.md §5:
  - Det mAP@0.5 / @0.5:0.95
  - Corner error (px)
  - Plate accuracy (exact string match rate, end-to-end)
  - Char accuracy (1 - CER)
  - Edit distance (normalized Levenshtein)
  - Latency p50/p95 per stage + E2E
  - Uplift table (never vs always vs gated enhancement)

And the ablation matrix A1-A10 from REVIEW_AND_RECOMMENDATIONS.md §6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
