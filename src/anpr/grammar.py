"""Grammar + state-code correction for Indian plates.

Implements v1.1 spec §4 — fixes the v1.0 BH-series regex bug and adds
all missing formats (VA, diplomatic, temporary, trade).

Grammar match is a FEATURE, not a filter — never hard-drop invalid reads.
State-code correction is logged (corrected=True) and reversible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# A-Z minus {I, O, Q} per Indian registration practice.
# I/O are visually confusable with 1/0; Q is excluded by RTO convention.
SERIES_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"   # 23 letters

# 38 state/UT codes (verified via vehicleinfo.io RTO directory).
# Includes legacy OR alias for OD (Odisha) — plates pre-2010 may use OR.
STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA",
    "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP",
    "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK",
    "TN", "TS", "TR", "UP", "UK", "WB", "OR",  # OR = legacy Odisha
})


@dataclass(frozen=True)
class GrammarSpec:
    name: str
    regex: re.Pattern[str]
    example: str
    description: str


# v1.1 corrected regexes — see REVIEW_AND_RECOMMENDATIONS.md §4.
GRAMMARS: tuple[GrammarSpec, ...] = (
    GrammarSpec(
        name="private",
        regex=re.compile(rf"^[A-Z]{{2}}\d{{1,2}}[{SERIES_LETTERS}]{{1,3}}\d{{4}}$"),
        example="MH12DE1434",
        description="Standard private/commercial: SS RR LLL NNNN",
    ),
    GrammarSpec(
        name="bh",   # FIXED: was ^BH\d{4}$ — wrong
        regex=re.compile(rf"^\d{{2}}BH\d{{4}}[{SERIES_LETTERS}]{{1,2}}$"),
        example="22BH1234PA",
        description="Bharat series: YY BH NNNN LL (since 2021)",
    ),
    GrammarSpec(
        name="va",
        regex=re.compile(rf"^[A-Z]{{2}}VA[{SERIES_LETTERS}]{{2}}\d{{4}}$"),
        example="MHVAAA1234",
        description="Vintage series: SS VA LL NNNN (since 2021)",
    ),
    GrammarSpec(
        name="diplomatic",
        regex=re.compile(r"^\d{1,3}(CD|CC|UN|IOD)\d{1,4}$"),
        example="199CD99",
        description="Diplomatic: XXX (CD|CC|UN|IOD) NNNN",
    ),
    GrammarSpec(
        name="temporary",
        regex=re.compile(r"^T\d{4}[A-Z]{2}\d{4}[A-Z]?$"),
        example="T0124AN0123A",
        description="Temporary: T YYYY SS NNNN L?",
    ),
    GrammarSpec(
        name="trade",
        regex=re.compile(r"^[A-Z]{2}\d{2}C\d{4}TC\d{4}$"),
        example="AN01C0123TC0123",
        description="Trade plate: SS RR C NNNN TC NNNN",
    ),
    GrammarSpec(
        name="defense",
        regex=re.compile(r"^\d{2}[A-Z]\d{6}[A-Z]$"),
        example="22B123456A",
        description="Defense (without broad arrow prefix): YY T NNNNNN C",
    ),
)


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance. Used for state-code correction only."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            substitute = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, substitute))
        prev = curr
    return prev[-1]


def correct_state_code(candidate: str, max_distance: int = 1) -> Optional[str]:
    """Return the closest Indian state code within Levenshtein distance.

    Conservative policy: if 2+ state codes tie at the minimum distance,
    return None (ambiguous — do not correct). This prevents silently
    turning a misread 'MG' into 'MH' when 'CG' is equally plausible.

    Returns None if no state code is within max_distance — caller must
    leave the read unchanged in that case (never drop).
    """
    if len(candidate) != 2 or not candidate.isalpha():
        return None
    cand = candidate.upper()
    if cand in STATE_CODES:
        return None   # already valid
    # Find all state codes at each distance, ascending.
    for dist in range(1, max_distance + 1):
        matches = [code for code in STATE_CODES
                   if levenshtein(cand, code) == dist]
        if len(matches) == 1:
            return matches[0]
        if len(matches) >= 2:
            return None   # ambiguous — refuse to correct
    return None


@dataclass
class GrammarResult:
    """Result of grammar validation + correction."""

    text: str                       # final (possibly corrected) text
    grammar_valid: bool
    grammar_kind: Optional[str]     # which GrammarSpec matched, if any
    corrected: bool                 # was a state-code correction applied?
    original_text: str              # the read as it came from the recognizer


def apply_grammar(raw_text: str) -> GrammarResult:
    """Validate + (optionally) correct a plate read.

    Pipeline:
      1. Strip whitespace, dots, dashes that OCR may include.
      2. Check against each grammar.
      3. If a grammar matches AND the state code is invalid (private / VA /
         trade format only), attempt unambiguous Levenshtein-1 correction.
      4. If no grammar matched, also attempt state-code correction to see
         if a corrected version would match.
      5. Otherwise, return original text flagged grammar_valid=False.

    NEVER drop the read. Auditability > cleanliness.
    State-code correction is conservative: ambiguous matches (2+ codes at
    equal distance) are NOT corrected — flagged but left as-is.
    """
    original = raw_text
    # Normalize: strip separators the OCR sometimes returns.
    cleaned = re.sub(r"[\s.\-_/\\]", "", raw_text).upper()

    # Step 2: try each grammar as-is
    matched_spec: Optional[GrammarSpec] = None
    for spec in GRAMMARS:
        if spec.regex.match(cleaned):
            matched_spec = spec
            break

    # Step 3: for private / va / trade formats, validate + correct state code
    if matched_spec is not None and matched_spec.name in ("private", "va", "trade"):
        state_code = cleaned[:2]
        if state_code not in STATE_CODES:
            fixed = correct_state_code(state_code, max_distance=1)
            if fixed is not None and fixed != state_code:
                corrected_text = fixed + cleaned[2:]
                # Re-validate the corrected text against the same grammar
                if matched_spec.regex.match(corrected_text):
                    return GrammarResult(
                        text=corrected_text,
                        grammar_valid=True,
                        grammar_kind=matched_spec.name,
                        corrected=True,
                        original_text=original,
                    )
        # State code is valid OR correction was ambiguous — accept as-is
        return GrammarResult(
            text=cleaned,
            grammar_valid=True,
            grammar_kind=matched_spec.name,
            corrected=False,
            original_text=original,
        )

    # Step 4: no grammar matched — try state-code correction to rescue
    if matched_spec is None and len(cleaned) >= 8 and cleaned[:2].isalpha():
        fixed_state = correct_state_code(cleaned[:2], max_distance=1)
        if fixed_state is not None and fixed_state != cleaned[:2]:
            candidate = fixed_state + cleaned[2:]
            for spec in GRAMMARS:
                if spec.regex.match(candidate):
                    return GrammarResult(
                        text=candidate,
                        grammar_valid=True,
                        grammar_kind=spec.name,
                        corrected=True,
                        original_text=original,
                    )

    # Step 5: matched a non-state grammar (bh/diplomatic/temporary/defense)
    # or no match at all after correction attempt
    if matched_spec is not None:
        return GrammarResult(
            text=cleaned,
            grammar_valid=True,
            grammar_kind=matched_spec.name,
            corrected=False,
            original_text=original,
        )
    return GrammarResult(
        text=cleaned,
        grammar_valid=False,
        grammar_kind=None,
        corrected=False,
        original_text=original,
    )
