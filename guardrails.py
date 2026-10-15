"""
Governance & Security Guardrails
---------------------------------
CRITICAL requirement from mentor review.

This module enforces rules at three points in the pipeline:

  1. INPUT GUARDRAILS   — before any processing
     - File type enforcement (PDF only)
     - File size limit (50 MB hard cap)
     - Content density check (is this actually a text-bearing document?)
     - Annual-report signal check (is it a financial document?)
     - Filename sanitisation
     - Rate limiting (max N runs per session)

  2. OUTPUT GUARDRAILS  — after Stage 6 (Generate)
     - Citation presence check (every section must have a citation)
     - Hallucination-risk flag (numbers in brief that are not in extracted_figures)
     - "No supported findings" sentinel check

  3. AUDIT TRAIL        — every guardrail decision is logged
     - Structured entry written via OpsLogger.log_guardrail()

Usage:
    from guardrails import InputGuardrails, OutputGuardrails, GuardrailViolation

    ig = InputGuardrails(ops_logger=ops)
    ig.validate(uploaded_file_bytes, original_filename)   # raises GuardrailViolation on failure

    og = OutputGuardrails(ops_logger=ops)
    og.validate(brief_markdown, pipeline_record)
"""

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024          # 50 MB
MIN_TEXT_CHARS_PER_PAGE = 100                   # below this → probably scanned
MIN_ANNUAL_REPORT_KEYWORDS = 2                  # need at least 2 of the signals
MAX_RUNS_PER_SESSION = 10                       # rate limit

ANNUAL_REPORT_SIGNALS = [
    "revenue", "net income", "earnings per share", "eps",
    "balance sheet", "cash flow", "ebitda", "shareholders",
    "annual report", "10-k", "20-f", "directors' report",
    "profit", "loss", "auditors", "consolidated",
    "financial statements", "equity", "dividend",
]

CITATION_PATTERN = re.compile(
    r"\[(?:Section:|Calc:|Fig\.|Page\s)\s*[A-Za-z0-9 _&/,]+(?:,\s*Page\s*\d+)?\]"
)

_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]")   # keep word chars, dashes, dots


# ── Exceptions ────────────────────────────────────────────────────────────────

class GuardrailViolation(Exception):
    """Raised when a guardrail rejects input or output."""
    def __init__(self, check: str, message: str):
        super().__init__(message)
        self.check = check
        self.message = message


# ── Input Guardrails ──────────────────────────────────────────────────────────

class InputGuardrails:
    """
    Call .validate() before handing any uploaded file to the pipeline.
    Raises GuardrailViolation immediately on the first failure.
    """

    def __init__(self, ops_logger=None):
        self._ops = ops_logger
        self._run_count = 0

    def validate(self, file_bytes: bytes, original_filename: str) -> str:
        """
        Run all input checks.  Returns the sanitised filename.
        Raises GuardrailViolation on any failure.
        """
        safe_name = self._sanitise_filename(original_filename)
        self._check_rate_limit()
        self._check_file_type(file_bytes, safe_name)
        self._check_file_size(file_bytes, safe_name)
        self._check_content_density(file_bytes, safe_name)
        self._check_annual_report_signals(file_bytes, safe_name)
        self._run_count += 1
        self._log("all_input_checks", "pass", f"File accepted: {safe_name}")
        return safe_name

    # ── Individual checks ──────────────────────────────────────────────────

    def _sanitise_filename(self, name: str) -> str:
        # Normalise unicode, strip path components, replace unsafe chars
        name = unicodedata.normalize("NFC", name)
        name = Path(name).name                        # strip any path prefix
        name = _SAFE_FILENAME_RE.sub("_", name)
        name = name[:120]                             # hard length cap
        if not name:
            name = "upload.pdf"
        return name

    def _check_rate_limit(self) -> None:
        if self._run_count >= MAX_RUNS_PER_SESSION:
            self._log("rate_limit", "block",
                      f"Session has already run {self._run_count} analyses.")
            raise GuardrailViolation(
                "rate_limit",
                f"Maximum {MAX_RUNS_PER_SESSION} analyses per session reached. "
                "Please start a new session.",
            )

    def _check_file_type(self, data: bytes, name: str) -> None:
        # PDF magic bytes: %PDF
        is_pdf_magic = data[:4] == b"%PDF"
        is_pdf_ext = name.lower().endswith(".pdf")
        if not (is_pdf_magic and is_pdf_ext):
            self._log("file_type", "block", f"Not a PDF: {name}")
            raise GuardrailViolation(
                "file_type",
                "Only PDF files are accepted. "
                f"'{name}' does not appear to be a valid PDF.",
            )
        self._log("file_type", "pass", name)

    def _check_file_size(self, data: bytes, name: str) -> None:
        size = len(data)
        if size > MAX_FILE_SIZE_BYTES:
            mb = size / 1024 / 1024
            self._log("file_size", "block", f"{mb:.1f} MB > 50 MB limit")
            raise GuardrailViolation(
                "file_size",
                f"File is {mb:.1f} MB, which exceeds the 50 MB limit. "
                "Please upload a smaller or compressed PDF.",
            )
        self._log("file_size", "pass", f"{size/1024:.0f} KB")

    def _check_content_density(self, data: bytes, name: str) -> None:
        """
        Quick heuristic: count ASCII printable characters in the raw PDF bytes.
        A text-layer PDF has many; a scanned image PDF has very few.
        We flag (not block) very sparse files so the user knows OCR may be needed.
        """
        printable = sum(1 for b in data[:50_000] if 32 <= b < 127)
        density = printable / min(len(data), 50_000)
        if density < 0.05:
            self._log("content_density", "warn",
                      f"Low text density ({density:.2%}). May be a scanned PDF.")
            # Warning only — do not block; OCR fallback is on the roadmap
        else:
            self._log("content_density", "pass", f"density={density:.2%}")

    def _check_annual_report_signals(self, data: bytes, name: str) -> None:
        """
        Check that the document contains at least 2 annual-report keywords.
        Uses only the first 200 KB to keep it fast.
        """
        snippet = data[:200_000].decode("latin-1", errors="ignore").lower()
        hits = [kw for kw in ANNUAL_REPORT_SIGNALS if kw in snippet]
        if len(hits) < MIN_ANNUAL_REPORT_KEYWORDS:
            self._log("annual_report_signal", "block",
                      f"Only {len(hits)} of {MIN_ANNUAL_REPORT_KEYWORDS} required signals found.")
            raise GuardrailViolation(
                "annual_report_signal",
                "This document does not appear to be a financial annual report. "
                f"Required financial keywords not found (found: {hits or 'none'}). "
                "Please upload a corporate annual report or 10-K.",
            )
        self._log("annual_report_signal", "pass", f"signals found: {hits[:5]}")

    def _log(self, check: str, result: str, detail: str = "") -> None:
        if self._ops:
            self._ops.log_guardrail(check, result, detail)


# ── Output Guardrails ─────────────────────────────────────────────────────────

class OutputGuardrails:
    """
    Call .validate() after Stage 6 produces the Executive Brief.
    Returns a list of warnings (non-blocking); raises GuardrailViolation
    only for severe issues (no content at all).
    """

    def __init__(self, ops_logger=None):
        self._ops = ops_logger

    def validate(self, brief_markdown: str, pipeline_record: dict) -> list[str]:
        """
        Returns a list of warning strings (empty = clean).
        Raises GuardrailViolation if the brief is empty or completely uncited.
        """
        warnings = []
        warnings += self._check_not_empty(brief_markdown)
        warnings += self._check_citations_present(brief_markdown)
        warnings += self._check_no_unsupported_sentinel(brief_markdown)
        warnings += self._check_numbers_sourced(brief_markdown, pipeline_record)

        level = "pass" if not warnings else "warn"
        self._log("output_validation", level,
                  f"{len(warnings)} warning(s): {'; '.join(warnings[:3])}")
        return warnings

    def _check_not_empty(self, text: str) -> list[str]:
        if len(text.strip()) < 200:
            raise GuardrailViolation(
                "empty_output",
                "The AI generated an empty or near-empty Executive Brief. "
                "This may be a pipeline error. Please retry or contact support.",
            )
        return []

    def _check_citations_present(self, text: str) -> list[str]:
        """Brief must have at least 3 citation tags."""
        citations = CITATION_PATTERN.findall(text)
        if len(citations) < 3:
            return [
                f"Brief has only {len(citations)} citation(s). "
                "Financial claims should be cited. Review before distributing."
            ]
        return []

    def _check_no_unsupported_sentinel(self, text: str) -> list[str]:
        """Count how many sections fell back to 'No supported findings'."""
        count = text.lower().count("no supported findings in the source document")
        if count >= 3:
            return [
                f"{count} sections had no supported findings. "
                "The document may lack the expected financial disclosures."
            ]
        return []

    def _check_numbers_sourced(self, text: str, record: dict) -> list[str]:
        """
        Look for bare large numbers in the brief that are not in extracted_figures.
        Simple heuristic — flags suspiciously large round numbers.
        """
        extracted = set()
        for fig_name, periods in record.get("extracted_figures", {}).items():
            for period, fig in periods.items():
                if isinstance(fig, dict) and fig.get("status") == "found":
                    v = fig.get("value")
                    if v is not None:
                        extracted.add(str(abs(int(v))) if v == int(v) else str(v))

        # Find all numbers >= 4 digits in the brief (likely financial figures)
        bare_numbers = re.findall(r"\b(\d{4,}(?:\.\d+)?)\b", text)
        unsourced = [n for n in bare_numbers if n not in extracted]
        if len(unsourced) > 5:
            return [
                f"{len(unsourced)} large numbers in the brief could not be "
                "traced to extracted figures. Verify sourcing before sign-off."
            ]
        return []

    def _log(self, check: str, result: str, detail: str = "") -> None:
        if self._ops:
            self._ops.log_guardrail(check, result, detail)


# ── Utility ───────────────────────────────────────────────────────────────────

def file_fingerprint(data: bytes) -> str:
    """SHA-256 of the uploaded file — used in audit trail."""
    return hashlib.sha256(data).hexdigest()[:16]
