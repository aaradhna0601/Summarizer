"""
Pipeline Orchestrator
---------------------
Runs all 7 stages of the Executive Brief pipeline in sequence.
Each stage has explicit error handling, status tracking, and structured logging.

This is the single entry point for the Streamlit app — it replaces
any ad-hoc function calls scattered across app.py.

Architecture
------------
  InputGuardrails → Stage1 → Stage2 → Stage3(Python) →
  Stage4 → Stage5 → Stage6 → OutputGuardrails → Stage7

Usage:
    orch = PipelineOrchestrator(openai_api_key="sk-...")
    result = orch.run(pdf_bytes=..., original_filename="reliance_2025.pdf")
    print(result.brief_markdown)   # the Executive Brief
    print(result.ops_summary)      # tokens, cost, latency
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import openai

from calc_engine import run as calc_run, rule_flags
from guardrails import InputGuardrails, OutputGuardrails, GuardrailViolation, file_fingerprint
from ops_logger import OpsLogger
from prompt_manager import PromptManager

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    run_id: str
    status: str                    # "ok" | "failed" | "guardrail_blocked"
    brief_markdown: str = ""
    stage_statuses: dict = field(default_factory=dict)
    output_warnings: list = field(default_factory=list)
    review_flags: list = field(default_factory=list)
    ops_summary: dict = field(default_factory=dict)
    error: str = ""
    error_stage: int | None = None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Instantiate once at app startup (it validates prompts on __init__).
    Call .run() for each uploaded document.
    """

    def __init__(
        self,
        openai_api_key: str,
        prompt_version: str = "v1",
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        max_chunk_chars: int = 3000,
    ):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.max_chunk_chars = max_chunk_chars
        self.pm = PromptManager(version=prompt_version)

        logger.info("PipelineOrchestrator ready | model=%s | prompts=%s",
                    model, self.pm.all_versions())

    # ── Public entry point ────────────────────────────────────────────────

    def run(self, pdf_bytes: bytes, original_filename: str) -> PipelineResult:
        run_id = str(uuid.uuid4())[:8]
        ops = OpsLogger(run_id=run_id)
        stage_statuses: dict[int, str] = {}

        # ── Input guardrails ──────────────────────────────────────────────
        ig = InputGuardrails(ops_logger=ops)
        try:
            safe_filename = ig.validate(pdf_bytes, original_filename)
        except GuardrailViolation as exc:
            return PipelineResult(
                run_id=run_id,
                status="guardrail_blocked",
                error=exc.message,
                ops_summary=ops.session_summary(),
            )

        fingerprint = file_fingerprint(pdf_bytes)
        logger.info("RUN %s | file=%s | sha256=%s", run_id, safe_filename, fingerprint)

        rec: dict[str, Any] = {
            "run_id": run_id,
            "file_name": safe_filename,
            "file_fingerprint": fingerprint,
            "extracted_figures": {},
            "risks": [],
            "review_flags": [],
        }

        # ── Extract raw text ──────────────────────────────────────────────
        try:
            raw_text, page_count = self._extract_pdf_text(pdf_bytes)
            chunks = self._chunk_text(raw_text)
            logger.info("RUN %s | pages=%d | chunks=%d", run_id, page_count, len(chunks))
        except Exception as exc:
            return self._fail(run_id, 0, str(exc), stage_statuses, ops)

        # ── Stage 1: Ground ───────────────────────────────────────────────
        try:
            stage_1_out = self._stage_1(ops, chunks)
            rec["section_map"] = stage_1_out.get("section_map", [])
            rec["queries"] = stage_1_out.get("queries", [])
            stage_statuses[1] = "ok"
        except Exception as exc:
            return self._fail(run_id, 1, str(exc), stage_statuses, ops)

        # ── Stage 2: Extract ──────────────────────────────────────────────
        try:
            retrieved = self._retrieve(chunks, rec["queries"])
            stage_2_out = self._stage_2(ops, retrieved)
            rec.update(stage_2_out)
            stage_statuses[2] = "ok"
        except Exception as exc:
            return self._fail(run_id, 2, str(exc), stage_statuses, ops)

        # ── Stage 3: Calculate (Python only) ─────────────────────────────
        try:
            rec = calc_run(rec)
            stage_statuses[3] = "ok"
        except Exception as exc:
            return self._fail(run_id, 3, str(exc), stage_statuses, ops)

        # ── Stage 4: Analyse ──────────────────────────────────────────────
        try:
            stage_4_out = self._stage_4(ops, rec, retrieved)
            rec["qualitative_findings"] = stage_4_out.get("qualitative_findings", [])
            rec["risks"] = stage_4_out.get("risks", [])
            stage_statuses[4] = "ok"
        except Exception as exc:
            return self._fail(run_id, 4, str(exc), stage_statuses, ops)

        # ── Stage 5: Verify ───────────────────────────────────────────────
        try:
            stage_5_out = self._stage_5(ops, rec)
            rec["verification"] = stage_5_out.get("verification", [])
            stage_statuses[5] = "ok"
        except Exception as exc:
            # Verification failure is non-fatal — continue with unverified brief
            logger.warning("RUN %s | Stage 5 failed: %s — continuing unverified", run_id, exc)
            rec["verification"] = []
            stage_statuses[5] = "skipped"

        # ── Stage 6: Generate ─────────────────────────────────────────────
        try:
            brief_markdown = self._stage_6(ops, rec)
            if not brief_markdown or len(brief_markdown.strip()) < 100:
                raise ValueError(
                    "Stage 6 returned an empty or near-empty brief. "
                    "Possible cause: token limit exceeded or malformed prompt output."
                )
            stage_statuses[6] = "ok"
        except Exception as exc:
            return self._fail(run_id, 6, str(exc), stage_statuses, ops)

        # ── Output guardrails ─────────────────────────────────────────────
        og = OutputGuardrails(ops_logger=ops)
        try:
            output_warnings = og.validate(brief_markdown, rec)
        except GuardrailViolation as exc:
            return PipelineResult(
                run_id=run_id,
                status="guardrail_blocked",
                brief_markdown=brief_markdown,
                stage_statuses=stage_statuses,
                error=exc.message,
                ops_summary=ops.session_summary(),
            )

        # ── Stage 7: Review ───────────────────────────────────────────────
        try:
            rec = rule_flags(rec)                       # Python rule flags
            stage_7_out = self._stage_7(ops, rec, brief_markdown)
            llm_flags = stage_7_out.get("review_flags", [])
            rec["review_flags"] = rec["review_flags"] + llm_flags
            stage_statuses[7] = "ok"
        except Exception as exc:
            logger.warning("RUN %s | Stage 7 failed: %s — brief still delivered", run_id, exc)
            stage_statuses[7] = "skipped"

        summary = ops.session_summary()
        logger.info("RUN %s | COMPLETE | cost=$%.4f | latency=%.1fs",
                    run_id, summary.get("total_cost_usd", 0), summary.get("total_latency_s", 0))

        return PipelineResult(
            run_id=run_id,
            status="ok",
            brief_markdown=brief_markdown,
            stage_statuses=stage_statuses,
            output_warnings=output_warnings,
            review_flags=rec.get("review_flags", []),
            ops_summary=summary,
        )

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_1(self, ops: OpsLogger, chunks: list[str]) -> dict:
        system, user = self.pm.render(1, {"document_chunks": "\n\n".join(chunks[:30])})
        meta = self.pm.get_meta(1)
        with ops.trace(1, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 2000))
            span.record_response(resp)
        return self._parse_json(resp, stage=1)

    def _stage_2(self, ops: OpsLogger, retrieved: str) -> dict:
        system, user = self.pm.render(2, {"retrieved_passages": retrieved})
        meta = self.pm.get_meta(2)
        with ops.trace(2, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 4000))
            span.record_response(resp)
        return self._parse_json(resp, stage=2)

    def _stage_4(self, ops: OpsLogger, rec: dict, retrieved: str) -> dict:
        system, user = self.pm.render(4, {
            "extracted_figures": json.dumps(rec.get("extracted_figures", {}), indent=2),
            "calculated_metrics": json.dumps(rec.get("calculated_metrics", {}), indent=2),
            "retrieved_narrative": retrieved,
        })
        meta = self.pm.get_meta(4)
        with ops.trace(4, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 4000))
            span.record_response(resp)
        return self._parse_json(resp, stage=4)

    def _stage_5(self, ops: OpsLogger, rec: dict) -> dict:
        system, user = self.pm.render(5, {
            "qualitative_findings": json.dumps(rec.get("qualitative_findings", []), indent=2),
            "risks": json.dumps(rec.get("risks", []), indent=2),
            "source_registry": json.dumps(rec.get("source_registry", {}), indent=2),
            "extracted_figures": json.dumps(rec.get("extracted_figures", {}), indent=2),
            "calculated_metrics": json.dumps(rec.get("calculated_metrics", {}), indent=2),
        })
        meta = self.pm.get_meta(5)
        with ops.trace(5, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 3000))
            span.record_response(resp)
        return self._parse_json(resp, stage=5)

    def _stage_6(self, ops: OpsLogger, rec: dict) -> str:
        """Returns Markdown string, not JSON."""
        open_flags = sum(1 for f in rec.get("review_flags", []) if f.get("analyst_decision") == "pending")
        unsupported = sum(1 for v in rec.get("verification", []) if v.get("status") == "unsupported")

        system, user = self.pm.render(6, {
            "verification": json.dumps(rec.get("verification", []), indent=2),
            "extracted_figures": json.dumps(rec.get("extracted_figures", {}), indent=2),
            "calculated_metrics": json.dumps(rec.get("calculated_metrics", {}), indent=2),
            "document_meta": json.dumps(rec.get("document_meta", {}), indent=2),
            "risks": json.dumps(rec.get("risks", []), indent=2),
            # Template substitution helpers
            "company": rec.get("document_meta", {}).get("company_name", "Unknown Company"),
            "document_type": rec.get("document_meta", {}).get("document_type", "Annual Report"),
            "current_period": rec.get("document_meta", {}).get("current_period", "FY"),
            "prior_period": rec.get("document_meta", {}).get("prior_period", "Prior FY"),
            "ccy": rec.get("document_meta", {}).get("currency", ""),
            "scale": rec.get("document_meta", {}).get("reporting_scale", ""),
            "accounting_standard": rec.get("document_meta", {}).get("accounting_standard", ""),
            "today": time.strftime("%Y-%m-%d"),
            "flag_count": str(open_flags),
            "unsupported_count": str(unsupported),
            "file_name": rec.get("file_name", ""),
        })
        meta = self.pm.get_meta(6)
        with ops.trace(6, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 3000))
            span.record_response(resp)
        # Stage 6 returns Markdown — strip any accidental code fences
        content = resp.choices[0].message.content or ""
        content = re.sub(r"^```(?:markdown)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content)
        return content.strip()

    def _stage_7(self, ops: OpsLogger, rec: dict, brief: str) -> dict:
        system, user = self.pm.render(7, {
            "brief_draft": brief,
            "qualitative_findings": json.dumps(rec.get("qualitative_findings", []), indent=2),
            "risks": json.dumps(rec.get("risks", []), indent=2),
            "calculated_metrics": json.dumps(rec.get("calculated_metrics", {}), indent=2),
        })
        meta = self.pm.get_meta(7)
        with ops.trace(7, meta["prompt_id"], meta["version"], self.model) as span:
            resp = self._llm(system, user, max_tokens=meta.get("max_tokens", 1500))
            span.record_response(resp)
        return self._parse_json(resp, stage=7)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _llm(self, system: str, user: str, max_tokens: int = 2000):
        return self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    def _parse_json(self, response, stage: int) -> dict:
        """Parse JSON from an LLM response; raises ValueError with a clear message on failure."""
        content = response.choices[0].message.content or ""
        # Strip markdown code fences if the model wrapped its JSON
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stage {stage} returned invalid JSON. "
                f"Parse error: {exc}. "
                f"First 300 chars: {content[:300]!r}"
            ) from exc

    def _extract_pdf_text(self, pdf_bytes: bytes) -> tuple[str, int]:
        """Extract text from PDF bytes. Returns (full_text, page_count)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text()
                pages.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(pages), len(doc)
        except ImportError:
            pass

        try:
            import pdfplumber
            import io
            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append(f"[Page {i+1}]\n{text}")
            return "\n\n".join(pages), len(pages)
        except ImportError:
            pass

        raise ImportError(
            "No PDF library available. "
            "Please add 'PyMuPDF' or 'pdfplumber' to requirements.txt."
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of roughly max_chunk_chars characters."""
        words = text.split()
        chunks, current, length = [], [], 0
        for word in words:
            current.append(word)
            length += len(word) + 1
            if length >= self.max_chunk_chars:
                chunks.append(" ".join(current))
                current, length = [], 0
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _retrieve(self, chunks: list[str], queries: list[dict]) -> str:
        """
        Simple keyword-based retrieval — returns the top 20 most relevant chunks
        as a single string. For production, replace with FAISS vector search.
        """
        if not queries:
            return "\n\n".join(chunks[:20])

        query_text = " ".join(q.get("query", "") for q in queries).lower()
        keywords = set(query_text.split())

        scored = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for kw in keywords if kw in chunk_lower)
            scored.append((score, i, chunk))

        scored.sort(key=lambda x: -x[0])
        top = [c for _, _, c in scored[:20]]
        return "\n\n---\n\n".join(top)

    @staticmethod
    def _fail(run_id, stage, error, stage_statuses, ops) -> PipelineResult:
        stage_statuses[stage] = "error"
        logger.error("RUN %s | Stage %s FAILED: %s", run_id, stage, error)
        return PipelineResult(
            run_id=run_id,
            status="failed",
            stage_statuses=stage_statuses,
            error=f"Stage {stage} failed: {error}",
            error_stage=stage,
            ops_summary=ops.session_summary(),
        )
