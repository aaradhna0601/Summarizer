"""
GenAI Ops Logger
----------------
Logs every LLM call to a structured JSONL file.
Satisfies the mentor requirement for a GenAI Ops framework:
  - Which prompt version was used
  - How many tokens were consumed (in + out)
  - Latency per stage
  - Model used
  - Estimated cost
  - Run ID for end-to-end tracing

All logs go to:  logs/genai_ops.jsonl
Summary stats are readable with OpsLogger.session_summary()

Usage:
    logger = OpsLogger(run_id="abc123")
    with logger.trace(stage=1, prompt_version="1.0", model="gpt-4o") as span:
        response = openai_client.chat.completions.create(...)
        span.record_response(response)
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

std_logger = logging.getLogger(__name__)

# Rough cost table (USD per 1K tokens) — update when OpenAI prices change
_COST_TABLE = {
    "gpt-4o":              {"input": 0.0025,  "output": 0.010},
    "gpt-4o-mini":         {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":         {"input": 0.010,   "output": 0.030},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "genai_ops.jsonl"


@dataclass
class Span:
    """Mutable context object populated during an LLM call."""
    run_id: str
    stage: int
    prompt_id: str
    prompt_version: str
    model: str
    start_ts: float = field(default_factory=time.perf_counter)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    status: str = "pending"   # pending | ok | error
    error: str = ""
    extra: dict = field(default_factory=dict)

    def record_response(self, response) -> None:
        """Extract token counts from an OpenAI ChatCompletion response."""
        usage = getattr(response, "usage", None)
        if usage:
            self.tokens_in = getattr(usage, "prompt_tokens", 0)
            self.tokens_out = getattr(usage, "completion_tokens", 0)
        self._compute_cost()
        self.status = "ok"

    def record_embedding(self, response) -> None:
        """Extract token counts from an OpenAI Embedding response."""
        usage = getattr(response, "usage", None)
        if usage:
            self.tokens_in = getattr(usage, "total_tokens", 0)
            self.tokens_out = 0
        self._compute_cost()
        self.status = "ok"

    def _compute_cost(self) -> None:
        table = _COST_TABLE.get(self.model, {"input": 0.0, "output": 0.0})
        self.cost_usd = (
            self.tokens_in / 1000 * table["input"]
            + self.tokens_out / 1000 * table["output"]
        )


class OpsLogger:
    """
    Thread-safe structured logger for GenAI operations.
    Each pipeline run creates one OpsLogger instance.
    """

    def __init__(self, run_id: str | None = None, log_file: Path | str | None = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.log_file = Path(log_file) if log_file else _LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._spans: list[Span] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def trace(
        self,
        stage: int,
        prompt_id: str = "",
        prompt_version: str = "",
        model: str = "gpt-4o",
        extra: dict | None = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager.  Yield a Span; write the log entry on exit.

            with ops.trace(stage=6, prompt_id="stage_6_generate",
                           prompt_version="1.0", model="gpt-4o") as span:
                resp = client.chat.completions.create(...)
                span.record_response(resp)
        """
        span = Span(
            run_id=self.run_id,
            stage=stage,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            model=model,
            extra=extra or {},
        )
        t0 = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = str(exc)[:500]
            raise
        finally:
            span.latency_s = round(time.perf_counter() - t0, 3)
            self._write(span)
            self._spans.append(span)

    def log_guardrail(self, check: str, result: str, detail: str = "") -> None:
        """Log a security / governance guardrail decision."""
        entry = {
            "event": "guardrail",
            "run_id": self.run_id,
            "ts": _ts(),
            "check": check,
            "result": result,
            "detail": detail[:300],
        }
        self._append(entry)

    def session_summary(self) -> dict:
        """Return total tokens, cost, and latency for this run."""
        if not self._spans:
            return {}
        return {
            "run_id": self.run_id,
            "stages_completed": [s.stage for s in self._spans if s.status == "ok"],
            "stages_failed": [s.stage for s in self._spans if s.status == "error"],
            "total_tokens_in": sum(s.tokens_in for s in self._spans),
            "total_tokens_out": sum(s.tokens_out for s in self._spans),
            "total_cost_usd": round(sum(s.cost_usd for s in self._spans), 6),
            "total_latency_s": round(sum(s.latency_s for s in self._spans), 2),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, span: Span) -> None:
        entry = {
            "event": "llm_call",
            "run_id": span.run_id,
            "ts": _ts(),
            "stage": span.stage,
            "prompt_id": span.prompt_id,
            "prompt_version": span.prompt_version,
            "model": span.model,
            "tokens_in": span.tokens_in,
            "tokens_out": span.tokens_out,
            "latency_s": span.latency_s,
            "cost_usd": span.cost_usd,
            "status": span.status,
            "error": span.error or None,
            **span.extra,
        }
        self._append(entry)
        std_logger.info(
            "Stage %s | %s tokens | %.2fs | $%.5f | %s",
            span.stage, span.tokens_in + span.tokens_out,
            span.latency_s, span.cost_usd, span.status,
        )

    def _append(self, entry: dict) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            std_logger.warning("OpsLogger: could not write log entry: %s", exc)


def _ts() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
