"""Source-backed annual report extraction and bounded analysis.

The offline mode is deliberately extractive: absence of evidence becomes N/A.
It is a teaching prototype, not an investment or audit system.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

import fitz


@dataclass
class Evidence:
    id: str
    page: int
    text: str
    section: str


@dataclass
class Metric:
    label: str
    current: float
    previous: float | None
    unit: str
    evidence_id: str
    page: int
    row: str
    change_pct: float | None


ALIASES = {
    "Revenue": [r"revenue from operations", r"total revenue", r"net sales"],
    "Operating profit": [r"operating profit", r"ebitda"],
    "Net profit": [r"profit after tax", r"net profit", r"profit for the year"],
    "Operating cash flow": [r"net cash (?:generated from|from|provided by) operating activities", r"cash flow from operations"],
    "Cash": [r"cash and cash equivalents"],
    "Borrowings": [r"total borrowings", r"total debt", r"borrowings"],
    "EPS": [r"(?:basic )?earnings per share", r"basic eps"],
}
NUM = re.compile(r"(?<![A-Za-z\d])\(?-?\d[\d,]*(?:\.\d+)?\)?(?![A-Za-z\d])")
RISK_TERMS = {
    "Liquidity / credit": r"liquidity|refinanc|covenant|credit risk|working capital",
    "Operations": r"supply chain|disruption|capacity constraint|raw material|operational risk",
    "Regulatory / legal": r"litigation|regulatory|compliance|legal proceeding",
    "Market / strategic": r"competition|demand slowdown|foreign exchange|market risk|inflation",
    "Governance / technology": r"cyber|data breach|internal control|governance",
    "ESG": r"climate risk|emission|water stress|environmental risk",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _number(s: str) -> float:
    return float(s.replace(",", "").replace("(", "-").replace(")", ""))


def _unit(text: str) -> str:
    m = re.search(r"(?:₹|Rs\.?|INR|USD|US\$|\$)\s*(?:in\s+)?(crore|lakh|million|billion)s?", text, re.I)
    if m:
        currency = "INR" if m.group(0).lower().startswith(("₹", "rs", "inr")) else "USD"
        return f"{currency} {m.group(1).lower()}"
    return "report units (check)"


def _section(lines: list[str], index: int) -> str:
    for line in reversed(lines[max(0, index - 12):index]):
        if re.search(r"consolidated|standalone|cash flow|balance sheet|profit and loss|financial highlights|risk|management", line, re.I):
            return _clean(line)[:90]
    return "Annual report"


def read_pdf(data: bytes) -> tuple[list[Evidence], list[str]]:
    """Extract page numbered lines. Never silently OCR an image-only PDF."""
    if not data.startswith(b"%PDF-") or len(data) > 30_000_000:
        raise ValueError("Upload a PDF smaller than 30 MB.")
    doc = fitz.open(stream=data, filetype="pdf")
    if doc.needs_pass:
        raise ValueError("Password-protected PDFs are not supported.")
    if not 1 <= len(doc) <= 500:
        raise ValueError("The PDF must have 1 to 500 pages.")
    evidence: list[Evidence] = []
    pages: list[str] = []
    for p, page in enumerate(doc, 1):
        text = page.get_text("text", sort=True)
        pages.append(text)
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        # Line-by-line evidence keeps page references stable and audit friendly.
        for i, line in enumerate(lines):
            if len(line) < 8:
                continue
            evidence.append(Evidence(f"P{p}L{i+1}", p, _clean(line), _section(lines, i)))
    doc.close()
    if sum(len(p.strip()) for p in pages) < 100:
        raise ValueError("No usable text was found. Please use a searchable PDF; scanned reports require OCR.")
    return evidence, pages


def _metric_candidates(evidence: list[Evidence], pages: list[str]) -> dict[str, Metric]:
    found: dict[str, Metric] = {}
    for label, aliases in ALIASES.items():
        candidates: list[tuple[int, Metric]] = []
        for e in evidence:
            match = next((re.match(r"^\s*(?:\d+[.)]\s*)?(?:" + alias + r")\b\s*[:\-–]?\s*", e.text, re.I) for alias in aliases if re.match(r"^\s*(?:\d+[.)]\s*)?(?:" + alias + r")\b\s*[:\-–]?\s*", e.text, re.I)), None)
            if not match:
                continue
            tail = e.text[match.end():]
            vals = NUM.findall(tail)
            # Require a financial statement style row. Years embedded in a label are excluded.
            if not vals or len(vals) > 4:
                continue
            current = _number(vals[0]); previous = _number(vals[1]) if len(vals) > 1 else None
            if abs(current) > 1e12 or (previous is not None and abs(previous) > 1e12):
                continue
            unit = "per share" if label == "EPS" else _unit(pages[e.page - 1][:3500])
            change = round((current - previous) / abs(previous) * 100, 1) if previous not in (None, 0) else None
            metric = Metric(label, current, previous, unit, e.id, e.page, e.text[:200], change)
            section = e.section.lower()
            score = 3 if "consolidated" in section else 2 if "financial highlights" in section else 1
            score += 1 if previous is not None else 0
            score -= 2 if "segment" in section or "standalone" in section else 0
            candidates.append((score, metric))
        if candidates:
            found[label] = max(candidates, key=lambda x: x[0])[1]
    return found


def _narratives(evidence: list[Evidence], pattern: str, limit: int = 3) -> list[dict[str, Any]]:
    result = []
    for e in evidence:
        if re.search(pattern, e.text, re.I) and len(e.text) >= 35 and not re.search(r"^\s*(?:table of contents|index)\b", e.text, re.I):
            # Prefer complete disclosure lines, retain the exact excerpt for checking.
            result.append({"text": e.text[:195], "evidence_id": e.id, "page": e.page, "kind": "Report excerpt"})
        if len(result) >= limit:
            break
    return result


def _item(text: str, metric: Metric, kind: str = "Calculated trend") -> dict[str, Any]:
    return {"text": text, "evidence_id": metric.evidence_id, "page": metric.page, "kind": kind}


def _fmt(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def analyze(data: bytes, filename: str, use_ai: bool = False) -> dict[str, Any]:
    evidence, pages = read_pdf(data)
    metrics = _metric_candidates(evidence, pages)
    first_lines = [x.strip() for x in pages[0].splitlines() if x.strip()]
    candidate = next((x for x in first_lines[:8] if re.search(r"limited|ltd\.?|inc\.?|corporation|corp\.?|plc", x, re.I)), "")
    title = (candidate.split(" - ")[0] if candidate else filename.rsplit(".", 1)[0].replace("_", " "))[:75]
    # Deterministic calculations; narrative lines are visibly labeled as excerpts.
    positives: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    for label in ("Revenue", "Operating profit", "Net profit", "Operating cash flow"):
        m = metrics.get(label)
        if m and m.change_pct is not None and abs(m.change_pct) >= 2:
            wording = f"{label} {'rose' if m.change_pct > 0 else 'fell'} {abs(m.change_pct):.1f}% vs prior period ({_fmt(m.current)} vs {_fmt(m.previous)} {m.unit})."
            (positives if m.change_pct > 0 else attention).append(_item(wording, m))
    debt = metrics.get("Borrowings")
    if debt and debt.change_pct is not None and abs(debt.change_pct) >= 10:
        (attention if debt.change_pct > 0 else positives).append(_item(f"Borrowings {'rose' if debt.change_pct > 0 else 'fell'} {abs(debt.change_pct):.1f}% vs prior period.", debt))
    if not positives:
        positives = _narratives(evidence, r"growth|improv|record|expanded|strong performance", 2)
    if not attention:
        attention = _narratives(evidence, r"declin|pressur|challeng|weaker|adverse", 2)

    risks: list[dict[str, Any]] = []
    if debt and debt.change_pct is not None and debt.change_pct >= 10:
        risks.append({"category": "Liquidity / credit", "signal": "Borrowings increased vs prior period", "severity": "Review", "evidence_id": debt.evidence_id, "page": debt.page})
    for category, pattern in RISK_TERMS.items():
        if len(risks) >= 3:
            break
        for e in evidence:
            if category not in {x["category"] for x in risks} and re.search(pattern, e.text, re.I) and len(e.text) >= 35 and e.id not in {x["evidence_id"] for x in risks}:
                risks.append({"category": category, "signal": e.text[:130], "severity": "Review", "evidence_id": e.id, "page": e.page})
                break
    outlook = _narratives(evidence, r"outlook|guidance|expect(?:s|ed)?|plan to|strategy|strategic priorit", 2)
    questions = []
    for r in risks[:3]:
        questions.append({"text": f"How is management monitoring {r['category'].lower()} exposure?", "evidence_id": r["evidence_id"], "page": r["page"], "kind": "AI/analyst question"})
    if not questions and metrics:
        m = next(iter(metrics.values()))
        questions = [_item(f"What explains the movement in {m.label.lower()}?", m, "Analyst question")]
    watch = [_item(f"Track next-period {m.label.lower()} and the cited disclosure.", m, "Analyst watchlist") for m in list(metrics.values())[:2]]
    result: dict[str, Any] = {
        "company": title, "source": filename, "pages": len(pages), "mode": "Offline evidence mode",
        "metrics": {k: asdict(v) for k, v in metrics.items()}, "positives": positives[:3],
        "attention": attention[:3], "risks": risks[:3], "outlook": outlook[:2],
        "questions": questions[:3], "watchlist": watch[:2],
        "warnings": ["Financial rows are candidates and require analyst verification, especially units, scope (consolidated vs standalone), and column year order."],
        "evidence": {e.id: asdict(e) for e in evidence},
    }
    if not metrics:
        result["warnings"].append("No financial statement rows matched the supported labels; metrics are unavailable.")
    if use_ai:
        result = _ai_enrich(result, evidence)
    return result


def _ai_enrich(result: dict[str, Any], evidence: list[Evidence]) -> dict[str, Any]:
    """Optional OpenAI-compatible chat endpoint; rejects unknown citation identifiers."""
    key = os.getenv("AI_API_KEY")
    model = os.getenv("AI_MODEL")
    endpoint = os.getenv("AI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions")
    if not key or not model:
        raise ValueError("AI mode needs AI_API_KEY and AI_MODEL environment variables.")
    # Bounded, relevant retrieval by exact phrase matching; not a full vector index.
    chosen = []
    terms = r"risk|challeng|growth|margin|outlook|guidance|cash flow|borrow|litigation|competition|segment"
    for e in evidence:
        if re.search(terms, e.text, re.I) and 25 <= len(e.text) <= 500:
            chosen.append(e)
        if len(chosen) == 60:
            break
    source = {e.id: e for e in chosen}
    prompt = {
        "metrics": result["metrics"],
        "retrieved_passages": [{"id": e.id, "page": e.page, "text": e.text} for e in chosen],
        "task": "Return JSON object with keys positives, attention, outlook, each arrays of up to 3 objects {text,evidence_id}. Use only supplied evidence ids and never compute figures. Keep interpretations explicitly tentative. Skip unsupported points.",
    }
    body = json.dumps({"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": "You are a cautious financial disclosure analyst. Only use the provided source passages; no investment recommendations. Return only valid JSON."},
        {"role": "user", "content": json.dumps(prompt)},
    ]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            raw = json.loads(response.read(1_000_000))
        content = raw["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        proposed = json.loads(content)
    except Exception as exc:
        raise ValueError(f"AI request failed: {type(exc).__name__}. Offline mode remains available.") from exc
    for section in ("positives", "attention", "outlook"):
        accepted = []
        for item in proposed.get(section, [])[:3]:
            if not isinstance(item, dict) or item.get("evidence_id") not in source or not isinstance(item.get("text"), str):
                continue
            e = source[item["evidence_id"]]
            accepted.append({"text": item["text"][:225], "evidence_id": e.id, "page": e.page, "kind": "AI interpretation - verify"})
        if accepted:
            result[section] = accepted
    result["mode"] = "AI-assisted (human review required)"
    result["warnings"].append("AI citations identify source passages, but do not establish that every interpretation follows from them. Review before use.")
    return result
