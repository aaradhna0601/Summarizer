"""Local, dependency-light web interface. Start with `python app.py`."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from engine import analyze
from export import create_pdf

HERE = Path(__file__).resolve().parent
PAGE = (HERE / "index.html").read_bytes()
DOCUMENTS: dict[str, tuple[dict, bytes]] = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str, filename: str | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        bits = route.strip("/").split("/")
        if len(bits) == 2 and bits[0] in ("pdf", "json") and bits[1] in DOCUMENTS:
            result, pdf = DOCUMENTS[bits[1]]
            if bits[0] == "pdf":
                return self._send(200, pdf, "application/pdf", "executive-brief.pdf")
            return self._send(200, json.dumps(result, indent=2).encode(), "application/json", "evidence-and-analysis.json")
        return self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, b"Not found", "text/plain")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 30_000_000:
                raise ValueError("Upload a PDF smaller than 30 MB.")
            filename = self.headers.get("X-Report-Name", "Annual report.pdf")
            filename = Path(filename).name[:100].replace('"', "")
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF reports are accepted.")
            report = analyze(self.rfile.read(size), filename, self.headers.get("X-Use-AI") == "true")
            pdf = create_pdf(report)
            ident = uuid4().hex
            if len(DOCUMENTS) >= 20:
                DOCUMENTS.pop(next(iter(DOCUMENTS)))
            DOCUMENTS[ident] = (report, pdf)
            cited_ids = {item["evidence_id"] for section in ("positives", "attention", "risks", "outlook", "questions", "watchlist") for item in report[section]}
            cited_ids.update(m["evidence_id"] for m in report["metrics"].values())
            report["cited_evidence"] = {eid: report["evidence"][eid] for eid in cited_ids}
            report = {k: v for k, v in report.items() if k != "evidence"}
            report["id"] = ident
            return self._send(200, json.dumps(report).encode(), "application/json")
        except Exception as exc:
            return self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    print(f"Annual Brief running at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
