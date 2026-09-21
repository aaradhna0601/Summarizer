"""Single-page A4 brief with source references."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import reportlab
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_DIR = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("Brief", str(FONT_DIR / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Brief-Bold", str(FONT_DIR / "VeraBd.ttf")))

INK = HexColor("#19324D")
MUTED = HexColor("#536879")
ACCENT = HexColor("#067C80")
PALE = HexColor("#EAF4F3")
RULE = HexColor("#D8E3E8")


def _wrap(s: str, max_width: float, font_size: int = 8) -> list[str]:
    words = str(s).replace("\n", " ").split()
    lines, line = [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if stringWidth(candidate, "Brief", font_size) > max_width and line:
            lines.append(line); line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def create_pdf(report: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin, gap = 34, 16
    colw = (w - 2 * margin - gap) / 2
    c.setFillColor(INK); c.rect(0, h - 96, w, 96, stroke=0, fill=1)
    c.setFillColor(HexColor("#A8DED8")); c.setFont("Brief-Bold", 9)
    c.drawString(margin, h - 35, "ANNUAL REPORT | EXECUTIVE BRIEF")
    c.setFillColor(HexColor("#FFFFFF")); c.setFont("Brief-Bold", 17)
    c.drawString(margin, h - 59, report["company"][:57])
    c.setFont("Brief", 8); c.drawString(margin, h - 77, f"{report['pages']} source pages  •  {report['mode']}")

    c.setFillColor(PALE); c.roundRect(margin, h - 168, w - 2 * margin, 58, 7, stroke=0, fill=1)
    order = ["Revenue", "Operating profit", "Net profit", "Operating cash flow", "Cash", "Borrowings"]
    metrics = report["metrics"]
    cell = (w - 2 * margin - 18) / 6
    for i, name in enumerate(order):
        x = margin + 9 + i * cell
        m = metrics.get(name)
        c.setFillColor(MUTED); c.setFont("Brief", 7); c.drawString(x, h - 130, name[:21])
        c.setFillColor(INK); c.setFont("Brief-Bold", 11)
        value = f"{m['current']:,.1f}" if m else "N/A"
        c.drawString(x, h - 147, value[:12])
        c.setFont("Brief", 6); c.setFillColor(MUTED)
        c.drawString(x, h - 158, f"p.{m['page']} | {m['unit'][:13]}" if m else "not found")

    def section(x: float, top: float, label: str, items: list, height: float, risk: bool = False) -> None:
        c.setFillColor(ACCENT); c.setFont("Brief-Bold", 9); c.drawString(x, top, label.upper())
        c.setStrokeColor(RULE); c.line(x, top - 5, x + colw, top - 5)
        y = top - 20
        if not items:
            c.setFillColor(MUTED); c.setFont("Brief", 8)
            c.drawString(x, y, "No supported item found in extracted text.")
            return
        for it in items[:3]:
            statement = f"{it['category']}: {it['signal']}" if risk else it["text"]
            suffix = f"  [p.{it['page']}]"
            lines = _wrap(statement + suffix, colw - 13)
            available = int((y - (top - height)) / 10)
            if available <= 0:
                break
            shown = lines[:min(3, available)]
            if len(shown) < len(lines):
                shown[-1] = shown[-1][:54].rstrip() + "..."
            c.setFillColor(ACCENT); c.circle(x + 3, y + 2, 2, stroke=0, fill=1)
            c.setFillColor(INK); c.setFont("Brief", 8)
            for line in shown:
                c.drawString(x + 12, y, line)
                y -= 10
            y -= 9

    left = margin; right = margin + colw + gap
    section(left, h - 192, "What went well", report["positives"], 143)
    section(left, h - 338, "Needs attention", report["attention"], 143)
    section(left, h - 484, "Management outlook", report["outlook"], 112)
    section(left, h - 600, "Watchlist", report["watchlist"], 96)
    section(right, h - 192, "Top risks / signals", report["risks"], 240, True)
    section(right, h - 435, "Decision / diligence questions", report["questions"], 190)
    c.setStrokeColor(RULE); c.line(margin, 90, w - margin, 90)
    c.setFillColor(MUTED); c.setFont("Brief", 7)
    for i, line in enumerate(_wrap("Review required: figures and year columns are extracted candidates; citations identify PDF pages, not assurance of factual accuracy. No investment recommendation.", w - 2 * margin, 7)[:2]):
        c.drawString(margin, 75 - 9 * i, line)
    c.drawString(margin, 44, "Source: " + report["source"][:88])
    c.drawRightString(w - margin, 44, "Page 1 / 1")
    c.showPage(); c.save()
    return buf.getvalue()
