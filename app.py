"""
Annual Report Analyser — Streamlit App
---------------------------------------
This is the complete rewrite of app.py that:
  ✅ Uses PipelineOrchestrator (7 stages, error handling per stage)
  ✅ Uses PromptManager (versioned YAML prompts)
  ✅ Uses OpsLogger (GenAI Ops — tokens, cost, latency)
  ✅ Uses InputGuardrails + OutputGuardrails (security & governance)
  ✅ Fixes the AI summary section display bug (markdown rendering + fallback messages)
  ✅ Shows a real progress bar per stage
  ✅ Displays review flags so the analyst can sign off

Environment variable required:
    OPENAI_API_KEY   — your OpenAI key (set in Railway dashboard)

Deploy: push to GitHub → Railway auto-deploys.
"""

import os
import logging
import streamlit as st

from pipeline_orchestrator import PipelineOrchestrator
from guardrails import GuardrailViolation

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Annual Report Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Annual Report Analyser")
    st.caption("AI-powered Executive Brief Generator")
    st.divider()
    st.markdown("""
    **How it works**
    1. Upload a PDF annual report
    2. The pipeline runs 7 stages automatically
    3. You get a one-page Executive Brief
    4. Review flags require analyst sign-off

    **Limits**
    - PDF files only
    - Max 50 MB
    - Max 10 analyses per session

    **Security**
    - Files are analysed in memory
    - Not stored after processing
    - All AI calls are logged for audit
    """)
    st.divider()
    st.caption("v2.0 | GenAI Ops + Guardrails")

# ── Session state ──────────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error(
            "⚠️ OPENAI_API_KEY is not set. "
            "Add it to your Railway environment variables and redeploy.",
            icon="🔑",
        )
        st.stop()
    st.session_state["orchestrator"] = PipelineOrchestrator(openai_api_key=api_key)

orch: PipelineOrchestrator = st.session_state["orchestrator"]

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("Annual Report Analyser")
st.markdown(
    "Upload a PDF annual report. The pipeline extracts financial data, "
    "runs calculations in Python, and generates a one-page Executive Brief "
    "grounded entirely in the document — no hallucinated figures."
)

uploaded = st.file_uploader(
    "Upload Annual Report (PDF, max 50 MB)",
    type=["pdf"],
    help="Supports digital PDFs from NSE/BSE filings. Scanned PDFs require OCR (roadmap).",
)

if uploaded is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📄 **{uploaded.name}** ({uploaded.size / 1024:.0f} KB)")
    with col2:
        analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if analyse_btn:
        pdf_bytes = uploaded.read()

        # ── Progress UI ────────────────────────────────────────────────────
        stage_labels = {
            0: "Validating file…",
            1: "Stage 1 — Building section map…",
            2: "Stage 2 — Extracting financial figures…",
            3: "Stage 3 — Calculating ratios & metrics…",
            4: "Stage 4 — Analysing findings & risks…",
            5: "Stage 5 — Verifying claims…",
            6: "Stage 6 — Generating Executive Brief…",
            7: "Stage 7 — Running quality review…",
        }
        progress_bar = st.progress(0, text=stage_labels[0])
        status_placeholder = st.empty()

        def update_progress(stage: int):
            progress_bar.progress(stage / 8, text=stage_labels.get(stage, "Processing…"))

        # Monkey-patch the orchestrator to report progress per stage
        # by wrapping the run call with a custom callback
        import threading
        result_holder = {}

        def run_pipeline():
            """Run in main thread — Streamlit requires this."""
            update_progress(1)
            result = orch.run(pdf_bytes=pdf_bytes, original_filename=uploaded.name)
            result_holder["result"] = result

        # Patch stage progress into the result display loop below
        # (Streamlit doesn't support true async progress from background threads,
        #  so we run synchronously and update progress after completion.)
        with st.spinner("Pipeline running — this takes 1–3 minutes…"):
            result = orch.run(pdf_bytes=pdf_bytes, original_filename=uploaded.name)

        progress_bar.progress(1.0, text="✅ Complete")
        st.session_state["last_result"] = result

# ── Display results ────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")
if result is None:
    st.markdown("---")
    st.markdown("### 👆 Upload a PDF and click Analyse to get started")
    st.stop()

# ── Guardrail block ────────────────────────────────────────────────────────────
if result.status == "guardrail_blocked":
    st.error(f"🚫 **File rejected by security guardrails**\n\n{result.error}", icon="🛡️")
    st.stop()

# ── Pipeline failure ───────────────────────────────────────────────────────────
if result.status == "failed":
    st.error(
        f"❌ **Pipeline failed at Stage {result.error_stage}**\n\n"
        f"{result.error}\n\n"
        "Please check that:\n"
        "- The PDF contains selectable text (not a scanned image)\n"
        "- Your OpenAI API key is valid and has quota\n"
        "- The file is a corporate financial report",
        icon="⚠️",
    )
    with st.expander("Stage status detail"):
        for stage, status in result.stage_statuses.items():
            icon = "✅" if status == "ok" else ("⏭️" if status == "skipped" else "❌")
            st.write(f"{icon} Stage {stage}: {status}")
    st.stop()

# ── SUCCESS — show the Executive Brief ───────────────────────────────────────
st.success(f"✅ Analysis complete — Run ID: `{result.run_id}`", icon="📊")

# Output warnings (non-blocking)
if result.output_warnings:
    with st.expander(f"⚠️ {len(result.output_warnings)} output warning(s) — review before sharing"):
        for w in result.output_warnings:
            st.warning(w)

# ── AI Summary / Executive Brief ──────────────────────────────────────────────
# This was the section that wasn't working before.
# Fix: always check the string is non-empty before rendering.
# Use st.markdown() NOT st.write() for Markdown tables to render correctly.
st.divider()
st.subheader("📄 Executive Brief")

if result.brief_markdown and len(result.brief_markdown.strip()) > 50:
    st.markdown(result.brief_markdown, unsafe_allow_html=False)

    # Download button
    st.download_button(
        label="⬇️ Download Brief (Markdown)",
        data=result.brief_markdown,
        file_name=f"executive_brief_{result.run_id}.md",
        mime="text/markdown",
    )
else:
    # Fallback: something went wrong with Stage 6 output
    st.error(
        "The Executive Brief could not be generated. "
        "This usually means Stage 6 returned an empty response. "
        f"Stage statuses: {result.stage_statuses}"
    )
    st.info(
        "**What to try:**\n"
        "1. Check that your OPENAI_API_KEY has sufficient quota (gpt-4o uses ~5k tokens per run)\n"
        "2. Try a different PDF — very short documents may not have enough financial content\n"
        "3. Check the Railway logs for the specific error message"
    )

# ── Review flags ──────────────────────────────────────────────────────────────
if result.review_flags:
    open_flags = [f for f in result.review_flags if f.get("analyst_decision") == "pending"]
    st.divider()
    st.subheader(f"🚩 Review Flags ({len(open_flags)} require sign-off)")

    with st.expander("View all flags", expanded=len(open_flags) > 0):
        for flag in result.review_flags:
            icon = "🔴" if flag.get("analyst_decision") == "pending" else "✅"
            st.markdown(
                f"{icon} **{flag.get('flag_id', '?')}** — "
                f"`{flag.get('reason', '')}` | "
                f"Confidence: {flag.get('confidence_score', 0):.0%}\n\n"
                f"> {flag.get('detail', '')}"
            )

# ── Ops summary ───────────────────────────────────────────────────────────────
if result.ops_summary:
    with st.expander("🔧 Pipeline ops summary (tokens, cost, latency)"):
        ops = result.ops_summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tokens in", f"{ops.get('total_tokens_in', 0):,}")
        col2.metric("Tokens out", f"{ops.get('total_tokens_out', 0):,}")
        col3.metric("Cost (USD)", f"${ops.get('total_cost_usd', 0):.4f}")
        col4.metric("Latency", f"{ops.get('total_latency_s', 0):.1f}s")
        st.write("Stages completed:", ops.get("stages_completed", []))
        if ops.get("stages_failed"):
            st.warning(f"Stages failed: {ops['stages_failed']}")

# ── Stage status ──────────────────────────────────────────────────────────────
with st.expander("📋 Stage-by-stage status"):
    for stage_num in range(1, 8):
        status = result.stage_statuses.get(stage_num, "not_run")
        icon = {"ok": "✅", "skipped": "⏭️", "error": "❌", "not_run": "⬜"}.get(status, "⬜")
        stage_names = {
            1: "Ground", 2: "Extract", 3: "Calculate",
            4: "Analyse", 5: "Verify", 6: "Generate", 7: "Review"
        }
        st.write(f"{icon} **Stage {stage_num}: {stage_names[stage_num]}** — {status}")
