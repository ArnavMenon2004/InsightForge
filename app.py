"""
InsightForge — Streamlit front-end.
Redesigned UI: research intelligence platform aesthetic.
All pipeline logic, schemas, and backend calls are unchanged.
"""

import html

import streamlit as st
from pypdf import PdfReader

from agents.base_agent import BaseAgent
from config.settings import PPT_PAPER_CHAR_LIMIT
from graphs.pipeline import pipeline
from prompts.ppt import PPT_OUTLINE_PROMPT, SlideOutline
from schemas.state import GraphState
from tools.export import PDFExporter
from tools.pdf_extractor import extract_paper, fit_to_limit
from tools.ppt_checker import unverified_numbers
from tools.ppt_generator import PPTGenerator
from utils.logger import logger


def friendly_error(error, prefix: str) -> str:
    """Turn a raw exception/error string into a short, user-readable message."""
    text = str(error)
    low = text.lower()
    if "500 internal" in low or "internal error" in low:
        why = ("Google's AI service hit a temporary internal error. "
               "Please try again in a minute or two.")
    elif "503" in text or "unavailable" in low or "high demand" in low:
        why = ("The AI model is temporarily overloaded on Google's side. "
               "This usually clears within a few minutes — please try again shortly.")
    elif "429" in text or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        why = "The API rate limit or daily quota was reached. Wait a bit, or check your API usage, then try again."
    elif "timed out" in low or "timeout" in low or "deadline" in low:
        why = "The AI model took too long to respond. Please try again."
    elif "api key" in low or "api_key" in low or "permission_denied" in low or "401" in text or "403" in text:
        why = "There is a problem with an API key. Check GEMINI_API_KEY and TAVILY_API_KEY in your .env file."
    else:
        why = "Something unexpected went wrong. Details are in the terminal log."
    logger.error(f"{prefix} {text}")
    return f"{prefix} {why}"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightForge",
    page_icon="◈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp { background-color: #0c0c0e; color: #e0e0f0; }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 860px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f0f12; border-right: 1px solid #1e1e26; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; max-width: 100%; }

    .if-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 500; letter-spacing: 0.12em;
        color: #a0a0c0; text-transform: uppercase; margin-bottom: 0.75rem;
    }
    .if-wordmark {
        font-size: 1.1rem; font-weight: 600; letter-spacing: 0.18em;
        color: #ffffff; text-transform: uppercase; font-family: 'Inter', sans-serif; margin-bottom: 0.1rem;
    }
    .if-tagline {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        letter-spacing: 0.1em; color: #8888aa; text-transform: uppercase; margin-bottom: 1.8rem;
    }

    /* Sidebar pipeline stages */
    .pipe-stage {
        display: flex; align-items: center; gap: 0.75rem; padding: 0.4rem 0;
        font-size: 0.9rem; color: #e0e0f0; font-family: 'Inter', sans-serif; font-weight: 400;
    }
    .pipe-stage .stage-num {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        color: #8888aa; min-width: 20px;
    }
    .pipe-stage .stage-dot { font-size: 0.6rem; color: #6060a0; }

    .sidebar-rule { border: none; border-top: 1px solid #1a1a22; margin: 1.2rem 0; }

    .sys-tag {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
        color: #c0c0d8; letter-spacing: 0.06em; padding: 0.2rem 0; display: block;
    }
    .sidebar-version {
        font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
        color: #50507a; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2rem;
    }

    /* Page header */
    .if-header { margin-bottom: 3rem; padding-bottom: 2rem; border-bottom: 1px solid #1a1a22; }
    .if-header-mark { font-size: 0.9rem; color: #7070b0; margin-bottom: 0.5rem; display: block; }
    .if-descriptor {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        color: #8888aa; letter-spacing: 0.08em; margin-top: 0.6rem;
    }

    /* Input workspace */
    .workspace-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
        letter-spacing: 0.14em; color: #a0a0c8; text-transform: uppercase; margin-bottom: 0.4rem;
    }
    .workspace-instruction { font-size: 1.0rem; color: #c0c0e0; margin-bottom: 1.2rem; font-weight: 300; }

    [data-testid="stTextInput"] input {
        background-color: #111118 !important; border: 1px solid #2a2a3a !important;
        border-radius: 4px !important; color: #ffffff !important;
        font-size: 1.05rem !important; padding: 0.9rem 1rem !important; font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stTextInput"] input:focus { border-color: #5a5aaa !important; box-shadow: none !important; }
    [data-testid="stTextInput"] input::placeholder { color: #40404e !important; }
    [data-testid="stTextInput"] label { display: none !important; }

    [data-testid="stSelectbox"] > div > div {
        background-color: #111118 !important; border: 1px solid #222230 !important;
        border-radius: 4px !important; color: #e0e0f0 !important; font-size: 0.9rem !important;
    }
    [data-testid="stSelectbox"] label {
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.65rem !important;
        letter-spacing: 0.1em !important; color: #a0a0c0 !important; text-transform: uppercase !important;
    }

    /* Run button */
    [data-testid="stButton"] > button {
        background-color: transparent !important; border: 1px solid #4a4a7a !important;
        border-radius: 3px !important; color: #c0c0e8 !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
        letter-spacing: 0.1em !important; padding: 0.65rem 1.8rem !important;
        text-transform: uppercase !important; transition: all 0.15s ease !important; font-weight: 500 !important;
    }
    [data-testid="stButton"] > button:hover {
        border-color: #7070aa !important; color: #ffffff !important; background-color: #18182a !important;
    }

    /* Pipeline status */
    .pipeline-status {
        background-color: #0f0f14; border: 1px solid #1e1e2a;
        border-radius: 4px; padding: 1.2rem 1.4rem; margin: 1.5rem 0;
    }
    .ps-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.9rem;
    }
    .ps-stages { display: flex; gap: 2rem; flex-wrap: wrap; }
    .ps-stage {
        display: flex; align-items: center; gap: 0.4rem;
        font-size: 0.82rem; color: #55556a; font-family: 'Inter', sans-serif;
    }
    .ps-stage.done { color: #7ab87a; }
    .ps-stage.active { color: #ffffff; }
    .ps-stage .ps-dot { font-size: 0.5rem; }

    /* Complete strip */
    .complete-strip {
        background-color: #0d0d12; border: 1px solid #1e2e1e; border-left: 3px solid #4a7a4a;
        border-radius: 3px; padding: 0.9rem 1.2rem; margin: 1.5rem 0;
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
    }
    .cs-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        letter-spacing: 0.14em; color: #6ab06a; text-transform: uppercase;
    }
    .cs-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #a0a0c0; letter-spacing: 0.05em; }

    /* Report header */
    .report-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        letter-spacing: 0.18em; color: #8080b0; text-transform: uppercase; margin-bottom: 0.7rem;
    }
    .report-title {
        font-size: 1.9rem; font-weight: 600; color: #ffffff;
        line-height: 1.25; letter-spacing: -0.01em; margin-bottom: 1.2rem; font-family: 'Inter', sans-serif;
    }
    .report-meta-strip { display: flex; gap: 2.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .rms-item { display: flex; flex-direction: column; gap: 0.15rem; }
    .rms-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
        letter-spacing: 0.12em; color: #7070a0; text-transform: uppercase;
    }
    .rms-value { font-size: 0.92rem; color: #e0e0f0; font-weight: 400; }

    /* Export */
    .export-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
        letter-spacing: 0.14em; color: #8080b0; text-transform: uppercase; margin-bottom: 0.4rem;
    }
    .export-sub { font-size: 0.82rem; color: #9090b0; margin-bottom: 1rem; font-weight: 300; }

    [data-testid="stDownloadButton"] > button {
        background-color: transparent !important; border: 1px solid #4a4a7a !important;
        border-radius: 3px !important; color: #c0c0e8 !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.72rem !important;
        letter-spacing: 0.1em !important; padding: 0.55rem 1.4rem !important;
        text-transform: uppercase !important; transition: all 0.15s ease !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        border-color: #7070aa !important; color: #ffffff !important; background-color: #14142a !important;
    }

    /* Executive summary */
    .exec-summary-block {
        background-color: #0f0f14; border: 1px solid #1e1e2a; border-left: 2px solid #4a4a8a;
        border-radius: 3px; padding: 1.4rem 1.6rem; margin: 1.5rem 0 2rem 0;
    }
    .exec-summary-block .if-eyebrow { margin-bottom: 0.8rem; }
    .exec-summary-body { font-size: 0.95rem; line-height: 1.8; color: #d0d0e8; font-weight: 300; max-width: 680px; }

    /* Report sections */
    .section-block { margin: 2.5rem 0; padding-bottom: 2.5rem; border-bottom: 1px solid #18181e; }
    .section-header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem; }
    .section-num { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #7070a0; min-width: 24px; }
    .section-title { font-size: 1.0rem; font-weight: 500; color: #ffffff; text-transform: uppercase; font-family: 'Inter', sans-serif; }
    .section-body { font-size: 0.92rem; line-height: 1.85; color: #c8c8e0; font-weight: 300; max-width: 700px; padding-left: 1.6rem; }
    .section-body p { margin-bottom: 1rem; }

    /* Evidence review */
    .evidence-header { display: flex; align-items: center; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .confidence-badge {
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.14em;
        padding: 0.28rem 0.75rem; border-radius: 2px; text-transform: uppercase; font-weight: 500;
    }
    .confidence-high { background-color: #0d1f0d; color: #6ac86a; border: 1px solid #1e3a1e; }
    .confidence-medium { background-color: #1a1a0a; color: #c8c840; border: 1px solid #2e2e10; }
    .confidence-low { background-color: #1f0d0d; color: #c86060; border: 1px solid #3a1a1a; }

    .verdict-tally { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.05em; }
    .tally-supported { color: #6ab06a; }
    .tally-unverifiable { color: #b0b040; }
    .tally-unsupported { color: #b05050; }

    .claim-card {
        background-color: #0f0f14; border: 1px solid #1a1a22;
        border-radius: 3px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    }
    .claim-verdict-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.5rem; }
    .verdict-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 0.12em;
        padding: 0.18rem 0.55rem; border-radius: 2px; text-transform: uppercase; font-weight: 500;
    }
    .vl-supported { background-color: #0a180a; color: #5aaa5a; border: 1px solid #1a2e1a; }
    .vl-unverifiable { background-color: #18180a; color: #aaaa30; border: 1px solid #28280e; }
    .vl-unsupported { background-color: #180a0a; color: #aa4040; border: 1px solid #2e1010; }

    .claim-text { font-size: 0.88rem; color: #c8c8e0; line-height: 1.6; font-weight: 300; margin-bottom: 0.5rem; }
    .claim-explanation { font-size: 0.8rem; color: #9090b8; line-height: 1.5; font-style: italic; }

    /* Sources */
    .source-item {
        display: flex; gap: 1rem; padding: 0.75rem 0;
        border-bottom: 1px solid #16161c; align-items: flex-start;
    }
    .source-num { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #6060a0; min-width: 24px; padding-top: 0.1rem; }
    .source-body { flex: 1; min-width: 0; }
    .source-title { font-size: 0.88rem; color: #a0a0d0; font-weight: 400; margin-bottom: 0.2rem; text-decoration: none; display: block; }
    .source-title:hover { color: #ffffff; }
    .source-domain { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #6868a0; letter-spacing: 0.04em; }
    .source-score { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #6060a0; white-space: nowrap; padding-top: 0.1rem; }

    /* Divider */
    .if-rule { border: none; border-top: 1px solid #141418; margin: 2.5rem 0; }

    .stAlert { background-color: #0f0f14 !important; border: 1px solid #2a1a1a !important; color: #cc8888 !important; border-radius: 3px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="if-wordmark">InsightForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="if-tagline">Research Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="if-eyebrow">Pipeline</div>', unsafe_allow_html=True)

    for num, name in [("01","Planner"),("02","Researcher"),("03","Writer"),("04","Fact Checker"),("05","Citation")]:
        st.markdown(
            f'<div class="pipe-stage"><span class="stage-num">{num}</span>'
            f'<span class="stage-dot">◦</span>{name}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.markdown('<div class="if-eyebrow">System</div>', unsafe_allow_html=True)

    for tag in ["LangGraph","LangChain","Gemini","Tavily","Pydantic"]:
        st.markdown(f'<span class="sys-tag">{tag}</span>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v0.5.0</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Research", "Paper to PPT"])

# Paper to PPT is filled first so a st.stop() in the Research tab cannot skip it.
with tab2:
    st.markdown('<div class="workspace-label">Paper to PPT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="workspace-instruction">Upload a research paper and generate a full conference presentation (10-12 slides).</div>',
        unsafe_allow_html=True,
    )

    paper_file = st.file_uploader("Research paper (PDF)", type=["pdf"])
    ppt_mode = "Full Conference PPT"
    ppt_theme = st.radio("Theme", ["Dark", "Light (better for printing)"], horizontal=True)
    ppt_button = st.button("Generate Slides →")

    if ppt_button:
        if paper_file is None:
            st.error("Please upload a PDF first.")
        else:
            st.session_state.pop("ppt", None)
            try:
                with st.spinner("Reading paper and generating slides..."):
                    paper_text, paper_title = extract_paper(PdfReader(paper_file))
                    if not paper_text:
                        raise ValueError("No extractable text found in this PDF (it may be scanned).")

                    outline = BaseAgent("PPT Agent")._run_gemini_chain(
                        schema=SlideOutline,
                        prompt=PPT_OUTLINE_PROMPT,
                        inputs={
                            "paper_text": fit_to_limit(paper_text, PPT_PAPER_CHAR_LIMIT),
                            "mode": ppt_mode,
                            "paper_title": paper_title or "Unknown",
                        },
                    )
                    st.session_state["ppt"] = {
                        "outline": outline,
                        "bytes": PPTGenerator(
                            theme="light" if ppt_theme.startswith("Light") else "dark"
                        ).generate(outline),
                        "flags": unverified_numbers(outline, paper_text),
                    }
            except Exception as exc:
                logger.exception("PPT generation failed.")
                st.error(friendly_error(exc, "Slide generation failed."))

    # Rendered from session_state so the deck survives the rerun a download click causes.
    ppt_result = st.session_state.get("ppt")
    if ppt_result:
        outline, flags = ppt_result["outline"], ppt_result["flags"]
        st.markdown('<hr class="if-rule">', unsafe_allow_html=True)
        st.download_button(
            label="Download PPTX",
            data=ppt_result["bytes"],
            file_name="insightforge_slides.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
        if flags:
            st.caption(
                "⚠ Some numbers could not be found in the paper text (they may be values the model "
                "calculated). Check the marked slides before presenting."
            )

        st.markdown(
            f'<div class="report-title" style="font-size:1.4rem;">{html.escape(outline.presentation_title)}</div>',
            unsafe_allow_html=True,
        )
        for idx, slide in enumerate(outline.slides):
            bullets_html = "".join(f"<p>• {html.escape(b)}</p>" for b in slide.bullets)
            warn_html = (
                f'<p style="color:#d9a441;font-size:0.8rem;">⚠ Not found in paper: '
                f'{html.escape(", ".join(flags[idx]))}</p>'
                if idx in flags
                else ""
            )
            st.markdown(
                f"""<div class="claim-card">
                    <div class="claim-verdict-row">
                        <span class="section-num">{idx + 1:02d}</span>
                        <span class="section-title">{html.escape(slide.title)}</span>
                    </div>
                    <div class="claim-text">{bullets_html}{warn_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )

with tab1:
    # ---------------------------------------------------------------------------
    # Page header
    # ---------------------------------------------------------------------------

    st.markdown(
        """
        <div class="if-header">
            <span class="if-header-mark">◈</span>
            <div class="if-wordmark" style="font-size:1.4rem;letter-spacing:0.22em;">INSIGHTFORGE</div>
            <div class="if-tagline" style="color:#50506a;margin-bottom:0.6rem;">AI RESEARCH WORKSPACE</div>
            <div class="if-descriptor">
                Multi-agent research &nbsp;·&nbsp; Evidence synthesis &nbsp;·&nbsp; Source verification
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Input workspace
    # ---------------------------------------------------------------------------

    st.markdown('<div class="workspace-label">New Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-instruction">What would you like to investigate?</div>', unsafe_allow_html=True)

    query = st.text_input("query", placeholder="e.g. The future of quantum computing in drug discovery", label_visibility="collapsed")

    col_audience, col_depth, col_empty = st.columns([1, 1, 1.5])
    with col_audience:
        audience = st.selectbox("AUDIENCE", options=["Auto","Beginner","Student","Developer","Executive"], index=0)
    with col_depth:
        depth = st.selectbox("DEPTH", options=["Auto","Basic","Intermediate","Advanced"], index=0)

    audience_override = None if audience == "Auto" else audience
    depth_override    = None if depth == "Auto" else depth

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    run_button = st.button("Begin Research →")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Pipeline status
    # ---------------------------------------------------------------------------

    status_placeholder = st.empty()

    def render_pipeline_status(done: set, active: str | None = None) -> str:
        stage_map = [
            ("plan",       "Planning"),
            ("document",   "Web Research"),
            ("draft",      "Synthesis"),
            ("fact_check", "Fact Verification"),
            ("report",     "Citation"),
        ]
        stage_html = ""
        for key, label in stage_map:
            if key in done:
                css, dot = "done", "✓"
            elif key == active:
                css, dot = "active", "●"
            else:
                css, dot = "", "○"
            stage_html += f'<div class="ps-stage {css}"><span class="ps-dot">{dot}</span> {label}</div>'

        label_color = "#6060a0" if active else "#4a8a4a"
        title       = "Research in Progress" if active else "Research Complete"

        return f"""
        <div class="pipeline-status">
            <div class="ps-label" style="color:{label_color};">{title}</div>
            <div class="ps-stages">{stage_html}</div>
        </div>"""

    # ---------------------------------------------------------------------------
    # Pipeline execution
    # ---------------------------------------------------------------------------

    if run_button:

        if not query.strip():
            st.error("Please enter a research topic.")
            st.stop()

        st.session_state.pop("result", None)
        logger.info(f"Pipeline started: '{query}' audience={audience_override} depth={depth_override}")

        status_placeholder.markdown(render_pipeline_status(set(), active="plan"), unsafe_allow_html=True)

        try:
            initial_state: GraphState = {"query": query, "audience": audience_override, "depth": depth_override}
            final_state: GraphState   = initial_state.copy()
            done_keys: set            = set()

            for chunk in pipeline.stream(initial_state, stream_mode="values"):
                final_state = chunk
                done_keys   = {k for k in ("plan","document","draft","fact_check","report") if chunk.get(k) is not None}
                all_stages  = ["plan","document","draft","fact_check","report"]
                active      = next((s for s in all_stages if s not in done_keys), None)
                status_placeholder.markdown(render_pipeline_status(done_keys, active), unsafe_allow_html=True)
                logger.info(f"Pipeline: done={done_keys} active={active}")

        except Exception as exc:
            logger.exception("Pipeline invocation failed.")
            st.error(friendly_error(exc, "The research pipeline failed."))
            st.stop()

        if final_state.get("error"):
            st.error(friendly_error(final_state["error"], "The research pipeline stopped."))
            st.stop()

        if not final_state.get("report"):
            st.error("Pipeline completed but no report was produced.")
            st.stop()

        logger.info("Pipeline completed successfully.")
        st.session_state["result"] = final_state

    # Rendered from session state so the report survives reruns
    # (download clicks, widget changes, reconnects after a long run).
    if st.session_state.get("result"):
        final_state = st.session_state["result"]
        report   = final_state["report"]
        document = final_state["document"]
        plan     = final_state["plan"]

        sources_total = sum(len(s.sources) for s in document.sections)

        status_placeholder.markdown(
            render_pipeline_status({"plan","document","draft","fact_check","report"}),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="complete-strip">
                <span class="cs-label">Research Complete</span>
                <span class="cs-meta">{sources_total} sources &nbsp;·&nbsp; {len(document.sections)} sections &nbsp;·&nbsp; {plan.depth or "—"} depth</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="if-rule">', unsafe_allow_html=True)

        # ── Report header ──────────────────────────────────────────────────────

        confidence_val = report.fact_check.overall_confidence if report.fact_check else "—"

        st.markdown('<div class="report-eyebrow">Research Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-title">{report.title}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="report-meta-strip">
                <div class="rms-item"><span class="rms-label">Audience</span><span class="rms-value">{plan.audience or "—"}</span></div>
                <div class="rms-item"><span class="rms-label">Depth</span><span class="rms-value">{plan.depth or "—"}</span></div>
                <div class="rms-item"><span class="rms-label">Sections</span><span class="rms-value">{len(document.sections)}</span></div>
                <div class="rms-item"><span class="rms-label">Sources</span><span class="rms-value">{sources_total}</span></div>
                <div class="rms-item"><span class="rms-label">Confidence</span><span class="rms-value">{confidence_val}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── PDF download only ──────────────────────────────────────────────────

        st.markdown('<div class="export-label">Export Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="export-sub">Download the completed research report.</div>', unsafe_allow_html=True)

        try:
            pdf_exporter = PDFExporter()
            pdf_bytes    = pdf_exporter.export(report, document)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"insightforge_{plan.topic[:40].replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.caption("PDF export requires reportlab — run: pip install reportlab")

        st.markdown('<hr class="if-rule">', unsafe_allow_html=True)

        # ── Executive summary ──────────────────────────────────────────────────

        st.markdown(
            f"""<div class="exec-summary-block">
                <div class="if-eyebrow">Executive Summary</div>
                <div class="exec-summary-body">{report.executive_summary}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Report sections ────────────────────────────────────────────────────

        import html
        import re

        for idx, raw in enumerate(re.split(r'\n---\n', report.markdown), start=1):
            raw = raw.strip()
            if not raw:
                continue
            m = re.match(r'^##\s+(.+)', raw)
            section_title = m.group(1).strip() if m else f"Section {idx}"
            body          = raw[m.end():].strip() if m else raw
            paragraphs    = [p.strip() for p in body.split("\n\n") if p.strip()]
            body_html     = "".join(f"<p>{p}</p>" for p in paragraphs)

            # Per-section confidence badge (hidden when no claims were matched)
            conf_badge_html = ""
            sc = (report.section_confidences or {}).get(section_title)
            if sc and sc.total > 0:
                sc_css = {"High":"confidence-high","Medium":"confidence-medium","Low":"confidence-low"}.get(sc.confidence_label,"confidence-medium")
                conf_badge_html = (
                    f'<span class="confidence-badge {sc_css}" style="font-size:0.55rem;">'
                    f'{sc.confidence_pct:.0f}% verified</span>'
                )

            # Student Guide Mode block
            guidance_html = ""
            sg = (report.student_guidance or {}).get(section_title)
            if sg:
                def _guide_row(label: str, items: list[str]) -> str:
                    return (
                        '<div style="margin-bottom:0.6rem;">'
                        f'<span class="rms-label">{label}</span>'
                        f'<div style="color:#9a9ac8;font-size:0.82rem;line-height:1.6;">{" &nbsp;·&nbsp; ".join(html.escape(i) for i in items)}</div>'
                        '</div>'
                    )
                guidance_html = (
                    '<div style="background-color:#0d0d10;border:1px solid #2a2a3a;border-radius:3px;'
                    'padding:1rem 1.2rem;margin:1.2rem 0 0 1.6rem;max-width:700px;">'
                    '<div class="if-eyebrow" style="margin-bottom:0.8rem;">How to Research Further</div>'
                    + _guide_row("Search terms", sg.search_terms)
                    + _guide_row("Source types", sg.source_types)
                    + _guide_row("Related subtopics", sg.related_subtopics)
                    + '</div>'
                )

            # Built without blank/whitespace-only lines: Markdown ends an HTML
            # block at a blank line and renders later indented lines as code.
            st.markdown(
                '<div class="section-block">'
                '<div class="section-header">'
                f'<span class="section-num">{idx:02d}</span>'
                f'<span class="section-title">{section_title}</span>'
                f'{conf_badge_html}'
                '</div>'
                f'<div class="section-body">{body_html}</div>'
                f'{guidance_html}'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="if-rule">', unsafe_allow_html=True)

        # ── Evidence review ────────────────────────────────────────────────────

        if report.fact_check:
            st.markdown('<div class="if-eyebrow">Evidence Review</div>', unsafe_allow_html=True)

            conf     = report.fact_check.overall_confidence
            conf_css = {"High":"confidence-high","Medium":"confidence-medium","Low":"confidence-low"}.get(conf,"confidence-medium")
            results  = report.fact_check.results
            n_sup    = sum(1 for r in results if r.verdict == "SUPPORTED")
            n_unv    = sum(1 for r in results if r.verdict == "UNVERIFIABLE")
            n_uns    = sum(1 for r in results if r.verdict == "UNSUPPORTED")

            st.markdown(
                f"""<div class="evidence-header">
                    <span class="confidence-badge {conf_css}">{conf} Confidence</span>
                    <span class="verdict-tally">
                        <span class="tally-supported">{n_sup} Supported</span> &nbsp;·&nbsp;
                        <span class="tally-unverifiable">{n_unv} Unverifiable</span> &nbsp;·&nbsp;
                        <span class="tally-unsupported">{n_uns} Unsupported</span>
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

            vdict = {"SUPPORTED":"vl-supported","UNVERIFIABLE":"vl-unverifiable","UNSUPPORTED":"vl-unsupported"}
            for fc in results:
                st.markdown(
                    f"""<div class="claim-card">
                        <div class="claim-verdict-row">
                            <span class="verdict-label {vdict.get(fc.verdict,'vl-unverifiable')}">{fc.verdict}</span>
                        </div>
                        <div class="claim-text">{fc.claim}</div>
                        <div class="claim-explanation">{fc.explanation}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="if-rule">', unsafe_allow_html=True)

        # ── Consistency check ──────────────────────────────────────────────────

        if report.contradiction_report:
            cr      = report.contradiction_report
            cr_css  = {"High":"confidence-high","Medium":"confidence-medium","Low":"confidence-low"}.get(cr.consistency_score,"confidence-medium")

            st.markdown('<div class="if-eyebrow">Consistency Check</div>', unsafe_allow_html=True)
            st.markdown(
                f"""<div class="evidence-header">
                    <span class="confidence-badge {cr_css}">{cr.consistency_score} Consistency</span>
                    <span class="verdict-tally">{len(cr.contradictions)} contradiction(s) found</span>
                </div>""",
                unsafe_allow_html=True,
            )

            if not cr.contradictions:
                st.markdown(
                    '<div class="claim-card"><div class="tally-supported" style="font-size:0.85rem;">'
                    'No contradictions detected across sections.</div></div>',
                    unsafe_allow_html=True,
                )
            for c in cr.contradictions:
                st.markdown(
                    f"""<div class="claim-card">
                        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                            <div style="flex:1;min-width:240px;">
                                <div class="rms-label">{html.escape(c.section_a)}</div>
                                <div class="claim-text">{html.escape(c.claim_a)}</div>
                            </div>
                            <div style="flex:1;min-width:240px;">
                                <div class="rms-label">{html.escape(c.section_b)}</div>
                                <div class="claim-text">{html.escape(c.claim_b)}</div>
                            </div>
                        </div>
                        <div class="claim-explanation">{html.escape(c.explanation)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<hr class="if-rule">', unsafe_allow_html=True)

        # ── Sources ────────────────────────────────────────────────────────────

        st.markdown(f'<div class="if-eyebrow">Sources &nbsp;—&nbsp; {sources_total} retrieved</div>', unsafe_allow_html=True)

        if report.source_diversity:
            sd     = report.source_diversity
            sd_css = {"High":"confidence-high","Medium":"confidence-medium","Low":"confidence-low"}.get(sd.diversity_score,"confidence-medium")
            st.markdown(
                f"""<div class="evidence-header">
                    <span class="confidence-badge {sd_css}">{sd.diversity_score} Diversity</span>
                    <span class="verdict-tally">
                        Academic: {sd.academic} &nbsp;·&nbsp; News: {sd.news} &nbsp;·&nbsp;
                        Industry: {sd.industry} &nbsp;·&nbsp; Gov/Edu: {sd.government} &nbsp;·&nbsp;
                        Other: {sd.other}
                    </span>
                    <span class="verdict-tally">Dominant: {sd.dominant_type}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        seen: set = set()
        idx = 1
        for section in document.sections:
            for source in section.sources:
                key = str(source.url)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(key).netloc.lstrip("www.")
                except Exception:
                    domain = key
                score_html = f'<span class="source-score">{source.score:.2f}</span>' if source.score is not None else ""
                st.markdown(
                    f"""<div class="source-item">
                        <span class="source-num">{idx:02d}</span>
                        <div class="source-body">
                            <a class="source-title" href="{source.url}" target="_blank" rel="noopener noreferrer">{source.title}</a>
                            <div class="source-domain">{domain}</div>
                        </div>
                        {score_html}
                    </div>""",
                    unsafe_allow_html=True,
                )
                idx += 1