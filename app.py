import streamlit as st
import time
import json
import io
import re
from pathlib import Path
from pypdf import PdfReader

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from retrieval_utils import load_vector_store, docs_to_records
from full_text_fetch import augment_with_full_text
from generate_report import generate_report
from web_search import search_web_fallback

# App Configuration
st.set_page_config(
    page_title="AgriSearch AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Dedicated Stylesheet
def load_stylesheet(css_file="style.css"):
    css_path = Path(css_file)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_stylesheet("style.css")


@st.cache_resource(show_spinner=False)
def get_cached_vector_store():
    return load_vector_store("faiss_index")


def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return ""


def build_pdf_report(title: str, markdown_text: str) -> bytes:
    """Compiles markdown text into a styled PDF document."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#17362a'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e5b3a'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374a3f'),
        spaceAfter=8
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374a3f'),
        leftIndent=15,
        spaceAfter=4
    )

    story = [
        Paragraph("AgriSearch AI — Scientific Research Synthesis", title_style),
        Paragraph(f"<b>Topic:</b> {title}", body_style),
        Spacer(1, 10)
    ]

    for line in markdown_text.split("\n"):
        clean_line = line.strip()
        if not clean_line:
            continue
        
        if clean_line.startswith("# ") or clean_line.startswith("## "):
            text = re.sub(r"^#+\s*", "", clean_line)
            story.append(Paragraph(f"<b>{text}</b>", h2_style))
        elif clean_line.startswith("### "):
            text = re.sub(r"^###\s*", "", clean_line)
            story.append(Paragraph(f"<b>{text}</b>", h2_style))
        elif clean_line.startswith("* ") or clean_line.startswith("- "):
            text = clean_line[2:].strip()
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
            story.append(Paragraph(f"• {text}", bullet_style))
        else:
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_line)
            story.append(Paragraph(text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Initialize Session State
if "report_data" not in st.session_state:
    st.session_state.report_data = None

TOP_K = 8
FALLBACK_THRESHOLD = 0.75

# Top Header
st.markdown(
    """
    <header class="site-header">
        <div class="field-lines" style="position: absolute; inset: 0;"></div>
        <div style="position: relative; z-index: 1; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.85rem;">
                <div style="display: flex; height: 2.75rem; width: 2.75rem; align-items: center; justify-content: center; border-radius: 1rem; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.1); font-size: 1.35rem;">
                    🌾
                </div>
                <div>
                    <h2 class="brand-serif" style="color: #ffffff; margin: 0; font-size: 1.35rem; font-weight: 700;">AgriSearch AI</h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #dce9d9; letter-spacing: 0.12em; text-transform: uppercase;">Agricultural RAG Research Assistant</p>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #dce9d9; background: rgba(255,255,255,0.08); padding: 0.4rem 0.85rem; border-radius: 9999px; border: 1px solid rgba(255,255,255,0.15);">
                AGRIS Knowledge Base + Live Web
            </div>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="brand-serif" style="font-size: 2.65rem; font-weight: 700; color: #17362a; margin-bottom: 0.5rem; line-height: 1.15;">
            Evidence-backed agricultural intelligence.
        </h1>
        <p style="font-size: 1.05rem; color: #4c6253; max-width: 760px; margin: 0 auto; line-height: 1.6;">
            Synthesize peer-reviewed agronomy literature, harvest full-text PDFs on demand, and fall back to live web intelligence when queries exceed corpus boundaries.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section (Rendered when no active report is in state)
if st.session_state.report_data is None:
    st.markdown(
        """
        <div class="paper-texture soft-shadow" style="margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <h3 class="brand-serif" style="color: #17362a; margin: 0; font-size: 1.35rem;">Research Inquiry</h3>
                <span class="badge-rag">Hybrid RAG Pipeline</span>
            </div>
            <p style="margin: 0; font-size: 0.85rem; color: #607365;">
                Enter a research query, attach a custom agricultural PDF, or synthesize both together.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_query, col_upload = st.columns([1.1, 0.9], gap="medium")

    with col_query:
        query_input = st.text_area(
            label="Research Query",
            placeholder="e.g., What are effective methods for controlling Cercospora leaf spot in sugar beets?",
            height=140,
            label_visibility="collapsed",
        )

    with col_upload:
        uploaded_pdf = st.file_uploader(
            "Attach Paper (PDF optional)",
            type=["pdf"],
            help="Upload a scientific paper to synthesize with your inquiry.",
            label_visibility="collapsed",
        )

    col_spacer1, col_center_btn, col_spacer2 = st.columns([1.2, 1, 1.2])
    with col_center_btn:
        generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)

    if generate_clicked and (query_input.strip() or uploaded_pdf is not None):
        start_time = time.time()
        effective_query = query_input.strip() if query_input.strip() else "Synthesize and summarize the core agricultural findings from the attached document."

        with st.status("Investigating agricultural knowledge base...", expanded=True) as status_box:
            retrieved_records = []
            use_fallback = False
            top_score = float("inf")
            raw_results = []

            # 1. Custom PDF Upload
            if uploaded_pdf is not None:
                st.write(f"-> Extracting text from uploaded PDF: `{uploaded_pdf.name}`...")
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                retrieved_records.append({
                    "title": uploaded_pdf.name.replace(".pdf", ""),
                    "abstract": pdf_text[:1200] + "..." if len(pdf_text) > 1200 else pdf_text,
                    "date": "Uploaded Document",
                    "subject": "Custom Ingestion",
                    "source_id": uploaded_pdf.name,
                    "full_text": pdf_text,
                })
                badge_html = '<span class="badge-pdf">📑 Route: Uploaded Custom PDF</span>'

            # 2. Local FAISS Retrieval
            if not retrieved_records:
                st.write("-> Querying FAISS index with `BAAI/bge-base-en-v1.5` embeddings...")
                vector_store = get_cached_vector_store()
                raw_results = vector_store.similarity_search_with_score(effective_query, k=TOP_K)

                if not raw_results:
                    use_fallback = True
                    st.write("-> FAISS index returned no matching chunks.")
                else:
                    top_score = raw_results[0][1]
                    st.write(f"-> Top match distance: **`{top_score:.4f}`** (Threshold: **`{FALLBACK_THRESHOLD:.2f}`**)")
                    if top_score > FALLBACK_THRESHOLD:
                        use_fallback = True
                        st.write("-> Distance exceeded threshold. Switching route to **Tavily Live Web Search**...")

                if use_fallback:
                    st.write("-> Executing live web search via Tavily API...")
                    retrieved_records = search_web_fallback(effective_query, max_results=5)
                    badge_html = '<span class="badge-web">🌐 Route: Tavily Live Web Search</span>'
                else:
                    st.write("-> Harvesting open-access full-text PDFs for matched papers...")
                    docs = [d for d, s in raw_results]
                    initial_records = docs_to_records(docs)
                    retrieved_records = augment_with_full_text(initial_records)
                    badge_html = '<span class="badge-rag">Route: Local AGRIS Corpus</span>'

            st.write("-> Synthesizing structured scientific literature report via Gemini 2.0 Flash...")
            report_markdown = generate_report(effective_query, retrieved_records)

            elapsed = time.time() - start_time
            status_box.update(label=f"Synthesis complete in {elapsed:.2f}s", state="complete", expanded=False)

        # Store in state so downloads won't cause the view to clear
        st.session_state.report_data = {
            "query": effective_query,
            "report_markdown": report_markdown,
            "badge_html": badge_html,
            "retrieved_records": retrieved_records,
            "raw_results": [(d.metadata, s, d.page_content) for d, s in raw_results],
            "use_fallback": use_fallback,
            "is_custom_pdf": uploaded_pdf is not None,
        }
        st.rerun()

# Display Persistent Report View
else:
    data = st.session_state.report_data

    st.markdown(
        f"""
        <div class="paper-texture soft-shadow" style="margin-top: 1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; border-bottom: 1px solid #d6dfd1; padding-bottom: 1.25rem; margin-bottom: 1.5rem;">
                <div>
                    <p style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em; color: #a36712; text-transform: uppercase; margin: 0;">
                        Research Report Workspace
                    </p>
                    <h2 class="brand-serif" style="color: #17362a; font-size: 1.75rem; font-weight: 700; margin: 0.35rem 0 0 0;">
                        {data['query']}
                    </h2>
                </div>
                <div>
                    {data['badge_html']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_report, tab_sources, tab_diagnostics = st.tabs(
        ["Synthesized Report", "Evidence & Source Explorer", "Retrieval Diagnostics"]
    )

    with tab_report:
        st.markdown(f'<div class="report-content">{data["report_markdown"]}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
        
        with col_d1:
            pdf_bytes = build_pdf_report(data['query'], data['report_markdown'])
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name="agrisearch_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        
        with col_d2:
            st.download_button(
                label="Export Report (Markdown)",
                data=data['report_markdown'],
                file_name="agrisearch_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            
        with col_d3:
            st.download_button(
                label="Export Evidence Metadata (JSON)",
                data=json.dumps(data['retrieved_records'], indent=2),
                file_name="retrieval_metadata.json",
                mime="application/json",
            )

        # Ask Again Button centered below the export buttons
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        col_sp1, col_ask_btn, col_sp2 = st.columns([1.2, 1, 1.2])
        with col_ask_btn:
            if st.button("Ask Again", type="secondary", use_container_width=True):
                st.session_state.report_data = None
                st.rerun()

    with tab_sources:
        st.markdown(f"#### Retrieved Evidence Sources ({len(data['retrieved_records'])})")
        for idx, rec in enumerate(data['retrieved_records'], 1):
            is_pdf = bool(rec.get("full_text"))
            pdf_tag = "Full-Text Attached" if is_pdf else "Abstract Only"
            raw_title = rec.get('title', 'Unknown Title').strip()
            # Clean header label prevents Streamlit markdown parsing collisions
            expander_title = f"Source {idx} — {raw_title}"
            
            with st.expander(expander_title, expanded=(idx == 1)):
                st.markdown(f"**Source ID / URL:** `{rec.get('source_id', 'N/A')}`")
                st.markdown(f"**Date:** {rec.get('date', 'N/A')} | **Subject:** {rec.get('subject', 'General')}")
                st.markdown(f"**Evidence Level:** `{pdf_tag}`")
                st.markdown("**Abstract / Content Summary:**")
                st.info(rec.get("abstract") or rec.get("full_text") or "No excerpt available.")
                if str(rec.get("source_id", "")).startswith("http"):
                    st.markdown(f"[🔗 Open Direct Source URL]({rec.get('source_id')})")

    with tab_diagnostics:
        st.markdown("#### FAISS Vector Search Diagnostics")
        if data['raw_results'] and not data['use_fallback']:
            for rank, (meta, score, content) in enumerate(data['raw_results'], 1):
                st.markdown(f"**Rank #{rank} | L2 Distance: `{score:.4f}`** — {meta.get('title', 'Untitled')}")
                st.caption(content[:250] + "...")
                st.divider()
        elif data['is_custom_pdf']:
            st.info("Direct uploaded document analysis was executed.")
        else:
            st.info("Tavily live web search was triggered because the best match exceeded the similarity threshold.")