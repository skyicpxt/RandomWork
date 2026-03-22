# Streamlit web app for AP World History: Modern essay grading.
# Wraps the existing grade_essay() function from grader.py with a UI that supports:
#   - Single combined Q&A text box (same format as the existing QandA .txt files)
#   - File upload (PDF, image, or .txt) as an alternative to typing
#   - Multi-file upload for DBQ source documents (PDF + JPEG/PNG), separate and optional
#   - On-screen results display matching the grading_report.txt format
#   - Downloadable .txt report
#
# Run: streamlit run streamlit_app.py

import base64
import io
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from grader import GradeResult, grade_essay

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR.parent / ".env")

_DEFAULT_MODEL = "gpt-5.4"
_SUPPORTED_MODELS = ["gpt-5.4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
_ESSAY_TYPES = ["DBQ", "LEQ", "SAQ"]
_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
_PDF_TYPE = ["pdf"]
_TXT_TYPE = ["txt"]
_QA_FILE_TYPES = _TXT_TYPE + _PDF_TYPE + _IMAGE_TYPES   # allowed for combined Q&A upload
_DOC_FILE_TYPES = _PDF_TYPE + _IMAGE_TYPES               # allowed for DBQ source docs

# ---------------------------------------------------------------------------
# Format examples shown in the UI
# ---------------------------------------------------------------------------

_FORMAT_LEQ = """\
CATEGORY: LEQ

Q: Evaluate the extent to which industrialization caused changes in the role of women.

A: The Industrial Revolution fundamentally altered women's roles...
   (continue your essay here)
"""

_FORMAT_DBQ = """\
CATEGORY: DBQ

Q: Evaluate the extent to which the Silk Roads facilitated cultural exchange.

A: The Silk Roads connected China to the Mediterranean world...
   (continue your essay here)

Note: Source documents are uploaded separately below (optional if pasted in Q&A).
You may also include a DOCS: section in this box if not uploading files.
"""

_FORMAT_SAQ = """\
CATEGORY: SAQ

(a)
Q: Briefly describe ONE cause of the Columbian Exchange.
A: The Columbian Exchange was caused primarily by Columbus's 1492 voyage...

(b)
Q: Briefly explain ONE effect of the Columbian Exchange on the Americas.
A: One significant effect was the catastrophic demographic collapse...

(c)
Q: Briefly explain ONE effect on Europe or Africa.
A: One major effect on Europe was the introduction of new crops...
"""

_FORMAT_EXAMPLES = {"DBQ": _FORMAT_DBQ, "LEQ": _FORMAT_LEQ, "SAQ": _FORMAT_SAQ}


# ---------------------------------------------------------------------------
# Q&A text parser
# ---------------------------------------------------------------------------

# Matches standalone sub-part labels like "(a)", "(b)", "(c)"
_SAQ_SUBPART_RE = re.compile(r"^\s*\(\s*([a-zA-Z])\s*\)\s*$")


def _parse_leq_dbq(lines: list[str]) -> tuple[str, str, Optional[str]]:
    """
    Parses LEQ or DBQ Q&A text (list of lines) using a state machine.
    Returns (question, answer, dbq_docs_from_text).
    dbq_docs_from_text is None if no DOCS: section is present.
    """
    state = "before_q"
    q_lines: list[str] = []
    a_lines: list[str] = []
    docs_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("CATEGORY:"):
            continue

        if upper.startswith("Q:"):
            state = "in_q"
            rest = stripped[2:].strip()
            if rest:
                q_lines.append(rest)
            continue

        if upper.startswith("DOCS:"):
            state = "in_docs"
            rest = stripped[5:].strip()
            if rest:
                docs_lines.append(rest)
            continue

        if upper.startswith("A:"):
            state = "in_a"
            rest = stripped[2:].strip()
            if rest:
                a_lines.append(rest)
            continue

        if state == "in_q":
            q_lines.append(line.rstrip())
        elif state == "in_docs":
            docs_lines.append(line.rstrip())
        elif state == "in_a":
            a_lines.append(line.rstrip())

    question = "\n".join(q_lines).strip()
    answer = "\n".join(a_lines).strip()
    docs = "\n".join(docs_lines).strip() or None
    return question, answer, docs


def _parse_saq(lines: list[str]) -> tuple[str, str, None]:
    """
    Parses SAQ Q&A text with (a)/(b)/(c) sub-parts.
    Returns (combined_question, combined_answer, None).
    Sub-parts are merged with Part A / Part B / Part C headers.
    """
    # Split into sub-part blocks
    SubPart = dict  # {letter, q_lines, a_lines}
    parts: list[SubPart] = []
    current: Optional[SubPart] = None
    sub_state = "before_part"

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("CATEGORY:"):
            continue

        m = _SAQ_SUBPART_RE.match(stripped)
        if m:
            current = {"letter": m.group(1).upper(), "q_lines": [], "a_lines": []}
            parts.append(current)
            sub_state = "before_q"
            continue

        if current is None:
            continue

        if upper.startswith("Q:"):
            sub_state = "in_q"
            rest = stripped[2:].strip()
            if rest:
                current["q_lines"].append(rest)
            continue

        if upper.startswith("A:"):
            sub_state = "in_a"
            rest = stripped[2:].strip()
            if rest:
                current["a_lines"].append(rest)
            continue

        if sub_state == "in_q":
            current["q_lines"].append(line.rstrip())
        elif sub_state == "in_a":
            current["a_lines"].append(line.rstrip())

    if not parts:
        # Fallback: no sub-part markers found — treat as plain Q/A
        q, a, _ = _parse_leq_dbq(lines)
        return q, a, None

    q_chunks: list[str] = []
    a_chunks: list[str] = []
    for p in parts:
        letter = p["letter"]
        q_text = "\n".join(p["q_lines"]).strip()
        a_text = "\n".join(p["a_lines"]).strip()
        q_chunks.append(f"Part {letter}\nQ: {q_text}")
        a_chunks.append(f"Part {letter}\nA: {a_text}")

    return "\n\n".join(q_chunks), "\n\n".join(a_chunks), None


def parse_qa_text(
    text: str,
    category: str,
) -> tuple[str, str, Optional[str]]:
    """
    Parses combined Q&A text in the same format as the existing QandA .txt files.
    Returns (question, answer, dbq_docs_from_text).
    dbq_docs_from_text is None unless a DOCS: section is found in the text.
    For SAQ, sub-parts (a)/(b)/(c) are merged into combined question/answer strings.
    Raises ValueError if question or answer cannot be extracted.
    """
    lines = text.splitlines()
    if category == "SAQ":
        question, answer, docs = _parse_saq(lines)
    else:
        question, answer, docs = _parse_leq_dbq(lines)

    if not question:
        raise ValueError(
            "Could not find a question (Q:) in the input. "
            "Make sure your text includes a line starting with 'Q:'."
        )
    if not answer:
        raise ValueError(
            "Could not find an answer (A:) in the input. "
            "Make sure your text includes a line starting with 'A:'."
        )
    return question, answer, docs


# ---------------------------------------------------------------------------
# File text extraction
# ---------------------------------------------------------------------------

def _pdf_to_text(file_bytes: bytes) -> str:
    """Extracts plain text from a PDF using pypdf."""
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _image_to_text_via_vision(file_bytes: bytes, mime_type: str, client: OpenAI, model: str) -> str:
    """
    Sends an image to the OpenAI vision API and returns the extracted text content.
    Used for JPEG/PNG question/answer sheets or DBQ source documents.
    """
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please extract and transcribe all text visible in this image exactly as written. "
                            "Preserve paragraph breaks. Do not add commentary or headings — output only the transcribed text."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content or ""


def extract_text_from_file(
    uploaded_file,
    client: OpenAI,
    model: str,
) -> str:
    """
    Extracts plain text from an uploaded Streamlit file object.
    .txt files are decoded directly.
    PDF files are processed locally with pypdf.
    Image files (JPEG/PNG/WebP) are sent to the OpenAI vision API for OCR.
    Returns the extracted text string.
    """
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        return _pdf_to_text(file_bytes)

    if name.endswith(".png"):
        mime = "image/png"
    elif name.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    return _image_to_text_via_vision(file_bytes, mime, client, model)


def extract_dbq_docs_text(
    uploaded_files: list,
    client: OpenAI,
    model: str,
) -> str:
    """
    Extracts and concatenates text from multiple DBQ source document uploads.
    Each file is labeled as Document 1, Document 2, … in the output string,
    which is passed directly to grade_essay() as the dbq_docs argument.
    Both PDF and image files are supported.
    """
    doc_texts: list[str] = []
    for i, f in enumerate(uploaded_files, start=1):
        text = extract_text_from_file(f, client, model)
        doc_texts.append(f"Document {i}:\n{text.strip()}")
    return "\n\n".join(doc_texts)


# ---------------------------------------------------------------------------
# Report formatting  (mirrors grading_report.txt style)
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20


def _score_bar(earned: int, possible: int) -> str:
    """Returns a filled/empty block bar string representing score fraction."""
    if possible == 0:
        return ""
    filled = round((earned / possible) * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _wrap(text: str, width: int = 58, indent: str = "  ") -> str:
    """Wraps text at width, indenting all lines."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def format_report_txt(result: GradeResult) -> str:
    """
    Formats a GradeResult as a plain-text report matching the style of grading_report.txt.
    Returns the full report string suitable for file download.
    """
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  AP WORLD HISTORY: MODERN — GRADING REPORT")
    lines.append(f"  Essay Type: {result.category}")
    lines.append(sep)
    lines.append("")

    # ── Section 1: Score Breakdown ──
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │             1. SCORE BREAKDOWN                  │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    bar = _score_bar(result.total_earned, result.total_possible)
    lines.append(f"  TOTAL: {bar} {result.total_earned}/{result.total_possible}")
    lines.append("")
    lines.append(f"  {'Criterion':<54} {'Earned':>6}  {'Max':>4}")
    lines.append(f"  {'-'*54} {'------':>6}  {'----':>4}")
    for cr in result.criteria_results:
        check = "✔" if cr.points_earned > 0 else "✘"
        lines.append(f"  {check} {cr.name:<53} {cr.points_earned:>6}  {cr.max_points:>4}")
    lines.append("")

    # ── Section 2: Evidence ──
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │        2. EVIDENCE THAT EARNED EACH POINT       │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    earned_criteria = [cr for cr in result.criteria_results if cr.points_earned > 0]
    if earned_criteria:
        for cr in earned_criteria:
            lines.append(f"  ✔ {cr.name} ({cr.points_earned}/{cr.max_points} pt{'s' if cr.max_points != 1 else ''})")
            if cr.evidence and cr.evidence != "N/A":
                lines.append(_wrap(f'"{cr.evidence}"'))
            if cr.evidence_comment and cr.evidence_comment != "N/A":
                lines.append(_wrap(f"↳ {cr.evidence_comment}"))
            lines.append("")
    else:
        lines.append("  No points were earned.")
        lines.append("")

    # ── Section 3: Points Not Earned ──
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │       3. POINTS NOT EARNED AND WHY              │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    missed = [cr for cr in result.criteria_results if cr.points_earned < cr.max_points]
    if missed:
        for cr in missed:
            missed_pts = cr.max_points - cr.points_earned
            lines.append(f"  ✘ {cr.name} (missed {missed_pts} of {cr.max_points} pt{'s' if cr.max_points != 1 else ''})")
            if cr.not_earned_reason:
                lines.append(_wrap(cr.not_earned_reason))
            lines.append("")
    else:
        lines.append("  All points were earned — excellent work!")
        lines.append("")

    # ── Section 4: Suggestions ──
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │         4. SUGGESTIONS TO IMPROVE               │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    for cr in result.criteria_results:
        if cr.suggestion:
            lines.append(f"  • [{cr.name}]")
            lines.append(_wrap(cr.suggestion))
            lines.append("")

    if result.overall_suggestions:
        lines.append("  ── Overall Feedback ──")
        lines.append(_wrap(result.overall_suggestions))
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# On-screen result rendering
# ---------------------------------------------------------------------------

def render_grade_result(result: GradeResult) -> None:
    """
    Renders a GradeResult in the Streamlit UI with expandable sections for
    score breakdown, evidence, missed points, and suggestions.
    """
    st.divider()

    pct = result.total_earned / result.total_possible if result.total_possible else 0
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            label=f"{result.category} Score",
            value=f"{result.total_earned} / {result.total_possible}",
        )
    with col2:
        st.progress(pct, text=f"{pct:.0%}")

    st.markdown("---")

    # ── Section 1: Score Breakdown ──
    with st.expander("1. Score Breakdown", expanded=True):
        st.caption("Each row is one independent scoring criterion. Earned / Max are shown on the right.")
        for cr in result.criteria_results:
            icon = "✅" if cr.points_earned > 0 else "❌"
            col_name, col_score = st.columns([5, 1])
            col_name.markdown(f"{icon} {cr.name}")
            col_score.markdown(
                f"**{cr.points_earned} / {cr.max_points}**",
                help=f"Earned {cr.points_earned} of {cr.max_points} available point(s) for this criterion.",
            )

    # ── Section 2: Evidence ──
    with st.expander("2. Evidence That Earned Each Point", expanded=True):
        earned = [cr for cr in result.criteria_results if cr.points_earned > 0]
        if earned:
            for cr in earned:
                st.markdown(f"**✅ {cr.name}** ({cr.points_earned}/{cr.max_points})")
                if cr.evidence and cr.evidence != "N/A":
                    st.markdown(f"> {cr.evidence}")
                if cr.evidence_comment and cr.evidence_comment != "N/A":
                    st.caption(f"💬 {cr.evidence_comment}")
        else:
            st.info("No points were earned.")

    # ── Section 3: Points Not Earned ──
    with st.expander("3. Points Not Earned and Why", expanded=True):
        missed = [cr for cr in result.criteria_results if cr.points_earned < cr.max_points]
        if missed:
            for cr in missed:
                missed_pts = cr.max_points - cr.points_earned
                st.markdown(f"**❌ {cr.name}** (missed {missed_pts} of {cr.max_points})")
                if cr.not_earned_reason:
                    st.markdown(cr.not_earned_reason)
        else:
            st.success("All points were earned — excellent work!")

    # ── Section 4: Suggestions ──
    with st.expander("4. Suggestions to Improve", expanded=True):
        for cr in result.criteria_results:
            if cr.suggestion:
                st.markdown(f"**• [{cr.name}]**")
                st.markdown(cr.suggestion)
        if result.overall_suggestions:
            st.markdown("---")
            st.markdown("**Overall Feedback**")
            st.info(result.overall_suggestions)


# ---------------------------------------------------------------------------
# Grading helper — parses Q&A, calls the API, and renders results in place.
# Extracted so it can be called from both the normal path and the mismatch-
# confirmation path without duplicating code.
# ---------------------------------------------------------------------------

def _grade_and_render(
    client: OpenAI,
    essay_type: str,
    raw_qa: str,
    dbq_docs_text: Optional[str],
    model: str,
) -> None:
    """Parses raw_qa, grades the essay, and renders the full results UI."""
    try:
        question, answer, docs_from_text = parse_qa_text(raw_qa, essay_type)
    except ValueError as e:
        st.error(str(e))
        return

    effective_docs: Optional[str] = dbq_docs_text or docs_from_text or None

    with st.spinner(f"Grading {essay_type} essay with {model}… this may take 15–30 seconds."):
        try:
            result: GradeResult = grade_essay(
                client=client,
                category=essay_type,
                question=question,
                answer=answer,
                dbq_docs=effective_docs,
                model=model,
            )
        except Exception as e:
            st.error(f"Grading failed: {e}")
            return

    st.success("Grading complete!")
    render_grade_result(result)

    report_txt = format_report_txt(result)

    col_dl, col_copy = st.columns(2)
    with col_dl:
        st.download_button(
            label="Download Report (.txt)",
            data=report_txt.encode("utf-8"),
            file_name=f"grading_report_{essay_type}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_copy:
        with st.expander("Copy Report Text", expanded=False):
            st.code(report_txt, language=None)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit AP World History grading app."""
    st.set_page_config(
        page_title="AP World History Grader",
        page_icon="📝",
        layout="wide",
    )
    st.title("AP World History: Modern — Essay Grader")
    st.caption("Powered by OpenAI using the official College Board rubrics.")

    # ── Sidebar: configuration ──
    with st.sidebar:
        st.header("Configuration")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-… (leave blank to use .env)",
            help="If blank, the app reads OPENAI_API_KEY from the .env file.",
        )
        model = st.selectbox("Model", _SUPPORTED_MODELS, index=0)
        st.divider()
        st.markdown(
            "**Supported input formats**\n"
            "- Type or paste text directly\n"
            "- Upload a `.txt`, PDF, JPEG, or PNG file\n\n"
            "DBQ source documents are uploaded separately and are optional."
        )

    resolved_key = api_key_input.strip() or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        st.warning("No OpenAI API key found. Enter one in the sidebar or set OPENAI_API_KEY in your .env file.")
        st.stop()

    client = OpenAI(api_key=resolved_key)

    # Pending mismatch confirmation: stores warnings + grading params until user decides.
    if "_mismatch_pending" not in st.session_state:
        st.session_state["_mismatch_pending"] = None

    # ── Essay type ──
    st.subheader("Essay Type")
    essay_type: str = st.radio(
        "Select the type of essay to grade:",
        _ESSAY_TYPES,
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Format guide ──
    with st.expander("Input Format Guide — click to expand", expanded=False):
        st.markdown(
            "Paste or type your Q&A in the box below using the format shown. "
            "The `CATEGORY:` line is optional (essay type is set above). "
            "Lines starting with `Q:` and `A:` mark the question and answer. "
        )
        if essay_type == "SAQ":
            st.markdown(
                "For SAQ, use `(a)`, `(b)`, `(c)` labels before each sub-part's `Q:` and `A:`."
            )
        if essay_type == "DBQ":
            st.markdown(
                "For DBQ, source documents can be uploaded separately below "
                "or included in the text with a `DOCS:` section."
            )
        st.code(_FORMAT_EXAMPLES[essay_type], language=None)

    st.divider()

    # ── Combined Q&A input ──
    st.subheader("Question & Answer")
    tab_text, tab_file = st.tabs(["Type / Paste", "Upload File (.txt, PDF, or Image)"])

    typed_qa = ""
    with tab_text:
        typed_qa = st.text_area(
            "Q&A input",
            key="qa_text",
            placeholder=(
                "Q: Your essay question here...\n\n"
                "A: Student's essay answer here..."
            ),
            height=380,
            label_visibility="collapsed",
        )

    uploaded_qa_text = ""
    with tab_file:
        uploaded_qa = st.file_uploader(
            "Upload Q&A file",
            type=_QA_FILE_TYPES,
            key="qa_file",
            label_visibility="collapsed",
        )
        if uploaded_qa is not None:
            with st.spinner(f"Reading {uploaded_qa.name}…"):
                try:
                    uploaded_qa_text = extract_text_from_file(uploaded_qa, client, model)
                    st.success(f"Loaded {len(uploaded_qa_text.split())} words from {uploaded_qa.name}")
                    with st.expander("Preview loaded text"):
                        st.text(uploaded_qa_text[:1200] + ("…" if len(uploaded_qa_text) > 1200 else ""))
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    # Uploaded file takes priority over typed text
    raw_qa = uploaded_qa_text.strip() if uploaded_qa_text.strip() else typed_qa.strip()

    # ── DBQ source documents (DBQ only, optional) ──
    dbq_docs_text: Optional[str] = None
    if essay_type == "DBQ":
        st.divider()
        st.subheader("DBQ Source Documents (optional)")
        st.caption(
            "Upload each source document as a separate file (PDF or image). "
            "Leave empty if you included a DOCS: section in the Q&A text above."
        )
        doc_files = st.file_uploader(
            "Upload source documents",
            type=_DOC_FILE_TYPES,
            accept_multiple_files=True,
            key="dbq_docs",
            label_visibility="collapsed",
        )
        if doc_files:
            with st.spinner(f"Extracting text from {len(doc_files)} document(s)…"):
                try:
                    dbq_docs_text = extract_dbq_docs_text(doc_files, client, model)
                    st.success(f"Extracted text from {len(doc_files)} document(s).")
                    with st.expander("Preview extracted documents"):
                        st.text(dbq_docs_text[:2000] + ("…" if len(dbq_docs_text) > 2000 else ""))
                except Exception as e:
                    st.error(f"Could not extract document text: {e}")

    # ── Grade button ──
    st.divider()
    grade_button = st.button("Grade Essay", type="primary", use_container_width=True)

    # ── Mismatch confirmation UI ──
    # Shown in place of grading when a mismatch was detected on the previous run.
    # The user must explicitly choose to continue or cancel before grading proceeds.
    _pending = st.session_state["_mismatch_pending"]
    if _pending is not None:
        st.divider()
        st.subheader("Essay Type Mismatch Detected")
        for w in _pending["warnings"]:
            st.warning(w)
        st.markdown("Would you like to continue grading with the selected essay type anyway?")
        col_yes, col_no = st.columns(2)
        _proceed = col_yes.button("Continue Anyway", type="primary", use_container_width=True)
        _cancel = col_no.button("Cancel — Go Back", use_container_width=True)

        if _cancel:
            st.session_state["_mismatch_pending"] = None
            st.rerun()

        if _proceed:
            _p = st.session_state["_mismatch_pending"]
            st.session_state["_mismatch_pending"] = None
            _grade_and_render(
                client, _p["essay_type"], _p["raw_qa"], _p["dbq_docs_text"], _p["model"]
            )

    elif grade_button:
        if not raw_qa:
            st.error("Please enter or upload the Q&A text before grading.")
            st.stop()

        # Heuristics for mismatch detection used across all three guards below.
        # Matches "Document 1", "Doc. 2", "Doc 3", "(Document 4)", etc.
        _doc_citation_re = re.compile(r"\b(?:document|doc\.?)\s*\d+", re.IGNORECASE)
        _has_doc_citations = bool(_doc_citation_re.search(raw_qa))
        _has_part_labels = bool(
            re.search(r"\(\s*[aAbBcC]\s*\)", raw_qa)
            or re.search(r"\bpart\s+[abc]\b", raw_qa, re.IGNORECASE)
        )
        _answer_word_count = len(raw_qa.split())

        _warnings: list[str] = []

        # Guard 1 — SAQ selected but input looks like a full essay.
        if essay_type == "SAQ" and not _has_part_labels and _answer_word_count > 150:
            _warnings.append(
                f"**SAQ mismatch:** You selected SAQ but the input looks like a full essay "
                f"(no (a)/(b)/(c) part labels found, and the text is {_answer_word_count} words long). "
                "SAQ answers should be short paragraphs labeled by part. "
                "If this is an LEQ or DBQ, please cancel and change the essay type above."
            )

        # Guard 2 — DBQ selected but answer contains no document citations.
        if essay_type == "DBQ" and not _has_doc_citations:
            _warnings.append(
                "**DBQ mismatch:** You selected DBQ but the answer contains no document "
                "citations (e.g. \"Document 1\", \"Doc. 2\"). DBQ essays must reference "
                "the provided source documents. If this is an LEQ, please cancel and "
                "change the essay type above. If you continue, Evidence from Documents "
                "points will almost certainly not be earned."
            )

        # Guard 3 — LEQ or SAQ selected but answer contains document citations.
        if essay_type in ("LEQ", "SAQ") and _has_doc_citations:
            _warnings.append(
                f"**{essay_type} mismatch:** You selected {essay_type} but the answer "
                "contains document citations (e.g. \"Document 1\"). "
                f"{essay_type} essays do not use source documents — only DBQ does. "
                "If this is a DBQ, please cancel and change the essay type above."
            )

        if _warnings:
            # Save params and rerun to show the blocking confirmation UI.
            st.session_state["_mismatch_pending"] = {
                "warnings": _warnings,
                "essay_type": essay_type,
                "raw_qa": raw_qa,
                "dbq_docs_text": dbq_docs_text,
                "model": model,
            }
            st.rerun()
        else:
            _grade_and_render(client, essay_type, raw_qa, dbq_docs_text, model)


if __name__ == "__main__":
    main()
