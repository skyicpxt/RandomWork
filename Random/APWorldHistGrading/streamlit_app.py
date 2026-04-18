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
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from grader import DEFAULT_MODEL, GradeResult, explain_changes, grade_essay, revise_answer
from qa_parser import (
    QAFormatError as _QAFormatError,
    has_multi_question_markers as _has_multi_question_markers,
    normalize_entry as _normalize_entry,
    parse_qa_text as _parse_qa_entries,
)
from report_formatter import format_grade_report, format_summary


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR.parent / ".env")

_SUPPORTED_MODELS = [DEFAULT_MODEL]
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

Question1
(a)
Q: Briefly describe ONE cause of the Columbian Exchange.
A: The Columbian Exchange was caused primarily by Columbus's 1492 voyage...

(b)
Q: Briefly explain ONE effect of the Columbian Exchange on the Americas.
A: One significant effect was the catastrophic demographic collapse...

(c)
Q: Briefly explain ONE effect on Europe or Africa.
A: One major effect on Europe was the introduction of new crops...

Question2
(a)
Q: Describe one change that resulted from the voyages depicted in the map.
A: One change resulting from Zheng He's voyages was China's status...

(b)
Q: Describe one continuity in the Indian Ocean basin from 1200-1450.
A: One continuity was reliance on monsoon winds...

(c)
Q: Describe one way the decision to end the voyages impacted China.
A: The end of Zheng He's voyages marked the end of China's dominance...
"""

_FORMAT_EXAMPLES = {"DBQ": _FORMAT_DBQ, "LEQ": _FORMAT_LEQ, "SAQ": _FORMAT_SAQ}


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
# Report formatting  (delegated to report_formatter.py, shared with main.py)
# ---------------------------------------------------------------------------


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
# Grading helpers — parse Q&A, call the API, and render results in place.
# _grade_and_render dispatches to single- or multi-question mode automatically.
# ---------------------------------------------------------------------------

def _grade_single_entry(
    client: OpenAI,
    cat: str,
    question: str,
    answer: str,
    docs: Optional[str],
    model: str,
    label: str = "",
) -> Optional[GradeResult]:
    """
    Grades a single prepared entry (question + answer already extracted) and renders results.
    Returns the GradeResult or None on failure.
    """
    with st.spinner(f"Grading {cat}{' (' + label + ')' if label else ''} with {model}… this may take 15–30 seconds."):
        try:
            result: GradeResult = grade_essay(
                client=client,
                category=cat,
                question=question,
                answer=answer,
                dbq_docs=docs,
                model=model,
            )
        except Exception as e:
            st.error(f"Grading failed: {e}")
            return None

    st.success(f"Grading complete! Score: {result.total_earned}/{result.total_possible}")
    render_grade_result(result)
    return result


def _render_download_buttons(report_txt: str, filename: str) -> None:
    """Renders the copy-text expander for a report."""
    with st.expander("Copy Report Text", expanded=False):
        st.code(report_txt, language=None)


# ---------------------------------------------------------------------------
# Revision helpers
# ---------------------------------------------------------------------------

def _extract_saq_parts(merged_text: str, label_key: str = "Q") -> list[tuple[str, str]]:
    """
    Extracts [(part_letter, text)] from a merged SAQ string such as:
    'Part A\\nQ: [text]\\n\\nPart B\\nQ: [text]\\n\\nPart C\\nQ: [text]'
    label_key is "Q" for questions or "A" for original answers.
    """
    parts: list[tuple[str, str]] = []
    for chunk in merged_text.split("\n\n"):
        lines = [ln for ln in chunk.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0].strip()
        if not first.upper().startswith("PART"):
            continue
        letter = first.split()[-1].lower()
        rest = "\n".join(lines[1:]).strip()
        # Strip optional "Q:" / "A:" prefix
        if rest.upper().startswith(f"{label_key.upper()}:"):
            rest = rest[len(label_key) + 1:].strip()
        parts.append((letter, rest))
    return parts


def _parse_revised_saq(revised: str) -> list[tuple[str, str]]:
    """
    Parses the model's revised SAQ output (expected format: '(a)\\n[text]\\n\\n(b)\\n[text]...')
    into [(letter, answer_text)]. Falls back to [('', full_text)] if no markers found.
    """
    subpart_re = re.compile(r"^\s*\(\s*([a-zA-Z])\s*\)\s*$", re.MULTILINE)
    chunks = subpart_re.split(revised)
    result: list[tuple[str, str]] = []
    if len(chunks) >= 3:
        i = 1
        while i + 1 < len(chunks):
            letter = chunks[i].strip().lower()
            text = chunks[i + 1].strip()
            if letter and text:
                result.append((letter, text))
            i += 2
    if not result:
        # Model didn't follow (a)/(b)/(c) structure — return the whole text as one block
        result = [("", revised.strip())]
    return result


def _build_revised_output(
    cat: str,
    question: str,
    revised: str,
    question_label: str = "",
) -> str:
    """
    Constructs the copyable revised-output text block in input format
    but with 'RA:' in place of 'A:'.
    """
    lines: list[str] = ["Revised answer", ""]
    if question_label:
        lines.append(question_label)
    if cat == "SAQ":
        q_parts = _extract_saq_parts(question, "Q")
        ra_parsed = _parse_revised_saq(revised)
        ra_dict = {letter: text for letter, text in ra_parsed if letter}
        fallback_ra = ra_parsed[0][1] if ra_parsed else revised
        if q_parts:
            for letter, q_text in q_parts:
                lines.append(f"({letter})")
                lines.append(f"Q: {q_text}")
                ra_text = ra_dict.get(letter, fallback_ra)
                lines.append(f"RA: {ra_text}")
                lines.append("")
        else:
            # Fallback: question not in merged Part A/B/C format.
            lines.append(f"Q: {question.strip()}")
            lines.append(f"RA: {revised.strip()}")
    else:
        lines.append(f"Q: {question.strip()}")
        lines.append(f"RA: {revised.strip()}")
    return "\n".join(lines).strip()


def _render_changes_explanation(
    client: OpenAI,
    cat: str,
    original: str,
    revised: str,
    model: str,
) -> str:
    """
    Calls explain_changes() and renders the result in a collapsible expander.
    Shows a spinner while the API call is in flight.
    Returns the explanation text so callers can include it in copyable output,
    or an empty string if the call failed.
    """
    with st.expander("What changed and why", expanded=True):
        with st.spinner("Explaining changes…"):
            try:
                explanation = explain_changes(client, cat, original, revised, model)
            except Exception as e:
                st.error(f"Could not generate change explanation: {e}")
                return ""
        st.markdown(explanation)
    return explanation


def _spaced_paragraphs(text: str) -> str:
    """
    Ensures a blank line between every paragraph so Streamlit markdown
    renders each paragraph as a separate block rather than collapsing them.
    Consecutive newlines are normalised to exactly two, then doubled to four
    so the markdown parser always sees a true paragraph break.
    """
    import re
    # Collapse 3+ newlines to 2, then replace every double-newline with 4
    # newlines so Streamlit's markdown renderer produces a visible gap.
    normalised = re.sub(r"\n{3,}", "\n\n", text.strip())
    return normalised.replace("\n\n", "\n\n\n\n")


def _render_revised_question(cat: str, question: str, revised: str) -> None:
    """
    Renders one revised Q/RA pair in Streamlit.
    Both Q: and RA: are shown together in the same block so the question is
    always visible alongside the generated revision. SAQ is rendered part-by-part.
    """
    if cat == "SAQ":
        q_parts = _extract_saq_parts(question, "Q")
        ra_parsed = _parse_revised_saq(revised)
        ra_dict = {letter: text for letter, text in ra_parsed if letter}
        fallback_ra = ra_parsed[0][1] if ra_parsed else revised
        if q_parts:
            for letter, q_text in q_parts:
                ra_text = _spaced_paragraphs(ra_dict.get(letter, fallback_ra))
                st.success(f"**Part {letter.upper()} — Q:** {q_text}\n\n**RA:** {ra_text}")
        else:
            # Fallback: question not in merged Part A/B/C format — show as a single block.
            st.success(f"**Q:** {question.strip()}\n\n**RA:** {_spaced_paragraphs(revised)}")
    else:
        st.success(f"**Q:** {question.strip()}\n\n**RA:** {_spaced_paragraphs(revised)}")


def _revise_and_render(
    client: OpenAI,
    essay_type: str,
    raw_qa: str,
    dbq_docs_text: Optional[str],
    model: str,
) -> None:
    """
    Parses raw_qa, calls revise_answer for each Q/A pair, displays the
    revised answers, and provides a copyable output block.
    """
    if _has_multi_question_markers(raw_qa):
        # ── Multi-question path ──
        try:
            entries = _parse_qa_entries(raw_qa, default_category=essay_type)
        except _QAFormatError as e:
            st.error(str(e))
            return
        if not entries:
            st.error("No questions found in the input. Check the format guide.")
            return

        st.info(f"Detected **{len(entries)} question(s)** — revising each separately…")
        output_blocks: list[str] = []
        all_ok = True

        for i, entry in enumerate(entries, 1):
            cat = entry["category"]
            q = entry["question"]
            a = entry["answer"]
            docs = entry.get("docs") or dbq_docs_text or None
            label = entry.get("question_label") or f"Question{i}"

            st.markdown("---")
            st.subheader(f"Question {i} — {label}")

            with st.spinner(f"Revising {cat} {label} with {model}…"):
                try:
                    revised = revise_answer(client, cat, q, a, model, docs)
                except Exception as e:
                    st.error(f"Revision failed for {label}: {e}")
                    all_ok = False
                    continue

            _render_revised_question(cat, q, revised)
            explanation = _render_changes_explanation(client, cat, a, revised, model)
            block = _build_revised_output(cat, q, revised, label)
            if explanation:
                block += f"\n\nWhat changed and why\n{explanation}"
            output_blocks.append(block)

        if output_blocks:
            full_output = f"CATEGORY: {essay_type}\n\n" + "\n\n".join(output_blocks)
            st.divider()
            st.subheader("Full Revised Output")
            with st.expander("Copy Revised Text", expanded=True):
                st.code(full_output, language=None)

    else:
        # ── Single-question path ──
        try:
            entries = _parse_qa_entries(raw_qa, default_category=essay_type)
        except _QAFormatError as e:
            st.error(str(e))
            return
        if not entries:
            st.error("No question found in the input. Check the format guide.")
            return
        entry = entries[0]
        question = entry["question"]
        answer = entry["answer"]
        docs_from_text = entry.get("docs") or None

        effective_docs = dbq_docs_text or docs_from_text or None

        with st.spinner(f"Revising {essay_type} with {model}…"):
            try:
                revised = revise_answer(client, essay_type, question, answer, model, effective_docs)
            except Exception as e:
                st.error(f"Revision failed: {e}")
                return

        _render_revised_question(essay_type, question, revised)
        explanation = _render_changes_explanation(client, essay_type, answer, revised, model)
        revised_block = _build_revised_output(essay_type, question, revised)
        if explanation:
            revised_block += f"\n\nWhat changed and why\n{explanation}"
        full_output = f"CATEGORY: {essay_type}\n\n" + revised_block
        st.divider()
        st.subheader("Full Revised Output")
        with st.expander("Copy Revised Text", expanded=True):
            st.code(full_output, language=None)


def _grade_and_render_single(
    client: OpenAI,
    essay_type: str,
    raw_qa: str,
    dbq_docs_text: Optional[str],
    model: str,
) -> None:
    """Parses raw_qa as a single question, grades it, and renders the full results UI."""
    try:
        entries = _parse_qa_entries(raw_qa, default_category=essay_type)
    except _QAFormatError as e:
        st.error(str(e))
        return
    if not entries:
        st.error("No question found in the input. Check the format guide.")
        return
    entry = entries[0]
    question = entry["question"]
    answer = entry["answer"]
    docs_from_text = entry.get("docs") or None

    effective_docs: Optional[str] = dbq_docs_text or docs_from_text or None
    result = _grade_single_entry(client, essay_type, question, answer, effective_docs, model)
    if result is None:
        return

    report_txt = format_grade_report(result)
    _render_download_buttons(report_txt, f"grading_report_{essay_type}.txt")


def _grade_and_render_multi(
    client: OpenAI,
    essay_type: str,
    raw_qa: str,
    dbq_docs_text: Optional[str],
    model: str,
) -> None:
    """
    Parses raw_qa as multiple Question1/Question2/… entries, grades each one
    separately, and renders per-question results plus a combined summary.
    """
    try:
        entries = _parse_qa_entries(raw_qa, default_category=essay_type)
    except _QAFormatError as e:
        st.error(str(e))
        return

    if not entries:
        st.error("No questions found in the input. Check the format guide.")
        return

    st.info(f"Detected **{len(entries)} question(s)** in your input. Grading each separately…")

    if len(entries) == 1:
        entry = entries[0]
        cat = entry["category"]
        q, a = _normalize_entry(entry)
        docs = entry.get("docs") or dbq_docs_text or None
        result = _grade_single_entry(client, cat, q, a, docs, model)
        if result:
            _render_download_buttons(format_grade_report(result), f"grading_report_{cat}.txt")
        return

    # Multiple questions — grade each and show results with a per-question header
    graded: list[tuple[int, str, GradeResult]] = []
    all_reports: list[str] = []

    for i, entry in enumerate(entries, 1):
        cat = entry["category"]
        q, a = _normalize_entry(entry)
        docs = entry.get("docs") or dbq_docs_text or None
        label = entry.get("question_label") or f"Question{i}"

        st.markdown(f"---")
        st.subheader(f"Question {i} — {cat} ({label})")

        result = _grade_single_entry(client, cat, q, a, docs, model, label=label)
        if result is None:
            continue

        graded.append((i, label, result))
        all_reports.append(format_grade_report(result))

    if not graded:
        return

    # Combined summary
    st.divider()
    total_e = sum(r.total_earned for _, _, r in graded)
    total_p = sum(r.total_possible for _, _, r in graded)
    pct = total_e / total_p if total_p else 0

    st.subheader("Combined Score Summary")
    col1, col2 = st.columns([1, 3])
    col1.metric("Total Score", f"{total_e} / {total_p}")
    col2.progress(pct, text=f"{pct:.0%} overall ({total_e}/{total_p} pts)")

    summary_data = [
        {"#": i, "Label": lbl, "Type": r.category, "Score": f"{r.total_earned}/{r.total_possible}"}
        for i, lbl, r in graded
    ]
    st.table(summary_data)

    all_reports.append(format_summary(graded))
    combined_report = "\n\n".join(all_reports)
    _render_download_buttons(combined_report, "grading_report_multi.txt")


def _grade_and_render(
    client: OpenAI,
    essay_type: str,
    raw_qa: str,
    dbq_docs_text: Optional[str],
    model: str,
) -> None:
    """
    Entry point for grading. Automatically detects multi-question format
    (Question1 / Question2 / … markers) and dispatches accordingly.
    """
    if _has_multi_question_markers(raw_qa):
        _grade_and_render_multi(client, essay_type, raw_qa, dbq_docs_text, model)
    else:
        _grade_and_render_single(client, essay_type, raw_qa, dbq_docs_text, model)


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
                "For SAQ, use `Question1`, `Question2`, … labels to separate multiple questions. "
                "Within each question, use `(a)`, `(b)`, `(c)` labels before each sub-part's `Q:` and `A:`. "
                "A single question without `Question1` labels is also accepted."
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

    # ── Action buttons ──
    st.divider()
    _btn_col_grade, _btn_col_revise = st.columns(2)
    grade_button = _btn_col_grade.button("Grade Essay", type="primary", use_container_width=True)
    revise_button = _btn_col_revise.button("✏️ Revise Answer", use_container_width=True)

    # ── Mismatch confirmation UI ──
    # Shown in place of grading/revision when a mismatch was detected on the previous run.
    # The user must explicitly choose to continue or cancel before the action proceeds.
    _pending = st.session_state["_mismatch_pending"]
    if _pending is not None:
        st.divider()
        st.subheader("Essay Type Mismatch Detected")
        for w in _pending["warnings"]:
            st.warning(w)
        _pending_action = _pending.get("action", "grade")
        _action_label = "Revise" if _pending_action == "revise" else "Grade"
        st.markdown(f"Would you like to continue {_action_label.lower()}ing with the selected essay type anyway?")
        col_yes, col_no = st.columns(2)
        _proceed = col_yes.button(f"Continue — {_action_label} Anyway", type="primary", use_container_width=True)
        _cancel = col_no.button("Cancel — Go Back", use_container_width=True)

        if _cancel:
            st.session_state["_mismatch_pending"] = None
            st.rerun()

        if _proceed:
            _p = st.session_state["_mismatch_pending"]
            st.session_state["_mismatch_pending"] = None
            if _p.get("action") == "revise":
                _revise_and_render(
                    client, _p["essay_type"], _p["raw_qa"], _p["dbq_docs_text"], _p["model"]
                )
            else:
                _grade_and_render(
                    client, _p["essay_type"], _p["raw_qa"], _p["dbq_docs_text"], _p["model"]
                )

    elif grade_button or revise_button:
        if not raw_qa:
            st.error("Please enter or upload the Q&A text before proceeding.")
            st.stop()

        # Multi-question format bypasses mismatch detection — the user has
        # explicitly structured input with Question1/Question2/… labels.
        _is_multi = _has_multi_question_markers(raw_qa)

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

        # Guard 2 always runs — DBQ selected but answer contains no document citations.
        # This catches accidentally picking DBQ when the content is an SAQ or LEQ,
        # regardless of whether the input uses multi-question (Question1/2/…) format.
        if essay_type == "DBQ" and not _has_doc_citations:
            _warnings.append(
                "**DBQ mismatch:** You selected DBQ but the answer contains no document "
                "citations (e.g. \"Document 1\", \"Doc. 2\"). DBQ essays must reference "
                "the provided source documents. If this is an SAQ or LEQ, please cancel and "
                "change the essay type above. If you continue, Evidence from Documents "
                "points will almost certainly not be earned."
            )

        if not _is_multi:
            # Guard 1 — SAQ selected but input looks like a full essay.
            if essay_type == "SAQ" and not _has_part_labels and _answer_word_count > 150:
                _warnings.append(
                    f"**SAQ mismatch:** You selected SAQ but the input looks like a full essay "
                    f"(no (a)/(b)/(c) part labels found, and the text is {_answer_word_count} words long). "
                    "SAQ answers should be short paragraphs labeled by part. "
                    "If this is an LEQ or DBQ, please cancel and change the essay type above."
                )

            # Guard 3 — LEQ or SAQ selected but answer contains document citations.
            if essay_type in ("LEQ", "SAQ") and _has_doc_citations:
                _warnings.append(
                    f"**{essay_type} mismatch:** You selected {essay_type} but the answer "
                    "contains document citations (e.g. \"Document 1\"). "
                    f"{essay_type} essays do not use source documents — only DBQ does. "
                    "If this is a DBQ, please cancel and change the essay type above."
                )

        _current_action = "revise" if revise_button else "grade"
        if _warnings:
            # Save params and rerun to show the blocking confirmation UI.
            st.session_state["_mismatch_pending"] = {
                "warnings": _warnings,
                "essay_type": essay_type,
                "raw_qa": raw_qa,
                "dbq_docs_text": dbq_docs_text,
                "model": model,
                "action": _current_action,
            }
            st.rerun()
        elif _current_action == "revise":
            _revise_and_render(client, essay_type, raw_qa, dbq_docs_text, model)
        else:
            _grade_and_render(client, essay_type, raw_qa, dbq_docs_text, model)


if __name__ == "__main__":
    main()
