# Shared Q&A parsing logic used by both main.py (CLI) and streamlit_app.py (web UI).
# Handles single-question (Q:/A:) and multi-question (Question1/Question2/...) formats
# for all three AP World History essay types: DBQ, LEQ, and SAQ (with (a)/(b)/(c) sub-parts).

import re
import textwrap
from pathlib import Path
from typing import Optional


class QAFormatError(Exception):
    """Raised when a QandA file or text block does not follow the expected format."""


_VALID_CATEGORIES = {"DBQ", "LEQ", "SAQ"}

# Matches a standalone line like "Question1", "Question 2", "QUESTION3", or "Question 1:"
_QUESTION_MARKER_RE = re.compile(r"^Question\s*\d+\s*:?\s*$", re.IGNORECASE)

# Line-leading SAQ sub-part labels: (a), (b), … or A), B), … (AP-style)
_SAQ_SUBPART_PAREN_RE = re.compile(r"^(\s*)\(([a-z])\)\s*(.*)$", re.IGNORECASE)
_SAQ_SUBPART_UPPER_RE = re.compile(r"^(\s*)([A-Z])\)\s*(.*)$")

# Standalone line that is only a sub-part label, e.g. "(a)" — followed by Q: then A:
_SAQ_SUBPART_STANDALONE_RE = re.compile(r"^\s*\(\s*([a-z])\s*\)\s*$", re.IGNORECASE)


_FORMAT_HINT = textwrap.dedent("""\
    Expected format (blocks separated by '---'):

    Single question per block:
        CATEGORY: DBQ
        Q: <question>
        DOCS:              ← optional; DBQ only
        ...
        A: <answer>

    Multiple questions in one block (each QuestionN on its own line above Q:):
        CATEGORY: SAQ
        Question1
        Q: ...
        A: ...
        Question2
        Q: ...
        A: ...

        CATEGORY: DBQ (per-question DOCS — not shared; use a full set of 7 documents per DBQ)
        Question1
        DOCS:
        DOCUMENT 1 … DOCUMENT 7
        Q: ...
        A: ...
        Question2
        DOCS:
        DOCUMENT 1 … DOCUMENT 7
        Q: ...
        A: ...

        Optional: DOCS: once before Question1 applies only to the first question if that
        question has no DOCS: of its own (legacy).

    SAQ sub-parts (each sub-part has its own Q: and A:):
        CATEGORY: SAQ
        Question1
        (a)
        Q: First prompt…
        A: First answer…
        (b)
        Q: Second prompt…
        A: Second answer…

        Legacy (no QuestionN): same (a)/(b)/… blocks after CATEGORY: SAQ.

    Optional: older single-block style with (a)… and A)… on one line each is still
    normalized to Part A/B/C when grading.

    ---
        CATEGORY: LEQ
        Q: ...
        A: ...
""")


def _merge_saq_subqa_pairs(pairs: list[tuple[str, str, str]]) -> tuple[str, str]:
    """
    Builds combined question and answer strings from sub-parts (letter, q, a) for one SAQ.
    Uses Part A/B/C headers so grading previews stay readable after normalization.
    """
    q_chunks: list[str] = []
    a_chunks: list[str] = []
    for letter, q, a in pairs:
        part = chr(ord("A") + ord(letter.lower()) - ord("a"))
        q_chunks.append(f"Part {part}\nQ: {q}")
        a_chunks.append(f"Part {part}\nA: {a}")
    return "\n\n".join(q_chunks), "\n\n".join(a_chunks)


def _consume_saq_subqa_pairs(
    lines: list[str],
    start: int,
    end: int,
    context: str,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Parses (a) Q:/A:, (b) Q:/A:, … between start and end (each sub-part has its own Q: and A:).
    Returns (pairs, errors).
    """
    errors: list[str] = []
    pairs: list[tuple[str, str, str]] = []
    i = start
    while i < end:
        while i < end and not lines[i].strip():
            i += 1
        if i >= end:
            break
        stripped = lines[i].strip()
        m = _SAQ_SUBPART_STANDALONE_RE.match(stripped)
        if not m:
            errors.append(
                f"    • {context}: expected standalone '(a)', '(b)', … line, got {stripped[:48]!r}."
            )
            return [], errors
        letter = m.group(1).lower()
        i += 1
        while i < end and not lines[i].strip():
            i += 1
        if i >= end:
            errors.append(f"    • {context}: missing 'Q:' after '({letter})'.")
            return [], errors
        st = lines[i].strip()
        if not st.upper().startswith("Q:"):
            errors.append(
                f"    • {context}: expected 'Q:' after '({letter})', found {st[:48]!r}."
            )
            return [], errors
        q_lines = [st[2:].strip()]
        i += 1
        while i < end:
            st = lines[i].strip()
            if st.upper().startswith("A:"):
                break
            if st and _SAQ_SUBPART_STANDALONE_RE.match(st):
                errors.append(f"    • {context}: missing 'A:' before next sub-part.")
                return [], errors
            q_lines.append(lines[i].rstrip())
            i += 1
        if i >= end:
            errors.append(f"    • {context}: missing 'A:' for sub-part ({letter}).")
            return [], errors
        st = lines[i].strip()
        if not st.upper().startswith("A:"):
            errors.append(f"    • {context}: expected 'A:' for ({letter}).")
            return [], errors
        a_lines = [st[2:].strip()]
        i += 1
        while i < end:
            st = lines[i].strip()
            if st and _SAQ_SUBPART_STANDALONE_RE.match(st):
                break
            if st.upper().startswith("Q:"):
                errors.append(f"    • {context}: unexpected 'Q:' inside answer for ({letter}).")
                return [], errors
            a_lines.append(lines[i].rstrip())
            i += 1
        pairs.append((letter, "\n".join(q_lines).strip(), "\n".join(a_lines).strip()))
    if not pairs:
        errors.append(f"    • {context}: no '(a)' / '(b)' / … sub-parts with Q: and A: found.")
        return [], errors
    return pairs, []


def _parse_classic_qa_segment(lines: list[str]) -> tuple[str, str, list[str]]:
    """Parses a single Q:/A: pair (continuation lines allowed). Returns (q, a, errors)."""
    section = None
    q_lines: list[str] = []
    a_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("Q:"):
            section = "Q"
            q_lines.append(stripped[2:].strip())
        elif stripped.upper().startswith("A:"):
            section = "A"
            a_lines.append(stripped[2:].strip())
        elif section == "Q":
            q_lines.append(stripped)
        elif section == "A":
            a_lines.append(stripped)
    errs: list[str] = []
    if section is None:
        return "", "", ["missing 'Q:'"]
    q = "\n".join(q_lines).strip()
    a = "\n".join(a_lines).strip()
    if not q:
        errs.append("missing question text after 'Q:'")
    if not a:
        errs.append("missing 'A:' or answer text")
    if errs:
        return "", "", errs
    return q, a, []


def _parse_multi_question_segment(segment: list[str], label: str) -> tuple[str, str, str, list[str]]:
    """
    Parses one QuestionN block: optional DOCS:, then either (a)…Q/A…(b)… or a single Q:/A: pair.
    Returns (question, answer, pair_docs, errors).
    """
    errors: list[str] = []
    pair_docs_lines: list[str] = []
    i = 0
    while i < len(segment) and not segment[i].strip():
        i += 1
    if i < len(segment) and segment[i].strip().upper() == "DOCS:":
        i += 1
        while i < len(segment):
            st = segment[i].strip()
            if not st:
                pair_docs_lines.append("")
                i += 1
                continue
            if st.upper() == "DOCS:":
                break
            if _SAQ_SUBPART_STANDALONE_RE.match(st):
                break
            if st.upper().startswith("Q:"):
                break
            pair_docs_lines.append(segment[i].rstrip())
            i += 1
    rest = segment[i:]
    if not rest:
        return "", "", "", [f"    • '{label}': empty segment after optional DOCS."]
    st0 = rest[0].strip()
    if _SAQ_SUBPART_STANDALONE_RE.match(st0):
        pairs, perr = _consume_saq_subqa_pairs(rest, 0, len(rest), label)
        if perr:
            return "", "", "", perr
        mq, ma = _merge_saq_subqa_pairs(pairs)
        return mq, ma, "\n".join(pair_docs_lines).strip(), errors
    q, a, cerr = _parse_classic_qa_segment(rest)
    if cerr:
        return "", "", "", [f"    • '{label}': " + c for c in cerr]
    return q, a, "\n".join(pair_docs_lines).strip(), errors


def _legacy_block_looks_like_saq_subqa(lines: list[str]) -> bool:
    """True when CATEGORY is SAQ and the first content line after CATEGORY is '(a)', '(b)', …"""
    seen_category = False
    category = ""
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CATEGORY:"):
            category = stripped.split(":", 1)[1].strip().upper()
            seen_category = True
            continue
        if not seen_category:
            continue
        if not stripped:
            continue
        if category != "SAQ":
            return False
        if stripped.upper() == "DOCS:":
            return False
        return bool(_SAQ_SUBPART_STANDALONE_RE.match(stripped))
    return False


def _parse_block_legacy_saq_subqa(
    lines: list[str],
    block_num: int,
) -> tuple[list[dict], list[str]]:
    """Legacy SAQ block: (a) Q:/A:, (b) Q:/A:, … without QuestionN lines."""
    prefix = f"  Block #{block_num}:\n"
    category = ""
    start = 0
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip().upper()
            start = idx + 1
            break
    segment = lines[start:]
    pairs, perr = _consume_saq_subqa_pairs(segment, 0, len(segment), "SAQ")
    if perr:
        return [], [prefix + "\n".join(perr)]
    mq, ma = _merge_saq_subqa_pairs(pairs)
    return [
        {
            "category": category,
            "question": mq,
            "docs": "",
            "answer": ma,
            "question_label": "",
        }
    ], []


def _normalize_saq_subpart_labels(text: str) -> str:
    """
    Rewrites line-leading (a)/(b)/… or A)/B)/… to Part A, Part B, … so prompts align
    with the SAQ rubric criterion names (Part A / Part B / Part C).
    """
    lines: list[str] = []
    for line in text.splitlines():
        m = _SAQ_SUBPART_PAREN_RE.match(line)
        if m:
            indent, letter, rest = m.groups()
            part = chr(ord("A") + ord(letter.lower()) - ord("a"))
            lines.append(f"{indent}Part {part}: {rest}")
            continue
        m = _SAQ_SUBPART_UPPER_RE.match(line)
        if m:
            indent, letter, rest = m.groups()
            lines.append(f"{indent}Part {letter}: {rest}")
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_question_marker_line(stripped: str) -> bool:
    """Returns True if this line is a QuestionN marker (e.g. 'Question1', 'Question 2')."""
    return bool(stripped and _QUESTION_MARKER_RE.fullmatch(stripped))


def _block_has_question_markers(lines: list[str]) -> bool:
    """True if this block uses Question1 / Question2 / ... labels."""
    return any(_is_question_marker_line(line.strip()) for line in lines if line.strip())


def _parse_block_legacy(lines: list[str], block_num: int) -> tuple[list[dict], list[str]]:
    """
    Parses a block with a single Q / A (optional DOCS: for DBQ).
    Returns (entries, errors).
    """
    if _legacy_block_looks_like_saq_subqa(lines):
        return _parse_block_legacy_saq_subqa(lines, block_num)

    category = ""
    question_lines: list[str] = []
    docs_lines: list[str] = []
    answer_lines: list[str] = []
    section = None  # "Q", "DOCS", or "A"

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CATEGORY:"):
            category = stripped.split(":", 1)[1].strip().upper()
        elif stripped.upper().startswith("Q:"):
            section = "Q"
            question_lines.append(stripped[2:].strip())
        elif stripped.upper() == "DOCS:":
            section = "DOCS"
        elif stripped.upper().startswith("A:"):
            section = "A"
            answer_lines.append(stripped[2:].strip())
        else:
            if section == "Q":
                question_lines.append(stripped)
            elif section == "DOCS":
                docs_lines.append(line.rstrip())
            elif section == "A":
                answer_lines.append(stripped)

    question = "\n".join(question_lines).strip()
    docs = "\n".join(docs_lines).strip()
    answer = "\n".join(answer_lines).strip()

    block_errors: list[str] = []
    prefix = f"  Block #{block_num}:\n"

    if not category:
        block_errors.append("    • Missing 'CATEGORY: DBQ/LEQ/SAQ' line.")
    elif category not in _VALID_CATEGORIES:
        block_errors.append(
            f"    • Invalid category '{category}'. Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}."
        )

    if not question:
        block_errors.append("    • Missing 'Q: ...' line (question is empty).")

    if not answer:
        block_errors.append("    • Missing 'A: ...' line (answer is empty).")

    if answer and not question:
        block_errors.append("    • Answer found but no question — 'Q:' line must come before 'A:'.")

    if block_errors:
        return [], [prefix + "\n".join(block_errors)]

    return [
        {
            "category": category,
            "question": question,
            "docs": docs,
            "answer": answer,
            "question_label": "",
        }
    ], []


def _parse_block_multi(lines: list[str], block_num: int) -> tuple[list[dict], list[str]]:
    """
    Parses a block with Question1 / Question2 / … lines above each question segment.
    Each segment is either a single Q:/A: pair or SAQ sub-parts: (a) Q:/A:, (b) Q:/A:, …

    DOCS: placement:
      * After CATEGORY, before the first QuestionN — shared fallback for any pair that has
        no per-question DOCS (backward compatible).
      * After a QuestionN line, before Q: or (a) — documents for that question only.
    """
    block_errors: list[str] = []
    prefix = f"  Block #{block_num}:\n"

    category = ""
    category_line = -1
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip().upper()
            category_line = idx
            break
    else:
        return [], [prefix + "    • Missing 'CATEGORY: DBQ/LEQ/SAQ' line."]

    q_indices = [j for j in range(len(lines)) if _is_question_marker_line(lines[j].strip())]
    if not q_indices:
        return [], [prefix + "    • No 'Question1' / 'Question2' / … lines in this block."]

    first_q = q_indices[0]
    global_docs_lines: list[str] = []
    j = category_line + 1
    while j < first_q:
        stripped = lines[j].strip()
        if not stripped:
            j += 1
            continue
        if stripped.upper() == "DOCS:":
            j += 1
            while j < first_q:
                st2 = lines[j].strip()
                if not st2:
                    global_docs_lines.append("")
                    j += 1
                    continue
                if st2.upper() == "DOCS:":
                    break
                if _is_question_marker_line(st2):
                    break
                global_docs_lines.append(lines[j].rstrip())
                j += 1
            continue
        block_errors.append(f"    • Unexpected text before Question1: {stripped[:50]!r}")
        j += 1

    pairs: list[tuple[str, str, str, str]] = []
    pair_flush_count = 0

    for k, q_idx in enumerate(q_indices):
        label = lines[q_idx].strip()
        start = q_idx + 1
        end = q_indices[k + 1] if k + 1 < len(q_indices) else len(lines)
        segment = lines[start:end]

        q_text, a_text, pair_docs, seg_errs = _parse_multi_question_segment(segment, label)
        if seg_errs:
            block_errors.extend(seg_errs)
            continue
        if not q_text or not a_text:
            block_errors.append(f"    • '{label}' has empty Q or A.")
            continue
        local = pair_docs.strip()
        if not local and pair_flush_count == 0 and global_docs_lines:
            local = "\n".join(global_docs_lines).strip()
        pairs.append((label, q_text, a_text, local))
        pair_flush_count += 1

    if not category:
        block_errors.append("    • Missing 'CATEGORY: DBQ/LEQ/SAQ' line.")
    elif category not in _VALID_CATEGORIES:
        block_errors.append(
            f"    • Invalid category '{category}'. Must be one of: {', '.join(sorted(_VALID_CATEGORIES))}."
        )

    if not pairs and not block_errors:
        block_errors.append(
            "    • No QuestionN segments parsed. Use 'Question1' then Q:/A: or (a) Q:/A: …"
        )

    if block_errors:
        return [], [prefix + "\n".join(block_errors)]

    entries = [
        {
            "category": category,
            "question": q,
            "docs": d,
            "answer": a,
            "question_label": label,
        }
        for label, q, a, d in pairs
    ]
    return entries, []


def parse_qa_file(filepath: Path) -> list[dict]:
    """
    Parses a QandA file and returns a list of entry dicts (one per graded essay).
    Each dict has keys: category, question, docs, answer, question_label.
    Raises QAFormatError with a descriptive message on any format problem.
    """
    text = filepath.read_text(encoding="utf-8")
    return parse_qa_text(text, source_name=filepath.name)


def parse_qa_text(
    text: str,
    source_name: str = "<input>",
    default_category: Optional[str] = None,
) -> list[dict]:
    """
    Parses Q&A text (as a string) and returns a list of entry dicts.
    Each dict has keys: category, question, docs, answer, question_label.

    If the text has no CATEGORY: line and default_category is provided,
    a CATEGORY line is automatically prepended before parsing.

    Blocks are separated by '---'. Raises QAFormatError on format problems.
    """
    if not text.strip():
        raise QAFormatError(f"Input is empty.\n\n{_FORMAT_HINT}")

    # Inject CATEGORY if missing and a default is provided
    if default_category and not re.search(r"(?i)^CATEGORY\s*:", text, re.MULTILINE):
        text = f"CATEGORY: {default_category}\n\n{text}"

    raw_blocks = [b.strip() for b in text.split("---") if b.strip()]
    if not raw_blocks:
        raise QAFormatError(
            f"No entries found in {source_name}. "
            f"Blocks must be separated by '---'.\n\n{_FORMAT_HINT}"
        )

    entries: list[dict] = []
    errors: list[str] = []

    for block_num, block in enumerate(raw_blocks, start=1):
        lines = block.splitlines()
        if _block_has_question_markers(lines):
            block_entries, block_errors = _parse_block_multi(lines, block_num)
        else:
            block_entries, block_errors = _parse_block_legacy(lines, block_num)
        errors.extend(block_errors)
        entries.extend(block_entries)

    if errors:
        error_lines = [
            f"\nFormat error(s) in '{source_name}':",
            *errors,
            "",
            _FORMAT_HINT,
        ]
        raise QAFormatError("\n".join(error_lines))

    return entries
