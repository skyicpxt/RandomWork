# AP World History: Modern — Grading Agent
# Reads Q&A pairs from type-specific files (DBQQandA.txt, LEQQandA.txt, SAQQandA.txt)
# and grades each essay using the official College Board rubric, then writes a report.
# DBQ source documents live inside DBQQandA.txt under a DOCS: section.
#
# Sample inputs:
#   * Single Q&A (default files):  --category DBQ | LEQ | SAQ  (or no flag → all three)
#   * Multi Question1/2/… format: --category DBQ_multi | LEQ_multi | SAQ_multi
#   * Custom file: --qa path/to/file.txt (optional --category filters essay type inside file)
#
# Reference: https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment
#
# Usage:
#   python main.py                         # all single: DBQ + LEQ + SAQ
#   python main.py --category DBQ          # DBQQandA.txt only
#   python main.py --category DBQ_multi    # DBQQandA_multi.txt only
#   python main.py --qa my.txt --category SAQ
#   python main.py --output results.txt

import argparse
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from grader import GradeResult, grade_essay

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

# Load .env from Random/ parent directory (same pattern as other projects here)
load_dotenv(_SCRIPT_DIR.parent / ".env")

# Maps CLI --category to the default Q&A file (single vs *_multi QuestionN format).
INPUT_MODE_FILES: dict[str, Path] = {
    "DBQ": _SCRIPT_DIR / "DBQQandA.txt",
    "DBQ_multi": _SCRIPT_DIR / "DBQQandA_multi.txt",
    "LEQ": _SCRIPT_DIR / "LEQQandA.txt",
    "LEQ_multi": _SCRIPT_DIR / "LEQQandA_multi.txt",
    "SAQ": _SCRIPT_DIR / "SAQQandA.txt",
    "SAQ_multi": _SCRIPT_DIR / "SAQQandA_multi.txt",
}

# When no --category: grade these three default single-format files.
DEFAULT_SINGLE_MODES = ("DBQ", "LEQ", "SAQ")

DEFAULT_MODEL = "gpt-4o"


def _rubric_category_for_filter(cli_category: Optional[str]) -> Optional[str]:
    """
    Maps CLI flag (e.g. DBQ_multi) to the essay type inside the file (DBQ) for filtering
    entries when using --qa. None means no filter.
    """
    if cli_category is None:
        return None
    if cli_category.endswith("_multi"):
        return cli_category[: -len("_multi")]
    return cli_category


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

class QAFormatError(Exception):
    """Raised when a QandA file does not follow the expected format."""


_VALID_CATEGORIES = {"DBQ", "LEQ", "SAQ"}

# Matches a standalone line like "Question1", "Question 2", "QUESTION3" (label above Q:)
_QUESTION_MARKER_RE = re.compile(r"^Question\s*\d+\s*$", re.IGNORECASE)

# Line-leading SAQ sub-part labels: (a), (b), … or A), B), … (AP-style)
_SAQ_SUBPART_PAREN_RE = re.compile(
    r"^(\s*)\(([a-z])\)\s*(.*)$",
    re.IGNORECASE,
)
_SAQ_SUBPART_UPPER_RE = re.compile(
    r"^(\s*)([A-Z])\)\s*(.*)$",
)

# Standalone line that is only a sub-part label, e.g. "(a)" — followed by Q: then A:
_SAQ_SUBPART_STANDALONE_RE = re.compile(r"^\s*\(\s*([a-z])\s*\)\s*$", re.IGNORECASE)


def _merge_saq_subqa_pairs(pairs: list[tuple[str, str, str]]) -> tuple[str, str]:
    """
    Builds combined question and answer strings from sub-parts (letter, q, a) for one SAQ.
    Uses Part A/B/C headers (not (a)) so grading previews stay readable after normalization.
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
    """Parses a single Q:/A: pair (continuation lines allowed)."""
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
    One QuestionN block: optional DOCS:, then either (a)…Q/A…(b)… or a single Q:/A: pair.
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
    """Returns True if this line is a QuestionN marker (the line above Q:)."""
    return bool(stripped and _QUESTION_MARKER_RE.fullmatch(stripped))


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


def _block_has_question_markers(lines: list[str]) -> bool:
    """True if this block uses Question1 / Question2 / ... labels."""
    return any(_is_question_marker_line(l.strip()) for l in lines if l.strip())


def parse_qa_file(filepath: Path) -> list[dict]:
    """
    Parses the QandA file and returns a list of dicts (one per graded essay):
      [{"category": "...", "question": "...", "docs": "...", "answer": "...",
        "question_label": "" or "Question1"}, ...]

    question_label is set when the block uses QuestionN lines; empty string otherwise.

    Raises QAFormatError with a descriptive message if the file is empty,
    contains no valid blocks, or any block is missing CATEGORY / Q / A.
    """
    text = filepath.read_text(encoding="utf-8")

    if not text.strip():
        raise QAFormatError(
            f"File is empty: {filepath.name}\n\n{_FORMAT_HINT}"
        )

    raw_blocks = [b.strip() for b in text.split("---") if b.strip()]
    if not raw_blocks:
        raise QAFormatError(
            f"No entries found in {filepath.name}. "
            f"Blocks must be separated by '---'.\n\n{_FORMAT_HINT}"
        )

    entries = []
    errors: list[str] = []

    for block_num, block in enumerate(raw_blocks, start=1):
        lines = block.splitlines()
        if _block_has_question_markers(lines):
            block_entries, block_errors = _parse_block_multi(lines, block_num)
            errors.extend(block_errors)
            entries.extend(block_entries)
        else:
            block_entries, block_errors = _parse_block_legacy(lines, block_num)
            errors.extend(block_errors)
            entries.extend(block_entries)

    if errors:
        error_lines = [
            f"\nFormat error(s) in '{filepath.name}':",
            *errors,
            "",
            _FORMAT_HINT,
        ]
        raise QAFormatError("\n".join(error_lines))

    return entries


def _parse_block_legacy(lines: list[str], block_num: int) -> tuple[list[dict], list[str]]:
    """
    Parses a block with a single Q / A (optional DOCS: for DBQ).
    Returns (entries, errors); errors are prefixed for this block.
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


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_SEPARATOR = "=" * 72
_THIN_SEP = "-" * 72
_INDENT = "  "

# Score bar: visual representation of points earned
def _score_bar(earned: int, possible: int, width: int = 20) -> str:
    """Returns a simple ASCII progress bar."""
    filled = int(round(earned / possible * width)) if possible > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {earned}/{possible}"


def _wrap(text: str, width: int = 68, indent: str = _INDENT) -> str:
    """Wraps and indents text for console output."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _preview_one_line(text: str, max_len: int) -> str:
    """Compresses whitespace for a single-line question preview in reports."""
    s = " ".join(text.split())
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def format_grade_report(
    result: GradeResult,
    entry_index: int,
    question_label: Optional[str] = None,
) -> str:
    """
    Formats a GradeResult into a human-readable report string.
    question_label is shown when the source file used Question1 / Question2 / ... lines.
    """
    lines = []
    lines.append("")
    lines.append(_SEPARATOR)
    label_part = f"  |  {question_label}" if question_label else ""
    lines.append(f"  ESSAY #{entry_index}{label_part}  |  TYPE: {result.category}")
    lines.append(_SEPARATOR)

    # Question preview (single line; newlines would break summary tables)
    q_preview = _preview_one_line(result.question, 120)
    lines.append(f"  QUESTION: {q_preview}")
    lines.append("")

    # ── 1. Score Breakdown ──────────────────────────────────────────────────
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │             1. SCORE BREAKDOWN                  │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    lines.append(f"  TOTAL: {_score_bar(result.total_earned, result.total_possible)}")
    lines.append("")
    lines.append(f"  {'Criterion':<44} {'Earned':>6}  {'Max':>4}")
    lines.append(f"  {'-'*44} {'-'*6}  {'-'*4}")

    for cr in result.criteria_results:
        status = "✔" if cr.points_earned > 0 else "✘"
        lines.append(f"  {status} {cr.name:<43} {cr.points_earned:>6}  {cr.max_points:>4}")

    lines.append("")

    # ── 2. Evidence That Earned Each Point ──────────────────────────────────
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │        2. EVIDENCE THAT EARNED EACH POINT       │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    for cr in result.criteria_results:
        if cr.points_earned > 0:
            lines.append(f"  ✔ {cr.name} ({cr.points_earned}/{cr.max_points} pt{'s' if cr.max_points > 1 else ''})")
            lines.append(_wrap(f'"{cr.evidence}"'))
            lines.append("")

    if not any(cr.points_earned > 0 for cr in result.criteria_results):
        lines.append(_wrap("No points were earned."))
        lines.append("")

    # ── 3. Points Not Earned & Why ──────────────────────────────────────────
    lines.append("  ┌─────────────────────────────────────────────────┐")
    lines.append("  │       3. POINTS NOT EARNED AND WHY              │")
    lines.append("  └─────────────────────────────────────────────────┘")
    lines.append("")
    missed = [cr for cr in result.criteria_results if cr.points_earned < cr.max_points]
    if missed:
        for cr in missed:
            missing = cr.max_points - cr.points_earned
            lines.append(f"  ✘ {cr.name} (missed {missing} of {cr.max_points} pt{'s' if cr.max_points > 1 else ''})")
            if cr.not_earned_reason:
                lines.append(_wrap(cr.not_earned_reason))
            lines.append("")
    else:
        lines.append(_wrap("All points were earned — excellent work!"))
        lines.append("")

    # ── 4. Suggestions to Improve ───────────────────────────────────────────
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

    lines.append(_SEPARATOR)
    return "\n".join(lines)


def print_summary(
    results: list[GradeResult],
    question_labels: Optional[list[Optional[str]]] = None,
) -> str:
    """
    Prints and returns a summary table of all graded essays.
    question_labels[i] matches results[i] when entries came from QuestionN markers.
    """
    lines = []
    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  GRADING SUMMARY")
    lines.append(_SEPARATOR)
    lines.append(f"  {'#':<4} {'Type':<6} {'Score':>10}  Label      Question (preview)")
    lines.append(f"  {'-'*4} {'-'*6} {'-'*10}  {'-'*10}  {'-'*30}")
    total_e, total_p = 0, 0
    for i, r in enumerate(results, 1):
        q_preview = _preview_one_line(r.question, 30)
        score_str = f"{r.total_earned}/{r.total_possible}"
        lbl = ""
        if question_labels and i - 1 < len(question_labels) and question_labels[i - 1]:
            lbl = (question_labels[i - 1] or "")[:10]
        lines.append(f"  {i:<4} {r.category:<6} {score_str:>10}  {lbl:<10}  {q_preview}")
        total_e += r.total_earned
        total_p += r.total_possible
    lines.append(f"  {'-'*4} {'-'*6} {'-'*10}  {'-'*10}  {'-'*30}")
    lines.append(f"  {'TOTAL':<22} {total_e}/{total_p}")
    lines.append(_SEPARATOR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AP World History: Modern — Grading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --category DBQ
              python main.py --category DBQ_multi
              python main.py --category SAQ_multi
              python main.py --qa custom.txt --category LEQ
              python main.py --output report.txt
        """),
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=None,
        help=(
            "Path to a custom Q&A file. If omitted, file is chosen from --category "
            "(DBQ → DBQQandA.txt, DBQ_multi → DBQQandA_multi.txt, etc.). "
            "With --qa, --category filters by essay type inside the file (DBQ_multi filters as DBQ)."
        ),
    )
    _cli_modes = sorted(INPUT_MODE_FILES.keys())
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=_cli_modes,
        metavar="MODE",
        help=(
            "Input preset: DBQ / LEQ / SAQ = single-format sample files; "
            "DBQ_multi / LEQ_multi / SAQ_multi = Question1/2/… sample files. "
            "Omit to grade all three single-format files (DBQ+LEQ+SAQ). "
            "With --qa, filters parsed entries to that essay type (e.g. SAQ or SAQ_multi → SAQ)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "grading_report.txt",
        help="Path to write the full report (default: grading_report.txt in the same folder)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to Random/.env or your environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Determine which files to read
    filter_type = _rubric_category_for_filter(args.category)

    if args.qa is not None:
        # Explicit file provided — read it and optionally filter by essay type
        if not args.qa.exists():
            print(f"ERROR: Q&A file not found: {args.qa}")
            sys.exit(1)
        try:
            entries = parse_qa_file(args.qa)
        except QAFormatError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if filter_type is not None:
            entries = [e for e in entries if e["category"] == filter_type]
    else:
        # Auto-select files from --category (single or *_multi) or all default singles
        modes = [args.category] if args.category else list(DEFAULT_SINGLE_MODES)
        entries = []
        for mode in modes:
            qa_file = INPUT_MODE_FILES[mode]
            if qa_file.exists():
                try:
                    entries.extend(parse_qa_file(qa_file))
                except QAFormatError as e:
                    print(f"ERROR: {e}")
                    sys.exit(1)
            else:
                print(f"  Warning: {qa_file.name} not found, skipping {mode}.")

    if not entries:
        print("No matching entries found in the Q&A file.")
        sys.exit(0)

    mode_display = args.category or "DBQ + LEQ + SAQ (single-format defaults)"
    print(f"\nAP World History: Modern — Grading Agent")
    print(f"Reference: https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment")
    print(f"Model:   {args.model}")
    print(f"Input:   {mode_display}")
    print(f"Output:  {args.output}")
    print(f"Entries: {len(entries)}")
    print(_THIN_SEP)

    # Grade all entries — collect output for the report file
    all_output: list[str] = []
    results: list[GradeResult] = []
    summary_labels: list[Optional[str]] = []

    for i, entry in enumerate(entries, 1):
        cat = entry["category"]
        qlabel = entry.get("question_label") or None
        label_hint = f" {qlabel}" if qlabel else ""
        print(f"  [{i}/{len(entries)}] Grading {cat}{label_hint}...", end="", flush=True)
        question_text = entry["question"]
        answer_text = entry["answer"]
        if cat == "SAQ":
            question_text = _normalize_saq_subpart_labels(question_text)
            answer_text = _normalize_saq_subpart_labels(answer_text)
        try:
            result = grade_essay(
                client=client,
                category=cat,
                question=question_text,
                answer=answer_text,
                dbq_docs=entry.get("docs") or None,
                model=args.model,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        results.append(result)
        summary_labels.append(qlabel)
        score_str = f"{result.total_earned}/{result.total_possible}"
        print(f" done  ({score_str} pts)")
        all_output.append(format_grade_report(result, i, question_label=qlabel))

    if not results:
        print("\nNo results to write.")
        sys.exit(0)

    # Append summary to report
    summary = print_summary(results, question_labels=summary_labels)
    all_output.append(summary)

    # Write full report to file
    report_text = "\n".join(all_output)
    args.output.write_text(report_text, encoding="utf-8")

    # Print summary + confirmation to terminal
    print(summary)
    print(_THIN_SEP)
    print(f"Full report written to: {args.output.resolve()}")
    print(_THIN_SEP)


if __name__ == "__main__":
    main()
