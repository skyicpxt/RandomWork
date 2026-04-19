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

# Sub-part marker line that may either be standalone "(a)" OR carry an inline
# question text after the closing paren, e.g. "(a) Briefly describe ONE cause".
# Group 1 = letter, Group 2 = optional inline question text (may be empty).
_SAQ_SUBPART_LINE_RE = re.compile(r"^\s*\(\s*([a-z])\s*\)\s*(.*)$", re.IGNORECASE)

# Bare sub-part marker — same as above but the opening paren is OPTIONAL,
# so "a)" or "a) text" also match. Used by the unified section parser to
# accept the AP-style "a) Q: ..." input without requiring the opening paren.
# Bare matches are accepted only when the letter matches the expected next
# sub-part in sequence (a → b → c …) to avoid false positives on prose like
# "z) something" inside an answer paragraph.
_SAQ_SUBPART_BARE_RE = re.compile(r"^\s*\(?\s*([a-z])\s*\)\s*(.*)$", re.IGNORECASE)


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

    SAQ with a shared stimulus (intro paragraph before the sub-parts):
        CATEGORY: SAQ
        Question1
        Use the following passage to answer parts (a), (b), and (c).
        "<a quote, document excerpt, or scenario the prompts refer to>"

        (a)
        Q: First prompt…
        A: First answer…
        (b) Second prompt inline on the same line as the marker
        A: Second answer…
        (c)
        Q: Third prompt…
        A: Third answer…

        The stimulus paragraph is shared by every sub-part and is included once
        in the merged question. Sub-parts may use either the standalone "(a)"
        + "Q:" + "A:" form or an inline "(a) prompt text" + "A:" form.

    Optional: older single-block style with (a)… and A)… on one line each is still
    normalized to Part A/B/C when grading.

    ---
        CATEGORY: LEQ
        Q: ...
        A: ...
""")


def _merge_saq_subqa_pairs(
    pairs: list[tuple[str, str, str]],
    stimulus: str = "",
) -> tuple[str, str]:
    """
    Builds combined question and answer strings from sub-parts (letter, q, a) for one SAQ.
    Uses Part A/B/C headers so grading previews stay readable after normalization.

    If a stimulus (intro paragraph shared by all sub-parts) is provided, it is
    prepended once to the merged question as a 'Stimulus:' block so the grader
    sees the shared context before each sub-part. The stimulus is NOT added to
    the merged answer (it's part of the prompt, not the student's response).
    Stimulus chunks do not start with 'Part', so downstream helpers that split
    the merged question by sub-part (e.g. _extract_saq_parts) safely skip them.
    """
    q_chunks: list[str] = []
    a_chunks: list[str] = []
    if stimulus.strip():
        q_chunks.append(f"Stimulus:\n{stimulus.strip()}")
    for letter, q, a in pairs:
        part = chr(ord("A") + ord(letter.lower()) - ord("a"))
        q_chunks.append(f"Part {part}\nQ: {q}")
        a_chunks.append(f"Part {part}\nA: {a}")
    return "\n\n".join(q_chunks), "\n\n".join(a_chunks)


# Parses the SAQ sub-parts between start and end. Accepts both the original
# format (standalone "(a)" then "Q:" then "A:") and a relaxed variant where
# the question text is inline on the same line as the sub-part marker
# (e.g. "(a) Briefly describe ONE cause."). Boundary detection between
# sub-parts uses _SAQ_SUBPART_LINE_RE so either style ends the previous A:.
def _consume_saq_subqa_pairs(
    lines: list[str],
    start: int,
    end: int,
    context: str,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    Parses one SAQ block's sub-parts between line indices start and end.

    Each sub-part begins with a line matching _SAQ_SUBPART_LINE_RE — either
    a bare label like "(a)" (followed by separate "Q:" then "A:" lines) OR
    a label with the question text inline like "(a) Briefly describe ONE
    cause." (followed directly by an "A:" line). When inline question text
    is present, an explicit "Q:" line is optional.

    Returns (pairs, errors) where each pair is (letter, question_text, answer_text).
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
        m = _SAQ_SUBPART_LINE_RE.match(stripped)
        if not m:
            errors.append(
                f"    • {context}: expected '(a)', '(b)', … line, got {stripped[:48]!r}."
            )
            return [], errors
        letter = m.group(1).lower()
        inline_q = m.group(2).strip()
        i += 1

        # Question collection — start with any inline text, then optionally
        # accept a "Q:" line and continuation lines until an "A:" line is hit.
        q_lines: list[str] = []
        if inline_q:
            q_lines.append(inline_q)

        while i < end and not lines[i].strip():
            i += 1

        if i < end and lines[i].strip().upper().startswith("Q:"):
            st = lines[i].strip()
            q_lines.append(st[2:].strip())
            i += 1
            while i < end:
                st = lines[i].strip()
                if st.upper().startswith("A:"):
                    break
                if st and _SAQ_SUBPART_LINE_RE.match(st):
                    errors.append(
                        f"    • {context}: missing 'A:' before next sub-part."
                    )
                    return [], errors
                q_lines.append(lines[i].rstrip())
                i += 1
        elif not inline_q:
            # No inline question text and no Q: line — we cannot recover the prompt.
            errors.append(
                f"    • {context}: missing question after '({letter})' "
                f"(provide either inline text on the '({letter})' line or a separate 'Q:' line)."
            )
            return [], errors

        if i >= end:
            errors.append(f"    • {context}: missing 'A:' for sub-part ({letter}).")
            return [], errors

        st = lines[i].strip()
        if not st.upper().startswith("A:"):
            errors.append(
                f"    • {context}: expected 'A:' for ({letter}), found {st[:48]!r}."
            )
            return [], errors

        a_lines = [st[2:].strip()]
        i += 1
        while i < end:
            st = lines[i].strip()
            if st and _SAQ_SUBPART_LINE_RE.match(st):
                break
            if st.upper().startswith("Q:"):
                errors.append(
                    f"    • {context}: unexpected 'Q:' inside answer for ({letter})."
                )
                return [], errors
            a_lines.append(lines[i].rstrip())
            i += 1
        pairs.append((letter, "\n".join(q_lines).strip(), "\n".join(a_lines).strip()))
    if not pairs:
        errors.append(
            f"    • {context}: no '(a)' / '(b)' / … sub-parts with Q: and A: found."
        )
        return [], errors
    return pairs, []


# Parses one classic Q:/A: pair, tolerating a preamble of free text before any
# explicit Q: or A: marker. The preamble (e.g. a stimulus paragraph or the
# question stem written under a 'QuestionN' label) is folded into the question.
def _parse_classic_qa_segment(lines: list[str]) -> tuple[str, str, list[str]]:
    """
    Parses a single Q:/A: pair, returning (question, answer, errors).

    Tolerated variants:
      * Continuation lines after Q: or A: are appended until the next marker.
      * Free text appearing BEFORE the first Q: / A: marker is treated as the
        question stem (a "preamble"). This lets users write the prompt /
        stimulus directly under 'QuestionN' without a leading 'Q:' label,
        and pair it with either a separate 'Q:' line, a separate 'A:' line,
        or both. When both preamble and Q: text are present they are joined
        (preamble first) so nothing is silently dropped.
      * 'Q:' may be omitted entirely as long as some preamble text exists and
        an 'A:' line is present.
    """
    section = None
    preamble_lines: list[str] = []
    q_lines: list[str] = []
    a_lines: list[str] = []
    saw_q_marker = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("Q:"):
            section = "Q"
            saw_q_marker = True
            q_lines.append(stripped[2:].strip())
        elif stripped.upper().startswith("A:"):
            section = "A"
            a_lines.append(stripped[2:].strip())
        elif section == "Q":
            q_lines.append(stripped)
        elif section == "A":
            a_lines.append(stripped)
        else:
            # Lines before any Q:/A: marker are kept as the question preamble.
            preamble_lines.append(stripped)

    preamble = "\n".join(preamble_lines).strip()
    q_section = "\n".join(q_lines).strip()
    a = "\n".join(a_lines).strip()

    # Combine preamble with the explicit Q: text so neither is dropped.
    if preamble and q_section:
        q = f"{preamble}\n{q_section}"
    else:
        q = preamble or q_section

    errs: list[str] = []
    if not q and not a:
        return "", "", ["missing both 'Q:' question and 'A:' answer"]
    if not q:
        if saw_q_marker:
            errs.append(
                "missing question text — 'Q:' was found but had no content "
                "and no preamble text was supplied above it."
            )
        else:
            errs.append("missing 'Q:' question text (and no preamble text was supplied).")
    if not a:
        errs.append("missing 'A:' or answer text")
    if errs:
        return "", "", errs
    return q, a, []


# Scans `lines[start:end]` for the first SAQ sub-part marker line and returns
# (stimulus_text, subpart_index). Anything before the first marker is treated
# as a shared stimulus paragraph that applies to every sub-part of the question.
def _extract_saq_stimulus(
    lines: list[str],
    start: int,
    end: int,
) -> tuple[str, Optional[int]]:
    """
    Returns (stimulus_text, subpart_index) for the segment.

    stimulus_text is the joined text of every line between start and the first
    line that matches _SAQ_SUBPART_LINE_RE (e.g. '(a)' or '(a) some text').
    A leading 'Q:' prefix on the stimulus is stripped so users who write
    'Q: <stimulus>' before the sub-parts get the natural result.

    subpart_index is the index of that first sub-part marker line, or None
    if no sub-part marker is present in the range (in which case the segment
    should be parsed as a classic single Q:/A: pair).
    """
    stimulus_lines: list[str] = []
    for j in range(start, end):
        st = lines[j].strip()
        if st and _SAQ_SUBPART_LINE_RE.match(st):
            stimulus = "\n".join(stimulus_lines).strip()
            # Tolerate users who lead the stimulus with a "Q:" prefix.
            if stimulus.upper().startswith("Q:"):
                stimulus = stimulus[2:].strip()
            return stimulus, j
        stimulus_lines.append(lines[j].rstrip())
    return "", None


# Parses an arbitrary slice of lines into a preamble plus a list of Q/A
# sections. A "section" starts at any of:
#   * an SAQ sub-part marker line such as "(a)", "(a) inline question",
#     "a)", or "a) inline question" (bare form is accepted only when the
#     letter matches the expected next sub-part to avoid false positives
#     on prose lines that happen to start with "x)");
#   * a "Q:" line (begins an implicit, letterless section);
#   * an "A:" line that appears before any other marker (begins an
#     implicit section with empty Q).
# A new "Q:" line that follows an "A:" line within the same section closes
# that section and starts a new one — this is what auto-splits the common
# "Q: … A: … Q: … A: … Q: … A: …" layout into separate SAQ sub-parts.
def _parse_qa_sections(
    lines: list[str],
    start: int,
    end: int,
) -> tuple[str, list[dict]]:
    """
    Returns (preamble, sections).

    preamble — joined text of every line before the first section started.
               Typically the SAQ stimulus or the question stem written
               directly under a "QuestionN" label (without a "Q:" prefix).

    sections — list of dicts, each with:
        "letter": str|None  — explicit sub-part letter ("a"/"b"/"c"/…) when
                              the section was opened by a sub-part marker,
                              else None (section opened by an implicit "Q:").
        "q":      str       — question text for the section (Q: line + any
                              continuation lines, plus inline text from the
                              sub-part marker if present).
        "a":      str       — answer text for the section.
    """
    preamble_lines: list[str] = []
    sections: list[dict] = []
    cur_letter: Optional[str] = None
    cur_q: list[str] = []
    cur_a: list[str] = []
    have_section = False
    state: Optional[str] = None  # "Q" or "A" within the current section

    def flush() -> None:
        nonlocal cur_letter, cur_q, cur_a, have_section, state
        if have_section:
            sections.append({
                "letter": cur_letter,
                "q": "\n".join(cur_q).strip(),
                "a": "\n".join(cur_a).strip(),
            })
        cur_letter = None
        cur_q = []
        cur_a = []
        have_section = False
        state = None

    for j in range(start, end):
        line = lines[j]
        stripped = line.strip()

        # Sub-part marker detection. Strict paren form is always accepted;
        # bare form is accepted only at section boundaries AND only when the
        # letter matches the expected next sub-part (a/b/c progression).
        m_paren = _SAQ_SUBPART_LINE_RE.match(stripped)
        m_bare = None
        if not m_paren:
            m_bare = _SAQ_SUBPART_BARE_RE.match(stripped)

        is_marker = False
        m = None
        if m_paren is not None:
            is_marker = True
            m = m_paren
        elif m_bare is not None:
            # Bare-form ("a)" with no opening paren) is only treated as a
            # sub-part marker when (1) we're at a clean section boundary
            # (no open section yet, or the open one already has an answer)
            # AND (2) the letter is the next one in the a/b/c progression.
            # This avoids false positives on prose lines like "z) something"
            # appearing inside an answer paragraph.
            letter_candidate = m_bare.group(1).lower()
            at_boundary = not have_section or bool(cur_a) or state == "A"
            next_expected = chr(ord("a") + len(sections) + (1 if have_section else 0))
            if at_boundary and letter_candidate == next_expected:
                is_marker = True
                m = m_bare

        if is_marker:
            letter = m.group(1).lower()
            inline_text = m.group(2).strip()
            # Tolerate "(a) Q: prompt" — strip a redundant "Q:" prefix from
            # the inline text so it becomes just "prompt".
            if inline_text.upper().startswith("Q:"):
                inline_text = inline_text[2:].strip()
            flush()
            cur_letter = letter
            if inline_text:
                cur_q = [inline_text]
            have_section = True
            state = "Q"
            continue

        if stripped.upper().startswith("Q:"):
            q_text = stripped[2:].strip()
            # A new Q: after an A: closes the previous section.
            if have_section and (cur_a or state == "A"):
                flush()
            if not have_section:
                have_section = True
            cur_q.append(q_text)
            state = "Q"
            continue

        if stripped.upper().startswith("A:"):
            a_text = stripped[2:].strip()
            if not have_section:
                have_section = True
            cur_a.append(a_text)
            state = "A"
            continue

        # Continuation line — belongs to the current section's Q/A or to the
        # preamble if no section has started yet.
        if not have_section:
            preamble_lines.append(stripped)
        elif state == "Q":
            cur_q.append(stripped)
        elif state == "A":
            cur_a.append(stripped)

    flush()

    preamble = "\n".join(preamble_lines).strip()
    return preamble, sections


# Parses one QuestionN segment into a (question, answer, docs, errors) tuple.
# Uses the unified _parse_qa_sections helper so it transparently handles:
#   * a single Q:/A: pair (LEQ/DBQ or single-prompt SAQ);
#   * SAQ sub-parts with explicit "(a)" / "a)" markers;
#   * SAQ sub-parts with NO markers at all — multiple Q:/A: pairs in the same
#     segment are auto-promoted to Part A / Part B / Part C when category is SAQ;
#   * a stimulus / question stem written before the first marker, captured as
#     a shared preamble that is kept in the merged question text.
def _parse_multi_question_segment(
    segment: list[str],
    label: str,
    category: str = "",
) -> tuple[str, str, str, list[str]]:
    """
    Parses one QuestionN block and returns (question, answer, pair_docs, errors).

    The optional `category` argument enables SAQ-specific behavior: when SAQ
    is selected and the segment contains either explicit (a)/(b)/(c) markers
    or simply 2+ Q:/A: pairs, the segment is auto-split into Part A / Part B
    / Part C sub-parts so the rest of the pipeline can render and grade each
    sub-part independently.
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
            if _SAQ_SUBPART_LINE_RE.match(st):
                break
            if st.upper().startswith("Q:"):
                break
            pair_docs_lines.append(segment[i].rstrip())
            i += 1
    rest_start = i
    if rest_start >= len(segment):
        return "", "", "", [f"    • '{label}': empty segment after optional DOCS."]

    preamble, sections = _parse_qa_sections(segment, rest_start, len(segment))

    if not sections:
        # Nothing recognisable — no Q:, no A:, no sub-part markers.
        # If we still have preamble text, treat it as a question stem missing
        # an answer; otherwise report the segment as empty.
        if preamble:
            return "", "", "", [
                f"    • '{label}': missing 'A:' or answer text "
                f"(question stem found but no answer)."
            ]
        return "", "", "", [
            f"    • '{label}': no Q:/A: lines or sub-part markers found in segment."
        ]

    has_explicit_letters = any(s["letter"] for s in sections)
    cat_upper = category.upper()

    # SAQ auto-promotion: explicit letters OR 2+ implicit Q:/A: pairs are
    # treated as SAQ sub-parts, with letters auto-assigned where missing.
    if cat_upper == "SAQ" and (has_explicit_letters or len(sections) >= 2):
        sub_pairs: list[tuple[str, str, str]] = []
        used: set[str] = set()
        for s in sections:
            letter = s["letter"]
            if not letter or letter in used:
                k = 0
                while chr(ord("a") + k) in used:
                    k += 1
                letter = chr(ord("a") + k)
            used.add(letter)
            sub_pairs.append((letter, s["q"], s["a"]))

        validation_errors: list[str] = []
        for letter, q, a in sub_pairs:
            if not q.strip():
                validation_errors.append(
                    f"    • '{label}': sub-part ({letter}) has no question text."
                )
            if not a.strip():
                validation_errors.append(
                    f"    • '{label}': sub-part ({letter}) has no 'A:' or answer text."
                )
        if validation_errors:
            return "", "", "", validation_errors

        mq, ma = _merge_saq_subqa_pairs(sub_pairs, stimulus=preamble)
        return mq, ma, "\n".join(pair_docs_lines).strip(), errors

    # Single-section path (LEQ, DBQ, or single-prompt SAQ). When more than
    # one section is present for a non-SAQ category we concatenate them so no
    # text is silently dropped (matches legacy behaviour).
    q_parts = [preamble] if preamble else []
    a_parts: list[str] = []
    for s in sections:
        if s["q"]:
            q_parts.append(s["q"])
        if s["a"]:
            a_parts.append(s["a"])
    q_text = "\n".join(q_parts).strip()
    a_text = "\n".join(a_parts).strip()

    block_errors: list[str] = []
    if not q_text and not a_text:
        block_errors.append(
            f"    • '{label}': missing both 'Q:' question and 'A:' answer."
        )
    elif not q_text:
        block_errors.append(
            f"    • '{label}': missing question text "
            f"(no preamble or 'Q:' content found)."
        )
    elif not a_text:
        block_errors.append(f"    • '{label}': missing 'A:' or answer text.")

    if block_errors:
        return "", "", "", block_errors

    return q_text, a_text, "\n".join(pair_docs_lines).strip(), errors


def _legacy_block_looks_like_saq_subqa(lines: list[str]) -> bool:
    """
    True when CATEGORY is SAQ and the block contains a sub-part marker line
    ('(a)', '(b)', '(a) inline text', …) anywhere after the CATEGORY line.

    This is intentionally permissive so that an SAQ block can include a leading
    stimulus paragraph or other intro text before the first '(a)' marker and
    still be routed to the SAQ sub-question parser. A block without any sub-part
    markers (classic Q:/A:) returns False so it falls through to the standard
    legacy parser.
    """
    seen_category = False
    category = ""
    has_any_qa_marker = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("CATEGORY:"):
            category = stripped.split(":", 1)[1].strip().upper()
            seen_category = True
            continue
        if not seen_category:
            continue
        if category != "SAQ":
            return False
        if not stripped:
            continue
        if _SAQ_SUBPART_LINE_RE.match(stripped):
            return True
        upper = stripped.upper()
        if upper.startswith("Q:") or upper.startswith("A:"):
            has_any_qa_marker = True
    # No sub-part marker found anywhere — must be classic Q:/A: SAQ
    # (or an empty block, which the classic parser will report).
    return False


def _parse_block_legacy_saq_subqa(
    lines: list[str],
    block_num: int,
) -> tuple[list[dict], list[str]]:
    """
    Legacy SAQ block: optional stimulus paragraph, then (a) Q:/A:, (b) Q:/A:, …
    without QuestionN lines. The stimulus (if any) is merged into the question
    text once at the top so the grader sees the shared context for every sub-part.
    """
    prefix = f"  Block #{block_num}:\n"
    category = ""
    start = 0
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip().upper()
            start = idx + 1
            break
    stimulus, subpart_idx = _extract_saq_stimulus(lines, start, len(lines))
    if subpart_idx is None:
        return [], [
            prefix
            + f"    • SAQ: no '(a)' / '(b)' / … sub-part markers found after CATEGORY."
        ]
    pairs, perr = _consume_saq_subqa_pairs(lines, subpart_idx, len(lines), "SAQ")
    if perr:
        return [], [prefix + "\n".join(perr)]
    mq, ma = _merge_saq_subqa_pairs(pairs, stimulus=stimulus)
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

        q_text, a_text, pair_docs, seg_errs = _parse_multi_question_segment(
            segment, label, category=category
        )
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


def has_multi_question_markers(text: str) -> bool:
    """Returns True if the text contains any Question1 / Question2 / … marker lines."""
    return _block_has_question_markers(text.splitlines())


def normalize_entry(entry: dict) -> tuple[str, str]:
    """
    Returns (question, answer) from a parsed entry dict.
    For SAQ entries, applies sub-part label normalization to align
    (a)/(b)/(c) labels with the rubric's Part A / Part B / Part C names.
    """
    q = entry["question"]
    a = entry["answer"]
    if entry.get("category") == "SAQ":
        q = _normalize_saq_subpart_labels(q)
        a = _normalize_saq_subpart_labels(a)
    return q, a


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
