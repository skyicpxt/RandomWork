# Shared report formatting utilities for AP World History grading.
# Used by both main.py (CLI) and streamlit_app.py (web UI) so that
# the plain-text report output is identical regardless of entry point.

import textwrap
from typing import Optional

from grader import GradeResult

_SEPARATOR = "=" * 72
_BAR_WIDTH = 20


def _score_bar(earned: int, possible: int) -> str:
    """Returns a filled/empty block bar string representing score fraction."""
    if possible == 0:
        return ""
    filled = round((earned / possible) * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    """Wraps text at width, indenting every line."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _preview_one_line(text: str, max_len: int = 200) -> str:
    """Compresses whitespace and truncates text to a single-line preview."""
    s = " ".join(text.split())
    return s[:max_len] + "..." if len(s) > max_len else s


def format_summary(graded: list[tuple[int, str, GradeResult]]) -> str:
    """
    Formats a GRADING SUMMARY table for a set of graded essays.
    graded is a list of (entry_index, label, GradeResult) tuples — the same
    structure used by both main.py and streamlit_app.py after grading each entry.
    Returns the formatted string for inclusion in a plain-text report.
    """
    lines: list[str] = []
    sep = _SEPARATOR
    lines.append("")
    lines.append(sep)
    lines.append("  GRADING SUMMARY")
    lines.append(sep)
    lines.append(f"  {'#':<4} {'Label':<12} {'Type':<6} {'Score':>10}  Question (preview)")
    lines.append(f"  {'-'*4} {'-'*12} {'-'*6} {'-'*10}  {'-'*30}")
    total_e = total_p = 0
    for i, lbl, r in graded:
        q_preview = _preview_one_line(r.question, 30)
        score_str = f"{r.total_earned}/{r.total_possible}"
        lines.append(f"  {i:<4} {lbl[:12]:<12} {r.category:<6} {score_str:>10}  {q_preview}")
        total_e += r.total_earned
        total_p += r.total_possible
    lines.append(f"  {'-'*4} {'-'*12} {'-'*6} {'-'*10}  {'-'*30}")
    lines.append(f"  {'TOTAL':<36} {total_e}/{total_p}")
    lines.append(sep)
    return "\n".join(lines)


def format_grade_report(
    result: GradeResult,
    entry_index: Optional[int] = None,
    question_label: Optional[str] = None,
) -> str:
    """
    Formats a GradeResult into a human-readable plain-text report string.

    entry_index controls the header style:
      - None  → standalone full-page header (used by streamlit_app.py).
      - int   → compact per-entry header "ESSAY #N | TYPE: X" (used by main.py
                for multi-essay report files). question_label is appended when given.

    Sections produced:
      1. Score Breakdown table
      2. Evidence That Earned Each Point (with evidence_comment where present)
      3. Points Not Earned and Why
      4. Suggestions to Improve
    """
    lines: list[str] = []
    sep = _SEPARATOR

    lines.append("")
    lines.append(sep)
    if entry_index is not None:
        label_part = f"  |  {question_label}" if question_label else ""
        lines.append(f"  ESSAY #{entry_index}{label_part}  |  TYPE: {result.category}")
    else:
        lines.append("  AP WORLD HISTORY: MODERN — GRADING REPORT")
        lines.append(f"  Essay Type: {result.category}")
    lines.append(sep)
    lines.append("")

    q_preview = _preview_one_line(result.question, 200)
    lines.append(f"  QUESTION: {q_preview}")
    lines.append("")

    # ── 1. Score Breakdown ──────────────────────────────────────────────────
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

    # ── 2. Evidence That Earned Each Point ──────────────────────────────────
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

    # ── 3. Points Not Earned & Why ──────────────────────────────────────────
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

    lines.append(sep)
    return "\n".join(lines)
