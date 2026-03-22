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
import sys
import textwrap
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from grader import GradeResult, grade_essay
from qa_parser import (
    QAFormatError,
    _normalize_saq_subpart_labels,
    parse_qa_file,
)

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

DEFAULT_MODEL = "gpt-5.4"


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
# File parsing — delegated to qa_parser.py (shared with streamlit_app.py)
# ---------------------------------------------------------------------------


# (All parsing functions live in qa_parser.py — imported at the top of this file.)


# ---------------------------------------------------------------------------
# Output formatting  (parsing is delegated to qa_parser.py)
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
