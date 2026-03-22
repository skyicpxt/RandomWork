# Unit tests for Q&A parsing (single vs multi-question) and grading report output shape.
# Run from this folder:
#   python -m unittest test_ap_world_grading -v
# Or:
#   cd APWorldHistGrading && python -m unittest test_ap_world_grading -v

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure imports resolve when the test runner's cwd is not this directory.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

# grader.py imports openai at import time; stub so unit tests run without openai installed.
if "openai" not in sys.modules:
    from unittest.mock import MagicMock

    _openai_mod = MagicMock()
    _openai_mod.OpenAI = MagicMock
    sys.modules["openai"] = _openai_mod

from grader import CriterionResult, GradeResult

import main as ap_main


def _write_qa_file(content: str) -> Path:
    """
    Writes content to a closed temp file and returns its path.
    The file handle must be closed before unlink on Windows.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="qa_test_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(content)
        return Path(f.name)


class TestParseSingleQandA(unittest.TestCase):
    """Legacy format: one Q / optional DOCS / one A per block (no QuestionN lines)."""

    def setUp(self) -> None:
        self._paths: list[Path] = []

    def tearDown(self) -> None:
        for p in self._paths:
            if p.exists():
                p.unlink()

    def _parse(self, text: str):
        path = _write_qa_file(text)
        self._paths.append(path)
        return ap_main.parse_qa_file(path)

    def test_single_saq_one_entry_empty_question_label(self) -> None:
        """Single SAQ block yields one entry with question_label ''."""
        text = """CATEGORY: SAQ

Q: Part A only?
A: My answer here.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["category"], "SAQ")
        self.assertEqual(e["question_label"], "")
        self.assertIn("Part A", e["question"])
        self.assertEqual(e["answer"].strip(), "My answer here.")
        self.assertEqual(e["docs"], "")

    def test_single_dbq_with_docs_order_q_docs_a(self) -> None:
        """DBQ with Q then DOCS then A attaches docs and no Question markers."""
        text = """CATEGORY: DBQ

Q: Evaluate trade.

DOCS:
DOCUMENT 1
Some excerpt.

A: Thesis and body.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["category"], "DBQ")
        self.assertEqual(e["question_label"], "")
        self.assertIn("Evaluate trade", e["question"])
        self.assertIn("DOCUMENT 1", e["docs"])
        self.assertIn("Some excerpt", e["docs"])
        self.assertIn("Thesis", e["answer"])

    def test_two_blocks_separated_by_dash_line(self) -> None:
        """--- separates two legacy blocks → two entries."""
        text = """CATEGORY: LEQ

Q: First prompt?
A: First answer.

---

CATEGORY: LEQ

Q: Second prompt?
A: Second answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["question_label"], "")
        self.assertEqual(entries[1]["question_label"], "")
        self.assertIn("First", entries[0]["question"])
        self.assertIn("Second", entries[1]["answer"])

    def test_legacy_saq_subqa_no_question_marker(self) -> None:
        """SAQ with (a) Q:/A: … without QuestionN is one merged entry."""
        text = """CATEGORY: SAQ
(a)
Q: First?
A: A1.

(b)
Q: Second?
A: A2.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question_label"], "")
        self.assertIn("Part A", entries[0]["question"])
        self.assertIn("A: A1.", entries[0]["answer"])


class TestParseMultiQandA(unittest.TestCase):
    """Format with Question1 / Question2 / … above each Q:."""

    def setUp(self) -> None:
        self._paths: list[Path] = []

    def tearDown(self) -> None:
        for p in self._paths:
            if p.exists():
                p.unlink()

    def _parse(self, text: str):
        path = _write_qa_file(text)
        self._paths.append(path)
        return ap_main.parse_qa_file(path)

    def test_multi_saq_three_entries_distinct_labels(self) -> None:
        """Three QuestionN blocks → three entries with shared category and labels."""
        text = """CATEGORY: SAQ

Question1
Q: Describe one cause.
A: Answer A.

Question2
Q: Effect on Americas.
A: Answer B.

Question 3
Q: Effect on Europe.
A: Answer C.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["question_label"], "Question1")
        self.assertEqual(entries[1]["question_label"], "Question2")
        self.assertEqual(entries[2]["question_label"], "Question 3")
        self.assertTrue(all(e["category"] == "SAQ" for e in entries))
        self.assertEqual(entries[0]["answer"].strip(), "Answer A.")
        self.assertEqual(entries[2]["question"].strip(), "Effect on Europe.")

    def test_multi_saq_subqa_question1_merged_parts(self) -> None:
        """Question1 with (a)/(b)/(c) each having Q: and A: merges into one entry."""
        text = """CATEGORY: SAQ

Question1
(a)
Q: Part one?
A: Ans one.

(b)
Q: Part two?
A: Ans two.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["question_label"], "Question1")
        self.assertIn("Part A", e["question"])
        self.assertIn("Q: Part one?", e["question"])
        self.assertIn("Part B", e["question"])
        self.assertIn("Q: Part two?", e["question"])
        self.assertIn("Part A", e["answer"])
        self.assertIn("A: Ans one.", e["answer"])
        self.assertIn("A: Ans two.", e["answer"])

    def test_multi_dbq_shared_docs(self) -> None:
        """DOCS before Question1 applies to the DBQ pair."""
        text = """CATEGORY: DBQ

DOCS:
DOCUMENT 1
Hello world.

Question1
Q: Use the docs.
A: My essay.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["question_label"], "Question1")
        self.assertIn("Hello world", e["docs"])
        self.assertIn("My essay", e["answer"])

    def test_multi_per_question_docs_attached(self) -> None:
        """DOCS: after QuestionN applies only to that question's entry."""
        text = """CATEGORY: SAQ

Question1
DOCS:
Only for Q1.

Q: Prompt one?
A: Answer one.

Question2
DOCS:
Only for Q2.

Q: Prompt two?
A: Answer two.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)
        self.assertIn("Only for Q1", entries[0]["docs"])
        self.assertIn("Only for Q2", entries[1]["docs"])
        self.assertNotIn("Q2", entries[0]["docs"])

    def test_multi_second_question_does_not_inherit_global_docs(self) -> None:
        """Leading DOCS before Question1 is only a fallback for the first pair."""
        text = """CATEGORY: SAQ

DOCS:
Global intro.

Question1
Q: First?
A: A1.

Question2
Q: Second?
A: A2.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)
        self.assertIn("Global intro", entries[0]["docs"])
        self.assertEqual(entries[1]["docs"], "")

    def test_legacy_single_block_without_question_marker_is_not_multi(self) -> None:
        """Only Q:/A: with no QuestionN line uses legacy parser (one entry, no error)."""
        text = """CATEGORY: SAQ

Q: No Question line above.
A: Still valid legacy format.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question_label"], "")


class TestNormalizeSaqSubpartLabels(unittest.TestCase):
    """(a)/(b)/(c) and A)/B)/C) line prefixes map to Part A / Part B / Part C."""

    def test_parenthetical_lowercase(self) -> None:
        raw = "(a) First\n(b) Second\n(c) Third"
        out = ap_main._normalize_saq_subpart_labels(raw)
        self.assertIn("Part A: First", out)
        self.assertIn("Part B: Second", out)
        self.assertIn("Part C: Third", out)

    def test_uppercase_letter_close_paren(self) -> None:
        raw = "A) One\nB) Two"
        out = ap_main._normalize_saq_subpart_labels(raw)
        self.assertIn("Part A: One", out)
        self.assertIn("Part B: Two", out)

    def test_preserves_indent(self) -> None:
        raw = "  (a) Indented"
        out = ap_main._normalize_saq_subpart_labels(raw)
        self.assertTrue(out.startswith("  Part A:"))

    def test_non_marker_lines_unchanged(self) -> None:
        raw = "Intro line\n(a) Sub\nMore detail without prefix"
        out = ap_main._normalize_saq_subpart_labels(raw)
        self.assertIn("Intro line", out)
        self.assertIn("More detail without prefix", out)


class TestPreviewOneLine(unittest.TestCase):
    """_preview_one_line must collapse newlines so report rows stay one line."""

    def test_collapses_whitespace_for_multiline_question(self) -> None:
        """Regression: SAQ merged text uses newlines; previews must not embed them."""
        out = ap_main._preview_one_line("Part A\nQ: x\n\nPart B\nQ: y", 200)
        self.assertNotIn("\n", out)
        self.assertIn("Part A", out)
        self.assertIn("Q: y", out)


class TestFormatGradeReportOutput(unittest.TestCase):
    """Report text must include the four numbered sections and score metadata."""

    def _sample_saq_result(self) -> GradeResult:
        """Build a minimal SAQ GradeResult matching rubric structure."""
        criteria = [
            CriterionResult("Part A", 1, 1, "Student wrote X.", "", "Keep it up."),
            CriterionResult("Part B", 1, 0, "N/A", "Missing B.", "Add B."),
            CriterionResult("Part C", 1, 1, "Student wrote Y.", "", ""),
        ]
        return GradeResult(
            category="SAQ",
            question="Sample SAQ stem?",
            answer="A) ... B) ... C) ...",
            total_earned=2,
            total_possible=3,
            criteria_results=criteria,
            overall_suggestions="Good effort overall.",
        )

    def test_report_contains_four_numbered_sections(self) -> None:
        """Output must include all four report parts the user expects."""
        result = self._sample_saq_result()
        report = ap_main.format_grade_report(result, entry_index=1)
        self.assertIn("1. SCORE BREAKDOWN", report)
        self.assertIn("2. EVIDENCE THAT EARNED EACH POINT", report)
        self.assertIn("3. POINTS NOT EARNED AND WHY", report)
        self.assertIn("4. SUGGESTIONS TO IMPROVE", report)

    def test_report_shows_total_score_bar(self) -> None:
        report = ap_main.format_grade_report(self._sample_saq_result(), 1)
        self.assertIn("2/3", report)
        self.assertIn("TYPE: SAQ", report)

    def test_report_includes_question_label_when_provided(self) -> None:
        """Multi-question label appears in the essay header line."""
        report = ap_main.format_grade_report(
            self._sample_saq_result(),
            entry_index=2,
            question_label="Question2",
        )
        self.assertIn("ESSAY #2", report)
        self.assertIn("Question2", report)
        self.assertIn("TYPE: SAQ", report)

    def test_report_without_label_no_extra_pipe_segment(self) -> None:
        """Legacy: header should not inject a blank Question label awkwardly."""
        report = ap_main.format_grade_report(self._sample_saq_result(), 1, question_label=None)
        self.assertRegex(report, r"ESSAY #1\s+\|\s+TYPE: SAQ")

    def test_report_question_preview_includes_all_parts_on_one_line(self) -> None:
        """Multi-line question text must be collapsed in QUESTION: row (not truncated at first \\n)."""
        result = GradeResult(
            category="SAQ",
            question="Part A\nQ: First stem\n\nPart B\nQ: Second stem",
            answer="A",
            total_earned=3,
            total_possible=3,
            criteria_results=[
                CriterionResult("Part A", 1, 1, "e", "", ""),
                CriterionResult("Part B", 1, 1, "e", "", ""),
                CriterionResult("Part C", 1, 1, "e", "", ""),
            ],
            overall_suggestions="",
        )
        report = ap_main.format_grade_report(result, 1)
        preview_row = report.split("QUESTION: ", 1)[1].split("\n", 1)[0]
        self.assertIn("Part B", preview_row)
        self.assertIn("Second stem", preview_row)



class TestPrintSummaryOutput(unittest.TestCase):
    """Summary table shape for multiple graded essays."""

    def test_summary_contains_header_and_totals(self) -> None:
        r1 = GradeResult(
            "SAQ", "Q1", "A1", 2, 3, [
                CriterionResult("Part A", 1, 1, "e", "", "s"),
                CriterionResult("Part B", 1, 1, "e", "", "s"),
                CriterionResult("Part C", 1, 0, "N/A", "n", "s"),
            ],
            "ok",
        )
        r2 = GradeResult(
            "SAQ", "Q2", "A2", 3, 3, [
                CriterionResult("Part A", 1, 1, "e", "", ""),
                CriterionResult("Part B", 1, 1, "e", "", ""),
                CriterionResult("Part C", 1, 1, "e", "", ""),
            ],
            "ok",
        )
        summary = ap_main.print_summary(
            [r1, r2],
            question_labels=["Question1", "Question2"],
        )
        self.assertIn("GRADING SUMMARY", summary)
        self.assertIn("Label", summary)
        self.assertIn("5/6", summary)  # total earned / total possible


class TestSampleFilesIfPresent(unittest.TestCase):
    """Optional: parse shipped sample files when they exist (integration smoke)."""

    def test_default_txt_files_parse_without_error(self) -> None:
        for name in ("DBQQandA.txt", "LEQQandA.txt", "SAQQandA.txt"):
            path = _PKG_DIR / name
            if not path.exists():
                self.skipTest(f"missing {name}")
            entries = ap_main.parse_qa_file(path)
            self.assertGreater(len(entries), 0)
            for e in entries:
                self.assertIn("category", e)
                self.assertIn("question", e)
                self.assertIn("answer", e)
                self.assertIn("docs", e)
                self.assertIn("question_label", e)

    def test_multi_txt_files_parse_without_error(self) -> None:
        for name in ("DBQQandA_multi.txt", "LEQQandA_multi.txt", "SAQQandA_multi.txt"):
            path = _PKG_DIR / name
            if not path.exists():
                self.skipTest(f"missing {name}")
            entries = ap_main.parse_qa_file(path)
            self.assertGreater(len(entries), 0)
            for e in entries:
                self.assertNotEqual(e["question"].strip(), "")
                self.assertNotEqual(e["answer"].strip(), "")


if __name__ == "__main__":
    unittest.main()
