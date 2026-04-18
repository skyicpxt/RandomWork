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

# Always import testing utilities so they're available in all test classes.
from unittest.mock import MagicMock, patch

# grader.py imports openai at import time; stub so unit tests run without openai installed.
if "openai" not in sys.modules:
    _openai_mod = MagicMock()
    _openai_mod.OpenAI = MagicMock
    sys.modules["openai"] = _openai_mod

from grader import CriterionResult, GradeResult, revise_answer
from qa_parser import _normalize_saq_subpart_labels
from report_formatter import format_summary

import main as ap_main

# streamlit_app imports streamlit/dotenv at module level; stub both so the module can be
# imported in headless test environments without a running Streamlit server.
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = MagicMock()

from streamlit_app import _build_revised_output, _extract_saq_parts, _parse_revised_saq, _spaced_paragraphs


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
        out = _normalize_saq_subpart_labels(raw)
        self.assertIn("Part A: First", out)
        self.assertIn("Part B: Second", out)
        self.assertIn("Part C: Third", out)

    def test_uppercase_letter_close_paren(self) -> None:
        raw = "A) One\nB) Two"
        out = _normalize_saq_subpart_labels(raw)
        self.assertIn("Part A: One", out)
        self.assertIn("Part B: Two", out)

    def test_preserves_indent(self) -> None:
        raw = "  (a) Indented"
        out = _normalize_saq_subpart_labels(raw)
        self.assertTrue(out.startswith("  Part A:"))

    def test_non_marker_lines_unchanged(self) -> None:
        raw = "Intro line\n(a) Sub\nMore detail without prefix"
        out = _normalize_saq_subpart_labels(raw)
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
            CriterionResult("Part A", 1, 1, "Student wrote X.", "", "", "Keep it up."),
            CriterionResult("Part B", 1, 0, "N/A", "", "Missing B.", "Add B."),
            CriterionResult("Part C", 1, 1, "Student wrote Y.", "", "", ""),
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
                CriterionResult("Part A", 1, 1, "e", "", "", ""),
                CriterionResult("Part B", 1, 1, "e", "", "", ""),
                CriterionResult("Part C", 1, 1, "e", "", "", ""),
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
                CriterionResult("Part A", 1, 1, "e", "", "", "s"),
                CriterionResult("Part B", 1, 1, "e", "", "", "s"),
                CriterionResult("Part C", 1, 0, "N/A", "", "n", "s"),
            ],
            "ok",
        )
        r2 = GradeResult(
            "SAQ", "Q2", "A2", 3, 3, [
                CriterionResult("Part A", 1, 1, "e", "", "", ""),
                CriterionResult("Part B", 1, 1, "e", "", "", ""),
                CriterionResult("Part C", 1, 1, "e", "", "", ""),
            ],
            "ok",
        )
        summary = format_summary([(1, "Question1", r1), (2, "Question2", r2)])
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


class TestReviseAnswer(unittest.TestCase):
    """revise_answer() builds the right prompt and returns the model's response."""

    def _make_client(self, content: str = "Revised answer.") -> MagicMock:
        """Returns a mock OpenAI client whose completions endpoint returns content."""
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def _user_message(self, client: MagicMock) -> str:
        """Extract the user-role message from the last API call."""
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        return next((m["content"] for m in messages if m["role"] == "user"), "")

    def _system_message(self, client: MagicMock) -> str:
        """Extract the system-role message from the last API call."""
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        return next(
            (m["content"] for m in messages if m["role"] in ("system", "developer")), ""
        )

    def test_returns_model_content(self) -> None:
        """Return value is exactly the text from the API response."""
        client = self._make_client("Excellent revised essay.")
        result = revise_answer(client, "LEQ", "What caused X?", "My draft.", "gpt-test")
        self.assertEqual(result, "Excellent revised essay.")

    def test_raises_value_error_on_empty_response(self) -> None:
        """An empty API response triggers a descriptive ValueError."""
        client = self._make_client("")
        with self.assertRaises(ValueError):
            revise_answer(client, "SAQ", "Q?", "A.", "gpt-test")

    def test_prompt_contains_question(self) -> None:
        """The question text is embedded in the user prompt."""
        client = self._make_client("ok")
        revise_answer(client, "LEQ", "Unique Question SENTINEL", "My answer.", "gpt-test")
        self.assertIn("Unique Question SENTINEL", self._user_message(client))

    def test_prompt_contains_original_answer(self) -> None:
        """The student's original answer is embedded in the user prompt."""
        client = self._make_client("ok")
        revise_answer(client, "LEQ", "Q?", "Unique Answer SENTINEL", "gpt-test")
        self.assertIn("Unique Answer SENTINEL", self._user_message(client))

    def test_prompt_contains_category(self) -> None:
        """The essay type label appears in the user prompt."""
        client = self._make_client("ok")
        revise_answer(client, "SAQ", "Q?", "A.", "gpt-test")
        self.assertIn("SAQ", self._user_message(client))

    def test_system_message_is_revision_role(self) -> None:
        """System prompt describes a tutor/revision role, not a grading role."""
        client = self._make_client("ok")
        revise_answer(client, "SAQ", "Q?", "A.", "gpt-test")
        sys_msg = self._system_message(client)
        self.assertIn("tutor", sys_msg.lower())

    def test_dbq_docs_included_when_provided(self) -> None:
        """DBQ source documents are passed through to the prompt."""
        client = self._make_client("ok")
        revise_answer(client, "DBQ", "Q?", "A.", "gpt-test", dbq_docs="DOCUMENT 1 text here")
        self.assertIn("DOCUMENT 1 text here", self._user_message(client))

    def test_dbq_docs_absent_when_none(self) -> None:
        """When dbq_docs is None no SOURCE DOCUMENTS block appears."""
        client = self._make_client("ok")
        revise_answer(client, "LEQ", "Q?", "A.", "gpt-test", dbq_docs=None)
        self.assertNotIn("SOURCE DOCUMENTS", self._user_message(client))

    def test_all_three_categories_accepted(self) -> None:
        """revise_answer works for DBQ, LEQ, and SAQ without raising."""
        for cat in ("DBQ", "LEQ", "SAQ"):
            client = self._make_client("ok")
            result = revise_answer(client, cat, "Q?", "A.", "gpt-test")
            self.assertEqual(result, "ok", f"Failed for category {cat}")


class TestExtractSaqParts(unittest.TestCase):
    """_extract_saq_parts splits a merged SAQ question/answer string into per-part tuples."""

    def test_three_parts_question(self) -> None:
        merged = "Part A\nQ: First stem?\n\nPart B\nQ: Second stem?\n\nPart C\nQ: Third stem?"
        parts = _extract_saq_parts(merged, "Q")
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ("a", "First stem?"))
        self.assertEqual(parts[1], ("b", "Second stem?"))
        self.assertEqual(parts[2], ("c", "Third stem?"))

    def test_two_parts_only(self) -> None:
        merged = "Part A\nQ: Alpha?\n\nPart B\nQ: Beta?"
        parts = _extract_saq_parts(merged, "Q")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], "a")
        self.assertEqual(parts[1][0], "b")

    def test_answer_label_key(self) -> None:
        """label_key='A' strips 'A:' prefix from each chunk."""
        merged = "Part A\nA: Answer one.\n\nPart B\nA: Answer two."
        parts = _extract_saq_parts(merged, "A")
        self.assertEqual(parts[0], ("a", "Answer one."))
        self.assertEqual(parts[1], ("b", "Answer two."))

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_extract_saq_parts("", "Q"), [])

    def test_skips_chunks_without_part_header(self) -> None:
        """Chunks that don't start with 'Part' are ignored."""
        merged = "Preamble text\n\nPart A\nQ: Real question?"
        parts = _extract_saq_parts(merged, "Q")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0], "a")


class TestParseRevisedSaq(unittest.TestCase):
    """_parse_revised_saq splits the model's (a)/(b)/(c) response into per-part tuples."""

    def test_standard_three_parts(self) -> None:
        revised = "(a)\nRevised part A.\n\n(b)\nRevised part B.\n\n(c)\nRevised part C."
        parts = _parse_revised_saq(revised)
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], ("a", "Revised part A."))
        self.assertEqual(parts[1], ("b", "Revised part B."))
        self.assertEqual(parts[2], ("c", "Revised part C."))

    def test_two_parts(self) -> None:
        revised = "(a)\nAnswer A.\n\n(b)\nAnswer B."
        parts = _parse_revised_saq(revised)
        self.assertEqual(len(parts), 2)

    def test_fallback_when_no_markers(self) -> None:
        """Text with no (a)/(b)/(c) markers returns a single tuple with empty letter."""
        revised = "This is a complete revised answer with no sub-part markers."
        parts = _parse_revised_saq(revised)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0], "")
        self.assertIn("complete revised answer", parts[0][1])

    def test_whitespace_around_markers(self) -> None:
        """Extra spaces/blank lines around markers are tolerated."""
        revised = "  (a)  \nAnswer A.\n\n  (b)  \nAnswer B."
        parts = _parse_revised_saq(revised)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], "a")
        self.assertEqual(parts[1][0], "b")

    def test_uppercase_markers_accepted(self) -> None:
        """(A) / (B) markers (uppercase) are handled the same as lowercase."""
        revised = "(A)\nUpper A.\n\n(B)\nUpper B."
        parts = _parse_revised_saq(revised)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], "a")


class TestBuildRevisedOutput(unittest.TestCase):
    """_build_revised_output constructs the copyable Q:/RA: text block."""

    def test_leq_format(self) -> None:
        """LEQ produces a simple 'Q:' then 'RA:' block with no bare 'A:' line."""
        out = _build_revised_output("LEQ", "Why did X happen?", "Better essay text.")
        self.assertIn("Q: Why did X happen?", out)
        self.assertIn("RA: Better essay text.", out)
        # Ensure no standalone "A:" label (distinct from "RA:" which contains "A:")
        self.assertNotIn("\nA:", out)
        self.assertFalse(out.startswith("A:"))

    def test_dbq_format(self) -> None:
        """DBQ produces the same simple two-line block."""
        out = _build_revised_output("DBQ", "Analyze trade.", "Revised DBQ essay.")
        self.assertIn("Q: Analyze trade.", out)
        self.assertIn("RA: Revised DBQ essay.", out)

    def test_saq_reconstructs_per_part(self) -> None:
        """SAQ output has (a)/(b)/(c) headers with Q: and RA: for each part."""
        question = "Part A\nQ: Stem A?\n\nPart B\nQ: Stem B?\n\nPart C\nQ: Stem C?"
        revised = "(a)\nRA text A.\n\n(b)\nRA text B.\n\n(c)\nRA text C."
        out = _build_revised_output("SAQ", question, revised)
        self.assertIn("(a)", out)
        self.assertIn("Q: Stem A?", out)
        self.assertIn("RA: RA text A.", out)
        self.assertIn("(b)", out)
        self.assertIn("Q: Stem B?", out)
        self.assertIn("RA: RA text B.", out)
        self.assertIn("(c)", out)
        # No bare "A:" label — only "RA:" should appear
        self.assertNotIn("\nA:", out)
        self.assertFalse(out.startswith("A:"))

    def test_question_label_prepended(self) -> None:
        """When question_label is provided it appears after the 'Revised answer' header."""
        out = _build_revised_output("LEQ", "Q?", "Revised.", question_label="Question 2")
        self.assertTrue(out.startswith("Revised answer"))
        self.assertIn("Question 2", out)

    def test_no_label_no_extra_header(self) -> None:
        """Output always starts with 'Revised answer' header and no leading blank line."""
        out = _build_revised_output("LEQ", "Q?", "Revised.")
        self.assertTrue(out.startswith("Revised answer"))
        self.assertFalse(out.startswith("\n"))

    def test_saq_fallback_ra_when_model_returns_no_markers(self) -> None:
        """If the model returns plain text without (a)/(b)/(c), each part still gets RA:."""
        question = "Part A\nQ: Stem A?\n\nPart B\nQ: Stem B?"
        revised = "Plain revised answer with no part markers."
        out = _build_revised_output("SAQ", question, revised)
        # Both parts should still have a RA: line
        self.assertEqual(out.count("RA:"), 2)
        self.assertIn("Plain revised answer", out)

    def test_saq_fallback_when_question_not_merged(self) -> None:
        """SAQ with a plain (non-merged) question still produces Q: and RA: lines."""
        # question without 'Part A/B/C' header — _extract_saq_parts returns []
        question = "What caused the decline of the Mongol Empire?"
        revised = "A full revised answer."
        out = _build_revised_output("SAQ", question, revised)
        self.assertIn("Q: What caused the decline of the Mongol Empire?", out)
        self.assertIn("RA: A full revised answer.", out)

    def test_output_mirrors_input_structure_replacing_only_answer_label(self) -> None:
        """
        End-to-end format check: the copyable output is structurally identical to the
        input except 'A:' is replaced by 'RA:'. Verifies CATEGORY line, question label,
        sub-part markers, and Q: lines are all preserved unchanged.
        """
        # Simulate what _revise_and_render assembles for a 2-question SAQ:
        #   entry 1: Question 1: with (a) and (b)
        #   entry 2: Question 2: with (a) only
        q1 = "Part A\nQ: First part question?\n\nPart B\nQ: Second part question?"
        ra1 = "(a)\nRevised answer A.\n\n(b)\nRevised answer B."
        block1 = _build_revised_output("SAQ", q1, ra1, "Question 1:")

        q2 = "Part A\nQ: Single-part question?"
        ra2 = "(a)\nRevised single answer."
        block2 = _build_revised_output("SAQ", q2, ra2, "Question 2:")

        full = "CATEGORY: SAQ\n\n" + "\n\n".join([block1, block2])

        # CATEGORY header is present
        self.assertTrue(full.startswith("CATEGORY: SAQ"))

        # Question labels are preserved
        self.assertIn("Question 1:", full)
        self.assertIn("Question 2:", full)

        # Sub-part markers are preserved
        self.assertIn("(a)", full)
        self.assertIn("(b)", full)

        # Q: lines are present (original question text unchanged)
        self.assertIn("Q: First part question?", full)
        self.assertIn("Q: Second part question?", full)
        self.assertIn("Q: Single-part question?", full)

        # RA: replaces A: — no bare A: on its own line
        self.assertIn("RA: Revised answer A.", full)
        self.assertIn("RA: Revised answer B.", full)
        self.assertIn("RA: Revised single answer.", full)
        self.assertNotIn("\nA:", full)
        self.assertFalse(full.startswith("A:"))


class TestSpacedParagraphs(unittest.TestCase):
    """_spaced_paragraphs ensures visible blank lines between paragraphs in Streamlit markdown."""

    def test_single_paragraph_unchanged_content(self) -> None:
        """Single-paragraph text has no double-newlines to expand; content is preserved."""
        out = _spaced_paragraphs("Hello world.")
        self.assertIn("Hello world.", out)

    def test_double_newline_becomes_four(self) -> None:
        """A paragraph break (\\n\\n) is doubled to \\n\\n\\n\\n for Streamlit rendering."""
        out = _spaced_paragraphs("Para one.\n\nPara two.")
        self.assertIn("Para one.", out)
        self.assertIn("Para two.", out)
        self.assertIn("\n\n\n\n", out)

    def test_triple_newline_normalised_then_doubled(self) -> None:
        """Three or more consecutive newlines are first collapsed to two, then doubled."""
        out = _spaced_paragraphs("A.\n\n\nB.")
        self.assertEqual(out.count("\n\n\n\n"), 1)
        self.assertNotIn("\n\n\n\n\n", out)

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace around the whole text is stripped."""
        out = _spaced_paragraphs("\n\nHello.\n\n")
        self.assertFalse(out.startswith("\n"))
        self.assertFalse(out.endswith("\n"))

    def test_empty_string_returns_empty(self) -> None:
        """Empty input produces an empty string without raising."""
        self.assertEqual(_spaced_paragraphs(""), "")

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only input is stripped to empty."""
        self.assertEqual(_spaced_paragraphs("   \n\n  "), "")


class TestBuildRevisedOutputHeader(unittest.TestCase):
    """'Revised answer' must appear as the first line for all essay types."""

    def test_leq_starts_with_revised_answer(self) -> None:
        out = _build_revised_output("LEQ", "Q?", "Essay text.")
        self.assertTrue(out.startswith("Revised answer"))

    def test_dbq_starts_with_revised_answer(self) -> None:
        out = _build_revised_output("DBQ", "Q?", "Essay text.")
        self.assertTrue(out.startswith("Revised answer"))

    def test_saq_starts_with_revised_answer(self) -> None:
        out = _build_revised_output("SAQ", "Part A\nQ: Stem?", "(a)\nAnswer.")
        self.assertTrue(out.startswith("Revised answer"))

    def test_header_present_with_question_label(self) -> None:
        """'Revised answer' header precedes the question label when one is given."""
        out = _build_revised_output("LEQ", "Q?", "Essay.", question_label="Question 3")
        lines = out.splitlines()
        self.assertEqual(lines[0], "Revised answer")
        self.assertIn("Question 3", out)
        self.assertGreater(out.index("Question 3"), out.index("Revised answer"))


if __name__ == "__main__":
    unittest.main()
