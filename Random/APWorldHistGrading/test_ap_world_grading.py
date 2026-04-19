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
from grader import _enforce_tier_dependency
from qa_parser import _normalize_saq_subpart_labels
from report_formatter import format_summary

import main as ap_main

# streamlit_app imports streamlit/dotenv at module level; stub both so the module can be
# imported in headless test environments without a running Streamlit server.
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = MagicMock()

from streamlit_app import (
    _build_revised_output,
    _extract_saq_parts,
    _extract_saq_stimulus_text,
    _parse_revised_saq,
    _spaced_paragraphs,
    _split_by_document_markers,
)


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

    def test_legacy_saq_subqa_with_stimulus_paragraph(self) -> None:
        """SAQ with a stimulus paragraph before the first (a) is parsed and stimulus preserved."""
        text = """CATEGORY: SAQ

Use the following passage to answer parts (a), (b), and (c).
"In 1492, Columbus sailed the ocean blue and the Columbian Exchange began."

(a)
Q: Briefly describe ONE cause.
A: Cause answer.

(b)
Q: Briefly describe ONE effect.
A: Effect answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["category"], "SAQ")
        self.assertIn("Stimulus:", e["question"])
        self.assertIn("Columbus sailed the ocean blue", e["question"])
        self.assertIn("Part A", e["question"])
        self.assertIn("Briefly describe ONE cause.", e["question"])
        self.assertIn("Part B", e["question"])
        # Stimulus should not appear in the answer side
        self.assertNotIn("Columbus sailed the ocean blue", e["answer"])
        self.assertIn("Cause answer.", e["answer"])
        self.assertIn("Effect answer.", e["answer"])

    def test_legacy_saq_subqa_with_inline_question_text(self) -> None:
        """SAQ accepts inline question text on the (a) line without a separate Q: line."""
        text = """CATEGORY: SAQ

(a) Briefly describe ONE cause of the Columbian Exchange.
A: My cause answer.

(b) Briefly describe ONE effect of the Columbian Exchange.
A: My effect answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertIn("Briefly describe ONE cause of the Columbian Exchange.", e["question"])
        self.assertIn("Briefly describe ONE effect of the Columbian Exchange.", e["question"])
        self.assertIn("My cause answer.", e["answer"])
        self.assertIn("My effect answer.", e["answer"])


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

    def test_multi_saq_subqa_with_stimulus_between_question_and_subparts(self) -> None:
        """A stimulus paragraph between QuestionN and (a) is captured as shared context."""
        text = """CATEGORY: SAQ

Question1
Use the following passage to answer parts (a), (b), and (c).
"Zheng He led seven voyages between 1405 and 1433 that reached as far as East Africa."

(a)
Q: Describe one change resulting from the voyages.
A: Change answer.

(b)
Q: Describe one continuity in Indian Ocean trade.
A: Continuity answer.

(c)
Q: Describe one impact of ending the voyages on China.
A: Impact answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["question_label"], "Question1")
        self.assertEqual(e["category"], "SAQ")
        self.assertIn("Stimulus:", e["question"])
        self.assertIn("Zheng He led seven voyages", e["question"])
        self.assertIn("Part A", e["question"])
        self.assertIn("Part B", e["question"])
        self.assertIn("Part C", e["question"])
        self.assertIn("Describe one change", e["question"])
        self.assertNotIn("Zheng He led seven voyages", e["answer"])
        self.assertIn("Change answer.", e["answer"])
        self.assertIn("Continuity answer.", e["answer"])
        self.assertIn("Impact answer.", e["answer"])

    def test_multi_saq_subqa_inline_question_text(self) -> None:
        """A QuestionN block whose sub-parts use inline 'prompt on (a) line' is parsed."""
        text = """CATEGORY: SAQ

Question1
(a) Briefly describe ONE cause of the Columbian Exchange.
A: My cause answer.

(b) Briefly describe ONE effect on the Americas.
A: My americas answer.

(c) Briefly describe ONE effect on Europe or Africa.
A: My eurafrica answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["question_label"], "Question1")
        self.assertIn("Briefly describe ONE cause of the Columbian Exchange.", e["question"])
        self.assertIn("Briefly describe ONE effect on the Americas.", e["question"])
        self.assertIn("Briefly describe ONE effect on Europe or Africa.", e["question"])
        self.assertIn("My cause answer.", e["answer"])
        self.assertIn("My americas answer.", e["answer"])
        self.assertIn("My eurafrica answer.", e["answer"])

    def test_multi_saq_question_stem_before_q_marker(self) -> None:
        """
        Tolerates: question stem written under 'QuestionN:' with an empty 'Q:' label
        (i.e. the stem is the preamble before Q:, and Q: itself has no text).
        Regression for: "missing question text after 'Q:'" on this layout.
        """
        text = """CATEGORY: SAQ

Question1:
Briefly explain ONE political cause of the French Revolution.

Q:
A: One key political cause was the absolutist monarchy of Louis XVI.

Question2:
Briefly explain ONE economic effect of the French Revolution.

Q:
A: A major economic effect was the abolition of feudal dues.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)
        self.assertIn("political cause of the French Revolution", entries[0]["question"])
        self.assertIn("absolutist monarchy", entries[0]["answer"])
        self.assertIn("economic effect of the French Revolution", entries[1]["question"])
        self.assertIn("abolition of feudal dues", entries[1]["answer"])

    def test_multi_saq_question_stem_without_q_marker(self) -> None:
        """Tolerates: 'Q:' line omitted entirely; the stem before A: is the question."""
        text = """CATEGORY: SAQ

Question1
Briefly describe ONE cause of the Columbian Exchange.
A: Columbus's 1492 voyage was a primary cause.

Question2
Briefly explain ONE effect of the Columbian Exchange.
A: A devastating effect was the demographic collapse of indigenous populations.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)
        self.assertIn("Briefly describe ONE cause", entries[0]["question"])
        self.assertIn("Columbus's 1492 voyage", entries[0]["answer"])
        self.assertIn("Briefly explain ONE effect", entries[1]["question"])
        self.assertIn("demographic collapse", entries[1]["answer"])

    def test_multi_saq_question_combines_preamble_and_q_text(self) -> None:
        """When both preamble and explicit Q: text are present, both are kept (preamble first)."""
        text = """CATEGORY: SAQ

Question1
Refer to the following short passage about the Industrial Revolution.

Q: Briefly describe ONE cause of industrialization.
A: One cause was the availability of coal in Britain.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        q = entries[0]["question"]
        self.assertIn("Refer to the following short passage", q)
        self.assertIn("Briefly describe ONE cause of industrialization.", q)
        # Preamble appears before the explicit Q: text in the merged question
        self.assertLess(
            q.index("Refer to the following short passage"),
            q.index("Briefly describe ONE cause"),
        )

    def test_multi_saq_auto_promote_multiple_qa_pairs_no_markers(self) -> None:
        """SAQ with stimulus + 3 bare Q:/A: pairs (no (a)/(b)/(c)) auto-splits to Part A/B/C."""
        text = """CATEGORY: SAQ

Question1
"The millionaires are a product of natural selection..."
— William Graham Sumner, 1883

Using the source above, answer parts a, b, and c below.

Q: Identify and explain ONE way the author applies evolutionary theory.
A: Natural selection answer for part a.

Q: Identify and explain ONE way principles influenced governmental policies 1865-1898.
A: Laissez-faire answer for part b.

Q: Identify and explain ONE way ideologies challenged the ideas 1750-1900.
A: Communism answer for part c.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["category"], "SAQ")
        # Stimulus + instruction line is captured as the shared preamble.
        self.assertIn("Stimulus:", e["question"])
        self.assertIn("millionaires are a product of natural selection", e["question"])
        self.assertIn("Using the source above", e["question"])
        # Each Q:/A: pair becomes its own Part with auto-assigned letter.
        self.assertIn("Part A", e["question"])
        self.assertIn("Part B", e["question"])
        self.assertIn("Part C", e["question"])
        self.assertIn("Identify and explain ONE way the author", e["question"])
        self.assertIn("Identify and explain ONE way principles", e["question"])
        self.assertIn("Identify and explain ONE way ideologies", e["question"])
        # Each answer is preserved under its own Part.
        self.assertIn("Part A", e["answer"])
        self.assertIn("Part B", e["answer"])
        self.assertIn("Part C", e["answer"])
        self.assertIn("Natural selection answer for part a.", e["answer"])
        self.assertIn("Laissez-faire answer for part b.", e["answer"])
        self.assertIn("Communism answer for part c.", e["answer"])

    def test_multi_saq_bare_subpart_markers_no_parens(self) -> None:
        """SAQ with bare 'a)' / 'b)' / 'c)' markers (no opening paren) splits correctly."""
        text = """CATEGORY: SAQ

Question1
Read the passage and answer the parts below.

a) Q: Briefly describe ONE cause.
A: Cause answer.

b) Q: Briefly describe ONE effect.
A: Effect answer.

c) Q: Briefly describe ONE continuity.
A: Continuity answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertIn("Stimulus:", e["question"])
        self.assertIn("Read the passage", e["question"])
        self.assertIn("Part A", e["question"])
        self.assertIn("Part B", e["question"])
        self.assertIn("Part C", e["question"])
        self.assertIn("Briefly describe ONE cause.", e["question"])
        self.assertIn("Briefly describe ONE effect.", e["question"])
        self.assertIn("Briefly describe ONE continuity.", e["question"])
        # Bare markers themselves don't leak into either field.
        self.assertNotIn("a)", e["answer"])
        self.assertNotIn("b)", e["answer"])
        self.assertNotIn("c)", e["answer"])
        self.assertIn("Cause answer.", e["answer"])
        self.assertIn("Effect answer.", e["answer"])
        self.assertIn("Continuity answer.", e["answer"])

    def test_multi_saq_bare_subpart_markers_inline_text(self) -> None:
        """SAQ with bare 'a) prompt' inline form (no parens, no Q:) splits correctly."""
        text = """CATEGORY: SAQ

Question1
Stimulus paragraph here.

a) Briefly describe ONE cause of industrialization.
A: Coal answer.

b) Briefly describe ONE effect of industrialization.
A: Urbanization answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertIn("Stimulus:", e["question"])
        self.assertIn("Stimulus paragraph here.", e["question"])
        self.assertIn("Part A", e["question"])
        self.assertIn("Part B", e["question"])
        self.assertIn("Briefly describe ONE cause of industrialization.", e["question"])
        self.assertIn("Briefly describe ONE effect of industrialization.", e["question"])
        self.assertIn("Coal answer.", e["answer"])
        self.assertIn("Urbanization answer.", e["answer"])

    def test_multi_saq_bare_marker_in_answer_does_not_split(self) -> None:
        """A bare letter+) inside an answer that breaks the a/b/c progression is kept as content."""
        text = """CATEGORY: SAQ

Question1
(a)
Q: First prompt?
A: First answer mentions option z) which should not split.

(b)
Q: Second prompt?
A: Second answer.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        # Out-of-sequence "z)" inside the answer must NOT terminate the section.
        self.assertIn("option z) which should not split.", e["answer"])
        self.assertIn("Part A", e["answer"])
        self.assertIn("Part B", e["answer"])

    def test_multi_saq_two_questions_one_with_stimulus(self) -> None:
        """Mixed: Question1 has no stimulus, Question2 has a stimulus paragraph."""
        text = """CATEGORY: SAQ

Question1
(a)
Q: First plain prompt?
A: First plain answer.

(b)
Q: Second plain prompt?
A: Second plain answer.

Question2
Examine the following primary source excerpt before answering each part.
"<historical excerpt about industrial revolution labor conditions>"

(a)
Q: Stimulus-based prompt one?
A: Stim answer one.

(b) Stimulus-based prompt two? (inline form)
A: Stim answer two.
"""
        entries = self._parse(text)
        self.assertEqual(len(entries), 2)

        e1 = entries[0]
        self.assertEqual(e1["question_label"], "Question1")
        self.assertNotIn("Stimulus:", e1["question"])
        self.assertIn("First plain prompt?", e1["question"])

        e2 = entries[1]
        self.assertEqual(e2["question_label"], "Question2")
        self.assertIn("Stimulus:", e2["question"])
        self.assertIn("industrial revolution labor conditions", e2["question"])
        self.assertIn("Stimulus-based prompt one?", e2["question"])
        self.assertIn("Stimulus-based prompt two? (inline form)", e2["question"])
        self.assertIn("Stim answer one.", e2["answer"])
        self.assertIn("Stim answer two.", e2["answer"])

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

    def test_no_diagnostic_section_when_grade_result_omitted(self) -> None:
        """When grade_result is not provided the prompt has no DIAGNOSTIC block."""
        client = self._make_client("ok")
        revise_answer(client, "LEQ", "Q?", "A.", "gpt-test")
        self.assertNotIn("DIAGNOSTIC FROM PRIOR GRADING", self._user_message(client))

    def _make_grade_result(self) -> GradeResult:
        """Builds a GradeResult with one earned and one unearned criterion for testing."""
        earned = CriterionResult(
            name="Thesis",
            max_points=1,
            points_earned=1,
            evidence="EARNED_EVIDENCE_SENTINEL",
            evidence_comment="Establishes a clear claim.",
            not_earned_reason="",
            suggestion="",
        )
        missed = CriterionResult(
            name="Contextualization",
            max_points=1,
            points_earned=0,
            evidence="N/A",
            evidence_comment="N/A",
            not_earned_reason="MISSING_REASON_SENTINEL",
            suggestion="FIX_SUGGESTION_SENTINEL",
        )
        return GradeResult(
            category="LEQ",
            question="Q?",
            answer="A.",
            total_earned=1,
            total_possible=2,
            criteria_results=[earned, missed],
            overall_suggestions="",
        )

    def test_diagnostic_section_lists_earned_and_unearned(self) -> None:
        """When grade_result is provided the prompt embeds earned evidence + missing reasons."""
        client = self._make_client("ok")
        gr = self._make_grade_result()
        revise_answer(client, "LEQ", "Q?", "A.", "gpt-test", grade_result=gr)
        prompt = self._user_message(client)
        self.assertIn("DIAGNOSTIC FROM PRIOR GRADING", prompt)
        self.assertIn("CRITERIA ALREADY EARNED", prompt)
        self.assertIn("EARNED_EVIDENCE_SENTINEL", prompt)
        self.assertIn("CRITERIA NOT YET EARNED", prompt)
        self.assertIn("MISSING_REASON_SENTINEL", prompt)
        self.assertIn("FIX_SUGGESTION_SENTINEL", prompt)

    def test_diagnostic_section_says_unchanged_when_all_earned(self) -> None:
        """When every criterion is earned the diagnostic instructs to return the answer unchanged."""
        client = self._make_client("ok")
        full = CriterionResult(
            name="Thesis",
            max_points=1,
            points_earned=1,
            evidence="quote",
            evidence_comment="ok",
            not_earned_reason="",
            suggestion="",
        )
        gr = GradeResult(
            category="LEQ",
            question="Q?",
            answer="A.",
            total_earned=1,
            total_possible=1,
            criteria_results=[full],
            overall_suggestions="",
        )
        revise_answer(client, "LEQ", "Q?", "A.", "gpt-test", grade_result=gr)
        prompt = self._user_message(client)
        self.assertIn("ALL CRITERIA ARE ALREADY EARNED", prompt)
        self.assertIn("UNCHANGED", prompt)


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


class TestExtractSaqStimulusText(unittest.TestCase):
    """_extract_saq_stimulus_text pulls the shared stimulus out of a merged SAQ question."""

    def test_returns_empty_for_question_without_stimulus(self) -> None:
        """A merged question that is just Part A/B/C with no Stimulus: prefix returns ''."""
        merged = "Part A\nQ: Stem A?\n\nPart B\nQ: Stem B?"
        self.assertEqual(_extract_saq_stimulus_text(merged), "")

    def test_returns_empty_for_plain_non_merged_question(self) -> None:
        """A plain (non-merged) single question with no Stimulus: header returns ''."""
        merged = "What caused the decline of the Mongol Empire?"
        self.assertEqual(_extract_saq_stimulus_text(merged), "")

    def test_extracts_single_paragraph_stimulus(self) -> None:
        """Single-paragraph stimulus is extracted with the 'Stimulus:' label stripped."""
        merged = (
            "Stimulus:\nThe millionaires are a product of natural selection.\n\n"
            "Part A\nQ: Stem A?"
        )
        self.assertEqual(
            _extract_saq_stimulus_text(merged),
            "The millionaires are a product of natural selection.",
        )

    def test_extracts_multi_paragraph_stimulus(self) -> None:
        """Multi-paragraph stimulus (paragraphs separated by blank lines) is preserved."""
        merged = (
            "Stimulus:\nFirst paragraph of source.\n\n"
            "Second paragraph of source.\n\n"
            "Part A\nQ: Stem A?"
        )
        out = _extract_saq_stimulus_text(merged)
        self.assertIn("First paragraph of source.", out)
        self.assertIn("Second paragraph of source.", out)


class TestBuildRevisedOutputStimulus(unittest.TestCase):
    """_build_revised_output renders the SAQ stimulus once at the top, not per sub-part."""

    def test_stimulus_emitted_once_before_parts(self) -> None:
        """Stimulus block appears exactly once, above the first (a) sub-part."""
        question = (
            "Stimulus:\nSumner argues that millionaires are naturally selected.\n\n"
            "Part A\nQ: Identify ONE way the author applies evolution.\n\n"
            "Part B\nQ: Identify ONE policy influenced by these ideas."
        )
        revised = "(a)\nAnswer for a.\n\n(b)\nAnswer for b."
        out = _build_revised_output("SAQ", question, revised)
        # Stimulus block present, with no leading "Q:" prefix
        self.assertIn("Stimulus:", out)
        self.assertIn("Sumner argues that millionaires", out)
        # The stimulus text must NOT be repeated as a Q: line
        self.assertEqual(out.count("Sumner argues that millionaires"), 1)
        # Per-part Q: and RA: lines for both sub-parts
        self.assertIn("Q: Identify ONE way the author applies evolution.", out)
        self.assertIn("RA: Answer for a.", out)
        self.assertIn("Q: Identify ONE policy influenced by these ideas.", out)
        self.assertIn("RA: Answer for b.", out)

    def test_no_stimulus_no_stimulus_header(self) -> None:
        """When the question has no Stimulus: prefix, the output must not invent one."""
        question = "Part A\nQ: Stem A?\n\nPart B\nQ: Stem B?"
        revised = "(a)\nA.\n\n(b)\nB."
        out = _build_revised_output("SAQ", question, revised)
        self.assertNotIn("Stimulus:", out)


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


class TestSplitByDocumentMarkers(unittest.TestCase):
    """_split_by_document_markers detects embedded doc headers and splits the text.
    Returns list[tuple[int, str]] — (doc_number, section_text)."""

    def _sections(self, text: str) -> list[str]:
        """Helper: return just the section strings from the result tuples."""
        return [s for _, s in _split_by_document_markers(text)]

    def _numbers(self, text: str) -> list[int]:
        """Helper: return just the document numbers from the result tuples."""
        return [n for n, _ in _split_by_document_markers(text)]

    def test_standard_document_headers(self) -> None:
        """'Document 1' / 'Document 2' splits into two (number, text) tuples."""
        text = "Document 1\nFirst source text.\n\nDocument 2\nSecond source text."
        parts = _split_by_document_markers(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], 1)
        self.assertEqual(parts[1][0], 2)
        self.assertIn("First source text.", parts[0][1])
        self.assertIn("Second source text.", parts[1][1])

    def test_case_insensitive_document(self) -> None:
        """'DOCUMENT 1' and 'document 2' are both recognised."""
        text = "DOCUMENT 1\nText A.\n\ndocument 2\nText B."
        self.assertEqual(len(_split_by_document_markers(text)), 2)

    def test_abbreviated_doc_header(self) -> None:
        """'Doc 1' and 'DOC 2' are recognised as markers with correct numbers."""
        text = "Doc 1\nAlpha.\n\nDOC 2\nBeta."
        parts = _split_by_document_markers(text)
        self.assertEqual(len(parts), 2)
        self.assertEqual(self._numbers(text), [1, 2])
        self.assertIn("Alpha.", self._sections(text)[0])
        self.assertIn("Beta.", self._sections(text)[1])

    def test_doc_with_period(self) -> None:
        """'Doc. 1' (with a period) is recognised and number extracted."""
        text = "Doc. 1\nGamma.\n\nDoc. 2\nDelta."
        self.assertEqual(self._numbers(text), [1, 2])

    def test_seven_documents(self) -> None:
        """A typical DBQ packet with seven documents splits correctly."""
        lines = [f"Document {n}\nContent of document {n}." for n in range(1, 8)]
        text = "\n\n".join(lines)
        parts = _split_by_document_markers(text)
        self.assertEqual(len(parts), 7)
        self.assertEqual([n for n, _ in parts], list(range(1, 8)))
        for n, section in parts:
            self.assertIn(f"Content of document {n}.", section)

    def test_header_with_trailing_punctuation(self) -> None:
        """'Document 1:' and 'Document 2 —' are still recognised."""
        text = "Document 1: Some title\nText one.\n\nDocument 2 — Another title\nText two."
        self.assertEqual(len(_split_by_document_markers(text)), 2)

    def test_no_markers_returns_empty_list(self) -> None:
        """Text without any document headers returns [] so caller falls back."""
        text = "This is just a single block of source text with no headers."
        self.assertEqual(_split_by_document_markers(text), [])

    def test_empty_string_returns_empty_list(self) -> None:
        self.assertEqual(_split_by_document_markers(""), [])

    def test_header_preserved_in_section(self) -> None:
        """Each returned section text begins with its original header line."""
        text = "Document 1\nBody one.\n\nDocument 2\nBody two."
        sections = self._sections(text)
        self.assertTrue(sections[0].startswith("Document 1"))
        self.assertTrue(sections[1].startswith("Document 2"))

    def test_mixed_capitalisation(self) -> None:
        """Mix of 'Document', 'Doc', and 'DOC' in the same file all split."""
        text = "Document 1\nA.\n\nDoc 2\nB.\n\nDOC 3\nC."
        self.assertEqual(len(_split_by_document_markers(text)), 3)

    def test_missing_document_numbers_detected(self) -> None:
        """When doc 2 is absent, numbers [1, 3] are returned so caller can warn."""
        text = "Document 1\nFirst.\n\nDocument 3\nThird."
        numbers = self._numbers(text)
        self.assertEqual(numbers, [1, 3])
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        self.assertEqual(missing, [2])

    def test_non_sequential_gap_of_two(self) -> None:
        """Documents 1 and 4 present — numbers 2 and 3 are missing."""
        text = "Document 1\nA.\n\nDocument 4\nD."
        numbers = self._numbers(text)
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        self.assertEqual(missing, [2, 3])

    # Regression test for centered/indented PDF headers: pypdf preserves the
    # leading whitespace used to center text on the page, so headers like
    # "    Document 4" must still be detected.
    def test_indented_headers_from_centered_pdf_text(self) -> None:
        """Headers with leading spaces or tabs (from centered PDF text) are still split."""
        text = (
            "    Document 1\nFirst.\n\n"
            "  Document 2\nSecond.\n\n"
            "\tDocument 3\nThird.\n\n"
            "Document 4\nFourth.\n\n"
            "      Document 5\nFifth."
        )
        numbers = self._numbers(text)
        self.assertEqual(numbers, [1, 2, 3, 4, 5])
        sections = self._sections(text)
        # Each captured header begins at "Document N" (leading whitespace stripped)
        self.assertTrue(sections[0].startswith("Document 1"))
        self.assertTrue(sections[1].startswith("Document 2"))
        self.assertTrue(sections[4].startswith("Document 5"))


# Helper: builds a CriterionResult quickly for the tier-dependency tests below.
def _cr(name: str, earned: int, max_pts: int = 1, evidence: str = "x") -> CriterionResult:
    """Construct a minimal CriterionResult for tier-dependency unit tests."""
    return CriterionResult(
        name=name,
        max_points=max_pts,
        points_earned=earned,
        evidence=evidence,
        evidence_comment="",
        not_earned_reason="",
        suggestion="",
    )


class TestEnforceTierDependency(unittest.TestCase):
    """_enforce_tier_dependency zeros out Tier 2 if its paired Tier 1 wasn't earned."""

    def test_zeros_tier2_when_tier1_missing(self) -> None:
        """DBQ Tier 2 awarded but Tier 1 missing — Tier 2 must be zeroed."""
        crs = [
            _cr("Evidence from Documents: Content (Tier 1)", earned=0),
            _cr("Evidence from Documents: Supports Argument (Tier 2)", earned=1),
        ]
        total = _enforce_tier_dependency(crs)
        self.assertEqual(crs[1].points_earned, 0)
        self.assertEqual(total, 0)
        self.assertIn("Tier 1", crs[1].not_earned_reason)

    def test_keeps_tier2_when_tier1_earned(self) -> None:
        """When Tier 1 is also earned, Tier 2 is preserved untouched."""
        crs = [
            _cr("Evidence from Documents: Content (Tier 1)", earned=1),
            _cr("Evidence from Documents: Supports Argument (Tier 2)", earned=1),
        ]
        total = _enforce_tier_dependency(crs)
        self.assertEqual(crs[1].points_earned, 1)
        self.assertEqual(total, 2)
        self.assertEqual(crs[1].not_earned_reason, "")

    def test_handles_leq_tier_pair(self) -> None:
        """LEQ uses 'Evidence: Specific Examples (Tier 1)' / '... Supports Argument (Tier 2)'."""
        crs = [
            _cr("Evidence: Specific Examples (Tier 1)", earned=0),
            _cr("Evidence: Supports Argument (Tier 2)", earned=1),
        ]
        total = _enforce_tier_dependency(crs)
        self.assertEqual(crs[1].points_earned, 0)
        self.assertEqual(total, 0)

    def test_preserves_original_rationale_when_present(self) -> None:
        """If Tier 2 already has a not_earned_reason, the dependency note is prepended."""
        crs = [
            _cr("Evidence from Documents: Content (Tier 1)", earned=0),
            _cr("Evidence from Documents: Supports Argument (Tier 2)", earned=1),
        ]
        crs[1].not_earned_reason = "Original rationale text."
        _enforce_tier_dependency(crs)
        self.assertIn("Tier 1", crs[1].not_earned_reason)
        self.assertIn("Original rationale text.", crs[1].not_earned_reason)

    def test_no_tier2_criteria_is_noop(self) -> None:
        """SAQ-style criteria (no Tier labels at all) are unaffected."""
        crs = [
            _cr("Part A", earned=1),
            _cr("Part B", earned=0),
            _cr("Part C", earned=1),
        ]
        total = _enforce_tier_dependency(crs)
        self.assertEqual(total, 2)
        self.assertEqual([c.points_earned for c in crs], [1, 0, 1])

    def test_tier2_with_zero_points_is_unchanged(self) -> None:
        """If the model already zeroed Tier 2, no warning or change is needed."""
        crs = [
            _cr("Evidence from Documents: Content (Tier 1)", earned=0),
            _cr("Evidence from Documents: Supports Argument (Tier 2)", earned=0),
        ]
        crs[1].not_earned_reason = "Only used 2 docs to support an argument."
        total = _enforce_tier_dependency(crs)
        self.assertEqual(total, 0)
        self.assertEqual(crs[1].not_earned_reason, "Only used 2 docs to support an argument.")


if __name__ == "__main__":
    unittest.main()
