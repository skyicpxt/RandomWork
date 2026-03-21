# Official AP World History: Modern rubric definitions for DBQ, LEQ, and SAQ.
# Source: https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment
# and College Board published scoring guidelines.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RubricCriterion:
    """A single scorable criterion within a rubric."""
    name: str
    max_points: int
    description: str
    earning_criteria: list[str]
    not_earning_criteria: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DBQ Rubric  (7 points total, worth 25% of exam score)
# Reference: AP World History: Modern Exam – AP Central / College Board
# ---------------------------------------------------------------------------
DBQ_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        name="Thesis/Claim",
        max_points=1,
        description=(
            "Responds to the prompt with a historically defensible thesis or claim "
            "that establishes a line of reasoning. The thesis must do more than restate "
            "the prompt; it should make a historically defensible claim and indicate the "
            "reason for the claim or establish analytical categories."
        ),
        earning_criteria=[
            "Makes a historically defensible claim that responds to the prompt.",
            "Establishes a line of reasoning (not just a restatement of the prompt).",
            "May appear in the introduction or conclusion of the essay.",
            "Identifies a relevant historical development within the time period.",
        ],
        not_earning_criteria=[
            "Merely restates or rephrases the prompt.",
            "States only a vague or generic claim with no line of reasoning.",
            "Appears only in the middle of the essay without framing an argument.",
        ],
    ),
    RubricCriterion(
        name="Contextualization",
        max_points=1,
        description=(
            "Accurately describes a broader historical context relevant to the prompt. "
            "Context must relate the topic of the prompt to broader historical events, "
            "developments, or processes that occur before, during, or continue after "
            "the time frame of the prompt. This is more than a single phrase or sentence."
        ),
        earning_criteria=[
            "Accurately describes a broader historical context relevant to the prompt.",
            "Relates the prompt's topic to broader historical events or processes.",
            "Contextualization goes beyond a brief mention — it is developed (multiple sentences).",
            "The context can be from before, during, or after the prompt's time frame.",
        ],
        not_earning_criteria=[
            "Merely mentions a historical fact without tying it to the prompt's argument.",
            "Context is limited to a single phrase or sentence.",
            "Restates the prompt's time period without broader context.",
        ],
    ),
    RubricCriterion(
        name="Evidence: Content from Documents (1 pt)",
        max_points=1,
        description=(
            "1 point: Uses the content of at least THREE documents to address the topic "
            "of the prompt. (Accurate content must be used, not just a citation.)"
        ),
        earning_criteria=[
            "Accurately describes the content of at least three documents.",
            "Uses document content to address the topic of the prompt.",
        ],
        not_earning_criteria=[
            "Merely cites documents without describing their content.",
            "Uses fewer than three documents accurately.",
            "Misidentifies or misrepresents document content.",
        ],
    ),
    RubricCriterion(
        name="Evidence: Content from Documents (2 pts)",
        max_points=1,
        description=(
            "2 points total for documents: Uses the content of at least SIX documents "
            "to support an argument in response to the prompt. (Requires the 1-pt tier.)"
        ),
        earning_criteria=[
            "Accurately describes the content of at least six documents.",
            "Uses those six documents to support an argument (not just address the topic).",
        ],
        not_earning_criteria=[
            "Uses fewer than six documents to support an argument.",
            "Describes document content without tying it to an argument.",
        ],
    ),
    RubricCriterion(
        name="Evidence: Evidence Beyond the Documents",
        max_points=1,
        description=(
            "Uses at least one piece of relevant evidence not found in the documents "
            "to support an argument about the prompt. This outside evidence must be "
            "specific, accurate, and used to support the argument."
        ),
        earning_criteria=[
            "Provides at least one piece of specific evidence not in the documents.",
            "The evidence is accurate and relevant to the argument.",
            "The evidence is used to support (not just mention) the argument.",
        ],
        not_earning_criteria=[
            "Only paraphrases or restates the documents.",
            "Outside evidence is vague or inaccurate.",
            "Outside evidence is mentioned but not used to support the argument.",
        ],
    ),
    RubricCriterion(
        name="Analysis & Reasoning: Sourcing",
        max_points=1,
        description=(
            "For at least THREE documents, explains how or why the document's point of view, "
            "purpose, historical situation, or audience is relevant to an argument. "
            "Sourcing must be tied to an argument, not just stated."
        ),
        earning_criteria=[
            "Explains POV, purpose, historical situation, or audience for at least three documents.",
            "Sourcing is connected to and supports an argument.",
            "Explanation goes beyond a surface-level observation.",
        ],
        not_earning_criteria=[
            "Simply identifies the author or document type without explanation.",
            "Sourcing is applied to fewer than three documents.",
            "Sourcing is stated but not tied to the argument.",
        ],
    ),
    RubricCriterion(
        name="Analysis & Reasoning: Complexity",
        max_points=1,
        description=(
            "Demonstrates a complex understanding of the historical development analyzed. "
            "This can be done through corroboration, qualification, modification of the "
            "argument, explaining both similarity AND difference, both continuity AND change, "
            "multiple causes, or both cause AND effect. The complexity must be present "
            "throughout the essay, not just in a single phrase."
        ),
        earning_criteria=[
            "Explains nuance by analyzing multiple variables (cause/effect, continuity/change, similarity/difference).",
            "Explains both the cause and effect of a historical development.",
            "Qualifies or modifies the argument with a counter-argument or exception.",
            "Connects the argument to a different time period, geographical area, or theme.",
            "Complexity is demonstrated throughout the essay, not just in one sentence.",
        ],
        not_earning_criteria=[
            "Complexity is limited to a single phrase like 'it was complex' without demonstration.",
            "Does not go beyond the basic argument to show nuance.",
            "Does not address counter-arguments or qualifications.",
        ],
    ),
]

DBQ_MAX_SCORE = 7
DBQ_EXAM_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# LEQ Rubric  (6 points total, worth 15% of exam score)
# Reference: AP World History: Modern Exam – AP Central / College Board
# ---------------------------------------------------------------------------
LEQ_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        name="Thesis/Claim",
        max_points=1,
        description=(
            "Responds to the prompt with a historically defensible thesis or claim "
            "that establishes a line of reasoning. The thesis must make a historically "
            "defensible claim that is more than a restatement of the prompt."
        ),
        earning_criteria=[
            "Makes a historically defensible claim that responds to the prompt.",
            "Establishes a line of reasoning beyond restating the prompt.",
            "May appear in the introduction or conclusion.",
        ],
        not_earning_criteria=[
            "Merely restates or rephrases the prompt.",
            "Makes only a vague or generic claim.",
            "No clear line of reasoning is established.",
        ],
    ),
    RubricCriterion(
        name="Contextualization",
        max_points=1,
        description=(
            "Accurately describes a broader historical context relevant to the prompt. "
            "Relates the topic to broader historical events, developments, or processes "
            "before, during, or after the time frame. Must be more than a brief phrase."
        ),
        earning_criteria=[
            "Accurately describes a broader historical context relevant to the prompt.",
            "Contextualization is developed (not just a single phrase or sentence).",
            "Links the broader context to the argument.",
        ],
        not_earning_criteria=[
            "Context is a single phrase or sentence without development.",
            "Mentions a fact without tying it to the prompt's topic or argument.",
        ],
    ),
    RubricCriterion(
        name="Evidence (1 pt): Specific Examples",
        max_points=1,
        description=(
            "Provides specific examples of evidence relevant to the topic of the prompt."
        ),
        earning_criteria=[
            "Provides at least one specific, accurate piece of evidence relevant to the prompt.",
            "Evidence is described (not merely mentioned) and relates to the topic.",
        ],
        not_earning_criteria=[
            "Evidence is vague or not specific.",
            "Evidence is inaccurate.",
            "Evidence is off-topic.",
        ],
    ),
    RubricCriterion(
        name="Evidence (2 pts): Supports Argument",
        max_points=1,
        description=(
            "Uses specific evidence to support an argument in response to the prompt."
        ),
        earning_criteria=[
            "Uses multiple specific, accurate pieces of evidence.",
            "Evidence is clearly tied to and supports the essay's argument.",
        ],
        not_earning_criteria=[
            "Evidence is listed but not tied to an argument.",
            "Only one piece of evidence is used to support the argument.",
        ],
    ),
    RubricCriterion(
        name="Analysis & Reasoning: Historical Reasoning",
        max_points=1,
        description=(
            "Uses a historical reasoning skill (Comparison, Causation, or Continuity "
            "and Change Over Time) to frame or structure an argument about the prompt."
        ),
        earning_criteria=[
            "Applies comparison, causation, or CCOT to frame the argument.",
            "Historical reasoning skill is used throughout the essay, not just mentioned.",
            "The reasoning skill explains the significance of evidence.",
        ],
        not_earning_criteria=[
            "Historical reasoning skill is mentioned but not applied.",
            "The essay describes events chronologically without applying a reasoning skill.",
        ],
    ),
    RubricCriterion(
        name="Analysis & Reasoning: Complexity",
        max_points=1,
        description=(
            "Demonstrates a complex understanding of the historical development. "
            "Achieved through corroboration, qualification, multiple causation, "
            "continuity AND change, similarity AND difference, or connections across "
            "time periods, themes, or geographic areas. Must be present throughout."
        ),
        earning_criteria=[
            "Explains nuance: multiple causes, continuity AND change, similarity AND difference.",
            "Qualifies or modifies the argument with exceptions or counter-arguments.",
            "Connects the argument to different time periods, geographic areas, or themes.",
            "Complexity is demonstrated throughout the essay.",
        ],
        not_earning_criteria=[
            "Complexity is only mentioned or stated, not demonstrated.",
            "Single-paragraph gesture at complexity without sustained analysis.",
        ],
    ),
]

LEQ_MAX_SCORE = 6
LEQ_EXAM_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# SAQ Rubric  (3 points per question, 3 questions = 9 total, worth 20% of score)
# Reference: AP World History: Modern Exam – AP Central / College Board
# ---------------------------------------------------------------------------
SAQ_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        name="Part A",
        max_points=1,
        description=(
            "Responds accurately to the 'describe/identify' or 'explain' prompt in Part A. "
            "Must provide a historically accurate response that directly addresses the question. "
            "Does not require a thesis or complex argument."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response to Part A.",
            "Directly addresses what is being asked (describe, identify, or explain).",
            "Response is specific, not vague.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not directly address the Part A question.",
            "Response is too vague to demonstrate specific knowledge.",
        ],
    ),
    RubricCriterion(
        name="Part B",
        max_points=1,
        description=(
            "Responds accurately to the 'explain' or 'describe' prompt in Part B. "
            "Must provide a historically accurate response that directly addresses the question."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response to Part B.",
            "Directly addresses what is being asked (explain or describe).",
            "Response demonstrates specific knowledge with an example or detail.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not address Part B.",
            "Response is a mere restatement with no specific knowledge.",
        ],
    ),
    RubricCriterion(
        name="Part C",
        max_points=1,
        description=(
            "Responds accurately to the 'explain' or 'describe' prompt in Part C. "
            "Must provide a historically accurate response that directly addresses the question."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response to Part C.",
            "Directly addresses what is being asked (explain, describe, or evaluate).",
            "Response demonstrates specific knowledge with an example or detail.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not address Part C.",
            "Response is vague or generic.",
        ],
    ),
]

SAQ_MAX_SCORE = 3
SAQ_EXAM_WEIGHT = 0.20  # per question; 3 questions total


def get_rubric(category: str) -> tuple[list[RubricCriterion], int]:
    """
    Returns (rubric_criteria, max_score) for the given essay category.
    category must be one of 'DBQ', 'LEQ', or 'SAQ'.
    """
    category = category.strip().upper()
    if category == "DBQ":
        return DBQ_RUBRIC, DBQ_MAX_SCORE
    elif category == "LEQ":
        return LEQ_RUBRIC, LEQ_MAX_SCORE
    elif category == "SAQ":
        return SAQ_RUBRIC, SAQ_MAX_SCORE
    else:
        raise ValueError(f"Unknown category: {category!r}. Must be 'DBQ', 'LEQ', or 'SAQ'.")


def format_rubric_for_prompt(rubric: list[RubricCriterion], max_score: int) -> str:
    """Formats rubric criteria into a structured string for use in a prompt."""
    lines = [f"TOTAL POSSIBLE POINTS: {max_score}\n"]
    for criterion in rubric:
        lines.append(f"CRITERION: {criterion.name} (max {criterion.max_points} pt{'s' if criterion.max_points > 1 else ''})")
        lines.append(f"  Description: {criterion.description}")
        lines.append("  To earn this point:")
        for item in criterion.earning_criteria:
            lines.append(f"    - {item}")
        if criterion.not_earning_criteria:
            lines.append("  Common reasons this point is NOT earned:")
            for item in criterion.not_earning_criteria:
                lines.append(f"    - {item}")
        lines.append("")
    return "\n".join(lines)
