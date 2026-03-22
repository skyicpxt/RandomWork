# Official AP World History: Modern rubric definitions for DBQ, LEQ, and SAQ.
# DBQ source: AP World History: Modern 2025 Scoring Guidelines (College Board)
#   https://apcentral.collegeboard.org/media/pdf/ap25-apc-world-history-dbq-set-1.pdf
# LEQ/SAQ source: AP World History: Modern Course and Exam Description (College Board)
#   https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment

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
# Source: AP World History: Modern 2025 Scoring Guidelines (College Board)
# Rows A–D as published; thresholds taken verbatim from the official document.
# ---------------------------------------------------------------------------
DBQ_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        name="Thesis/Claim",
        max_points=1,
        description=(
            "Row A (0–1 pts). Responds to the prompt with a historically defensible "
            "thesis/claim that establishes a line of reasoning. The thesis must consist "
            "of one or more sentences located in one place, either in the introduction "
            "or the conclusion. It must identify a relevant development in the period, "
            "and must either provide some indication of the reason for the claim OR "
            "establish categories of the argument."
        ),
        earning_criteria=[
            "Provides a historically defensible thesis or claim that responds to the prompt.",
            "Establishes a line of reasoning — not merely a restatement or rephrasing of the prompt.",
            "Either indicates the reason for the claim OR establishes analytical categories.",
            "Located in one place in the introduction or conclusion (not scattered through the essay).",
        ],
        not_earning_criteria=[
            "Is not historically defensible.",
            "Only restates or rephrases the prompt.",
            "Does not respond to the prompt.",
            "Does not establish a line of reasoning.",
            "Is overgeneralized.",
        ],
    ),
    RubricCriterion(
        name="Contextualization",
        max_points=1,
        description=(
            "Row B (0–1 pts). Describes a broader historical context relevant to the prompt. "
            "The response must describe broader historical events, developments, or processes "
            "that occur before, during, or continue after the time frame of the question and "
            "are relevant to the topic. Context must be more than a phrase or reference — "
            "it must be developed with elaboration."
        ),
        earning_criteria=[
            "Accurately describes a broader historical context relevant to the prompt's topic.",
            "Context describes events, developments, or processes before, during, or after the time frame.",
            "Provides more than a passing phrase or brief reference — context is elaborated.",
        ],
        not_earning_criteria=[
            "Provides an overgeneralized statement about the time period referenced in the prompt.",
            "Provides context that is not relevant to the prompt.",
            "Provides only a passing phrase or reference without elaboration.",
        ],
    ),
    RubricCriterion(
        name="Evidence from Documents: Content (Tier 1)",
        max_points=1,
        description=(
            "Row C — first tier (0–1 pts). Uses the content of at least THREE documents "
            "to address the topic of the prompt. Must accurately describe — rather than simply "
            "quote — each document's content. Documents addressed collectively rather than "
            "separately do not earn this point."
        ),
        earning_criteria=[
            "Accurately describes — rather than simply quotes — the content of at least three documents.",
            "Uses the described document content to address the topic of the prompt.",
            "Each document is considered separately (not collectively as a group).",
        ],
        not_earning_criteria=[
            "Uses evidence from fewer than three documents.",
            "Misinterprets the content of a document.",
            "Quotes document content without an accompanying description or analysis.",
            "Addresses documents collectively rather than considering each separately.",
        ],
    ),
    RubricCriterion(
        name="Evidence from Documents: Supports Argument (Tier 2)",
        max_points=1,
        description=(
            "Row C — second tier (1 additional point beyond Tier 1). Supports an argument "
            "in response to the prompt using at least FOUR documents. The four documents do not "
            "have to support a single argument — they can be used across sub-arguments or to "
            "address counterarguments. Earning this point requires first earning Tier 1."
        ),
        earning_criteria=[
            "Supports an argument in response to the prompt by accurately using at least four documents.",
            "Evidence from the documents is explicitly tied to and advances the argument.",
            "The four documents may support sub-arguments or address counterarguments.",
        ],
        not_earning_criteria=[
            "Uses fewer than four documents to support an argument.",
            "Describes document content accurately but does not use it to support an argument.",
        ],
    ),
    RubricCriterion(
        name="Evidence Beyond the Documents",
        max_points=1,
        description=(
            "Row C — evidence beyond documents (0–1 pts). Uses at least one additional piece "
            "of specific historical evidence, beyond that found in the documents, relevant to "
            "an argument in response to the prompt. The evidence must be specific (more than a "
            "phrase or reference), accurate, relevant to an argument, different from the "
            "evidence used for contextualization, and not a repeat of information in the prompt "
            "or documents."
        ),
        earning_criteria=[
            "Provides at least one specific piece of historical evidence not found in the documents.",
            "The evidence is accurate and relevant to an argument addressing the prompt.",
            "The evidence is elaborated beyond a passing phrase or reference.",
            "The evidence is distinct from what was used to earn the contextualization point.",
        ],
        not_earning_criteria=[
            "Provides evidence not relevant to an argument about the prompt.",
            "Provides evidence outside the time period or region specified in the prompt.",
            "Repeats information already specified in the prompt or any of the documents.",
            "Provides only a passing phrase or reference without elaboration.",
        ],
    ),
    RubricCriterion(
        name="Analysis and Reasoning: Sourcing",
        max_points=1,
        description=(
            "Row D — sourcing (0–1 pts). For at least TWO documents, explains how or why "
            "the document's point of view, purpose, historical situation, and/or audience "
            "is relevant to an argument. Must explain — not merely identify — the relevance. "
            "Summarizing the document's content without connecting it to POV/purpose/situation/"
            "audience does not earn this point."
        ),
        earning_criteria=[
            "Explains how or why — not merely identifies — the POV, purpose, historical situation, "
            "or audience of at least two documents is relevant to an argument.",
            "The sourcing explanation is connected to an argument that addresses the prompt.",
            "Each sourcing attempt goes beyond summarizing document content.",
        ],
        not_earning_criteria=[
            "Explains sourcing for fewer than two documents.",
            "Identifies POV, purpose, historical situation, or audience but does not explain "
            "how or why it is relevant to an argument.",
            "Summarizes the content or argument of the document without explaining relevance "
            "to POV, purpose, historical situation, or audience.",
        ],
    ),
    RubricCriterion(
        name="Analysis and Reasoning: Complex Understanding",
        max_points=1,
        description=(
            "Row D — complex understanding (0–1 pts). Demonstrates a complex understanding "
            "of the historical development that is the focus of the prompt through sophisticated "
            "argumentation and/or effective use of evidence. Must be part of the argument and "
            "more than merely a phrase or reference. May be demonstrated anywhere in the response."
        ),
        earning_criteria=[
            "Explains multiple themes or perspectives to explore complexity or nuance.",
            "Explains multiple causes or effects, multiple similarities or differences, "
            "or multiple continuities or changes.",
            "Explains both cause and effect, both similarity and difference, or both "
            "continuity and change.",
            "Explains relevant and insightful connections within and across periods or "
            "geographical areas, clearly related to an argument.",
            "Effectively uses all seven documents to support an argument (use may be "
            "unevenly developed in one or two instances).",
            "Explains how the POV, purpose, historical situation, and/or audience of at "
            "least four documents supports an argument.",
            "Uses documents and evidence beyond the documents to demonstrate sophisticated "
            "understanding of different perspectives.",
        ],
        not_earning_criteria=[
            "Complexity is only a phrase or passing reference — not part of the argument.",
            "Gestures at complexity in a single sentence without sustained demonstration.",
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
        name="Evidence: Specific Examples (Tier 1)",
        max_points=1,
        description=(
            "Tier 1 (1 pt). Provides specific examples of evidence relevant to the topic of the prompt."
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
        name="Evidence: Supports Argument (Tier 2)",
        max_points=1,
        description=(
            "Tier 2 (1 additional pt beyond Tier 1). Uses specific evidence to support an argument "
            "in response to the prompt. Requires Tier 1 to also be earned."
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
            "Does not require a thesis or complex argument. "
            "SAQ answers are SHORT — typically 3–5 sentences per part, not a full multi-paragraph essay. "
            "The answer must clearly be structured as Part A of a 3-part SAQ, not a standalone essay."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response specifically to Part A.",
            "Directly addresses what is being asked (describe, identify, or explain).",
            "Response is specific, not vague.",
            "Answer is a focused short-answer paragraph (not a full multi-paragraph essay).",
            "The student's response is clearly labeled or structured as Part A of an SAQ.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not directly address the Part A question.",
            "Response is too vague to demonstrate specific knowledge.",
            "Response is a full multi-paragraph essay (LEQ/DBQ style) rather than a short-answer paragraph — SAQ requires a brief, focused response per part.",
            "No clearly identifiable Part A response exists in the student's answer.",
        ],
    ),
    RubricCriterion(
        name="Part B",
        max_points=1,
        description=(
            "Responds accurately to the 'explain' or 'describe' prompt in Part B. "
            "Must provide a historically accurate response that directly addresses the question. "
            "SAQ answers are SHORT — typically 3–5 sentences per part, not a full multi-paragraph essay. "
            "The answer must clearly be structured as Part B of a 3-part SAQ."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response specifically to Part B.",
            "Directly addresses what is being asked (explain or describe).",
            "Response demonstrates specific knowledge with an example or detail.",
            "Answer is a focused short-answer paragraph (not a full multi-paragraph essay).",
            "The student's response is clearly labeled or structured as Part B of an SAQ.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not address Part B.",
            "Response is a mere restatement with no specific knowledge.",
            "Response is a full multi-paragraph essay (LEQ/DBQ style) rather than a short-answer paragraph — SAQ requires a brief, focused response per part.",
            "No clearly identifiable Part B response exists in the student's answer.",
        ],
    ),
    RubricCriterion(
        name="Part C",
        max_points=1,
        description=(
            "Responds accurately to the 'explain' or 'describe' prompt in Part C. "
            "Must provide a historically accurate response that directly addresses the question. "
            "SAQ answers are SHORT — typically 3–5 sentences per part, not a full multi-paragraph essay. "
            "The answer must clearly be structured as Part C of a 3-part SAQ."
        ),
        earning_criteria=[
            "Provides a historically accurate and relevant response specifically to Part C.",
            "Directly addresses what is being asked (explain, describe, or evaluate).",
            "Response demonstrates specific knowledge with an example or detail.",
            "Answer is a focused short-answer paragraph (not a full multi-paragraph essay).",
            "The student's response is clearly labeled or structured as Part C of an SAQ.",
        ],
        not_earning_criteria=[
            "Response is inaccurate.",
            "Response does not address Part C.",
            "Response is vague or generic.",
            "Response is a full multi-paragraph essay (LEQ/DBQ style) rather than a short-answer paragraph — SAQ requires a brief, focused response per part.",
            "No clearly identifiable Part C response exists in the student's answer.",
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
