# Core AP World History grading logic.
# Sends essay (Q+A) to OpenAI with the official rubric embedded in the prompt
# and returns a structured GradeResult with score breakdown, evidence, and suggestions.
# Reference rubric: https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment


import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from rubrics import RubricCriterion, get_rubric, format_rubric_for_prompt


@dataclass
class CriterionResult:
    """Grading result for a single rubric criterion."""
    name: str
    max_points: int
    points_earned: int
    evidence: str           # Quote or paraphrase from essay that earned the point
    evidence_comment: str   # Brief explanation of WHY the evidence satisfies the criterion
    not_earned_reason: str  # Empty string if the point was earned
    suggestion: str         # Specific advice to improve this criterion


@dataclass
class GradeResult:
    """Full grading result for one Q&A pair."""
    category: str
    question: str
    answer: str
    total_earned: int
    total_possible: int
    criteria_results: list[CriterionResult]
    overall_suggestions: str
    raw_model_response: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert AP World History: Modern grader trained on the official \
College Board rubrics. You grade student essays fairly and constructively, following the official \
scoring guidelines from https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment.

Your output is always a single valid JSON object — no markdown fences, no extra text.
"""

_GRADING_INSTRUCTIONS = """
TASK: Grade the student's essay response using the rubric provided below.

ESSAY TYPE: {category}
QUESTION: {question}

{dbq_docs_section}

STUDENT ANSWER:
{answer}

---
OFFICIAL RUBRIC:
{rubric_text}

---
OUTPUT FORMAT (respond ONLY with this JSON, no other text):
{{
  "criteria_results": [
    {{
      "name": "<criterion name exactly as in rubric>",
      "max_points": <int>,
      "points_earned": <int — 0 or up to max_points>,
      "evidence": "<see EVIDENCE RULES below>",
      "evidence_comment": "<see EVIDENCE COMMENT RULES below>",
      "not_earned_reason": "<see NOT-EARNED RULES below>",
      "suggestion": "<specific, actionable advice to improve this criterion>"
    }}
  ],
  "overall_suggestions": "<2-4 sentences of holistic feedback on how to strengthen the essay overall>"
}}

GRADING RULES:
1. Award points ONLY if the student's essay clearly meets the criterion — be honest, not generous.
5. Do NOT invent or fabricate content the student did not write.
6. Do NOT penalize grammar or spelling unless it obscures historical content.
7. The DBQ complexity point is the hardest to earn — require evidence of nuance throughout the essay, not just a single sentence.

EVIDENCE RULES (the "evidence" field):
- If the point WAS earned: provide the specific sentence(s) or passage from the student's essay
  that directly satisfied the criterion. Prefer a direct quote (with quotation marks) over a
  paraphrase. If you must paraphrase, begin with "The student writes that…".
- If the point was NOT earned: set "evidence" to "N/A".
- Do NOT fabricate or embellish — only cite text that actually appears in the essay.
- For multi-part criteria (e.g., Evidence from Documents tiers), cite evidence from each
  document or piece of evidence that contributed to earning the point.
- Keep evidence concise: one to three sentences maximum. If the relevant passage is long,
  quote the most diagnostic phrase and summarize the rest.

EVIDENCE COMMENT RULES (the "evidence_comment" field):
- If the point WAS earned: write 1–2 sentences explaining specifically WHY the cited evidence
  satisfies this criterion — what element of the rubric requirement it fulfills and how.
  Do NOT restate the evidence; instead explain its significance to the criterion.
  Example: "This establishes a line of reasoning by identifying trade as the mechanism
  through which industrialization spread, going beyond a mere restatement of the prompt."
- If the point was NOT earned: set "evidence_comment" to "N/A".

NOT-EARNED REASON RULES (the "not_earned_reason" field):
- If the point WAS earned: set "not_earned_reason" to "" (empty string).
- If the point was NOT earned: write a specific, diagnostic explanation that tells the student
  exactly what was missing or wrong. Do NOT write generic statements like "insufficient evidence"
  or "needs more development." Instead:
    * Name the specific gap: which document was missing, which sourcing element was absent,
      what the thesis lacked, why the context was too brief, etc.
    * Reference the rubric criterion explicitly (e.g., "The thesis does not establish a line
      of reasoning — it restates the prompt without indicating why or how the claim is true.").
    * For documents criteria, state how many were used vs. how many are required.
    * For sourcing, identify which documents were sourced and what the explanation was missing
      (e.g., "Document 2 is identified as a photograph but the student does not explain how
      the photographer's purpose or audience shapes what the image conveys.").
- Limit to 2-4 sentences. Be specific, factual, and constructive — not discouraging.
"""

_DBQ_DOCS_SECTION_TEMPLATE = """SOURCE DOCUMENTS PROVIDED TO THE STUDENT:
{docs}
"""


# ---------------------------------------------------------------------------
# Answer revision
# ---------------------------------------------------------------------------

_REVISION_SYSTEM_PROMPT = (
    "You are an expert AP World History: Modern tutor. "
    "Rewrite the student's answer so it earns every available point on the official AP rubric. "
    "Preserve the student's ideas and historical examples where they already satisfy the rubric; "
    "add or strengthen only what is needed to earn the remaining points. "
    "Output ONLY the revised answer text — no preamble, labels, or explanation."
)

_REVISION_INSTRUCTIONS = """\
ESSAY TYPE: {category}

QUESTION:
{question}

{dbq_docs_section}STUDENT'S ORIGINAL ANSWER:
{answer}

OFFICIAL RUBRIC:
{rubric_text}

---
Rewrite the student's answer above so it earns EVERY rubric point.
{format_hint}

Output ONLY the revised answer text — no commentary, no preamble.\
"""

_REVISION_FORMAT_HINTS: dict[str, str] = {
    "SAQ": (
        "Return the revised answer using exactly this structure (one block per sub-part):\n"
        "(a)\n[revised answer for part a]\n\n"
        "(b)\n[revised answer for part b]\n\n"
        "(c)\n[revised answer for part c]"
    ),
    "LEQ": (
        "Write a complete essay with: a clear thesis establishing a line of reasoning, "
        "body paragraphs each providing specific evidence with reasoning connecting it to the argument, "
        "and contextualization."
    ),
    "DBQ": (
        "Write a complete essay with: a clear thesis, body paragraphs analyzing documents "
        "and providing sourcing (purpose/audience/POV/situation) for at least three, "
        "contextualization, and outside evidence."
    ),
}


# Produces a revised student answer that earns full marks on the AP rubric.
def revise_answer(
    client: OpenAI,
    category: str,
    question: str,
    answer: str,
    model: str,
    dbq_docs: Optional[str] = None,
) -> str:
    """
    Calls the OpenAI API to produce a revised version of the student's answer
    that earns full marks on the official AP rubric.
    Returns the revised answer as plain text.
    """
    rubric, max_score = get_rubric(category)
    rubric_text = format_rubric_for_prompt(rubric, max_score)
    dbq_docs_section = ""
    if dbq_docs and dbq_docs.strip():
        dbq_docs_section = _DBQ_DOCS_SECTION_TEMPLATE.format(docs=dbq_docs.strip()) + "\n"
    prompt = _REVISION_INSTRUCTIONS.format(
        category=category,
        question=question.strip(),
        dbq_docs_section=dbq_docs_section,
        answer=answer.strip(),
        rubric_text=rubric_text,
        format_hint=_REVISION_FORMAT_HINTS.get(category, ""),
    )
    messages = [
        {"role": "system", "content": _REVISION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return _call_with_retry(client, messages, model)


def _build_grading_prompt(
    category: str,
    question: str,
    answer: str,
    rubric_text: str,
    dbq_docs: Optional[str] = None,
) -> str:
    """Builds the full grading prompt string."""
    dbq_docs_section = ""
    if dbq_docs and dbq_docs.strip():
        dbq_docs_section = _DBQ_DOCS_SECTION_TEMPLATE.format(docs=dbq_docs.strip())

    return _GRADING_INSTRUCTIONS.format(
        category=category,
        question=question.strip(),
        dbq_docs_section=dbq_docs_section,
        answer=answer.strip(),
        rubric_text=rubric_text,
    )


# ---------------------------------------------------------------------------
# OpenAI call with retry
# ---------------------------------------------------------------------------

# Calls the OpenAI chat completions API with exponential backoff on rate limit errors.
def _call_with_retry(
    client: OpenAI,
    messages: list[dict],
    model: str,
    max_retries: int = 4,
) -> str:
    """
    Calls the OpenAI chat completions API with exponential backoff on rate limit errors.
    Uses reasoning_effort="high" and omits temperature/seed (not supported by reasoning models).
    Returns the raw response text, or raises ValueError if the response is empty.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort="high",
                max_completion_tokens=16000,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(
                    "The model returned an empty response. "
                    "This may mean the model hit its output token limit or encountered "
                    "an internal error. Try again or reduce the length of the input."
                )
            return content
        except Exception as e:
            last_err = e
            is_rate_limit = (
                getattr(e, "status_code", None) == 429
                or "rate" in str(type(e).__name__).lower()
                or "429" in str(e)
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = (2 ** attempt) + 1
                print(f"  Rate limit hit, retrying in {wait}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise last_err


# ---------------------------------------------------------------------------
# Core grading function
# ---------------------------------------------------------------------------

# Grades one essay via the API using the rubric for category.
def grade_essay(
    client: OpenAI,
    category: str,
    question: str,
    answer: str,
    dbq_docs: Optional[str] = None,
    model: str = "gpt-5.4",
) -> GradeResult:
    """
    Grades a single essay (question + answer) against the official AP rubric for
    the given category ('DBQ', 'LEQ', or 'SAQ').

    For DBQ, optionally pass the source documents text as dbq_docs.
    Returns a GradeResult with full score breakdown, evidence, and suggestions.
    """
    rubric, max_score = get_rubric(category)
    rubric_text = format_rubric_for_prompt(rubric, max_score)

    prompt = _build_grading_prompt(category, question, answer, rubric_text, dbq_docs)

    messages = [
        {"role": "developer", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    print(f"  Grading {category} essay with {model}...")
    raw = _call_with_retry(client, messages, model)

    # Parse JSON response — strip accidental markdown fences and give a clear error if parsing fails.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            snippet = raw[:300] if raw else "<empty>"
            raise ValueError(
                f"The model's response could not be parsed as JSON. "
                f"This usually means the model returned plain text instead of the expected JSON format. "
                f"Try switching to a different model.\n\nResponse snippet: {snippet!r}"
            ) from exc

    # Build CriterionResult list — match model output back to rubric criteria.
    #
    # Matching strategy (in priority order):
    #   1. Exact name match
    #   2. Case-insensitive / whitespace-normalised name match
    #   3. Position-based fallback (model returned criteria in rubric order)
    #
    # A warning is printed whenever a criterion name from the model does not
    # match the expected rubric name, so silent score drops are visible.
    criteria_results: list[CriterionResult] = []
    total_earned = 0

    raw_model_criteria: list[dict] = data.get("criteria_results", [])

    def _normalise(s: str) -> str:
        """Lowercase and collapse whitespace for fuzzy name comparison."""
        return " ".join(s.lower().split())

    # Build lookup by normalised name for fast matching
    model_by_norm_name: dict[str, dict] = {
        _normalise(r.get("name", "")): r for r in raw_model_criteria
    }

    for idx, criterion in enumerate(rubric):
        # 1. Exact match
        model_r = next(
            (r for r in raw_model_criteria if r.get("name", "") == criterion.name),
            None,
        )
        # 2. Normalised-name match
        if model_r is None:
            model_r = model_by_norm_name.get(_normalise(criterion.name))

        # 3. Position-based fallback
        if model_r is None and idx < len(raw_model_criteria):
            model_r = raw_model_criteria[idx]
            returned_name = model_r.get("name", "<unnamed>")
            print(
                f"  WARNING: criterion '{criterion.name}' not found by name in model output; "
                f"using position-{idx} entry '{returned_name}' as fallback."
            )

        if model_r is None:
            # Truly missing — award 0 and warn loudly
            print(
                f"  WARNING: criterion '{criterion.name}' is completely missing from model output; "
                f"defaulting to 0 points. Check the raw response."
            )
            model_r = {}

        points_earned = int(model_r.get("points_earned", 0))
        points_earned = max(0, min(points_earned, criterion.max_points))
        total_earned += points_earned

        criteria_results.append(CriterionResult(
            name=criterion.name,
            max_points=criterion.max_points,
            points_earned=points_earned,
            evidence=model_r.get("evidence", "N/A"),
            evidence_comment=model_r.get("evidence_comment", "N/A"),
            not_earned_reason=model_r.get("not_earned_reason", ""),
            suggestion=model_r.get("suggestion", ""),
        ))

    return GradeResult(
        category=category,
        question=question,
        answer=answer,
        total_earned=total_earned,
        total_possible=max_score,
        criteria_results=criteria_results,
        overall_suggestions=data.get("overall_suggestions", ""),
        raw_model_response=raw,
    )
