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

# Default model used by both main.py (CLI) and streamlit_app.py (web UI).
DEFAULT_MODEL = "gpt-5.4"


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
   When in doubt about whether a criterion is met, err on the side of NOT awarding the point and
   explain in not_earned_reason what would have been needed. The official rule is: "if it is not
   clearly demonstrated, it is not earned."
2. Do NOT invent or fabricate content the student did not write.
3. Do NOT penalize grammar or spelling unless it obscures historical content.
4. The Complexity criteria — "Analysis and Reasoning: Complex Understanding" (DBQ Row D, second
   tier) and "Analysis & Reasoning: Complexity" (LEQ) — are the hardest to earn. Require sustained
   demonstration of nuance throughout the essay, not just a single sentence. However, see rule 6.
5. EXHAUSTIVE DOCUMENT SCAN (DBQ ONLY — applies to "Evidence from Documents: Content (Tier 1)",
   "Evidence from Documents: Supports Argument (Tier 2)", and "Analysis and Reasoning: Sourcing"):
   Before deciding points_earned for ANY of these three criteria, you MUST first scan the entire
   student essay and consider EVERY document the essay references (Document 1, 2, 3, … through 7).
   For each document, classify the student's use as:
     (a) CONTENT — the student accurately describes the document's content (counts toward Tier 1).
     (b) ARGUMENT — the student ties the document's content to a specific argument (counts toward Tier 2).
     (c) SOURCING — the student explains how or why the document's POV, purpose, historical
         situation, or audience is relevant to an argument (counts toward Sourcing).
   A single document can satisfy more than one of (a), (b), (c) independently.
   Do NOT stop scanning after the first qualifying document. Do NOT assume sourcing only exists
   where the student explicitly uses words like "purpose" or "audience" — paraphrased explanations
   of why the author wrote the document, who it was for, or what circumstances shaped it also count.
   Only AFTER this full enumeration may you decide whether each criterion's threshold is met.
6. COMPLEXITY MULTI-PATH: The Complexity criteria in the rubric list MULTIPLE alternative ways
   to earn the point (e.g., explaining nuance, multiple causes/effects, sustained corroboration
   or qualification, similarity AND difference, connections across periods/themes/regions,
   effective use of all 7 docs, etc.). Earning ANY ONE of those paths — demonstrated throughout
   the essay, not just in a single sentence — is sufficient. Do NOT require all paths.
7. TIER DEPENDENCY: For any criterion whose name contains "Tier 2", you may award the point ONLY
   if the corresponding "Tier 1" criterion was also earned (per the official rubric). If Tier 1
   was NOT earned, set Tier 2 points_earned = 0 and state the dependency in not_earned_reason.
8. LEQ EVIDENCE ENUMERATION (LEQ ONLY — applies to "Evidence: Specific Examples (Tier 1)" and
   "Evidence: Supports Argument (Tier 2)"): Before deciding either point, enumerate every
   concrete piece of historical evidence the student mentions (specific people, places, events,
   dates, policies, treaties, technologies, etc. — NOT vague generalizations like "trade increased").
   For each, classify whether (a) it is specific and accurate enough for Tier 1, and (b) it is
   explicitly tied to the essay's argument for Tier 2. Only AFTER this full enumeration may you
   decide each criterion.
9. BIAS NEUTRALITY: Do NOT award credit because an essay is long, well-written, uses
   sophisticated vocabulary, or sounds confident. Award credit ONLY when the rubric criteria are
   clearly met. Conversely, do NOT penalize a short essay or simple prose if the rubric criteria
   are met.
10. SELF-CONSISTENCY: Your fields must agree with points_earned. If points_earned == max_points:
    not_earned_reason MUST be "" and evidence MUST be a real citation (never "N/A"). If
    points_earned == 0: not_earned_reason MUST be filled with a specific diagnostic, and evidence
    MUST be either "N/A" or (for sourcing) the per-document attempt list. Never contradict
    yourself across these fields.

EVIDENCE RULES (the "evidence" field):
- If the point WAS earned: provide the specific sentence(s) or passage from the student's essay
  that directly satisfied the criterion. Prefer a direct quote (with quotation marks) over a
  paraphrase. If you must paraphrase, begin with "The student writes that…".
- If the point was NOT earned: set "evidence" to "N/A".
- Do NOT fabricate or embellish — only cite text that actually appears in the essay.
- For multi-part criteria (e.g., Evidence from Documents tiers), cite evidence from each
  document or piece of evidence that contributed to earning the point.
- For "Analysis and Reasoning: Sourcing" specifically, the evidence field MUST enumerate
  EVERY document the student references anywhere in the essay (Doc 1 through Doc 7). Do NOT
  pre-filter to documents you consider "real" sourcing attempts — list all referenced docs and
  classify each. Use exactly one of these three formats per line:
    "Doc N — QUALIFIES: '<quote/paraphrase>' — <how POV/purpose/situation/audience supports the argument>"
    "Doc N — PARTIAL: '<quote/paraphrase>' — <what's there and what's missing to fully qualify>"
    "Doc N — NO SOURCING: <one phrase noting how the doc was used (content only / not at all)>"
  After listing every referenced doc, end with a one-line tally:
    "Tally: X QUALIFIES, Y PARTIAL, Z NO SOURCING — rubric requires 2 QUALIFIES."
  This applies whether or not the point was earned.
- Keep each individual evidence snippet concise: one to three sentences maximum. If the relevant
  passage is long, quote the most diagnostic phrase and summarize the rest.

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
    * For sourcing, you MUST address EVERY document the student references anywhere in the
      essay — NOT just the documents you would call a "sourcing attempt" on first reading.
      A doc may have content use only, or may have a marginal sourcing gesture you initially
      dismissed; list it anyway. Use the same per-doc format and final tally line specified
      in the EVIDENCE RULES above (QUALIFIES / PARTIAL / NO SOURCING + Tally line). Only
      after listing every referenced doc may you draw the conclusion (e.g., "Only 1 doc fully
      qualifies; 2 are required for the point.").
- Limit to 2-4 sentences UNLESS the criterion is sourcing, in which case the per-document
  enumeration may extend the length as needed. Be specific, factual, and constructive — not discouraging.
"""

_DBQ_DOCS_SECTION_TEMPLATE = """SOURCE DOCUMENTS PROVIDED TO THE STUDENT:
{docs}
"""


# ---------------------------------------------------------------------------
# Answer revision
# ---------------------------------------------------------------------------

_REVISION_SYSTEM_PROMPT = (
    "You are an expert AP World History: Modern tutor. "
    "Your job is to produce the SMALLEST POSSIBLE patch to the student's answer so it earns every "
    "available point on the official AP rubric. "
    "Treat the student's original text as authoritative: copy it VERBATIM, word-for-word and "
    "sentence-for-sentence, wherever it already satisfies a rubric criterion. "
    "Only insert new sentences, replace specific phrases, or correct factual errors that block a "
    "rubric point. Never paraphrase, condense, polish, restructure, or 'improve' wording that "
    "already works — even if you could phrase it more elegantly. "
    "Preserve the student's voice, vocabulary, sentence order, and paragraph order. "
    "Do NOT correct grammar, spelling, or stylistic choices unless they actually obscure the "
    "historical content (the AP rubric does not penalize grammar). "
    "If the original already earns full marks, return it UNCHANGED. "
    "Output ONLY the revised answer text — no preamble, labels, diff markers, or explanation."
)

_REVISION_INSTRUCTIONS = """\
ESSAY TYPE: {category}

QUESTION:
{question}

{dbq_docs_section}STUDENT'S ORIGINAL ANSWER:
{answer}

OFFICIAL RUBRIC:
{rubric_text}
{diagnostic_section}
---
PROCESS (do this internally, then output only the final revised answer):
1. For EACH rubric criterion, decide whether the student's original answer already earns it.
2. For criteria that ARE earned: identify the supporting sentences and copy them into your output
   verbatim — character-for-character, including the student's original wording, vocabulary,
   punctuation, and any harmless errors. Do not change a single word, even to make them flow better.
3. For criteria that are NOT earned: write the smallest possible insertion or replacement that
   earns the point, placed in the most natural location within the student's existing structure.
   Prefer adding ONE sentence over a paragraph; prefer ONE clause over a sentence; prefer
   replacing a single phrase over a whole sentence — whichever is sufficient.

HARD CONSTRAINTS:
- Preserve the student's original sentences word-for-word wherever the rubric is already satisfied.
- Do NOT reorder sentences or paragraphs.
- Do NOT rephrase, condense, polish, or "tighten" wording that already earns its point.
- Do NOT change the student's voice, tone, vocabulary, or stylistic choices except where strictly
  required by a missing rubric point.
- Do NOT correct grammar, spelling, capitalization, or style — the AP rubric does not penalize
  these unless they obscure historical content.
- Do NOT add any new content (definitions, transitions, framing, conclusions, etc.) that is not
  required by a specific missing rubric criterion.
- If a paragraph already earns its rubric points, return that paragraph EXACTLY as written.
{format_hint}

Output ONLY the revised answer text — no commentary, no preamble, no diff markers.\
"""

_REVISION_FORMAT_HINTS: dict[str, str] = {
    "SAQ": (
        "Return the revised answer using exactly this structure (one block per sub-part):\n"
        "(a)\n[revised answer for part a]\n\n"
        "(b)\n[revised answer for part b]\n\n"
        "(c)\n[revised answer for part c]\n"
        "Within each sub-part, preserve the student's original answer verbatim and only patch "
        "what is strictly needed for the missing rubric point."
    ),
    "LEQ": (
        "Do NOT rewrite the essay from scratch. Keep every paragraph that already supports a "
        "rubric criterion EXACTLY as the student wrote it — no rephrasing, no reordering. "
        "Patch only what is missing: "
        "if the thesis sentence does not earn the thesis point, revise THAT ONE sentence; "
        "if a specific piece of evidence is absent, insert ONE sentence in the appropriate paragraph; "
        "if contextualization is missing, prepend (or append) ONE short paragraph; "
        "if reasoning/complexity is missing, add a single sentence in the most natural location. "
        "Never re-engineer paragraphs that already earn their points."
    ),
    "DBQ": (
        "Do NOT rewrite the essay from scratch. Keep every paragraph that already supports a "
        "rubric criterion EXACTLY as the student wrote it — no rephrasing, no reordering. "
        "Patch only what is missing: "
        "if the thesis sentence does not earn the thesis point, revise THAT ONE sentence; "
        "if document analysis or sourcing (purpose / audience / POV / situation) is absent for a "
        "specific document, insert ONE targeted sentence inline next to that document's existing usage; "
        "if contextualization or outside evidence is missing, add ONE short paragraph in the natural location. "
        "Never re-engineer paragraphs or document discussions that already earn their points."
    ),
}


# Builds the optional "DIAGNOSTIC FROM PRIOR GRADING" block that tells the revision model
# exactly which criteria are already earned (and must be preserved verbatim) and which ones
# need patches. Returns "" when no grade_result is supplied.
def _build_diagnostic_section(grade_result: Optional["GradeResult"]) -> str:
    """
    Formats per-criterion diagnostics from a prior grading pass into a prompt section
    the revision model can use to focus its edits.

    The returned string is either empty (when no grade_result is provided) or a
    multi-line block listing earned vs. not-earned criteria with their evidence
    and not-earned reasons.
    """
    if grade_result is None:
        return ""

    earned: list[CriterionResult] = [
        cr for cr in grade_result.criteria_results if cr.points_earned > 0
    ]
    not_earned: list[CriterionResult] = [
        cr for cr in grade_result.criteria_results if cr.points_earned < cr.max_points
    ]

    lines: list[str] = [
        "",
        "---",
        "DIAGNOSTIC FROM PRIOR GRADING:",
        f"A prior grading pass scored this answer {grade_result.total_earned}/"
        f"{grade_result.total_possible}. Use the per-criterion findings below to guide your edits.",
        "",
    ]

    if earned:
        lines.append(
            "CRITERIA ALREADY EARNED — the supporting text below MUST be preserved VERBATIM "
            "in your revised answer. Do not touch these sentences:"
        )
        for cr in earned:
            evidence = (cr.evidence or "").strip()
            if not evidence or evidence.upper() == "N/A":
                evidence = "(no quote captured — preserve the relevant passage as written)"
            lines.append(f"- {cr.name} ({cr.points_earned}/{cr.max_points})")
            lines.append(f"  Supporting text: {evidence}")
        lines.append("")

    if not_earned:
        lines.append(
            "CRITERIA NOT YET EARNED — apply the SMALLEST POSSIBLE patch to address each one. "
            "Do not patch anything else:"
        )
        for cr in not_earned:
            reason = (cr.not_earned_reason or "").strip() or "(no reason captured)"
            lines.append(
                f"- {cr.name} ({cr.points_earned}/{cr.max_points} — needs "
                f"{cr.max_points - cr.points_earned} more)"
            )
            lines.append(f"  What was missing: {reason}")
            if cr.suggestion and cr.suggestion.strip():
                lines.append(f"  Suggested fix: {cr.suggestion.strip()}")
        lines.append("")
    else:
        lines.append(
            "ALL CRITERIA ARE ALREADY EARNED — return the student's original answer UNCHANGED."
        )
        lines.append("")

    lines.append(
        "Make patches ONLY for the 'NOT YET EARNED' criteria above. Do NOT alter any sentence "
        "that supports an 'ALREADY EARNED' criterion."
    )
    return "\n".join(lines)


# Produces a revised student answer that earns full marks on the AP rubric.
# When grade_result is supplied, the prompt includes a per-criterion diagnostic so the
# model patches only the unearned criteria and leaves earned passages verbatim — this
# keeps the revised version much closer to the student's original.
def revise_answer(
    client: OpenAI,
    category: str,
    question: str,
    answer: str,
    model: str,
    dbq_docs: Optional[str] = None,
    grade_result: Optional["GradeResult"] = None,
) -> str:
    """
    Calls the OpenAI API to produce a revised version of the student's answer
    that earns full marks on the official AP rubric.

    If grade_result is provided (e.g. from a prior call to grade_essay), the
    revision prompt is augmented with a per-criterion diagnostic listing which
    rubric points are already earned (and must be preserved verbatim) and which
    are missing (where the model must patch). Supplying grade_result typically
    produces a revision much closer to the student's original answer because
    the model no longer has to re-derive the rubric assessment itself.

    Returns the revised answer as plain text.
    """
    rubric, max_score = get_rubric(category)
    rubric_text = format_rubric_for_prompt(rubric, max_score)
    dbq_docs_section = ""
    if dbq_docs and dbq_docs.strip():
        dbq_docs_section = _DBQ_DOCS_SECTION_TEMPLATE.format(docs=dbq_docs.strip()) + "\n"
    diagnostic_section = _build_diagnostic_section(grade_result)
    prompt = _REVISION_INSTRUCTIONS.format(
        category=category,
        question=question.strip(),
        dbq_docs_section=dbq_docs_section,
        answer=answer.strip(),
        rubric_text=rubric_text,
        diagnostic_section=diagnostic_section,
        format_hint=_REVISION_FORMAT_HINTS.get(category, ""),
    )
    messages = [
        {"role": "system", "content": _REVISION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return _call_with_retry(client, messages, model)


_EXPLAIN_CHANGES_SYSTEM_PROMPT = (
    "You are an expert AP World History: Modern tutor explaining edits to a student. "
    "Be concise, specific, and encouraging. Focus on the rubric criteria that were addressed."
)

_EXPLAIN_CHANGES_INSTRUCTIONS = """\
ESSAY TYPE: {category}

OFFICIAL RUBRIC:
{rubric_text}

STUDENT'S ORIGINAL ANSWER:
{original}

REVISED ANSWER:
{revised}

---
Compare the two answers above. List ONLY the changes that were actually made.
For each change:
- Quote or paraphrase the specific part that changed (keep quotes short).
- State which rubric criterion it addresses.
- Explain in one sentence why the change earns that point.

If the original already earned a point that was kept unchanged, do NOT mention it.
Format as a numbered list. Be concise — one bullet per change.\
"""


# Produces a bullet-list explanation of what changed between original and revised answer.
def explain_changes(
    client: OpenAI,
    category: str,
    original: str,
    revised: str,
    model: str,
) -> str:
    """
    Calls the OpenAI API to produce a numbered list of the specific changes
    made between original and revised, and the rubric reason for each change.
    Returns plain text suitable for display.
    """
    rubric, max_score = get_rubric(category)
    rubric_text = format_rubric_for_prompt(rubric, max_score)
    prompt = _EXPLAIN_CHANGES_INSTRUCTIONS.format(
        category=category,
        rubric_text=rubric_text,
        original=original.strip(),
        revised=revised.strip(),
    )
    messages = [
        {"role": "system", "content": _EXPLAIN_CHANGES_SYSTEM_PROMPT},
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
                max_completion_tokens=40000,
            )
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", "unknown")
            usage = getattr(response, "usage", None)
            # Surface token usage to the terminal so empty-content failures are diagnosable.
            if usage is not None:
                completion_tokens = getattr(usage, "completion_tokens", "?")
                prompt_tokens = getattr(usage, "prompt_tokens", "?")
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
                print(
                    f"  Token usage — prompt: {prompt_tokens}, completion: {completion_tokens}"
                    + (f", reasoning: {reasoning_tokens}" if reasoning_tokens is not None else "")
                    + f" (finish_reason={finish_reason})"
                )
            if not content:
                # Build an actionable error message that names the actual cause when the
                # output budget was consumed by reasoning tokens (the common case for
                # reasoning models with reasoning_effort="high").
                hint = ""
                if finish_reason == "length":
                    hint = (
                        " The model hit its max_completion_tokens limit before producing any "
                        "visible output — reasoning tokens consumed the entire budget. "
                        "Increase max_completion_tokens, lower reasoning_effort, or shorten the input."
                    )
                else:
                    hint = (
                        " The API returned no content (finish_reason="
                        f"{finish_reason!r}). Try again or switch models."
                    )
                raise ValueError("The model returned an empty response." + hint)
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

    # Defense-in-depth: enforce the rubric's Tier 1 → Tier 2 dependency in code.
    # If the model awarded Tier 2 without awarding Tier 1, zero out Tier 2.
    # (The prompt also instructs the model to do this, but reasoning models occasionally slip.)
    total_earned = _enforce_tier_dependency(criteria_results)

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


# Mapping of Tier 2 criterion names to the Tier 1 criterion they depend on.
# These names must match the canonical names defined in rubrics.py exactly.
# Tier 1 / Tier 2 criteria have different middle parts (e.g. "Content" vs
# "Supports Argument"), so a string-replace on "Tier 2" → "Tier 1" is not enough.
_TIER2_DEPENDS_ON_TIER1: dict[str, str] = {
    "Evidence from Documents: Supports Argument (Tier 2)":
        "Evidence from Documents: Content (Tier 1)",
    "Evidence: Supports Argument (Tier 2)":
        "Evidence: Specific Examples (Tier 1)",
}


# Enforces the official rubric rule that Tier 2 evidence/document points cannot be awarded
# unless the corresponding Tier 1 point was also earned. Mutates the criteria list in place
# and returns the recomputed total earned.
def _enforce_tier_dependency(criteria_results: list[CriterionResult]) -> int:
    """
    Zeros out any Tier 2 criterion that was awarded points while its paired
    Tier 1 criterion (looked up via _TIER2_DEPENDS_ON_TIER1) earned 0.

    Annotates the zeroed criterion's not_earned_reason so the report explains
    why the point was withheld, and prints a warning to the terminal so silent
    score changes are visible. Returns the total points earned after enforcement.
    """
    by_name: dict[str, CriterionResult] = {cr.name: cr for cr in criteria_results}
    for cr in criteria_results:
        if cr.points_earned <= 0:
            continue
        tier1_name = _TIER2_DEPENDS_ON_TIER1.get(cr.name)
        if tier1_name is None:
            continue
        tier1 = by_name.get(tier1_name)
        if tier1 is None or tier1.points_earned > 0:
            continue
        print(
            f"  WARNING: '{cr.name}' was awarded {cr.points_earned} pt(s) but "
            f"'{tier1_name}' was not earned. Zeroing Tier 2 per rubric requirement."
        )
        cr.points_earned = 0
        dependency_note = (
            f"This Tier 2 point cannot be awarded without first earning '{tier1_name}', "
            f"per the official AP rubric."
        )
        cr.not_earned_reason = (
            dependency_note if not cr.not_earned_reason
            else f"{dependency_note}\n\nOriginal model rationale: {cr.not_earned_reason}"
        )
    return sum(cr.points_earned for cr in criteria_results)
