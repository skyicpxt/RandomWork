# AP World History: Modern — Grading Tool

Python tool that grades **DBQ**, **LEQ**, and **SAQ** responses using rubrics aligned with the [AP World History: Modern exam description](https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment). Each essay is sent to the OpenAI API and results include a score breakdown, supporting evidence, missed-point explanations, and suggestions.

Two interfaces are available:

| Interface | How to run | Output |
|-----------|-----------|--------|
| **Web app** (recommended) | `streamlit run streamlit_app.py` | Interactive browser UI |
| **CLI** | `python main.py` | `grading_report.txt` |

---

## Setup

1. Python 3.10+ required.
2. Install dependencies (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the **parent** folder (`Random/.env`) containing your OpenAI API key:

   ```
   OPENAI_API_KEY="sk-..."
   ```

   Alternatively, set it as an environment variable. The Streamlit Cloud deployment uses [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) instead.

---

## Web App

```bash
streamlit run streamlit_app.py
```

Opens in your browser at `http://localhost:8501`. Features:

- **Grade Essay** — grades a Q&A against the official rubric and displays a full report (score breakdown, evidence, suggestions). Supports single questions and multi-question SAQ input.
- **✏️ Revise Answer** — rewrites the student's answer so it earns full rubric points. Output is in the same format as the input, with `RA:` in place of `A:`, and can be copied directly.
- File upload for Q&A input (`.txt`, PDF, or image).
- DBQ source document upload (PDF or image, one file per document).
- Essay-type mismatch detection with confirmation prompt before grading proceeds.
- "Copy Report Text" expander for easy clipboard access to the full report.

---

## CLI

From the `APWorldHistGrading` folder:

```bash
# Grade the three default single-format sample files (DBQ + LEQ + SAQ)
python main.py

# One preset (maps to the matching .txt sample file)
python main.py --category DBQ
python main.py --category SAQ_multi

# Custom input file
python main.py --qa my_answers.txt

# Custom report output path
python main.py --output my_report.txt
```

### `--category` presets

| Value | File |
|-------|------|
| `DBQ` | `DBQQandA.txt` |
| `DBQ_multi` | `DBQQandA_multi.txt` |
| `LEQ` | `LEQQandA.txt` |
| `LEQ_multi` | `LEQQandA_multi.txt` |
| `SAQ` | `SAQQandA.txt` |
| `SAQ_multi` | `SAQQandA_multi.txt` |

Omit `--category` to run `DBQ` + `LEQ` + `SAQ` (single-format defaults only).

Results are written to **`grading_report.txt`** (overwritten each run).

---

## Input File Format

- Blocks separated by a line `---`.
- Each block starts with `CATEGORY: DBQ` | `LEQ` | `SAQ`, followed by `Q:` and `A:` (multi-line allowed).
- **DBQ:** place source document excerpts in a `DOCS:` section between `Q:` and `A:`.
- **Multi-question:** prefix each Q&A pair with a line like `Question1`, `Question 2`, or `Question 1:` (on its own line, with optional colon/space).
- **SAQ sub-parts:** label each part `(a)`, `(b)`, `(c)` (or `A)`, `B)`, `C)`) on its own line above the `Q:` / `A:` for that part.

Invalid format raises a clear error and exits (CLI) or shows an error message (web app).

### Example — single SAQ with sub-parts

```
CATEGORY: SAQ

(a)
Q: Describe one cause of the Silk Road's decline after 1450.
A: The rise of maritime trade routes...

(b)
Q: Explain one effect on China.
A: China experienced a shift in trade partners...

(c)
Q: Evaluate the extent to which the decline changed the global economy.
A: The decline significantly altered global trade patterns...
```

### Example — multi-question SAQ

```
CATEGORY: SAQ

Question 1:
(a)
Q: ...
A: ...

(b)
Q: ...
A: ...

Question 2:
(a)
Q: ...
A: ...
```

---

## Project Layout

| File | Role |
|------|------|
| `streamlit_app.py` | Web app (Grade Essay + Revise Answer UI) |
| `main.py` | CLI entry point, report formatting |
| `grader.py` | OpenAI API calls, `GradeResult` dataclass, answer revision |
| `qa_parser.py` | Q&A text parsing (single and multi-question formats) |
| `rubrics.py` | Official DBQ / LEQ / SAQ rubric criteria |
| `test_ap_world_grading.py` | Unit tests (parsing, grading output, revision helpers) |
| `DBQQandA.txt`, `LEQQandA.txt`, `SAQQandA.txt` | Single-question sample files |
| `DBQQandA_multi.txt`, `LEQQandA_multi.txt`, `SAQQandA_multi.txt` | Multi-question sample files |

---

## Tests

Run after any code change:

```bash
python -m unittest test_ap_world_grading -v
```

All tests should report **`OK`**. The test suite stubs `openai`, `streamlit`, and `dotenv` so all tests run without a live API connection or a running Streamlit server.

---

## Disclaimer

This is a **study / feedback** tool, not an official College Board product. Scores are model-generated approximations; use official rubrics and teacher feedback for high-stakes decisions.
