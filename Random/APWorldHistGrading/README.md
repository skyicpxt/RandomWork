# AP World History: Modern — Grading Agent

Python tool that grades sample **DBQ**, **LEQ**, and **SAQ** responses using rubrics aligned with the [AP World History: Modern exam description](https://apstudents.collegeboard.org/courses/ap-world-history-modern/assessment). Each essay is sent to the OpenAI API; results are written to **`grading_report.txt`** (overwritten each run) with score breakdown, evidence, missed points, and suggestions.

## Setup

1. Python 3.10+ recommended.
2. Install dependencies (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

3. Set **`OPENAI_API_KEY`** in `Random/.env` (parent of this folder), or in your environment.

## Run

From **`APWorldHistGrading`**:

```bash
# Grade all three single-format sample files (DBQ + LEQ + SAQ)
python main.py

# One preset (maps to the matching .txt file)
python main.py --category DBQ
python main.py --category DBQ_multi
python main.py --category SAQ_multi

# Custom input file; optional filter by essay type inside the file
python main.py --qa my_answers.txt --category LEQ

# Custom report path
python main.py --output my_report.txt
```

### `--category` presets

| Value | File |
|--------|------|
| `DBQ` | `DBQQandA.txt` |
| `DBQ_multi` | `DBQQandA_multi.txt` |
| `LEQ` | `LEQQandA.txt` |
| `LEQ_multi` | `LEQQandA_multi.txt` |
| `SAQ` | `SAQQandA.txt` |
| `SAQ_multi` | `SAQQandA_multi.txt` |

Omit `--category` to run **`DBQ` + `LEQ` + `SAQ`** (single-format defaults only).

With **`--qa`**, `DBQ_multi` / `LEQ_multi` / `SAQ_multi` still filter parsed entries by essay type (`DBQ`, `LEQ`, `SAQ`).

## Input file format

- Blocks separated by a line **`---`**.
- Each block: **`CATEGORY: DBQ`** | **`LEQ`** | **`SAQ`**, then **`Q:`** and **`A:`** (multi-line allowed).
- **DBQ (single file):** optional **`DOCS:`** between **`Q:`** and **`A:`** for source excerpts.
- **DBQ (multi):** for each **`Question1`**, **`Question2`**, … put **`DOCS:`** *after* that label and *before* **`Q:`**. Use a **full set of 7 documents** per DBQ question; document sets are **not** shared across questions. Optional leading **`DOCS:`** before the first **`QuestionN`** only backfills the **first** question if it has no **`DOCS:`** of its own.
- **Multi questions:** a line like **`Question1`**, **`Question 2`** on its own line above each **`Q:`** / **`A:`** pair.

Invalid format raises a clear error and exits.

## Project layout

| File | Role |
|------|------|
| `main.py` | CLI, parsing, report output |
| `grader.py` | OpenAI call + `GradeResult` |
| `rubrics.py` | DBQ / LEQ / SAQ criteria |
| `test_ap_world_grading.py` | Unit tests |
| `DBQQandA.txt`, `LEQQandA.txt`, `SAQQandA.txt` | Single-Q&A samples |
| `*_multi.txt` | Multi-question samples |

## Tests (run after changes)

After editing code in this folder, run:

```bash
cd APWorldHistGrading
python -m unittest test_ap_world_grading -v
```

All tests should report **`OK`**. The test module stubs `openai` if the package is missing so parsing/report tests still run.

## Disclaimer

This is a **study / feedback** tool, not an official College Board product. Scores are model-generated approximations; use official rubrics and teacher feedback for high-stakes decisions.
