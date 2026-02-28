# https://developers.openai.com/cookbook/examples/file_search_responses
#
# RATE LIMIT (429): OpenAI limits requests per minute and tokens per minute per account.
# If you see RateLimitError/429: (1) we retry with backoff; (2) uploads use fewer
# parallel workers; (3) add short delays between sequential API calls. To raise limits
# see https://platform.openai.com/account/limits

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import PyPDF2
import os
import pandas as pd
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Resolve pdfs dir relative to this script so it works from any cwd
_dir_pdfs = Path(__file__).resolve().parent / "pdfs"
dir_pdfs = _dir_pdfs
pdf_files = [str(_dir_pdfs / f) for f in os.listdir(_dir_pdfs)]


def _call_with_retry(func, *args, max_retries: int = 4, **kwargs):
    """Call func(*args, **kwargs); on 429 (rate limit) wait and retry with exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            is_rate_limit = (
                getattr(e, "status_code", None) == 429
                or "rate" in str(type(e).__name__).lower()
                or "429" in str(e)
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = (2 ** attempt) + 1  # 2, 3, 5, 9 sec
                print(f"Rate limit hit, waiting {wait}s before retry ({attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_err


def upload_single_pdf(file_path: str, vector_store_id: str):
    """Upload one PDF to OpenAI Files and attach it to the given vector store; uses retry on rate limit."""
    file_name = os.path.basename(file_path)
    try:
        def _upload_file():
            with open(file_path, "rb") as f:
                return client.files.create(file=f, purpose="assistants")
        file_response = _call_with_retry(_upload_file)
        _call_with_retry(
            lambda: client.vector_stores.files.create(
                vector_store_id=vector_store_id, file_id=file_response.id
            )
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_list = [str(dir_pdfs / f) for f in os.listdir(dir_pdfs)]
    stats = {"total_files": len(pdf_list), "successful_uploads": 0, "failed_uploads": 0, "errors": []}
    print(f"{len(pdf_list)} PDF files to process. Uploading (low concurrency to avoid rate limit)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(upload_single_pdf, fp, vector_store_id): fp for fp in pdf_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_list)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

def create_vector_store(store_name: str) -> dict:
    """Create an empty vector store; uses retry on rate limit."""
    try:
        vector_store = _call_with_retry(lambda: client.vector_stores.create(name=store_name))
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def wait_for_vector_store_ready(vector_store_id: str, expected_files: int, timeout_sec: int = 120, poll_interval_sec: int = 5):
    """Poll vector store until file_counts.completed >= expected_files or timeout."""
    import time
    start = time.monotonic()
    while (time.monotonic() - start) < timeout_sec:
        vs = client.vector_stores.retrieve(vector_store_id)
        completed = getattr(vs.file_counts, "completed", 0) or 0
        if completed >= expected_files:
            print(f"Vector store ready: {completed} file(s) processed.")
            return
        print(f"Waiting for vector store to process files ({completed}/{expected_files})...")
        time.sleep(poll_interval_sec)
    print("Timeout waiting for vector store processing.")

store_name = "skyi_pdf_store"
vector_store_details = create_vector_store(store_name)
upload_stats = upload_pdf_files_to_vector_store(vector_store_details["id"])
print(f"Upload finished: {upload_stats['successful_uploads']} successful, {upload_stats['failed_uploads']} failed.")
expected = upload_stats["total_files"]
if expected > 0:
    wait_for_vector_store_ready(vector_store_details["id"], expected)

query = "How to train a large language model?"
# # directly search the vector store
# search_results = client.vector_stores.search(
#     vector_store_id=vector_store_details['id'],
#     query=query
# )

# print(f"Search returned {len(search_results.data)} result(s) for query: {query!r}")
# for result in search_results.data:
#     if result.content and len(result.content) > 0:
#         text = result.content[0].text
#         print(f"{len(text)} characters of content from {result.filename} (score: {result.score})")
#     else:
#         print(f"(no content) from {result.filename} (score: {result.score})")

# # use as a tool (with retry on rate limit)
response = _call_with_retry(
    lambda: client.responses.create(
        input=query,
        model="gpt-4o-mini",
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_details["id"]]}],
    )
)

# --- Dump response structure as fully as possible with clear labels ---
#
# RESPONSE STRUCTURE EXPLAINED (using a typical file_search example)
# ==================================================================
#
# 1) RESPONSE TOP-LEVEL ATTRIBUTES
#    ------------------------------
#    These describe the whole API response object:
#
#    response.id        - Unique ID for this response (e.g. "resp_abc123"). Use for retrieval/logging.
#    response.model    - Model that generated the response (e.g. "gpt-4o-mini").
#    response.object   - API object type (e.g. "response").
#    response.created  - Unix timestamp when the response was created.
#    response.status   - Lifecycle status (e.g. "completed", "in_progress", "incomplete").
#    response.usage    - Token usage: .input_tokens, .output_tokens (and optionally .total_tokens).
#                        Use these for cost estimation and rate limiting.
#
# 2) RESPONSE.OUTPUT (list of output items)
#    --------------------------------------
#    The model's reply is built from a sequence of "output items". Order matters.
#    Typical order for a file_search request: [file_search_call, message].
#
#    Output item [0] – type = 'file_search_call'
#    -------------------------------------------
#    Represents the file_search tool call: the API ran a search over your vector store
#    and attached the retrieved chunks to the context before the model replied.
#
#      .id       - Unique ID for this tool call (e.g. "call_xyz").
#      .queries  - List of search query strings the model/API used to search your files.
#                  The first is usually your input; the API may add paraphrases/expansions.
#      .status   - e.g. "completed" when the search finished successfully.
#      .results  - (if present) Number of retrieved chunks; details may be in the object.
#
#    Output item [1] – type = 'message'
#    ----------------------------------
#    The assistant's reply: one or more content blocks (usually one text block).
#
#      .id     - Unique ID for this message.
#      .role   - Always "assistant" for model output.
#      .status - e.g. "completed" when the message is fully generated.
#      .phase  - Optional: "commentary" (intermediate) or "final_answer".
#
#      .content - List of content blocks that make up the message.
#      Why a list: one message can have multiple parts (e.g. several text segments,
#      or text + image, or text + refusal). Each part is one block. For a simple
#      one-shot answer you typically get a single block.
#      Block types:
#        - output_text: a segment of reply text (most common).
#        - refusal: model refused to answer; .refusal explains why.
#        - (others possible: image, file, etc. for mixed content.)
#
#        block[0] .type = 'output_text'
#        -------------------------------
#        A text segment of the reply.
#
#          .text        - The actual reply text (in the dump we show first 200 chars).
#          .annotations - Citations: file_citation (file_id, filename), url_citation (url, title),
#                         etc. These link parts of .text to the sources (e.g. your PDFs).
#          .logprobs    - (Optional) Token-level log probabilities if requested via include.
#
#        block[0] .type = 'refusal' (only if the model refused to answer)
#        -----------------------------------------------------------------
#          .refusal - Explanation of why the model refused.
#
#    Other possible output item types (not shown in every response):
#    - reasoning   - Chain-of-thought / reasoning steps (e.g. for o1/o3).
#    - web_search_call - Web search tool call (action, status, sources).
#    - function_call  - Custom function call (name, arguments, call_id).
#
def _safe_str(x):
    """Return a safe string for printing; avoid repr() errors on API objects."""
    try:
        return repr(x)
    except Exception:
        return f"<{type(x).__name__}>"

def _dump_response(response):
    """Print the response object structure and all available attributes with labels."""
    try:
        print("\n" + "=" * 60, flush=True)
        print("RESPONSE TOP-LEVEL ATTRIBUTES")
        print("=" * 60)
        for attr in ("id", "model", "object", "created", "status", "usage"):
            if hasattr(response, attr):
                val = getattr(response, attr)
                print(f"  response.{attr} = {_safe_str(val)}")

        if hasattr(response, "output") and response.output is not None:
            print("\n" + "=" * 60)
            print("RESPONSE.OUTPUT (list of output items)")
            print("=" * 60)
            for i, item in enumerate(response.output):
                item_type = getattr(item, "type", type(item).__name__)
                print(f"\n  --- Output item [{i}] type = {_safe_str(item_type)} ---")
                for attr in ("id", "role", "status", "phase"):
                    if hasattr(item, attr):
                        val = getattr(item, attr)
                        print(f"    .{attr} = {_safe_str(val)}")

                if hasattr(item, "content") and item.content is not None:
                    print(f"    .content (list of {len(item.content)} block(s)):")
                    for j, block in enumerate(item.content):
                        block_type = getattr(block, "type", type(block).__name__)
                        print(f"      block[{j}] .type = {_safe_str(block_type)}")
                        if hasattr(block, "text"):
                            text = block.text
                            preview = (text[:200] + "...") if text and len(text) > 200 else (text or "")
                            print(f"      block[{j}] .text = {_safe_str(preview)}")
                        if hasattr(block, "refusal"):
                            print(f"      block[{j}] .refusal = {_safe_str(block.refusal)}")
                        if hasattr(block, "annotations") and block.annotations:
                            print(f"      block[{j}] .annotations = {_safe_str(block.annotations)}")
                            for a in block.annotations:
                                a_type = getattr(a, "type", None)
                                if a_type == "file_citation":
                                    print(f"        citation: file_id={getattr(a,'file_id',None)} filename={getattr(a,'filename',None)}")
                                elif a_type == "url_citation":
                                    print(f"        url_citation: url={getattr(a,'url',None)} title={getattr(a,'title',None)}")
                        if hasattr(block, "logprobs") and block.logprobs is not None:
                            print(f"      block[{j}] .logprobs = (present, len={len(block.logprobs)})")

                if item_type == "file_search_call":
                    # id and status already printed in common attributes above
                    if hasattr(item, "queries"):
                        print(f"    .queries = {_safe_str(getattr(item, 'queries'))}")
                    if hasattr(item, "results") and item.results:
                        print(f"    .results = {len(item.results)} result(s)")

                if item_type == "web_search_call":
                    # status already printed in common attributes above
                    if hasattr(item, "action"):
                        print(f"    .action = {_safe_str(item.action)}")

                if item_type == "function_call":
                    for attr in ("id", "call_id", "name", "arguments"):
                        if hasattr(item, attr):
                            print(f"    .{attr} = {_safe_str(getattr(item, attr))}")

                if item_type == "reasoning":
                    if hasattr(item, "summary") and item.summary:
                        print(f"    .summary = (list of {len(item.summary)} part(s))")
                    if hasattr(item, "content") and item.content:
                        print(f"    .content = (reasoning content, len={len(item.content)})")

        print("\n" + "=" * 60)
        print("END RESPONSE DUMP")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[Dump failed: {e}]", flush=True)
        import traceback
        traceback.print_exc()

print(f"Dumping response for this request (query you sent): {query!r}", flush=True)
_dump_response(response)

# Find the message output (has .content); output order can be [tool_call, message] or vary
message_output = None
for item in response.output:
    if hasattr(item, "content") and item.content:
        message_output = item
        break

if message_output is None:
    print("No message content in response.")
else:
    content_block = message_output.content[0]
    annotations = getattr(content_block, "annotations", None) or []
    retrieved_files = {a.filename for a in annotations if hasattr(a, "filename")}
    print(f"Files used: {retrieved_files}")
    print("Response:")
    print(content_block.text)

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with open(pdf_path, "rb") as f:
#             reader = PyPDF2.PdfReader(f)
#             for page in reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#     except Exception as e:
#         print(f"Error reading {pdf_path}: {e}")
#     return text

# def generate_questions(pdf_path, max_input_chars: int = 100_000):
#     """Generate a question answerable only from the given PDF; truncates long docs to fit context."""
#     text = extract_text_from_pdf(pdf_path)
#     if len(text) > max_input_chars:
#         text = text[:max_input_chars] + "\n\n[Document truncated for length.]"

#     prompt = (
#         "Can you generate a question that can only be answered from this document?:\n"
#         f"{text}\n\n"
#     )

#     response = _call_with_retry(
#         lambda: client.responses.create(input=prompt, model="gpt-4o")
#     )
#     question = response.output[0].content[0].text
#     return question


# # Generate questions for each PDF and store in a dictionary (delay between calls to avoid rate limit)
# questions_dict = {}
# for idx, pdf_path in enumerate(pdf_files):
#     if idx > 0:
#         time.sleep(2)
#     questions_dict[os.path.basename(pdf_path)] = generate_questions(pdf_path)
# # Print the questions dictionary
# print(questions_dict)

# rows = []
# for filename, query in questions_dict.items():
#     rows.append({"query": query, "_id": filename.replace(".pdf", "")})

# # Metrics evaluation parameters
# k = 5
# total_queries = len(rows)
# correct_retrievals_at_k = 0
# reciprocal_ranks = []
# average_precisions = []

# def process_query(row):
#     query = row["query"]
#     expected_filename = row["_id"] + ".pdf"
#     # Call file_search via Responses API (with retry on rate limit)
#     response = _call_with_retry(
#         lambda: client.responses.create(
#             input=query,
#             model="gpt-4o-mini",
#             tools=[{
#                 "type": "file_search",
#                 "vector_store_ids": [vector_store_details["id"]],
#                 "max_num_results": k,
#             }],
#             tool_choice="required",
#         )
#     )
#     # Extract annotations from the response
#     annotations = None
#     if hasattr(response.output[1], 'content') and response.output[1].content:
#         annotations = response.output[1].content[0].annotations
#     elif hasattr(response.output[1], 'annotations'):
#         annotations = response.output[1].annotations

#     if annotations is None:
#         print(f"No annotations for query: {query}")
#         return False, 0, 0

#     # Get top-k retrieved filenames
#     retrieved_files = [result.filename for result in annotations[:k]]
#     if expected_filename in retrieved_files:
#         rank = retrieved_files.index(expected_filename) + 1
#         rr = 1 / rank
#         correct = True
#     else:
#         rr = 0
#         correct = False

#     # Calculate Average Precision
#     precisions = []
#     num_relevant = 0
#     for i, fname in enumerate(retrieved_files):
#         if fname == expected_filename:
#             num_relevant += 1
#             precisions.append(num_relevant / (i + 1))
#     avg_precision = sum(precisions) / len(precisions) if precisions else 0
    
#     if expected_filename not in retrieved_files:
#         print("Expected file NOT found in the retrieved files!")
        
#     if retrieved_files and retrieved_files[0] != expected_filename:
#         print(f"Query: {query}")
#         print(f"Expected file: {expected_filename}")
#         print(f"First retrieved file: {retrieved_files[0]}")
#         print(f"Retrieved files: {retrieved_files}")
#         print("-" * 50)
    
    
#     return correct, rr, avg_precision


# print("before process_query")
# print(process_query(rows[0]))