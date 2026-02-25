from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import PyPDF2
import os
import pandas as pd
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
dir_pdfs = "pdfs"
pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]

def upload_single_pdf(file_path:str, vector_store_id:str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, "rb"), purpose="assistants")
        attach_response = client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_response.id)
        return {'file':file_name, "status": "success"}
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return {'file':file_name, "status": "failed","error":str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
    stats = {"total_files": len(pdf_files), "successful_uploads": 0, "failed_uploads": 0, "errors": []}
    
    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdf_files)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats

def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
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

store_name = "shh_pdf_store"
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


response = client.responses.create(
    input= query,
    model="gpt-4o-mini",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store_details['id']],
    }]
)

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