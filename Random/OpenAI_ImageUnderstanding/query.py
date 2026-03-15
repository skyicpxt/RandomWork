# https://developers.openai.com/cookbook/examples/multimodal/image_understanding_with_rag/


import base64
from io import BytesIO
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
# from IPython.display import display, Image
from tqdm import tqdm
from dotenv import load_dotenv

# Paths relative to this script so it works regardless of cwd
_script_dir = Path(__file__).resolve().parent
cache_dir = _script_dir / '.local_cache'
cache_dir.mkdir(parents=True, exist_ok=True)



# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(_script_dir.parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text_vector_store_id = "vs_69b395ce43248191ae22c39461f2920f"
text_image_vector_store_id = "vs_69b395ced724819192c082658a5fa58d"

# Query the vector store for spaghetti reviews in July
query = "Were there any reviews for pizza, and if so, was the pizza burnt?"
print(f"🔍 Query: {query}\n")

# Execute the search with filtering
response = client.responses.create(
    model="gpt-5",
    input=query,
    tools=[{
        "type": "file_search",
        "vector_store_ids": [text_vector_store_id, text_image_vector_store_id],
        "filters": {
            "type": "eq",
            "key": "month",
            "value": "july"
        }
    }]
)

# Display the results
print("📝 Response:")
print("-" * 40)
print(response.output_text)