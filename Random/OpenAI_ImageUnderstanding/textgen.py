
import base64
from io import BytesIO
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
# from IPython.display import display, Image
from tqdm.notebook import tqdm
from dotenv import load_dotenv

cache_dir = Path('.local_cache')
cache_dir.mkdir(parents=True, exist_ok=True)



# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_food_delivery_review(sentiment: str = 'positive') -> str:
    """
    Generate a synthetic food delivery review with the specified sentiment.
    
    Args:
        sentiment: An adjective such as 'positive' or 'negative'.
    
    Returns:
        Generated review text
    """
    prompt = "Write a very concise, realistic customer review for a recent food delivery."
    prompt += f" The review should reflect a {sentiment} experience."
    
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "minimal"},
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text


review = generate_food_delivery_review()
print(review)

