import base64
from io import BytesIO
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.notebook import tqdm
from dotenv import load_dotenv

# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

cache_dir = Path(__file__).resolve().parent / ".local_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    cache_dir_path: Path | None = None,
) -> Path:
    """Generate an image from prompt via OpenAI; use cache if already generated. Returns path to PNG file."""
    cache_dir_path = cache_dir_path or cache_dir
    cache_path = cache_dir_path / f"{hash(prompt)}.png"
    if not cache_path.exists():
        response = client.images.generate(model=model, prompt=prompt, size=size)
        with open(cache_path, "wb") as f:
            f.write(base64.b64decode(response.data[0].b64_json))
        print(f"Generated and cached: {cache_path}")
    else:
        print(f"Loading from cache: {cache_path}")
    return cache_path


def show_image(path: str | Path) -> None:
    """Display the image at path: in notebook uses IPython; otherwise matplotlib window or OS default viewer."""
    path = Path(path).resolve()
    if not path.exists():
        print(f"Image not found: {path}")
        return
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display, Image
            display(Image(filename=str(path)))
            return
    except (ImportError, Exception):
        pass
    # Run as script: show in matplotlib window (works on Windows/terminal)
    img = plt.imread(str(path))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Generated image")
    plt.show()


if __name__ == "__main__":
    prompt = (
        "Gourmet pasta neatly plated with garnish and sides on a white ceramic plate, "
        "photographed from above on a restaurant table. Soft shadows and vibrant colors."
    )
    cache_path = generate_image(prompt)
    show_image(cache_path)