# Download the image understanding RAG dataset (CSV + images) into .local_cache.

import urllib.request
from pathlib import Path

# Cache lives next to this script
BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / ".local_cache"
IMAGES_DIR = CACHE_DIR / "images"

BASE_URL = "https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data"


def download_file(url: str, dest: Path) -> None:
    """Download url to dest, creating parent directories if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded: {dest.name}")


def download_dataset() -> None:
    """Download df.csv and image files 1–7 into .local_cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    download_file(f"{BASE_URL}/df.csv", CACHE_DIR / "df.csv")
    for i in range(1, 8):
        download_file(
            f"{BASE_URL}/images/{i}.png",
            IMAGES_DIR / f"{i}.png",
        )
    print("Done.")


if __name__ == "__main__":
    download_dataset()
