"""Download the labeled Jigsaw train.csv from Hugging Face."""
from pathlib import Path
import requests

URL = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv"
DEST = Path(__file__).resolve().parent / "data" / "raw" / "train.csv"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.is_file() and DEST.stat().st_size > 1_000_000:
        print(f"Dataset already exists: {DEST} ({DEST.stat().st_size:,} bytes)")
        return
    if DEST.exists() and DEST.is_dir():
        raise IsADirectoryError(
            f"{DEST} is a folder, not a CSV file. Rename/remove that mistaken folder first, "
            "then rerun this script."
        )
    print("Downloading the Jigsaw training CSV. This requires internet access and may take a few minutes...")
    with requests.get(URL, stream=True, timeout=(30, 180), allow_redirects=True) as response:
        response.raise_for_status()
        temp = DEST.with_suffix(".csv.part")
        total = 0
        with open(temp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\rDownloaded {total:,} bytes", end="", flush=True)
        print()
        if total < 1_000_000:
            temp.unlink(missing_ok=True)
            raise RuntimeError("Download looks too small to be the full dataset. Check your internet connection/source.")
        temp.replace(DEST)
    print(f"Saved dataset to: {DEST}")

if __name__ == "__main__":
    main()
