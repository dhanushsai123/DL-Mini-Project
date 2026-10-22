from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, LABEL_COLUMNS

def load_raw_dataset(path=None):
    path = Path(path) if path else RAW_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put the actual Jigsaw CSV file at data/raw/train.csv or run: python download_dataset.py"
        )
    if path.is_dir():
        raise IsADirectoryError(
            f"{path} is a folder, not a CSV file. Remove/rename the mistaken folder and copy the real train.csv file there."
        )
    df = pd.read_csv(path)
    required = {"comment_text", *LABEL_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}. Check that you used the labeled Jigsaw train.csv.")
    print(f"Loaded {len(df):,} rows from {path}")
    return df
