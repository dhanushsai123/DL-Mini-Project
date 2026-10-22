import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RAW_DIR, LABEL_COLUMNS, FIGURE_DIR
from src.data_loader import load_raw_dataset

def main():
    df = load_raw_dataset()
    print("\\nDataset shape:", df.shape)
    print("\\nMissing values:\\n", df[["comment_text", *LABEL_COLUMNS]].isna().sum())
    print("\\nPositive examples per label:\\n", df[LABEL_COLUMNS].sum().sort_values(ascending=False))
    plt.figure(figsize=(10, 5))
    df[LABEL_COLUMNS].sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Positive samples per toxicity label")
    plt.ylabel("Number of comments")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution.png", dpi=160)
    plt.close()
    lengths = df["comment_text"].fillna("").astype(str).str.split().str.len()
    plt.figure(figsize=(9, 5))
    sns.histplot(lengths.clip(upper=250), bins=50)
    plt.title("Comment length distribution (capped at 250 words)")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "comment_length.png", dpi=160)
    plt.close()
    print(f"EDA figures saved to {FIGURE_DIR}")

if __name__ == "__main__":
    main()
