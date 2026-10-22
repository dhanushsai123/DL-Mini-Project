import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DIR, LABEL_COLUMNS, RANDOM_SEED
from src.data_loader import load_raw_dataset

def clean_dataset(df):
    df = df.copy()
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    df["comment_text"] = df["comment_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["comment_text"].ne("")].copy()
    for col in LABEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=LABEL_COLUMNS)
    for col in LABEL_COLUMNS:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Column {col} contains values other than 0/1.")
        df[col] = df[col].astype("int64")
    df = df.drop_duplicates(subset=["comment_text"]).reset_index(drop=True)
    return df

def main():
    df = clean_dataset(load_raw_dataset())
    train_df, remaining = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    val_df, test_df = train_test_split(remaining, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)
    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        out = PROCESSED_DIR / f"{name}.csv"
        part.to_csv(out, index=False)
        print(f"Saved {name}: {len(part):,} rows -> {out}")
    print("Preprocessing completed.")
    print("Note: random split is used; iterative multi-label stratification can improve rigor for a research-grade experiment.")

if __name__ == "__main__":
    main()
