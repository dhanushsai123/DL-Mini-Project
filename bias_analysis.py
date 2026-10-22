import pandas as pd
from src.config import PROCESSED_DIR, LABEL_COLUMNS, OUTPUT_DIR
from src.predict import ToxicPredictor

IDENTITY_TERMS = ["muslim", "christian", "jewish", "black", "white", "woman", "man", "gay", "transgender", "immigrant"]

def main():
    path = PROCESSED_DIR / "test.csv"
    if not path.exists(): raise FileNotFoundError("Run preprocessing first.")
    df = pd.read_csv(path)
    predictor = ToxicPredictor()
    records = []
    for term in IDENTITY_TERMS:
        subset = df[df["comment_text"].str.contains(rf"\\b{term}\\b", case=False, regex=True, na=False)]
        if subset.empty:
            records.append({"term": term, "n": 0, "mean_toxic_score": None, "observed_toxic_rate": None})
            continue
        scores = [predictor.predict(t) for t in subset["comment_text"].head(1000)]
        toxic_scores = [next(x["score"] for x in row if x["label"] == "toxic") for row in scores]
        observed = subset["toxic"].head(len(toxic_scores)).mean() if "toxic" in subset.columns else None
        records.append({"term": term, "n": len(subset), "mean_toxic_score": sum(toxic_scores)/len(toxic_scores), "observed_toxic_rate": float(observed) if observed is not None else None})
    out = OUTPUT_DIR / "exploratory_bias_analysis.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Saved {out}")
    print("Caution: this term-based analysis is exploratory and cannot establish causal or comprehensive fairness.")

if __name__ == "__main__": main()
