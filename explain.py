from pathlib import Path
import numpy as np
from lime.lime_text import LimeTextExplainer
from src.config import LABEL_COLUMNS, OUTPUT_DIR
from src.predict import ToxicPredictor

def main():
    predictor = ToxicPredictor()
    example = "You are a terrible person and nobody wants you here."
    def probabilities(texts):
        return np.array([[r["score"] for r in predictor.predict(t)] for t in texts])
    explainer = LimeTextExplainer(class_names=LABEL_COLUMNS)
    result = predictor.predict(example)
    label_index = max(range(len(result)), key=lambda i: result[i]["score"])
    explanation = explainer.explain_instance(example, probabilities, labels=[label_index], num_features=10, num_samples=500)
    out = OUTPUT_DIR / "lime_explanation.html"
    explanation.save_to_file(str(out))
    print("Example:", example)
    print("Scores:", result)
    print("Saved LIME explanation:", out)

if __name__ == "__main__": main()
