import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, roc_auc_score, classification_report, multilabel_confusion_matrix
from src.config import PROCESSED_DIR, MODEL_DIR, LABEL_COLUMNS, MAX_LENGTH, EVAL_BATCH_SIZE, DEVICE, OUTPUT_DIR, THRESHOLD
from src.tokenizer_utils import ToxicCommentDataset

def main():
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists(): raise FileNotFoundError("Run preprocessing first.")
    if not (MODEL_DIR / "config.json").exists(): raise FileNotFoundError("Trained model not found. Run `python -m src.train` first.")
    df = pd.read_csv(test_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    loader = DataLoader(ToxicCommentDataset(df, tokenizer), batch_size=EVAL_BATCH_SIZE)
    probs, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            probs.append(torch.sigmoid(model(**batch).logits).cpu().numpy())
            ys.append(labels.cpu().numpy())
    p, y = np.vstack(probs), np.vstack(ys)
    pred = (p >= THRESHOLD).astype(int)
    auc = {}
    for i, label in enumerate(LABEL_COLUMNS):
        try: auc[label] = float(roc_auc_score(y[:, i], p[:, i]))
        except ValueError: auc[label] = None
    metrics = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y, pred)),
        "roc_auc_per_label": auc,
        "classification_report": classification_report(y, pred, target_names=LABEL_COLUMNS, zero_division=0, output_dict=True),
        "multilabel_confusion_matrices": multilabel_confusion_matrix(y, pred).tolist()
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2)
    pd.DataFrame(p, columns=[f"{c}_probability" for c in LABEL_COLUMNS]).to_csv(OUTPUT_DIR / "test_probabilities.csv", index=False)
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report" and k != "multilabel_confusion_matrices"}, indent=2))
    print("\\nPer-label report:")
    print(pd.DataFrame(metrics["classification_report"]).T.round(3))

if __name__ == "__main__": main()
