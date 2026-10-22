import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_DIR, LABEL_COLUMNS, MAX_LENGTH, DEVICE, THRESHOLD

class ToxicPredictor:
    def __init__(self):
        if not (MODEL_DIR / "config.json").exists():
            raise FileNotFoundError("Trained model missing. Run `python -m src.train` first.")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        self.model.eval()

    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        with torch.no_grad():
            probs = torch.sigmoid(self.model(**enc).logits).cpu().numpy()[0]
        return [{"label": label, "score": float(score), "flagged": bool(score >= THRESHOLD)} for label, score in zip(LABEL_COLUMNS, probs)]
