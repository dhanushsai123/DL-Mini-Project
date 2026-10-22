import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, LABEL_COLUMNS

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

class ToxicCommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH, include_labels=True):
        self.texts = dataframe["comment_text"].fillna("").astype(str).tolist()
        self.include_labels = include_labels and all(c in dataframe.columns for c in LABEL_COLUMNS)
        self.labels = dataframe[LABEL_COLUMNS].astype(float).values if self.include_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.include_labels:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item
