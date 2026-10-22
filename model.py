from transformers import AutoConfig, AutoModelForSequenceClassification
from src.config import MODEL_NAME, LABEL_COLUMNS

def create_model():
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABEL_COLUMNS)},
        label2id={label: i for i, label in enumerate(LABEL_COLUMNS)},
    )
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
