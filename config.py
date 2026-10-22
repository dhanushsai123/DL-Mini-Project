from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models" / "distilbert-toxic"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

LABEL_COLUMNS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for directory in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
