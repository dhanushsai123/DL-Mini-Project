from pathlib import Path
from src.config import LABEL_COLUMNS, RAW_DIR

def test_six_labels_are_configured():
    assert len(LABEL_COLUMNS) == 6

def test_raw_folder_exists():
    assert RAW_DIR.exists()
