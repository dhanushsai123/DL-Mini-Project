# Explainable Transformer-Based Toxic Comment Detection with Bias Analysis

This project fine-tunes DistilBERT for six-label, multi-label toxicity classification using the Jigsaw Toxic Comment Classification dataset.

## Important: dataset
The full Jigsaw `train.csv` is about 69 MB and is **not bundled in this ZIP** because this environment could not download the remote file. Do not use a folder named `train.csv`; the path must point to an actual CSV file. See `data/raw/README.txt` or run `python download_dataset.py` (requires internet access).

Expected raw file:
`data/raw/train.csv`

Expected columns:
`id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Setup (Windows / VS Code)
Open the inner project folder (the one containing `requirements.txt`) in VS Code, then use its terminal:

```bash
python -m pip install -r requirements.txt
python download_dataset.py
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.bias_analysis
streamlit run app/streamlit_app.py
```

Run one command at a time. Training downloads pretrained DistilBERT weights on first run and may be slow on a CPU.

## Project layout
- `data/raw/`: original dataset CSV
- `data/processed/`: reproducible train/validation/test splits generated from the labeled training CSV
- `src/`: preprocessing, EDA, model, training, evaluation, prediction, explanation and bias analysis
- `models/distilbert-toxic/`: fine-tuned model saved by training
- `outputs/`: reports and plots

## Methodology notes
- This is multi-label classification; one comment can have several labels.
- The official competition `test.csv` has no toxicity labels. The scripts split the labeled `train.csv` into train/validation/test so evaluation can calculate metrics.
- Bias analysis is exploratory, not proof of fairness. Results depend on selected identity terms and the dataset.
- Use macro/micro F1, per-label precision/recall, Hamming loss, and ROC-AUC where defined. Do not report performance numbers until you have actually run evaluation.
