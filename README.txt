Place the REAL Jigsaw training CSV here and name it exactly train.csv.
The file must be a CSV file, not a folder. Required columns:
id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate.

Download source:
https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/blob/main/train.csv

You may also run `python download_dataset.py` from the project root (requires internet).
The original competition test.csv is unlabeled and is not needed for the train/validation/test split used by this project.
