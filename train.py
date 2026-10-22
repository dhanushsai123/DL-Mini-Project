import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    MODEL_NAME,
    MAX_LENGTH,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    LABEL_COLUMNS,
    DEVICE,
)

from src.tokenizer_utils import load_tokenizer, ToxicCommentDataset
from src.model import create_model


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# --------------------------------------------------
# Validation
# --------------------------------------------------

@torch.no_grad()
def validation_f1(model, loader):
    model.eval()

    probabilities = []
    true_labels = []

    for batch in tqdm(loader, desc="Validating"):
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        labels = batch.pop("labels")

        outputs = model(**batch)
        logits = outputs.logits

        probabilities.append(
            torch.sigmoid(logits).cpu().numpy()
        )

        true_labels.append(
            labels.cpu().numpy()
        )

    probabilities = np.vstack(probabilities)
    true_labels = np.vstack(true_labels)

    predictions = (probabilities >= 0.5).astype(int)

    score = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return score


# --------------------------------------------------
# Main training function
# --------------------------------------------------

def main():
    set_seed()

    # Find the preprocessed dataset files
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "validation.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Processed datasets were not found. "
            "Run: python -m src.preprocessing"
        )

    print("Loading datasets...")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # ----------------------------------------------
    # QUICK TRAINING MODE
    # Use a small sample to reduce CPU training time.
    # Increase these limits for final experiments.
    # ----------------------------------------------

    TRAIN_SAMPLE_SIZE = 1000
    VAL_SAMPLE_SIZE = 300

    # Keep the random seed fixed for reproducibility
    train_df = train_df.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train_df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    val_df = val_df.sample(
        n=min(VAL_SAMPLE_SIZE, len(val_df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print("\nQuick training mode enabled.")
    print(f"Training samples:   {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Device:             {DEVICE}")
    print(f"Maximum token length: {MAX_LENGTH}")
    print(f"Epochs:             {EPOCHS}")

    # Check that the required columns are present
    required_columns = ["comment_text"] + list(LABEL_COLUMNS)

    for column in required_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Required column '{column}' is missing "
                "from the training dataset."
            )

        if column not in val_df.columns:
            raise ValueError(
                f"Required column '{column}' is missing "
                "from the validation dataset."
            )

    # Load tokenizer and prepare datasets
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    print("Preparing training dataset...")
    train_ds = ToxicCommentDataset(train_df, tokenizer)

    print("Preparing validation dataset...")
    val_ds = ToxicCommentDataset(val_df, tokenizer)

    # Use a smaller batch size for limited-memory systems.
    # A batch size of 8 is generally a reasonable starting point.
    train_batch_size = min(TRAIN_BATCH_SIZE, 8)
    eval_batch_size = min(EVAL_BATCH_SIZE, 8)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nLoading DistilBERT model...")
    model = create_model().to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    history = []

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for batch in progress_bar:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            progress_bar.set_postfix(
                loss=f"{np.mean(losses):.4f}"
            )

        # Evaluate after each epoch
        print("\nRunning validation...")
        val_macro_f1 = validation_f1(model, val_loader)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": float(val_macro_f1),
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
        }

        history.append(epoch_result)

        print("\nEpoch results:")
        print(epoch_result)

        # Save the best model
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1

            MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            print(f"\nBest model saved to: {MODEL_DIR}")

    # Save training history
    history_path = MODEL_DIR.parent / "training_history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best validation Macro-F1: {best_f1:.4f}")
    print(f"Training history saved to: {history_path}")
    print(f"Model saved to: {MODEL_DIR}")
    print(
        "\nNote: This was a quick experiment using a small "
        "dataset sample. Report the sample sizes when "
        "presenting the results."
    )


if __name__ == "__main__":
    main()
