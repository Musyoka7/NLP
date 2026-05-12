"""Fine-tune DistilBERT for climate stance classification.
Run with: python scripts/train_distilbert.py

Designed for an NVIDIA GPU (CUDA). Falls back to CPU if no GPU is found.
On a 3070 expect ~5-8 minutes total for 3 epochs.

Saves:
  results/predictions/distilbert.parquet
  results/metrics.csv  (row appended)
  results/figures/distilbert/{cm_distilbert,training_curves}.png
  results/models/distilbert/  (HuggingFace model folder)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import load_splits
from src.eval import LABELS, append_metrics, evaluate, pretty_report

MODEL_NAME = "distilbert-base-uncased"
MODEL_KEY = "distilbert"
MAX_LEN = 96
EPOCHS = 3
LR = 2e-5
BATCH_TRAIN = 64
BATCH_EVAL = 128

FIG_DIR = PROJECT_ROOT / "results" / "figures" / MODEL_KEY
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "results" / "models" / MODEL_KEY
MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    print(f"torch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)

    label_to_int = {"activist": 0, "sceptic": 1}
    int_to_label = {0: "activist", 1: "sceptic"}

    train, val, test = load_splits()
    for df in (train, val, test):
        df["label"] = df["stance"].map(label_to_int)
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
    print(f"train: {len(train):,}, val: {len(val):,}, test: {len(test):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["clean_text"], truncation=True, max_length=MAX_LEN, padding=False)

    def to_ds(df):
        ds = Dataset.from_pandas(df[["clean_text", "label"]], preserve_index=False)
        return ds.map(tok, batched=True)

    train_ds, val_ds, test_ds = to_ds(train), to_ds(val), to_ds(test)
    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=int_to_label, label2id=label_to_int,
    )

    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "results" / "models" / f"_tmp_{MODEL_KEY}"),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("training ...", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"train time: {(time.time() - t0) / 60:.1f} min", flush=True)

    # Save training history plot
    history = trainer.state.log_history
    train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    eval_steps = [(h["step"], h["eval_loss"], h.get("eval_accuracy"), h.get("eval_f1_macro"))
                  for h in history if "eval_loss" in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    if train_loss:
        axes[0].plot([s for s, _ in train_loss], [l for _, l in train_loss], label="train loss")
    if eval_steps:
        axes[0].plot([s for s, _, _, _ in eval_steps], [l for _, l, _, _ in eval_steps],
                     marker="o", label="val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("step"); axes[0].legend()
    if eval_steps:
        axes[1].plot([s for s, _, _, _ in eval_steps], [a for _, _, a, _ in eval_steps],
                     marker="o", label="val accuracy")
        axes[1].plot([s for s, _, _, _ in eval_steps], [f for _, _, _, f in eval_steps],
                     marker="s", label="val f1_macro")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("step"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves.png", bbox_inches="tight")
    plt.close()

    # Test eval
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    y_pred_int = logits.argmax(axis=-1)
    y_pred_str = np.array([int_to_label[i] for i in y_pred_int])
    y_test_str = test["stance"].values
    y_proba_sceptic = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    metrics, cm = evaluate(MODEL_KEY, y_test_str, y_pred_str, y_proba=y_proba_sceptic)
    print(pretty_report(y_test_str, y_pred_str), flush=True)
    append_metrics(metrics)

    pd.DataFrame({
        "y_true": y_test_str,
        "y_pred": y_pred_str,
        "proba_sceptic": y_proba_sceptic,
    }).to_csv(PRED_DIR / f"{MODEL_KEY}.csv", index=False)
    # Also save as parquet for notebook 07 (uses fastparquet via pandas default)
    try:
        pd.DataFrame({
            "y_true": y_test_str,
            "y_pred": y_pred_str,
            "proba_sceptic": y_proba_sceptic,
        }).to_parquet(PRED_DIR / f"{MODEL_KEY}.parquet", index=False)
    except Exception as e:
        print(f"parquet save skipped: {e}", flush=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{MODEL_KEY} — confusion (test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cm_{MODEL_KEY}.png", bbox_inches="tight")
    plt.close()

    trainer.save_model(str(MODEL_DIR))
    print(f"saved model + predictions + figures + metrics", flush=True)


if __name__ == "__main__":
    main()
