"""Train BiLSTM + GloVe baseline. Run with: python scripts/train_bilstm.py

Saves:
  results/predictions/bilstm_glove.parquet
  results/metrics.csv  (row appended)
  results/figures/bilstm/{training_curves,cm_bilstm}.png
  results/models/bilstm_glove.keras
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # no GUI backend for headless script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

import gensim.downloader as api

from src.data import load_splits
from src.eval import LABELS, append_metrics, evaluate, pretty_report

tf.random.set_seed(42)
np.random.seed(42)

FIG_DIR = PROJECT_ROOT / "results" / "figures" / "bilstm"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    print(f"tensorflow: {tf.__version__}", flush=True)

    # 1. Load splits
    train, val, test = load_splits()
    label_to_int = {"activist": 0, "sceptic": 1}
    int_to_label = {v: k for k, v in label_to_int.items()}

    y_train = train["stance"].map(label_to_int).values
    y_val = val["stance"].map(label_to_int).values
    y_test = test["stance"].map(label_to_int).values

    X_train_text = train["clean_text"].astype(str).values
    X_val_text = val["clean_text"].astype(str).values
    X_test_text = test["clean_text"].astype(str).values
    print(f"train: {len(X_train_text):,}, val: {len(X_val_text):,}, test: {len(X_test_text):,}", flush=True)

    # 2. Tokenise + pad
    VOCAB_SIZE = 30_000
    MAX_LEN = 64
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    vectorizer.adapt(tf.constant(X_train_text, dtype=tf.string))
    vocab = vectorizer.get_vocabulary()
    print(f"vocab: {len(vocab):,}", flush=True)

    # 3. GloVe
    GLOVE_NAME = "glove-wiki-gigaword-100"
    EMBED_DIM = 100
    print(f"loading {GLOVE_NAME} ...", flush=True)
    t0 = time.time()
    glove = api.load(GLOVE_NAME)
    print(f"  glove loaded in {time.time() - t0:.1f}s", flush=True)

    embedding_matrix = np.zeros((len(vocab), EMBED_DIM), dtype=np.float32)
    hits = 0
    for i, word in enumerate(vocab):
        if word in glove.key_to_index:
            embedding_matrix[i] = glove[word]
            hits += 1
    print(f"GloVe coverage: {hits:,}/{len(vocab):,}  ({hits / len(vocab):.2%})", flush=True)

    # 4. Build model
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = layers.Embedding(
        input_dim=len(vocab),
        output_dim=EMBED_DIM,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
        mask_zero=True,
    )(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="bilstm_glove")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: print(s, flush=True))

    # 5. Train
    EPOCHS = 5
    BATCH = 256
    X_train_t = tf.constant(X_train_text, dtype=tf.string)
    X_val_t = tf.constant(X_val_text, dtype=tf.string)

    early = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True,
    )

    print("training ...", flush=True)
    t0 = time.time()
    history = model.fit(
        X_train_t, y_train,
        validation_data=(X_val_t, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early],
        verbose=2,
    )
    print(f"train time: {time.time() - t0:.1f}s", flush=True)

    # 6. Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_curves.png", bbox_inches="tight")
    plt.close()

    # 7. Test evaluation
    X_test_t = tf.constant(X_test_text, dtype=tf.string)
    y_proba_test = model.predict(X_test_t, batch_size=512, verbose=0).ravel()
    y_pred_int = (y_proba_test >= 0.5).astype(int)
    y_pred_str = np.array([int_to_label[i] for i in y_pred_int])
    y_test_str = np.array([int_to_label[i] for i in y_test])

    metrics, cm = evaluate("bilstm_glove", y_test_str, y_pred_str, y_proba=y_proba_test)
    print(pretty_report(y_test_str, y_pred_str), flush=True)
    append_metrics(metrics)

    pd.DataFrame({
        "y_true": y_test_str,
        "y_pred": y_pred_str,
        "proba_sceptic": y_proba_test,
    }).to_parquet(PRED_DIR / "bilstm_glove.parquet", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("BiLSTM+GloVe — confusion (test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cm_bilstm.png", bbox_inches="tight")
    plt.close()

    model.save(MODEL_DIR / "bilstm_glove.keras")
    print(f"saved model + predictions + figures + metrics", flush=True)


if __name__ == "__main__":
    main()
