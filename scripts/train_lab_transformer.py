"""Train the lab's from-scratch transformer baseline.
Run with: python scripts/train_lab_transformer.py

Reproduces the architecture from the Lab 4 brief: a custom Keras TransformerBlock +
TokenAndPositionEmbedding, no pre-training. Hyperparameters: 128-dim embeddings,
2 attention heads, 512-unit feed-forward, dropout 0.1.

Saves:
  results/predictions/lab_transformer.parquet
  results/metrics.csv  (row appended)
  results/figures/lab_transformer/{training_curves,cm_lab_transformer}.png
  results/models/lab_transformer.keras
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
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from src.data import load_splits
from src.eval import LABELS, append_metrics, evaluate, pretty_report

tf.random.set_seed(42)
np.random.seed(42)

FIG_DIR = PROJECT_ROOT / "results" / "figures" / "lab_transformer"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs, inputs)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(inputs + attn)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kw):
        super().__init__(**kw)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)


def main() -> None:
    print(f"tensorflow: {tf.__version__}", flush=True)

    train, val, test = load_splits()
    label_to_int = {"activist": 0, "sceptic": 1}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_train = train["stance"].map(label_to_int).values
    y_val = val["stance"].map(label_to_int).values
    y_test = test["stance"].map(label_to_int).values
    X_train_text = train["clean_text"].astype(str).values
    X_val_text = val["clean_text"].astype(str).values
    X_test_text = test["clean_text"].astype(str).values

    VOCAB_SIZE = 30_000
    MAX_LEN = 96
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    vectorizer.adapt(tf.constant(X_train_text, dtype=tf.string))
    vocab = vectorizer.get_vocabulary()
    print(f"vocab: {len(vocab):,}, max_len: {MAX_LEN}", flush=True)

    EMBED_DIM = 128
    NUM_HEADS = 2
    FF_DIM = 512
    DROPOUT = 0.1

    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = TokenAndPositionEmbedding(MAX_LEN, len(vocab), EMBED_DIM)(x)
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, rate=DROPOUT)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="lab_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: print(s, flush=True))

    EPOCHS = 4
    BATCH = 256
    X_train_t = tf.constant(X_train_text, dtype=tf.string)
    X_val_t = tf.constant(X_val_text, dtype=tf.string)
    early = callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)

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

    X_test_t = tf.constant(X_test_text, dtype=tf.string)
    y_proba_test = model.predict(X_test_t, batch_size=512, verbose=0).ravel()
    y_pred_int = (y_proba_test >= 0.5).astype(int)
    y_pred_str = np.array([int_to_label[i] for i in y_pred_int])
    y_test_str = np.array([int_to_label[i] for i in y_test])

    metrics, cm = evaluate("lab_transformer", y_test_str, y_pred_str, y_proba=y_proba_test)
    print(pretty_report(y_test_str, y_pred_str), flush=True)
    append_metrics(metrics)

    pd.DataFrame({
        "y_true": y_test_str,
        "y_pred": y_pred_str,
        "proba_sceptic": y_proba_test,
    }).to_parquet(PRED_DIR / "lab_transformer.parquet", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Lab transformer — confusion (test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cm_lab_transformer.png", bbox_inches="tight")
    plt.close()

    model.save(MODEL_DIR / "lab_transformer.keras")
    print("saved model + predictions + figures + metrics", flush=True)


if __name__ == "__main__":
    main()
