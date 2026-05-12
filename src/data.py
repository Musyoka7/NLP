"""Data loading, cleaning, and splitting utilities."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "climate_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]")


def clean_tweet(text: str, *, drop_hashtags: bool = True, drop_emojis: bool = True) -> str:
    """Lowercases, strips URLs/mentions, optionally strips hashtags and emojis.

    Hashtag removal matters here: stance labels were derived from hashtags
    in the original dataset, so leaving them in inflates accuracy unfairly.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if drop_hashtags:
        text = HASHTAG_RE.sub(" ", text)
    else:
        text = HASHTAG_RE.sub(r"\1", text)
    if drop_emojis:
        text = text.encode("ascii", "ignore").decode("ascii")
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "stance"]).copy()
    df["text"] = df["text"].astype(str)
    df["stance"] = df["stance"].astype(str)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features used in error analysis later."""
    df = df.copy()
    df["raw_len"] = df["text"].str.len()
    df["n_hashtags"] = df["text"].str.count(r"#\w+")
    df["n_mentions"] = df["text"].str.count(r"@\w+")
    df["n_urls"] = df["text"].str.count(URL_RE)
    df["has_emoji"] = df["text"].apply(_contains_emoji)
    return df


def _contains_emoji(text: str) -> bool:
    if not isinstance(text, str):
        return False
    try:
        text.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def make_splits(
    df: pd.DataFrame,
    *,
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 split on the stance label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["stance"], random_state=seed
    )
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["stance"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    # CSV (not parquet): pandas 3.0 hard-imports pyarrow when read_parquet is
    # called, and pyarrow's bundled abseil clashes with TensorFlow's bundled
    # abseil on macOS, deadlocking model.fit. CSV avoids loading libarrow at all.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(PROCESSED_DIR / "train.csv"),
        pd.read_csv(PROCESSED_DIR / "val.csv"),
        pd.read_csv(PROCESSED_DIR / "test.csv"),
    )
