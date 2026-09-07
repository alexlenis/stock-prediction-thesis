"""
Shared ML utilities for honest, leak-free train/test handling.

The dataset (final_dataset.csv) carries an `is_train` column produced by
merge_dataset.py: a per-ticker chronological 80/20 split. Every training
script and the backtest read this SAME column so that:
  * the hold-out is identical across all 7 models (fair ensemble comparison),
  * the hold-out is genuinely out-of-sample for each ticker,
  * there is no cross-ticker leakage and no in-sample evaluation.

This module also builds LSTM sequences WITHIN each ticker so a lookback window
never splices one company's history onto another's label, and assigns each
sequence to train/test by its label row's is_train flag.
"""
import numpy as np
import pandas as pd

SENTIMENT_FEATURES = [
    "sent_mean", "sent_std", "sent_count", "avg_reliability",
    "sentiment_lag_1", "sentiment_lag_2", "sentiment_3d",
    "sentiment_7d", "sentiment_momentum",
]

# Columns that are never model features
NON_FEATURE_COLS = ["date", "ticker", "future_return", "signal", "is_train"]


def feature_columns(df):
    """Return the model feature columns (everything that isn't metadata/target)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train_test_masks(df):
    """Boolean masks for the chronological per-ticker split stored in the data."""
    if "is_train" not in df.columns:
        raise ValueError(
            "final_dataset.csv has no 'is_train' column — rerun merge_dataset.py."
        )
    train_mask = df["is_train"].astype(bool).values
    return train_mask, ~train_mask


def build_sequences_per_ticker(df, X_scaled, y, seq_len):
    """
    Build LSTM sequences within each ticker (never crossing boundaries) and
    split them by the label row's is_train flag.

    df        : DataFrame row-aligned with X_scaled / y, holding 'ticker' & 'is_train'.
                MUST already be sorted by ['ticker', 'date'] and index-reset.
    X_scaled  : 2D ndarray (n_rows, n_features), already scaled.
    y         : ndarray (n_rows,) or (n_rows, k) of labels/targets per row.
    seq_len   : lookback window length.

    Returns X_train, y_train, X_test, y_test as float32 arrays.
    """
    X_tr, y_tr, X_te, y_te = [], [], [], []
    y = np.asarray(y)

    for _, grp in df.groupby("ticker", sort=False):
        pos = grp.index.values                 # contiguous (sorted by ticker,date)
        if len(pos) <= seq_len:
            continue
        Xg = X_scaled[pos]
        yg = y[pos]
        is_tr = grp["is_train"].astype(bool).values
        for i in range(seq_len, len(pos)):
            seq = Xg[i - seq_len:i]
            label = yg[i]
            if is_tr[i]:
                X_tr.append(seq); y_tr.append(label)
            else:
                X_te.append(seq); y_te.append(label)

    return (
        np.array(X_tr, dtype=np.float32), np.array(y_tr, dtype=np.float32),
        np.array(X_te, dtype=np.float32), np.array(y_te, dtype=np.float32),
    )
