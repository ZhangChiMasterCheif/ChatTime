"""
PTF (Paris Traffic Flow) loader for ChatTime conformal prediction.

CGTSF subset, hourly traffic flow at a Paris highway, with weather + calendar
text per window. Each CSV row is one pre-built (history, future, text) triple,
so window construction is trivial — we only need to parse strings and split
chronologically.

Data source
-----------
  dataset/PTF_full.csv  (downloaded from ChengsenWang/CGTSF)

Splits (row-index based; rows appear chronologically by Idx):
  train       : first 60%  (8 months — only used if we ever fine-tune)
  calibration : next  20%  (~3 months — for fitting conformal scores)
  test        : last  20%  (~3 months — for evaluating coverage)
"""

import ast
import numpy as np
import pandas as pd
from typing import List, Tuple

PTF_CSV = "dataset/PTF_full.csv"

# Match the ChatTime demo's split lengths
HIST_LEN = 120     # 5 days of hourly history
PRED_LEN = 24      # 1 day ahead

CAL_FRACTION  = 0.20
TEST_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ptf() -> pd.DataFrame:
    """
    Load PTF_full.csv. The Hist and Pred columns are strings of float lists;
    we parse them into actual numpy arrays.
    """
    df = pd.read_csv(PTF_CSV)
    df["Hist"] = df["Hist"].apply(ast.literal_eval).apply(np.asarray)
    df["Pred"] = df["Pred"].apply(ast.literal_eval).apply(np.asarray)
    df = df.sort_values("Idx").reset_index(drop=True)
    return df


def split_ptf(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Row-index split into train / cal / test (chronological)."""
    N = len(df)
    n_test = int(N * TEST_FRACTION)
    n_cal  = int(N * CAL_FRACTION)
    n_train = N - n_cal - n_test

    train = df.iloc[:n_train].reset_index(drop=True)
    cal   = df.iloc[n_train : n_train + n_cal].reset_index(drop=True)
    test  = df.iloc[n_train + n_cal :].reset_index(drop=True)
    return train, cal, test


def to_window_lists(
    df: pd.DataFrame,
    hist_len: int = HIST_LEN,
    pred_len: int = PRED_LEN,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Convert a split DataFrame to (contexts, futures, texts) lists.

    We trim Hist to its last `hist_len` values and Pred to its first `pred_len`
    values, matching the ChatTime demo's convention.
    """
    contexts, futures, texts = [], [], []
    for _, row in df.iterrows():
        h = np.asarray(row["Hist"], dtype=np.float32)[-hist_len:]
        p = np.asarray(row["Pred"], dtype=np.float32)[:pred_len]
        if len(h) < hist_len or len(p) < pred_len:
            continue                       # drop short rows
        contexts.append(h)
        futures.append(p)
        texts.append(row["Text"])
    return contexts, futures, texts


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_ptf()
    print(f"Loaded {len(df)} rows from {PTF_CSV}")
    print(f"Columns: {list(df.columns)}")
    print(f"Hist[0] length: {len(df.Hist.iloc[0])}, Pred[0] length: {len(df.Pred.iloc[0])}")
    print(f"Text[0] (first 200 chars):\n  {df.Text.iloc[0][:200]}\n")

    train, cal, test = split_ptf(df)
    print(f"train: {len(train)} rows")
    print(f"cal  : {len(cal)} rows")
    print(f"test : {len(test)} rows")

    cal_ctx, cal_fut, cal_txt = to_window_lists(cal)
    print(f"\ncal windows: {len(cal_ctx)}  (hist={HIST_LEN}, pred={PRED_LEN})")
    print(f"cal_ctx[0][:5]:  {cal_ctx[0][:5]}")
    print(f"cal_fut[0][:5]:  {cal_fut[0][:5]}")
