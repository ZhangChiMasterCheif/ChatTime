"""
ChatTime COVID dataset loader — weekly hospitalizations + weekly GPT trend
descriptions, aligned by (state, week_start).

Why weekly?
  - Text data is published weekly (one trend description per state per week).
  - ChatTime is slow (full LLaMA generation per prediction); operating at
    weekly resolution makes the experiment tractable.
  - Weekly hospitalization is also a less noisy signal than daily.

Splits (same calendar boundaries as covid_dataset.py for cross-comparability):
  train:       up to     2021-12-31     (for any future fine-tuning experiments)
  calibration: 2022-01-01 to 2022-06-30 (for fitting conformal scores)
  test:        2022-07-01 onwards       (for evaluating coverage)

The text alignment policy: for a prediction window whose history ends at
week W, the *text context* is the description of week W (the most recent
text known at prediction time — no leakage).

Data sources
------------
  hosp_daily_state.csv   : daily hospitalizations per state (wide format)
  hosp_gpt_trend.jsonl   : weekly text trend descriptions per state
"""

import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

HOSP_CSV  = "../Covid/covid_data/timeseries/hosp_daily_state.csv"
TEXT_PATH = "../Covid/covid_data/text/hosp_gpt_trend.jsonl"
HOSP_COL  = "Confirmed COVID-19 Hospital Admissions"

TRAIN_END  = "2021-12-31"
CAL_START  = "2022-01-01"
CAL_END    = "2022-06-30"
TEST_START = "2022-07-01"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_weekly_hosp() -> pd.DataFrame:
    """
    Load daily hospitalizations and aggregate to weekly (Mon → Sun) per state.

    Returns a wide DataFrame:
      - rows: weeks, indexed by Monday (week_start)
      - columns: state names
      - values: weekly hospitalization sum (float; NaN if any day missing)
    """
    daily = pd.read_csv(HOSP_CSV, parse_dates=["date"])
    daily_wide = daily.pivot_table(
        index="date", columns="state", values=HOSP_COL, aggfunc="first"
    ).sort_index()

    # reindex to a full daily calendar so missing dates are NaN
    full = pd.date_range(daily_wide.index.min(), daily_wide.index.max(), freq="D")
    daily_wide = daily_wide.reindex(full)

    # weekly sum, anchored to weeks ending Sunday → label by Monday (week_start)
    # `label="left"` puts the index at the Monday that starts the week.
    weekly = daily_wide.resample("W-SUN", label="left", closed="left").sum(min_count=7)
    # weekly.index is now Mondays of each week_start

    return weekly


def load_text() -> Dict[Tuple[str, pd.Timestamp], str]:
    """
    Load weekly trend descriptions. Returns a dict keyed by (state, week_start)
    where week_start is a pandas Timestamp (the Monday of the week).
    """
    text_map: Dict[Tuple[str, pd.Timestamp], str] = {}
    with open(TEXT_PATH) as f:
        for line in f:
            r = json.loads(line)
            week_start = pd.to_datetime(r["week_start"])
            state      = r["state"]
            text_map[(state, week_start)] = r["trend_description"]
    return text_map


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_weekly(weekly: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train / calibration / test split, same dates as covid_dataset."""
    train = weekly.loc[:TRAIN_END]
    cal   = weekly.loc[CAL_START:CAL_END]
    test  = weekly.loc[TEST_START:]
    return train, cal, test


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def make_text_windows(
    weekly: pd.DataFrame,
    text_map: Dict[Tuple[str, pd.Timestamp], str],
    context_weeks: int,
    pred_weeks: int,
    states: Optional[List[str]] = None,
    stride: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[Tuple[str, pd.Timestamp]]]:
    """
    Slide windows over each state's weekly series and pair with the text
    description of the *last context week* (the most recent text known at
    prediction time).

    Parameters
    ----------
    weekly        : wide DataFrame from load_weekly_hosp(), restricted to a split
    text_map      : output of load_text()
    context_weeks : weeks of history fed as the model's input
    pred_weeks    : weeks to predict / evaluate
    states        : optional list of states to include (default: all)
    stride        : weeks between consecutive window starts

    Returns
    -------
    contexts : list of 1D tensors, each length context_weeks
    futures  : list of 1D tensors, each length pred_weeks
    texts    : list of strings, one per window  (empty string if missing)
    keys     : list of (state, week_at_prediction_start) tuples for bookkeeping
    """
    if states is None:
        states = list(weekly.columns)

    contexts, futures, texts, keys = [], [], [], []

    for state in states:
        series = weekly[state]
        T = len(series)
        idx_start = 0
        while idx_start + context_weeks + pred_weeks <= T:
            ctx_slice = series.iloc[idx_start : idx_start + context_weeks]
            fut_slice = series.iloc[idx_start + context_weeks
                                    : idx_start + context_weeks + pred_weeks]

            # context_weeks ago up to the prediction-start week (exclusive).
            # The "last context week" is at index idx_start + context_weeks - 1.
            last_ctx_week_start = ctx_slice.index[-1]
            text = text_map.get((state, last_ctx_week_start), "")

            # Skip windows with NaN gaps (incomplete weeks). ChatTime can
            # technically handle NaN, but keeping clean windows simplifies
            # interpretation.
            if ctx_slice.isna().any() or fut_slice.isna().any():
                idx_start += stride
                continue

            contexts.append(torch.tensor(ctx_slice.values.astype(np.float32)))
            futures.append(torch.tensor(fut_slice.values.astype(np.float32)))
            texts.append(text)
            keys.append((state, ctx_slice.index[-1]))

            idx_start += stride

    return contexts, futures, texts, keys


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading weekly hosp ...")
    weekly = load_weekly_hosp()
    print(f"  shape={weekly.shape}  (weeks × states)")
    print(f"  date range: {weekly.index[0].date()} → {weekly.index[-1].date()}")
    print(f"  states:     {list(weekly.columns)[:5]} ...")

    print("\nLoading text ...")
    text_map = load_text()
    print(f"  text entries: {len(text_map)}")
    sample_key = next(iter(text_map.keys()))
    print(f"  sample key:   {sample_key}")
    print(f"  sample text:  {text_map[sample_key][:120]}...")

    print("\nSplits:")
    train, cal, test = split_weekly(weekly)
    for name, df in [("train", train), ("cal", cal), ("test", test)]:
        if len(df) > 0:
            print(f"  {name}: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} weeks)")

    print("\nMaking calibration windows (context=12 weeks, pred=4 weeks, stride=1) ...")
    ctxs, futs, texts, keys = make_text_windows(
        cal, text_map, context_weeks=12, pred_weeks=4, stride=1
    )
    print(f"  N windows = {len(ctxs)}")
    if ctxs:
        print(f"  example  : state={keys[0][0]}, week_at_pred={keys[0][1].date()}")
        print(f"             ctx[:5]={ctxs[0][:5].tolist()}")
        print(f"             text='{texts[0][:100]}...'" if texts[0] else "             text=<missing>")

    n_with_text = sum(1 for t in texts if t)
    print(f"\nText coverage: {n_with_text}/{len(texts)} windows have text "
          f"({n_with_text/max(len(texts),1):.1%})")
