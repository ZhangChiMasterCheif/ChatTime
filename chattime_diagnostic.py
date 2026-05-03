"""
Diagnostic for ChatTime sample distributions.

Picks a small set of representative test windows (2 falling, 2 stable, 2 rising)
and generates many samples per window so we can see what's actually going wrong:
  - Are samples wildly variable (sampling noise dominates)?
  - Are they tightly clustered around a wrong median (bias dominates)?
  - Does adding text shift the distribution helpfully?

Outputs
-------
  diagnostic_samples.npz   : per-window raw samples (no_text and with_text)
  diagnostic_trajectories.png : sample paths overlaid on truth
  diagnostic_stats.csv     : per-window mean / median / std / coverage

Knob
----
  Increase DIAG_NUM_SAMPLES (default 64) if you suspect quantile noise.

Usage:
    python chattime_diagnostic.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from model.model import ChatTime

from chattime_dataset import (
    load_weekly_hosp, load_text, split_weekly, make_text_windows,
)
from chattime_method_naive import predict_samples

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH        = "ChengsenWang/ChatTime-1-7B-Chat"
CONTEXT_WEEKS     = 12
PRED_WEEKS        = 4
DIAG_NUM_SAMPLES  = 64       # << bumped vs the main experiment's 8

# We pick a handful of windows from the test set, diversifying by recent slope.
# 2 from each regime → 6 total panels.
N_PER_REGIME = 2

EXAMPLE_STATES = [
    "California", "Texas", "Florida", "New York",
    "Illinois", "Pennsylvania", "Ohio", "Georgia",
    "North Carolina", "Michigan",
]

ALPHAS_TO_REPORT = [0.1, 0.2, 0.3]

CACHE_PATH = "diagnostic_samples.npz"
PLOT_PATH  = "diagnostic_trajectories.png"
CSV_PATH   = "diagnostic_stats.csv"


# ---------------------------------------------------------------------------
# Pick representative windows
# ---------------------------------------------------------------------------

def slope_pct(ctx: np.ndarray, lookback: int = 4) -> float:
    """Slope (in % of recent mean per week) over the last `lookback` weeks."""
    recent = ctx[-lookback:]
    valid  = ~np.isnan(recent)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(lookback)[valid].astype(float)
    y = recent[valid]
    m = y.mean()
    if m <= 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0] / m)


def pick_windows(contexts, futures, texts, keys, n_per_regime=N_PER_REGIME):
    """Return `3 * n_per_regime` indices: n falling, n stable, n rising."""
    THRESH = 0.02   # 2% per week

    falling, stable, rising = [], [], []
    for i, ctx in enumerate(contexts):
        s = slope_pct(ctx.numpy())
        if   s < -THRESH: falling.append(i)
        elif s >  THRESH: rising.append(i)
        else:             stable.append(i)

    # take evenly-spaced picks from each bucket so we don't all cluster
    def pick(lst):
        if len(lst) == 0:
            return []
        idxs = np.linspace(0, len(lst) - 1, n_per_regime).round().astype(int)
        return [lst[i] for i in idxs]

    return {
        "falling": pick(falling),
        "stable":  pick(stable),
        "rising":  pick(rising),
    }


# ---------------------------------------------------------------------------
# Sample generation (cached)
# ---------------------------------------------------------------------------

def get_samples_for_window(model, ctx, txt, num_samples):
    """Return (no_text_samples, with_text_samples), each (num_samples, pred_len)."""
    s_no_text   = predict_samples(model, ctx, PRED_WEEKS, None, num_samples)
    s_with_text = predict_samples(model, ctx, PRED_WEEKS, txt or None, num_samples)
    return s_no_text, s_with_text


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panels(picks, contexts, futures, texts, keys,
                samples_no_text, samples_with_text):
    """One row per regime, two panels per window (no_text vs with_text)."""
    n_total = sum(len(v) for v in picks.values())
    fig, axes = plt.subplots(n_total, 2, figsize=(13, 3.2 * n_total),
                             sharex=False)
    if n_total == 1:
        axes = axes.reshape(1, 2)

    row = 0
    for regime, idxs in picks.items():
        for win_idx in idxs:
            ctx = contexts[win_idx].numpy()
            fut = futures[win_idx].numpy()
            key = keys[win_idx]

            for col, (label, samples) in enumerate([
                ("no text",   samples_no_text[win_idx]),
                ("with text", samples_with_text[win_idx]),
            ]):
                ax = axes[row, col]

                # x axis: context weeks (negative) followed by pred weeks (positive)
                ctx_x = np.arange(-len(ctx), 0)
                fut_x = np.arange(0, len(fut))

                ax.plot(ctx_x, ctx, color="black", lw=1.2, label="context")
                ax.plot(fut_x, fut, color="black", lw=1.5, label="truth (future)")
                ax.axvline(-0.5, color="grey", ls=":", lw=0.8)

                # plot every sample as a faded line
                for s_path in samples:
                    ax.plot(fut_x, s_path, color="steelblue", alpha=0.18, lw=0.6)
                # median sample as a bold line
                med = np.nanmedian(samples, axis=0)
                ax.plot(fut_x, med, color="darkorange", lw=1.6, label="sample median")

                ax.set_title(f"{regime} | {key[0]} | {key[1].date()} | {label}",
                             fontsize=9)
                ax.set_xlabel("Weeks (0 = first prediction step)")
                ax.set_ylabel("Weekly admissions")
                if row == 0 and col == 0:
                    ax.legend(loc="upper left", fontsize=8)

            row += 1

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(picks, futures, samples_no_text, samples_with_text):
    """For each window × text condition, report sample mean/med/std and per-α coverage."""
    rows = []
    for regime, idxs in picks.items():
        for win_idx in idxs:
            fut = futures[win_idx].numpy()                       # (pred_len,)
            for label, samples in [
                ("no_text",   samples_no_text[win_idx]),
                ("with_text", samples_with_text[win_idx]),
            ]:
                # samples: (num_samples, pred_len)
                med  = np.nanmedian(samples, axis=0)
                bias = float(np.nanmean(med - fut))                  # signed bias
                std  = float(np.nanstd(samples))                     # global std
                rng  = float(np.nanmax(samples) - np.nanmin(samples))

                row = {
                    "regime": regime, "window_idx": win_idx, "text": label,
                    "truth_mean":   float(fut.mean()),
                    "median_mean":  float(np.nanmean(med)),
                    "bias_signed":  bias,
                    "sample_std":   std,
                    "sample_range": rng,
                }
                # per-α coverage on this window
                for a in ALPHAS_TO_REPORT:
                    lo = np.nanquantile(samples, a / 2,     axis=0)
                    hi = np.nanquantile(samples, 1 - a / 2, axis=0)
                    cov = float(((fut >= lo) & (fut <= hi)).mean())
                    row[f"cov@{a}"] = cov
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    weekly = load_weekly_hosp()
    texts  = load_text()
    _, _, test_weekly = split_weekly(weekly)

    contexts, futures, txts, keys = make_text_windows(
        test_weekly, texts, CONTEXT_WEEKS, PRED_WEEKS,
        states=EXAMPLE_STATES, stride=1,
    )
    print(f"Test windows: {len(contexts)}")

    picks = pick_windows(contexts, futures, txts, keys)
    flat_idxs = [i for v in picks.values() for i in v]
    print(f"Picked windows by regime: "
          f"falling={picks['falling']}, stable={picks['stable']}, rising={picks['rising']}")

    if os.path.exists(CACHE_PATH):
        print(f"Loading cache: {CACHE_PATH}")
        d = np.load(CACHE_PATH, allow_pickle=True)
        samples_no_text   = {int(k): d[f"nt_{k}"] for k in flat_idxs}
        samples_with_text = {int(k): d[f"wt_{k}"] for k in flat_idxs}
    else:
        print(f"\nLoading ChatTime: {MODEL_PATH}")
        model = ChatTime(
            model_path=MODEL_PATH,
            hist_len=CONTEXT_WEEKS,
            pred_len=PRED_WEEKS,
            num_samples=DIAG_NUM_SAMPLES,
        )

        samples_no_text   = {}
        samples_with_text = {}
        for i in flat_idxs:
            print(f"  Sampling window {i} ({keys[i][0]}, {keys[i][1].date()}) ...")
            s_nt, s_wt = get_samples_for_window(
                model, contexts[i].numpy(), txts[i], DIAG_NUM_SAMPLES
            )
            samples_no_text[i]   = s_nt
            samples_with_text[i] = s_wt

        # cache
        save_kwargs = {}
        for i in flat_idxs:
            save_kwargs[f"nt_{i}"] = samples_no_text[i]
            save_kwargs[f"wt_{i}"] = samples_with_text[i]
        np.savez(CACHE_PATH, **save_kwargs)
        print(f"Cached to {CACHE_PATH}")

    plot_panels(picks, contexts, futures, txts, keys,
                samples_no_text, samples_with_text)

    df = compute_stats(picks, futures, samples_no_text, samples_with_text)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}\n")

    # Pretty-print summary
    show_cols = ["regime", "text", "truth_mean", "median_mean",
                 "bias_signed", "sample_std", "sample_range",
                 *[f"cov@{a}" for a in ALPHAS_TO_REPORT]]
    print(df[show_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
