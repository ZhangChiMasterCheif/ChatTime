"""
Codebook-Distance CQR for ChatTime.

Same as chattime_method_cqr.py but the nonconformity score is measured in
codebook bin-index space rather than real-value space.

ChatTime's discretizer:
  - Fits a MinMaxScaler on the context (per-window).
  - Maps each value into one of n_tokens bins (centers in [-1, +1]).
  - The "bin index" is what np.digitize returns; bins[i] = centers[i].

To compute bin distance for samples and futures we need to use the SAME
scaler that was fitted on the corresponding history. We don't reuse the
internal Discretizer state (it's mutated per-call); instead we re-implement
the bin-id computation with an explicitly passed scaler.
"""

import sys
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, ".")
from model.model import ChatTime
from utils.tools import Discretizer

from chattime_dataset import (
    load_weekly_hosp, load_text, split_weekly, make_text_windows,
)
from chattime_method_naive import (
    predict_samples, compute_metrics,
    MODEL_PATH, CONTEXT_WEEKS, PRED_WEEKS, NUM_SAMPLES, ALPHAS,
    EXAMPLE_STATES, STRIDE,
)
from chattime_method_cqr import cqr_quantile


# ---------------------------------------------------------------------------
# Bin-id helpers — replicate Discretizer's internals so we can reuse a
# fitted scaler across samples / futures of the same window.
# ---------------------------------------------------------------------------

def fit_scaler(history: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on the history (same as Discretizer.discretize does)."""
    scaler = MinMaxScaler()
    scaler.fit(history.reshape(-1, 1))
    return scaler


def to_bins(values: np.ndarray, scaler: MinMaxScaler, boundaries: np.ndarray
            ) -> np.ndarray:
    """
    Map values to bin indices using a pre-fitted scaler and the discretizer's
    boundaries. Replicates the math inside Discretizer.discretize.

    values  : any shape, last axis is time
    scaler  : already fitted on the corresponding history
    boundaries : the discretizer's bin boundaries (1D array)

    Returns: integer bin IDs, same shape as `values`.
    """
    flat   = values.reshape(-1, 1)
    scaled = scaler.transform(flat).reshape(values.shape) - 0.5
    bin_ids = np.digitize(x=scaled, bins=boundaries, right=True)
    return bin_ids


# ---------------------------------------------------------------------------
# Codebook conformal core
# ---------------------------------------------------------------------------

def codebook_scores(bin_lo, bin_hi, true_bins):
    """
    Per-step CQR score in bin-index space. Returns (N * pred_len,).
    Same per-step semantics as cqr_scores in chattime_method_cqr.py.
    """
    return np.maximum(bin_lo - true_bins, true_bins - bin_hi).flatten()


def apply_codebook_correction(bin_lo, bin_hi, Q_hat, n_tokens):
    """Inflate bin interval by Q̂ and clamp to [0, n_tokens-1]."""
    lo = np.clip(bin_lo - Q_hat, 0, n_tokens - 1)
    hi = np.clip(bin_hi + Q_hat, 0, n_tokens - 1)
    return lo, hi


def bins_to_values(bin_ids: np.ndarray, scaler: MinMaxScaler, centers: np.ndarray
                   ) -> np.ndarray:
    """
    Convert bin IDs back to real values:  centers[bin_id]  →  inverse_scaler.
    Inverse of `to_bins`.
    """
    centered = centers[np.clip(bin_ids, 0, len(centers) - 1)]   # bin → center value
    flat     = (centered + 0.5).reshape(-1, 1)
    values   = scaler.inverse_transform(flat).reshape(centered.shape)
    return values


# ---------------------------------------------------------------------------
# Per-window pipeline: get bin-space quantiles + true bins
# ---------------------------------------------------------------------------

def per_window_bins(
    model: ChatTime,
    discretizer: Discretizer,
    history: np.ndarray,
    future:  np.ndarray,
    pred_len: int,
    alpha: float,
    text: str,
    use_text: bool,
    num_samples: int,
):
    """
    For one window, return:
      bin_lo (pred_len,), bin_hi (pred_len,),
      true_bins (pred_len,),
      scaler  (fitted on this window's history)
    """
    scaler     = fit_scaler(history)
    boundaries = discretizer.boundaries

    # samples in real-value space, then convert to bins
    ctx_text = text if (use_text and text) else None
    samples  = predict_samples(model, history, pred_len, ctx_text, num_samples)
    # samples: (num_samples, pred_len) — may contain NaN

    sample_bins = to_bins(np.nan_to_num(samples, nan=history.mean()),
                          scaler, boundaries).astype(np.float32)
    # (num_samples, pred_len)

    bin_lo = np.quantile(sample_bins, alpha / 2,     axis=0)
    bin_hi = np.quantile(sample_bins, 1 - alpha / 2, axis=0)

    true_bins = to_bins(future, scaler, boundaries)
    return bin_lo, bin_hi, true_bins, scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    weekly = load_weekly_hosp()
    texts  = load_text()
    _, cal_weekly, test_weekly = split_weekly(weekly)

    cal_ctx_t,  cal_fut_t,  cal_txt,  _ = make_text_windows(
        cal_weekly, texts, CONTEXT_WEEKS, PRED_WEEKS,
        states=EXAMPLE_STATES, stride=STRIDE,
    )
    test_ctx_t, test_fut_t, test_txt, _ = make_text_windows(
        test_weekly, texts, CONTEXT_WEEKS, PRED_WEEKS,
        states=EXAMPLE_STATES, stride=STRIDE,
    )

    cal_ctx_np  = [c.numpy() for c in cal_ctx_t]
    test_ctx_np = [c.numpy() for c in test_ctx_t]
    cal_fut_np  = np.stack([f.numpy() for f in cal_fut_t])
    test_fut_np = np.stack([f.numpy() for f in test_fut_t])

    print(f"Cal: {len(cal_ctx_np)} windows | Test: {len(test_ctx_np)} windows")

    print(f"\nLoading ChatTime: {MODEL_PATH}")
    model = ChatTime(
        model_path=MODEL_PATH,
        hist_len=CONTEXT_WEEKS,
        pred_len=PRED_WEEKS,
        num_samples=NUM_SAMPLES,
    )

    discretizer = model.discretizer
    centers     = discretizer.centers
    n_tokens    = len(centers)

    print(f"Codebook size: {n_tokens} bins")

    print("\n--- Codebook-CQR for ChatTime ---")
    print(f"{'Method':>25}  {'Alpha':>6}  {'Coverage':>10}  {'Width':>10}  {'Q̂(bins)':>10}")
    print("-" * 70)

    for use_text in (False, True):
        label = "Codebook-CQR (text)" if use_text else "Codebook-CQR (no text)"
        for alpha in ALPHAS:
            # ---- Calibration: collect per-window bins, build scores ----
            cal_bin_lo, cal_bin_hi, cal_true = [], [], []
            cal_scalers = []
            for ctx, fut, txt in zip(cal_ctx_np, cal_fut_np, cal_txt):
                bl, bh, tb, sc = per_window_bins(
                    model, discretizer, ctx, fut, PRED_WEEKS, alpha,
                    txt, use_text, NUM_SAMPLES,
                )
                cal_bin_lo.append(bl); cal_bin_hi.append(bh)
                cal_true.append(tb);   cal_scalers.append(sc)

            cal_bin_lo = np.stack(cal_bin_lo)
            cal_bin_hi = np.stack(cal_bin_hi)
            cal_true   = np.stack(cal_true)
            scores     = codebook_scores(cal_bin_lo, cal_bin_hi, cal_true)
            Q_hat      = cqr_quantile(scores, alpha)

            # ---- Test: get bin intervals, inflate, convert back ----
            lo_arr = np.empty((len(test_ctx_np), PRED_WEEKS))
            hi_arr = np.empty((len(test_ctx_np), PRED_WEEKS))
            for i, (ctx, fut, txt) in enumerate(zip(test_ctx_np, test_fut_np, test_txt)):
                bl, bh, _, sc = per_window_bins(
                    model, discretizer, ctx, fut, PRED_WEEKS, alpha,
                    txt, use_text, NUM_SAMPLES,
                )
                lo_b, hi_b = apply_codebook_correction(bl, bh, Q_hat, n_tokens)
                lo_arr[i]  = bins_to_values(lo_b.astype(int), sc, centers)
                hi_arr[i]  = bins_to_values(hi_b.astype(int), sc, centers)

            m = compute_metrics(lo_arr, hi_arr, test_fut_np)
            print(f"{label:>25}  {alpha:>6.2f}  "
                  f"{m['coverage']:>10.3f}  {m['width']:>10.2f}  {Q_hat:>10.1f}")


if __name__ == "__main__":
    main()
