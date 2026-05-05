"""
Conformalized Quantile Regression (CQR) for ChatTime.

Same procedure as method_cqr.py for Chronos:
  1. On calibration set, get α/2 and 1-α/2 sample quantiles q_lo, q_hi.
  2. Compute nonconformity scores  s = max( q_lo − y,  y − q_hi ),  max-over-horizon.
  3. Q̂ = ⌈(1−α)(N+1)⌉ / N  empirical quantile of scores.
  4. On test, inflate quantile interval by Q̂.

Supports `use_text=True/False` so we can compare:
  - CQR no-text  : strict baseline
  - CQR with-text: does textual context tighten the interval at the same coverage?
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from model.model import ChatTime

from chattime_dataset import (
    load_weekly_hosp, load_text, split_weekly, make_text_windows,
)
from chattime_method_naive import (
    predict_intervals, predict_samples, compute_metrics,
    MODEL_PATH, CONTEXT_WEEKS, PRED_WEEKS, NUM_SAMPLES, ALPHAS,
    EXAMPLE_STATES, STRIDE,
)


# ---------------------------------------------------------------------------
# Conformal core (works in real-value space)
# ---------------------------------------------------------------------------

def cqr_scores(q_lo: np.ndarray, q_hi: np.ndarray, futures: np.ndarray) -> np.ndarray:
    """
    Per-step CQR score: each (window, horizon) pair contributes one calibration
    sample.  Returns (N * pred_len,).

    With per-step scoring the conformal Q̂ is calibrated for marginal per-step
    coverage (≥ 1−α at every horizon), instead of the much stricter joint
    "all horizons inside" coverage produced by max-aggregating.
    """
    return np.maximum(q_lo - futures, futures - q_hi).flatten()


def cqr_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid (1-α)-quantile from Romano et al. 2019."""
    N     = len(scores)
    level = float(np.clip(np.ceil((1 - alpha) * (N + 1)) / N, 0.0, 1.0))
    return float(np.quantile(scores, level))


def apply_correction(q_lo, q_hi, Q_hat):
    return q_lo - Q_hat, q_hi + Q_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    weekly  = load_weekly_hosp()
    texts   = load_text()
    _, cal_weekly, test_weekly = split_weekly(weekly)

    cal_ctx_t, cal_fut_t, cal_txt, _ = make_text_windows(
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

    print(f"Cal windows : {len(cal_ctx_np)}  (with text: {sum(1 for t in cal_txt if t)})")
    print(f"Test windows: {len(test_ctx_np)} (with text: {sum(1 for t in test_txt if t)})")

    print(f"\nLoading ChatTime: {MODEL_PATH}")
    model = ChatTime(
        model_path=MODEL_PATH,
        hist_len=CONTEXT_WEEKS,
        pred_len=PRED_WEEKS,
        num_samples=NUM_SAMPLES,
    )

    print("\n--- CQR for ChatTime ---")
    print(f"{'Method':>20}  {'Alpha':>6}  {'Coverage':>10}  {'Width':>10}  {'Q̂':>10}")
    print("-" * 65)

    for use_text in (False, True):
        label = "CQR (with text)" if use_text else "CQR (no text)"
        for alpha in ALPHAS:
            # Step 1: base quantile intervals on calibration set
            cal_lo, cal_hi, _ = predict_intervals(
                model, cal_ctx_np, cal_txt, PRED_WEEKS, alpha, use_text=use_text
            )
            # Step 2: scores on calibration
            scores = cqr_scores(cal_lo, cal_hi, cal_fut_np)
            # Step 3: Q̂
            Q_hat  = cqr_quantile(scores, alpha)
            # Step 4: base intervals on test set
            test_lo, test_hi, _ = predict_intervals(
                model, test_ctx_np, test_txt, PRED_WEEKS, alpha, use_text=use_text
            )
            # Step 5: inflate
            lo, hi = apply_correction(test_lo, test_hi, Q_hat)

            m = compute_metrics(lo, hi, test_fut_np)
            print(f"{label:>20}  {alpha:>6.2f}  "
                  f"{m['coverage']:>10.3f}  {m['width']:>10.2f}  {Q_hat:>10.2f}")


if __name__ == "__main__":
    main()
