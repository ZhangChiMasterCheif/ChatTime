"""
Conformal PID Control for ChatTime — Angelopoulos, Candès, Tibshirani (2023).
https://arxiv.org/abs/2307.16895

Online adaptive conformal prediction. A PI controller adjusts the interval
half-width q at every step based on running coverage error:

    err_t  = α − 1{y_t ∉ Ĉ_t}
    q_{t+1} = clip(q_t − K_I · err_t, 0, ∞)

If we keep covering, q shrinks (intervals tighten). If we miss, q grows.
At steady state, joint per-window coverage equals the nominal rate (1 − α).

This implementation is purely a post-processing step on already-cached
ChatTime samples — no model calls. We:
  1. Use the cal samples to warm-start q₀ (the (1−α)-quantile of cal residuals).
  2. Walk through test windows in their list order, maintaining a single
     running q; for each window we form a symmetric interval around the
     sample median, observe coverage, and update q.

Notes
-----
- Coverage is JOINT (all pred_len steps inside) for the PI update, matching
  the Chronos PID implementation. The metric reported by ptf_evaluate is
  per-step (marginal) coverage, which will be ≥ the joint rate the controller
  is calibrating for. This is consistent with how the other methods are
  reported.
- This module supports use_text via whichever sample cache you pass in
  (`cal_s` / `test_s` from the no-text or with-text condition).
"""

import numpy as np

K_I = 0.005      # integral gain (from the paper)


def warmstart_q0(cal_s: np.ndarray, cal_fut: np.ndarray, alpha: float) -> float:
    """
    Compute a sensible starting q from calibration residuals.

    cal_s   : (N, num_samples, pred_len) — cached ChatTime samples
    cal_fut : (N, pred_len)              — calibration ground truth
    """
    median = np.nanmedian(cal_s, axis=1)                     # (N, pred_len)
    errors = np.abs(cal_fut - median).max(axis=1)            # (N,)  worst-step
    return float(np.nanquantile(errors, 1.0 - alpha))


def run_pid(
    cal_s:   np.ndarray,
    test_s:  np.ndarray,
    cal_fut: np.ndarray,
    test_fut: np.ndarray,
    alpha:   float,
    k_i:     float = K_I,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Online conformal PI control over the test windows.

    Returns
    -------
    lo, hi : (N_test, pred_len)  — per-step prediction interval bounds
    """
    q = warmstart_q0(cal_s, cal_fut, alpha)

    test_median = np.nanmedian(test_s, axis=1)               # (N, pred_len)
    N, pred_len = test_median.shape

    lo_arr = np.empty((N, pred_len))
    hi_arr = np.empty((N, pred_len))

    for t in range(N):
        med = test_median[t]
        lo  = med - q
        hi  = med + q
        lo_arr[t] = lo
        hi_arr[t] = hi

        # joint coverage check: all steps inside
        fut     = test_fut[t]
        covered = bool(np.all((fut >= lo) & (fut <= hi)))

        # PI update on q (symmetric half-width)
        err = alpha - (0 if covered else 1)
        q   = float(np.clip(q - k_i * err, a_min=0.0, a_max=None))

    return lo_arr, hi_arr
