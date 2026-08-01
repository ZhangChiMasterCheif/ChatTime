"""
Shared conformal utilities for the rebuttal experiments.

This module is model-agnostic and dataset-agnostic: everything operates on
cached sample paths, so nothing here needs a GPU.

Drop an identical copy in ChatTime/ so both repos import the same code.

Core objects
------------
    samples : (N, K, H)  cached sample paths (may contain NaN)
    futures : (N, H)     ground truth
    lo, hi  : (N, H)     interval bounds

What this adds over the original per-method files
-------------------------------------------------
1. `evaluate_intervals`  — reports BOTH joint and per-step coverage, plus the
   Winkler (interval) score and non-negativity diagnostics. The submitted paper
   calibrates for joint coverage but reports per-step, which makes every
   conformal method look like it overcovers by 8-10 points.
2. `cqr_perstep`         — CQR calibrated per horizon step, so the reported
   metric and the calibration target finally agree.
3. `aci`                 — Adaptive Conformal Inference (Gibbs & Candes 2021),
   the standard online baseline the submission cites but never runs.
4. `pid` with `gain_scale` — the original code used an absolute step size of
   0.005 in target units. On series measured in thousands the controller cannot
   move, so "PID" reduced to a fixed-width warm start. Gain is now expressed as
   a fraction of the calibration score scale.
5. `bootstrap_ci`        — paired bootstrap over test windows, so coverage and
   width differences come with error bars.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_intervals(lo, hi, futures, alpha, nonneg=False):
    """
    Full metric set for one (method, alpha) cell.

    Parameters
    ----------
    lo, hi   : (N, H) interval bounds
    futures  : (N, H) ground truth
    alpha    : miscoverage level, used for the Winkler penalty
    nonneg   : if True, also report metrics after clipping lo at 0. Use for
               strictly non-negative targets (hospital admissions, traffic
               counts), where a negative lower bound is uninformative.

    Returns
    -------
    dict with per-step coverage, joint (all-H-steps) coverage, mean width,
    median width, Winkler score, and the fraction of interval bounds that fall
    below zero.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(futures, dtype=float)

    covered = (y >= lo) & (y <= hi)
    width = hi - lo

    # Winkler / interval score: width plus a 2/alpha penalty on each violation.
    # Lower is better, and unlike coverage-plus-width it is a proper score, so
    # it ranks methods that trade width against misses on a single axis.
    winkler = (
        width
        + (2.0 / alpha) * np.maximum(lo - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    )

    out = {
        "coverage_perstep": float(covered.mean()),
        "coverage_joint": float(covered.all(axis=1).mean()),
        "width": float(width.mean()),
        "width_median": float(np.median(width)),
        "winkler": float(winkler.mean()),
        "frac_lo_negative": float((lo < 0).mean()),
        "coverage_perstep_by_h": covered.mean(axis=0),
        "width_by_h": width.mean(axis=0),
    }

    if nonneg:
        lo_c = np.maximum(lo, 0.0)
        cov_c = (y >= lo_c) & (y <= hi)
        w_c = hi - lo_c
        wink_c = (
            w_c
            + (2.0 / alpha) * np.maximum(lo_c - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - hi, 0.0)
        )
        out.update(
            {
                "coverage_perstep_nonneg": float(cov_c.mean()),
                "coverage_joint_nonneg": float(cov_c.all(axis=1).mean()),
                "width_nonneg": float(w_c.mean()),
                "winkler_nonneg": float(wink_c.mean()),
            }
        )
    return out


def bootstrap_ci(lo, hi, futures, alpha, n_boot=2000, seed=0, stat="coverage_joint"):
    """
    Percentile bootstrap CI for one metric, resampling test WINDOWS (not
    individual horizon steps -- steps within a window are strongly dependent,
    so resampling them would give dishonestly narrow intervals).

    Returns (point_estimate, lo_2.5%, hi_97.5%).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(futures, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    point = evaluate_intervals(lo, hi, y, alpha)[stat]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = evaluate_intervals(lo[idx], hi[idx], y[idx], alpha)[stat]
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(lo_a, hi_a, lo_b, hi_b, futures, alpha,
                          stat="width", n_boot=2000, seed=0):
    """
    Paired bootstrap on the difference (method A - method B) for one metric.
    Both methods must be evaluated on the SAME test windows in the same order.

    Use this for the with-text vs no-text width comparison: the submission
    reports a 10-25% width reduction with no uncertainty attached to it, which
    is the first thing a reviewer will ask about at N=200 windows.

    Returns (diff, lo_2.5%, hi_97.5%, frac_of_bootstrap_draws_favoring_A).

    Note: the calibration threshold Q-hat is held FIXED across resamples, so
    this is a statement about test-set uncertainty only. When two methods share
    a base interval and differ solely by a constant Q-hat, the width difference
    is deterministic and the CI collapses to a point -- that is correct, not a
    bug. It is informative for the text-condition contrasts, where the two arms
    have genuinely different sample paths.
    """
    y = np.asarray(futures, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    def s(l, h, idx):
        return evaluate_intervals(l[idx], h[idx], y[idx], alpha)[stat]

    full = np.arange(n)
    point = s(lo_a, hi_a, full) - s(lo_b, hi_b, full)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = s(lo_a, hi_a, idx) - s(lo_b, hi_b, idx)
    return (
        float(point),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        float((diffs < 0).mean()),
    )


# ---------------------------------------------------------------------------
# Conformal quantile
# ---------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """
    Finite-sample valid conformal threshold: the ceil((1-alpha)(N+1))/N
    empirical quantile of the calibration scores.

    Returns +inf when the level exceeds 1 (N too small for the requested
    alpha), which is the correct degenerate answer rather than silently
    clipping to the max score.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    level = np.ceil((1 - alpha) * (n + 1)) / n
    if level > 1.0:
        return np.inf
    return float(np.quantile(scores, level))


# ---------------------------------------------------------------------------
# Base quantiles from cached samples
# ---------------------------------------------------------------------------

def _has_nan(a):
    """Cheap NaN test; np.isnan over the whole array is ~40x cheaper than
    paying for the nan-aware reduction when there is nothing to skip."""
    return bool(np.isnan(a).any())


def sample_quantiles(samples, alpha):
    """
    (N, K, H) samples -> (q_lo, q_hi), each (N, H).

    Dispatches to np.quantile when the array is NaN-free. On a (2350, 100, 14)
    Chronos cache np.nanquantile costs ~4 s against ~0.1 s for np.quantile, and
    this helper is called dozens of times per analysis, so the branch is the
    difference between a script that runs in seconds and one that runs in tens
    of minutes. ChatTime caches genuinely do contain NaN (failed completions),
    so the nan-aware path is kept for them -- their arrays are small enough that
    the cost does not matter.
    """
    q = np.nanquantile if _has_nan(samples) else np.quantile
    return q(samples, alpha / 2.0, axis=1), q(samples, 1 - alpha / 2.0, axis=1)


def sample_median(samples):
    """Per-step median over sample paths, with the same NaN fast path."""
    return (np.nanmedian if _has_nan(samples) else np.median)(samples, axis=1)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def cqr_joint(cal_s, cal_fut, test_s, alpha):
    """
    Split CQR with a max-over-horizon score. This is the submission's method:
    it targets JOINT coverage over all H steps.
    """
    cal_lo, cal_hi = sample_quantiles(cal_s, alpha)
    scores = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi).max(axis=1)
    q = conformal_quantile(scores, alpha)

    test_lo, test_hi = sample_quantiles(test_s, alpha)
    return test_lo - q, test_hi + q


def cqr_perstep(cal_s, cal_fut, test_s, alpha):
    """
    Split CQR calibrated SEPARATELY at each horizon step h.

    This targets marginal per-step coverage, which is the quantity the
    submission's tables actually report. Adding this row removes the
    apples-to-oranges gap that makes every method look 8-10 points
    overcovering, and it is strictly tighter than the joint variant.
    """
    cal_lo, cal_hi = sample_quantiles(cal_s, alpha)
    per_step = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi)  # (N, H)

    test_lo, test_hi = sample_quantiles(test_s, alpha)
    lo = np.empty_like(test_lo)
    hi = np.empty_like(test_hi)
    for h in range(per_step.shape[1]):
        q_h = conformal_quantile(per_step[:, h], alpha)
        lo[:, h] = test_lo[:, h] - q_h
        hi[:, h] = test_hi[:, h] + q_h
    return lo, hi


def cqr_bonferroni(cal_s, cal_fut, test_s, alpha):
    """
    Per-step calibration at level alpha/H, which gives a JOINT guarantee by a
    union bound. Included as the honest alternative route to joint coverage:
    it is the natural competitor to the max-over-horizon score and is usually
    either tighter or wider depending on how correlated the horizon steps are.
    """
    H = cal_fut.shape[1]
    return cqr_perstep(cal_s, cal_fut, test_s, alpha / H)


def aci(cal_s, cal_fut, test_s, test_fut, alpha, gamma_scale=0.05):
    """
    Adaptive Conformal Inference (Gibbs & Candes, 2021).

    Maintains a running effective miscoverage alpha_t and re-reads the
    calibration score distribution at that level each step:

        alpha_{t+1} = alpha_t + gamma * (alpha - 1{miss_t})

    The submission cites ACI but never runs it, and PID alone is a thin online
    baseline. `gamma_scale` is a plain rate on alpha (dimensionless), so unlike
    the PID gain it needs no unit conversion.
    """
    cal_lo, cal_hi = sample_quantiles(cal_s, alpha)
    cal_scores = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi).max(axis=1)
    cal_scores = cal_scores[np.isfinite(cal_scores)]

    test_lo, test_hi = sample_quantiles(test_s, alpha)
    n, H = test_lo.shape
    lo = np.empty((n, H))
    hi = np.empty((n, H))

    a_t = alpha
    for t in range(n):
        a_eff = float(np.clip(a_t, 1e-3, 1 - 1e-3))
        q = conformal_quantile(cal_scores, a_eff)
        if not np.isfinite(q):
            q = float(cal_scores.max())
        lo[t] = test_lo[t] - q
        hi[t] = test_hi[t] + q

        covered = bool(np.all((test_fut[t] >= lo[t]) & (test_fut[t] <= hi[t])))
        a_t = a_t + gamma_scale * (alpha - (0.0 if covered else 1.0))
    return lo, hi


def nexcp(cal_s, cal_fut, test_s, alpha, rho=0.99):
    """
    NexCP -- nonexchangeable conformal prediction (Barber, Candes, Ramdas &
    Tibshirani, 2023).

    Instead of the unweighted empirical quantile, use weights that decay
    geometrically into the past,

        w_i  proportional to  rho^(N - i),

    so recent calibration windows count for more. This is the standard
    distribution-shift-tolerant split conformal method and it is the one the
    submission cites (as "decayed weights to recent calibration scores") but
    never runs. Reviewer sFBd asked specifically for stronger time-series
    conformal baselines; this is the cheapest strong one, and like everything
    else here it is pure post-processing on the cached samples.

    Calibration windows must be in chronological order for the weighting to
    mean anything -- which they are, since the split is chronological.

    rho = 0.99 gives an effective memory of roughly 100 windows; rho = 1
    recovers ordinary split CQR.
    """
    cal_lo, cal_hi = sample_quantiles(cal_s, alpha)
    scores = np.maximum(cal_lo - cal_fut, cal_fut - cal_hi).max(axis=1)
    ok = np.isfinite(scores)
    scores = scores[ok]
    n = len(scores)
    if n == 0:
        raise ValueError("no finite calibration scores")

    w = rho ** np.arange(n - 1, -1, -1, dtype=float)
    # The test point carries weight 1 in the Barber et al. construction; the
    # threshold is the smallest score whose cumulative weight reaches 1-alpha.
    w = w / (w.sum() + 1.0)

    order = np.argsort(scores)
    cw = np.cumsum(w[order])
    idx = np.searchsorted(cw, 1 - alpha)
    q = float(scores[order][min(idx, n - 1)]) if idx < n else float(scores.max())

    test_lo, test_hi = sample_quantiles(test_s, alpha)
    return test_lo - q, test_hi + q


def pid(cal_s, cal_fut, test_s, test_fut, alpha, gain_frac=0.05):
    """
    Conformal PI control with a SCALE-CORRECTED gain.

    The original implementation used k_i = 0.005 in absolute target units. On
    PTF (thousands of vehicles/hour) and on COVID admissions this makes the
    controller effectively frozen: over 200 test windows q can move by at most
    200 * 0.005 = 1.0 unit against widths of ~550, so "conformal PID" in the
    submission is really a fixed-width method equal to its warm start.

    Here the gain is `gain_frac` times the warm-start half-width, so the
    controller moves on the scale of the data, as intended in Angelopoulos et
    al. (2023). gain_frac=0.05 means a single miss widens the interval by ~5%.

    Returns (lo, hi, q_trace) so the controller path can be plotted -- a
    frozen-controller plot is the clearest way to show the fix mattered.
    """
    med_cal = sample_median(cal_s)
    resid = np.abs(cal_fut - med_cal).max(axis=1)
    q0 = float(np.nanquantile(resid, 1.0 - alpha))
    eta = gain_frac * q0

    med_test = sample_median(test_s)
    n, H = med_test.shape
    lo = np.empty((n, H))
    hi = np.empty((n, H))
    q_trace = np.empty(n)

    q = q0
    for t in range(n):
        lo[t] = med_test[t] - q
        hi[t] = med_test[t] + q
        q_trace[t] = q
        covered = bool(np.all((test_fut[t] >= lo[t]) & (test_fut[t] <= hi[t])))
        q = float(np.clip(q + eta * ((0.0 if covered else 1.0) - alpha), 0.0, None))
    return lo, hi, q_trace


# ---------------------------------------------------------------------------
# Codebook diagnostics
# ---------------------------------------------------------------------------

def clipping_diagnostics(bin_lo_raw, bin_hi_raw, true_bins, n_tokens):
    """
    How often does codebook clipping actually bind, and how often does it cost
    coverage?

    Section 4.5 of the submission argues that clipping to the codebook boundary
    "can only expand the intervals ... and never contract them". That is
    backwards: clip(bin_hi + Q, 0, V-1) TRUNCATES the upper edge at V-1, so
    once Q pushes past the boundary the interval stops growing. Any target
    whose bin index lies outside [0, V-1] can then never be covered, no matter
    how large Q is, and the finite-sample guarantee does not survive.

    This function quantifies the exposure so the claim can be replaced with a
    measured statement rather than simply deleted.

    Parameters
    ----------
    bin_lo_raw, bin_hi_raw : (N, H) inflated bin bounds BEFORE clipping
    true_bins              : (N, H) tokenized targets
    n_tokens               : codebook size V

    Returns dict of rates.
    """
    lo = np.asarray(bin_lo_raw, dtype=float)
    hi = np.asarray(bin_hi_raw, dtype=float)
    tb = np.asarray(true_bins, dtype=float)

    hi_clipped = hi > (n_tokens - 1)
    lo_clipped = lo < 0
    truth_above = tb >= (n_tokens - 1)
    truth_below = tb <= 0
    unreachable = truth_above | truth_below

    return {
        "frac_upper_clipped": float(hi_clipped.mean()),
        "frac_lower_clipped": float(lo_clipped.mean()),
        "frac_windows_any_clip": float((hi_clipped | lo_clipped).any(axis=1).mean()),
        "frac_target_at_codebook_edge": float(unreachable.mean()),
        "frac_windows_with_unreachable_target": float(unreachable.any(axis=1).mean()),
    }


def codebook_cqr_with_fallback(bin_lo, bin_hi, q_bins, n_tokens,
                               bins_to_values_fn, residual_q=0.0):
    """
    Codebook-CQR variant that restores the coverage guarantee.

    Rather than truncating at the codebook edge, any inflation that would run
    past the boundary is converted back to value space and applied as an
    additive residual expansion there. The interval stays finite (the codebook
    range is still the bulk of it) but is no longer capped, so the standard
    split-conformal argument goes through unchanged.

    `bins_to_values_fn(bin_array) -> value_array` supplies the model-specific
    inverse tokenizer (Chronos centers*scale, ChatTime centers+0.5 then
    inverse MinMax).
    """
    lo_over = np.maximum(0.0 - (bin_lo - q_bins), 0.0)
    hi_over = np.maximum((bin_hi + q_bins) - (n_tokens - 1), 0.0)

    lo_clipped = np.clip(bin_lo - q_bins, 0, n_tokens - 1)
    hi_clipped = np.clip(bin_hi + q_bins, 0, n_tokens - 1)

    lo_val = bins_to_values_fn(lo_clipped)
    hi_val = bins_to_values_fn(hi_clipped)

    # Value-space width of one bin at the boundary, used to convert the
    # overflow in bin units into a value-space expansion.
    edge_lo = bins_to_values_fn(np.zeros_like(lo_clipped))
    edge_lo1 = bins_to_values_fn(np.ones_like(lo_clipped))
    edge_hi = bins_to_values_fn(np.full_like(hi_clipped, n_tokens - 1))
    edge_hi1 = bins_to_values_fn(np.full_like(hi_clipped, n_tokens - 2))

    step_lo = np.abs(edge_lo1 - edge_lo)
    step_hi = np.abs(edge_hi - edge_hi1)

    lo_val = lo_val - lo_over * step_lo - residual_q
    hi_val = hi_val + hi_over * step_hi + residual_q
    return lo_val, hi_val
