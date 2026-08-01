"""
Finite-sample correction for sample-quantile ("self-consistency") intervals.

Answers reviewer tjor's Q2 and metareview point 4:

  "With only K=16 samples, empirical 5th/95th percentiles need not define a 90%
   interval even under a perfectly calibrated model, so a finite-sample
   correction or a sweep over K is needed before calling the observed 76%
   coverage severe miscalibration."

The reviewer is right, and the effect is large. Two facts, both distribution-free
or nearly so:

1. HARD CEILING. For K exchangeable samples and an independent target from the
   same distribution, the interval spanned by the smallest and largest samples
   covers the target with probability exactly (K-1)/(K+1). Nothing built from K
   order statistics can do better. For K=16 that is 15/17 = 0.882, so a 90%
   interval is IMPOSSIBLE from 16 samples no matter how well calibrated the
   model is.

2. INTERPOLATION LOSS. numpy's default linear-interpolation quantile is
   noticeably narrower than the order-statistic bound. Simulated expected
   coverage of the [alpha/2, 1-alpha/2] sample quantiles under a perfectly
   calibrated model:

       K     a=0.10   a=0.20   a=0.30
       8     0.723    0.645    0.546
      16     0.806    0.715    0.622
      32     0.851    0.753    0.660
      64     0.874    0.775    0.677
     100     0.882    0.785    0.686

   These vary by under 1.5 points across Gaussian, lognormal, t(3), exponential
   and bimodal targets, so they are safe to quote as model-free reference values.

Consequences for the paper
--------------------------
The correct benchmark for naive self-consistency is not the nominal level but
the ORACLE level above. Recomputing the gaps:

  ChatTime/PTF, K=16, alpha=0.10: oracle 0.806 vs observed 0.759 -> 4.7 points,
  not the 14 points implied by comparing against 0.90.
  Chronos/COVID, K=100, alpha=0.10: oracle 0.882 vs observed 0.788 -> 9.4 points.

The undercoverage is still real, still present at every alpha, and still present
on both models -- but roughly a third of the ChatTime gap and none of the
"ten or more percentage points" framing survives as stated. Section 1 and
Section 5.4 need rewording, and this module provides the corrected baseline to
report alongside.

`corrected_level` additionally solves for the quantile level whose expected
coverage equals the nominal target, giving a properly finite-K-corrected naive
method rather than only a diagnostic.

Usage
-----
    python finite_k.py                    # print the reference tables
"""

import numpy as np

_CACHE = {}


def max_order_statistic_coverage(K):
    """
    Exact coverage of [min, max] of K samples for an independent target from
    the same continuous distribution: (K-1)/(K+1). Distribution-free.

    Any interval built from the K order statistics is contained in this one, so
    this is a hard upper bound on achievable coverage at sample budget K.
    """
    return (K - 1) / (K + 1)


def _interp_quantile_sorted(s_sorted, beta):
    """
    numpy's default linear-interpolation quantile, evaluated on an array that is
    ALREADY sorted along axis 1.

    Identical to `np.quantile(s, beta, axis=1)` to floating-point precision, but
    it skips the internal partition/sort. That matters because the bisection in
    `corrected_level` evaluates this tens of times on the same array, and a full
    np.quantile call on (50000, 100) costs ~1 s against ~0.02 s here.
    """
    K = s_sorted.shape[1]
    pos = float(beta) * (K - 1)
    i = int(np.floor(pos))
    i = max(0, min(i, K - 1))
    j = min(i + 1, K - 1)
    f = pos - i
    return s_sorted[:, i] + f * (s_sorted[:, j] - s_sorted[:, i])


def _reference_draw(K, n_sim, seed):
    """Sorted samples plus an independent target, cached across calls."""
    key = ("draw", K, n_sim, seed)
    if key not in _CACHE:
        rng = np.random.default_rng(seed)
        s = np.sort(rng.standard_normal((n_sim, K)), axis=1)
        y = rng.standard_normal(n_sim)
        _CACHE[key] = (s, y)
    return _CACHE[key]


def oracle_coverage(K, alpha, n_sim=200000, seed=0, method="linear"):
    """
    Expected coverage of the [alpha/2, 1-alpha/2] sample-quantile interval when
    the K samples and the target are i.i.d. from the SAME distribution, i.e.
    when the model is perfectly calibrated.

    Simulated under a standard normal. The result is a probability integral
    transform of the order statistics, so it is distribution-free up to the
    interpolation between neighbouring order statistics; the residual
    dependence is under 1.5 points across very different shapes (see module
    docstring).
    """
    key = (K, round(alpha, 6), n_sim, seed, method)
    if key in _CACHE:
        return _CACHE[key]

    s, y = _reference_draw(K, n_sim, seed)
    lo = _interp_quantile_sorted(s, alpha / 2)
    hi = _interp_quantile_sorted(s, 1 - alpha / 2)

    cov = float(((y >= lo) & (y <= hi)).mean())
    _CACHE[key] = cov
    return cov


_LEVEL_CACHE = {}


def corrected_level(K, alpha, n_sim=50000, seed=0, tol=1e-3, max_iter=30):
    """
    Solve for the level beta such that the [beta, 1-beta] sample-quantile
    interval has expected coverage 1-alpha under a perfectly calibrated model.

    This yields a finite-K CORRECTED naive baseline: use `corrected_level` in
    place of alpha/2 when forming the interval, and the resulting method is
    genuinely targeting 1-alpha at sample budget K.

    Returns
    -------
    beta : float, or None when the target is unattainable at this K (i.e.
           1-alpha exceeds (K-1)/(K+1)), in which case no choice of sample
           quantiles can reach the nominal level and the honest answer is to
           report the ceiling instead.
    """
    key = (K, round(alpha, 6), n_sim, seed)
    if key in _LEVEL_CACHE:
        return _LEVEL_CACHE[key]

    target = 1.0 - alpha
    if target > max_order_statistic_coverage(K):
        _LEVEL_CACHE[key] = None
        return None

    s, y = _reference_draw(K, n_sim, seed)

    def coverage(beta):
        lo = _interp_quantile_sorted(s, beta)
        hi = _interp_quantile_sorted(s, 1 - beta)
        return float(((y >= lo) & (y <= hi)).mean())

    lo_b, hi_b = 0.0, alpha / 2.0            # smaller beta -> wider -> more coverage
    for _ in range(max_iter):
        mid = 0.5 * (lo_b + hi_b)
        c = coverage(mid)
        if abs(c - target) < tol:
            _LEVEL_CACHE[key] = mid
            return mid
        if c < target:
            hi_b = mid
        else:
            lo_b = mid
    beta = 0.5 * (lo_b + hi_b)
    _LEVEL_CACHE[key] = beta
    return beta


def corrected_interval(samples, K_effective, alpha, beta=None):
    """
    Finite-K corrected naive interval from cached sample paths.

    Parameters
    ----------
    samples     : (N, K, H)
    K_effective : sample budget to correct for (usually samples.shape[1])
    beta        : precomputed level from `corrected_level`; solved if omitted.

    Returns (lo, hi, beta). When the nominal level is unattainable at this K,
    falls back to the widest available interval (min and max of the samples) and
    returns beta = 0.0, which is the most any K-sample method can do.
    """
    if beta is None:
        beta = corrected_level(K_effective, alpha)
    nan = bool(np.isnan(samples).any())
    if beta is None:
        mn, mx = (np.nanmin, np.nanmax) if nan else (np.min, np.max)
        return (mn(samples, axis=1), mx(samples, axis=1), 0.0)
    # np.nanquantile is ~40x slower and has nothing to skip on a NaN-free cache.
    q = np.nanquantile if nan else np.quantile
    return q(samples, beta, axis=1), q(samples, 1 - beta, axis=1), beta


def gap_table(observed, K, alphas):
    """
    Re-express observed naive coverages as gaps against the finite-K oracle
    rather than against the nominal level.

    `observed` maps alpha -> measured per-step coverage.
    """
    rows = []
    for a in alphas:
        orc = oracle_coverage(K, a)
        obs = observed.get(a)
        rows.append(dict(
            alpha=a, nominal=round(1 - a, 3),
            oracle_finite_K=round(orc, 4),
            observed=None if obs is None else round(obs, 4),
            gap_vs_nominal=None if obs is None else round(obs - (1 - a), 4),
            gap_vs_oracle=None if obs is None else round(obs - orc, 4),
            ceiling=round(max_order_statistic_coverage(K), 4),
            nominal_attainable=(1 - a) <= max_order_statistic_coverage(K),
        ))
    return rows


def _print_tables():
    alphas = [0.05, 0.10, 0.20, 0.30, 0.40]
    ks = [8, 16, 32, 64, 100]

    print("Expected coverage of naive sample-quantile intervals under a")
    print("PERFECTLY CALIBRATED model (linear-interpolation quantiles).\n")
    head = "".join(f"{1 - a:>9.2f}" for a in alphas)
    print(f"{'K':>5}{head}{'ceiling':>10}")
    print("-" * (5 + 9 * len(alphas) + 10))
    for K in ks:
        row = "".join(f"{oracle_coverage(K, a):>9.4f}" for a in alphas)
        print(f"{K:>5}{row}{max_order_statistic_coverage(K):>10.4f}")

    print("\n'ceiling' = coverage of [min, max] of K samples = (K-1)/(K+1),")
    print("a hard distribution-free upper bound at that sample budget.\n")

    print("Finite-K corrected level beta (use in place of alpha/2):\n")
    print(f"{'K':>5}{head}")
    print("-" * (5 + 9 * len(alphas)))
    for K in ks:
        cells = []
        for a in alphas:
            b = corrected_level(K, a, n_sim=60000)
            cells.append(f"{b:>9.4f}" if b is not None else f"{'n/a':>9}")
        print(f"{K:>5}" + "".join(cells))
    print("\n'n/a' means the nominal level is unattainable from K samples.\n")

    print("Re-scored submission numbers:\n")
    print("ChatTime / PTF, K=16, naive no-text per-step coverage")
    for r in gap_table({0.1: 0.7592, 0.2: 0.6565, 0.3: 0.5715}, 16,
                       [0.1, 0.2, 0.3]):
        print(f"  alpha={r['alpha']:.2f}  nominal={r['nominal']:.2f}  "
              f"oracle={r['oracle_finite_K']:.3f}  observed={r['observed']:.3f}  "
              f"gap vs nominal={r['gap_vs_nominal']:+.3f}  "
              f"gap vs oracle={r['gap_vs_oracle']:+.3f}"
              + ("" if r["nominal_attainable"] else "   [nominal UNATTAINABLE at K=16]"))

    print("\nChronos / COVID, K=100, naive per-step coverage")
    for r in gap_table({0.05: 0.8453, 0.1: 0.7876, 0.2: 0.6826,
                        0.3: 0.5910, 0.4: 0.4998}, 100, alphas):
        print(f"  alpha={r['alpha']:.2f}  nominal={r['nominal']:.2f}  "
              f"oracle={r['oracle_finite_K']:.3f}  observed={r['observed']:.3f}  "
              f"gap vs nominal={r['gap_vs_nominal']:+.3f}  "
              f"gap vs oracle={r['gap_vs_oracle']:+.3f}")


if __name__ == "__main__":
    _print_tables()
