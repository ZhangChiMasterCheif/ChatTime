"""
Rebuttal analyses for the ChatTime / Paris Traffic Flow experiment.

CPU-only: everything runs off the .npz sample caches that ptf_evaluate.py,
ptf_text_ablation.py and ptf_random_ablation.py already wrote. No model load,
no GPU, seconds to run.

Analyses
--------
P1  Joint AND per-step coverage plus the Winkler interval score for every
    method and every text condition. The submission calibrates for joint
    coverage but reports per-step, which is the source of the 0.99-1.00
    coverage numbers in Tables 2 and 3.
P2  Per-step-calibrated CQR, so the reported metric matches the target.
P3  Bootstrap CIs on the with-text vs no-text width gap. The submission claims
    a 10-25% reduction at N=200 windows with no uncertainty attached; this
    gives a paired-bootstrap interval on the difference and the fraction of
    resamples in which text actually helps.
P4  The missing PID/shuffled-text cell in Table 3 (printed as "---").
P5  Sample-budget sweep over K, subsampling the 16 cached paths.
P6  Parse-failure and NaN diagnostics. `predict_samples` leaves a row as NaN
    when a completion fails to parse, and nanquantile then silently computes
    the interval from fewer paths. If failures correlate with the text
    condition, the width comparison across conditions is confounded.
P7  Codebook range diagnostics. ChatTime fits a MinMaxScaler on the context,
    so the codebook spans only [hist_min - 0.5R, hist_max + 1.5R] where R is
    the context range. Targets outside that band are unreachable at any Q,
    which is why Codebook-CQR undercovers on the COVID/ChatTime run
    (chattime_results.csv: 0.879 at a 0.90 nominal level).
P8  Text-condition pairwise comparison with CIs, i.e. the actual statistical
    backing for the "two clusters" claim in Section 5.7.

Usage
-----
    python rebuttal_ptf.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, ".")
import conformal_lib as cl
import finite_k as fk
import codebook_repair as cr
from ptf_dataset import load_ptf, split_ptf, to_window_lists, PRED_LEN

ALPHAS = [0.1, 0.2, 0.3]
N_CAL_WINDOWS = 200
N_TEST_WINDOWS = 200
SUBSAMPLE_SEED = 42

# text condition -> (cal cache, test cache); must match the generating scripts
CONDITIONS = {
    "no_text": ("ptf_samples_no_text_cal.npz", "ptf_samples_no_text_test.npz"),
    "with_text": ("ptf_samples_with_text_cal.npz", "ptf_samples_with_text_test.npz"),
    "shuffled_text": ("cal_ptf_samples_shuffled_text.npz",
                      "test_ptf_samples_shuffled_text.npz"),
    "random_text": ("cal_ptf_samples_random_text.npz",
                    "test_ptf_samples_random_text.npz"),
}

# ChatTime Discretizer defaults (utils/tools.py), replicated so we do not need
# to construct the 7B model just to read its bin edges.
DISC_LOW, DISC_HIGH, DISC_NTOKENS = -1, 1, 10002


def discretizer_arrays():
    boundaries = np.linspace(DISC_LOW, DISC_HIGH, DISC_NTOKENS - 1)
    centers = (boundaries[1:] + boundaries[:-1]) / 2
    centers = np.concatenate((centers[:1], centers, centers[-1:]))
    return boundaries, centers


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_windows():
    """Reproduce exactly the window subsample used by ptf_evaluate.py."""
    df = load_ptf()
    _, cal, test = split_ptf(df)
    cal_ctx, cal_fut, cal_txt = to_window_lists(cal)
    test_ctx, test_fut, test_txt = to_window_lists(test)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    ci = rng.choice(len(cal_ctx), size=min(N_CAL_WINDOWS, len(cal_ctx)), replace=False)
    ti = rng.choice(len(test_ctx), size=min(N_TEST_WINDOWS, len(test_ctx)), replace=False)

    return (
        [cal_ctx[i] for i in ci], np.stack([cal_fut[i] for i in ci]),
        [cal_txt[i] for i in ci],
        [test_ctx[i] for i in ti], np.stack([test_fut[i] for i in ti]),
        [test_txt[i] for i in ti],
    )


def load_condition(cond):
    cal_p, test_p = CONDITIONS[cond]
    if not (os.path.exists(cal_p) and os.path.exists(test_p)):
        return None, None
    return np.load(cal_p)["samples"], np.load(test_p)["samples"]


# ---------------------------------------------------------------------------
# Codebook-CQR from cached samples
# ---------------------------------------------------------------------------

def codebook_cqr(cal_s, cal_ctx, cal_fut, test_s, test_ctx, alpha, fallback=False):
    boundaries, centers = discretizer_arrays()
    n_tokens = len(centers)

    def window_bins(samples_w, ctx, fut=None):
        sc = MinMaxScaler().fit(np.asarray(ctx).reshape(-1, 1))

        def to_bins(v):
            v = np.asarray(v, dtype=float)
            scaled = sc.transform(v.reshape(-1, 1)).reshape(v.shape) - 0.5
            return np.digitize(scaled, boundaries, right=True)

        clean = np.nan_to_num(samples_w, nan=float(np.mean(ctx)))
        sb = to_bins(clean).astype(np.float32)
        blo = np.quantile(sb, alpha / 2, axis=0)
        bhi = np.quantile(sb, 1 - alpha / 2, axis=0)
        tb = to_bins(fut) if fut is not None else None
        return blo, bhi, tb, sc

    cal_lo, cal_hi, cal_true = [], [], []
    for i in range(len(cal_ctx)):
        a, b, t, _ = window_bins(cal_s[i], cal_ctx[i], cal_fut[i])
        cal_lo.append(a); cal_hi.append(b); cal_true.append(t)
    cal_lo, cal_hi, cal_true = np.stack(cal_lo), np.stack(cal_hi), np.stack(cal_true)

    scores = np.maximum(cal_lo - cal_true, cal_true - cal_hi).max(axis=1)
    q = cl.conformal_quantile(scores, alpha)

    n = len(test_ctx)
    lo = np.empty((n, PRED_LEN))
    hi = np.empty((n, PRED_LEN))
    raw_lo = np.empty((n, PRED_LEN))
    raw_hi = np.empty((n, PRED_LEN))

    for i in range(n):
        blo, bhi, _, sc = window_bins(test_s[i], test_ctx[i])
        raw_lo[i], raw_hi[i] = blo - q, bhi + q

        def b2v(b):
            c = centers[np.clip(np.asarray(b).astype(int), 0, len(centers) - 1)]
            return sc.inverse_transform((c + 0.5).reshape(-1, 1)).reshape(c.shape)

        if fallback:
            l, h = cl.codebook_cqr_with_fallback(
                blo[None, :], bhi[None, :], q, n_tokens,
                lambda x: b2v(x[0])[None, :],
            )
            lo[i], hi[i] = l[0], h[0]
        else:
            lo[i] = b2v(np.clip(blo - q, 0, n_tokens - 1))
            hi[i] = b2v(np.clip(bhi + q, 0, n_tokens - 1))

    return lo, hi, raw_lo, raw_hi, q, n_tokens


def codebook_cqr_phi(cal_s, cal_ctx, cal_fut, test_s, test_ctx, alpha):
    """
    Repaired codebook-CQR for ChatTime, in the monotone bin embedding.

    ChatTime's Discretizer applies a per-window MinMax fit on the context and
    then subtracts 0.5, i.e. an affine map z = (y - lo) / (hi - lo) - 0.5. We
    apply that map ourselves, run the embedding CQR of codebook_repair on the
    scaled values, and invert. Because both the affine map and phi are strictly
    increasing, coverage in value space equals coverage in phi space exactly.
    """
    grid = cr.BinGrid.from_chattime()

    def affine(ctx):
        lo = float(np.min(ctx))
        hi = float(np.max(ctx))
        rng = hi - lo if hi > lo else 1.0
        return lo, rng

    def fwd(v, ctx):
        lo, rng = affine(ctx)
        return (np.asarray(v, dtype=np.float64) - lo) / rng - 0.5

    def inv(z, ctx):
        lo, rng = affine(ctx)
        return (np.asarray(z, dtype=np.float64) + 0.5) * rng + lo

    scores = []
    for i in range(len(cal_ctx)):
        s_w = np.nan_to_num(cal_s[i], nan=float(np.mean(cal_ctx[i])))
        z = fwd(s_w, cal_ctx[i])
        p_lo = grid.phi(np.quantile(z, alpha / 2, axis=0))
        p_hi = grid.phi(np.quantile(z, 1 - alpha / 2, axis=0))
        p_y = grid.phi(fwd(cal_fut[i], cal_ctx[i]))
        scores.append(np.maximum(p_lo - p_y, p_y - p_hi).max())
    q = cl.conformal_quantile(np.asarray(scores), alpha)

    n = len(test_ctx)
    lo_arr = np.empty((n, PRED_LEN))
    hi_arr = np.empty((n, PRED_LEN))
    for i in range(n):
        s_w = np.nan_to_num(test_s[i], nan=float(np.mean(test_ctx[i])))
        z = fwd(s_w, test_ctx[i])
        p_lo = grid.phi(np.quantile(z, alpha / 2, axis=0))
        p_hi = grid.phi(np.quantile(z, 1 - alpha / 2, axis=0))
        lo_arr[i] = inv(grid.phi_inv(p_lo - q), test_ctx[i])
        hi_arr[i] = inv(grid.phi_inv(p_hi + q), test_ctx[i])
    return lo_arr, hi_arr


# ---------------------------------------------------------------------------
# P1/P2/P4 : main table
# ---------------------------------------------------------------------------

def table_main(windows):
    cal_ctx, cal_fut, _, test_ctx, test_fut, _ = windows
    rows = {}
    records = []

    for cond in CONDITIONS:
        cal_s, test_s = load_condition(cond)
        if cal_s is None:
            print(f"  [skip] no cache for condition '{cond}'")
            continue

        for alpha in ALPHAS:
            methods = {}
            lo, hi = cl.sample_quantiles(test_s, alpha)
            methods["Naive"] = (lo, hi)

            # At K=16 a 90% interval is not attainable from sample order
            # statistics at all (ceiling 15/17 = 0.882), so the nominal level is
            # the wrong comparator for the naive row. This variant widens the
            # quantile level so a perfectly calibrated model would hit 1-alpha.
            lo, hi, _ = fk.corrected_interval(test_s, test_s.shape[1], alpha)
            methods["Naive (finite-K corrected)"] = (lo, hi)

            methods["CQR (joint)"] = cl.cqr_joint(cal_s, cal_fut, test_s, alpha)
            methods["CQR (per-step)"] = cl.cqr_perstep(cal_s, cal_fut, test_s, alpha)
            methods["CQR (Bonferroni)"] = cl.cqr_bonferroni(cal_s, cal_fut, test_s, alpha)

            l, h, *_ = codebook_cqr(cal_s, cal_ctx, cal_fut, test_s, test_ctx, alpha)
            methods["Codebook-CQR"] = (l, h)
            l, h, *_ = codebook_cqr(cal_s, cal_ctx, cal_fut, test_s, test_ctx,
                                    alpha, fallback=True)
            methods["Codebook-CQR (fallback)"] = (l, h)

            # The repaired variant: CQR in the monotone bin embedding, which is
            # exactly valid in value space and always finite. ChatTime scales
            # per window with MinMax on the context, so the "scale" handed to
            # the repair is the context range and the offset is the context min.
            l, h = codebook_cqr_phi(cal_s, cal_ctx, cal_fut, test_s, test_ctx, alpha)
            methods["Codebook-CQR (phi, repaired)"] = (l, h)

            methods["ACI"] = cl.aci(cal_s, cal_fut, test_s, test_fut, alpha)
            methods["NexCP"] = cl.nexcp(cal_s, cal_fut, test_s, alpha)
            l, h, _ = cl.pid(cal_s, cal_fut, test_s, test_fut, alpha, gain_frac=0.05)
            methods["PID (scaled gain)"] = (l, h)

            for name, (l, h) in methods.items():
                rows[(cond, name, alpha)] = (l, h)
                m = cl.evaluate_intervals(l, h, test_fut, alpha, nonneg=True)
                records.append(
                    dict(
                        text=cond, method=name, alpha=alpha, nominal=round(1 - alpha, 2),
                        K=int(test_s.shape[1]),
                        oracle_finite_K=round(
                            fk.oracle_coverage(test_s.shape[1], alpha), 4),
                        nominal_attainable=(1 - alpha) <= fk.max_order_statistic_coverage(
                            test_s.shape[1]),
                        cov_perstep=m["coverage_perstep"], cov_joint=m["coverage_joint"],
                        width=m["width"], winkler=m["winkler"],
                        frac_lo_neg=m["frac_lo_negative"],
                        width_nonneg=m["width_nonneg"],
                    )
                )
    return pd.DataFrame(records), rows


# ---------------------------------------------------------------------------
# P3/P8 : text-condition comparisons with CIs
# ---------------------------------------------------------------------------

def table_text_contrasts(rows, test_fut, alpha=0.10):
    """
    Paired bootstrap on width differences between text conditions.

    The submission's Section 5.7 conclusion rests on real ~ shuffled and
    random ~ no_text. Those are two claims of NO difference and one claim of a
    difference; none currently carry an interval. At N=200 windows the honest
    version of the claim needs exactly this table.
    """
    contrasts = [
        ("with_text", "no_text"),
        ("shuffled_text", "no_text"),
        ("random_text", "no_text"),
        ("with_text", "shuffled_text"),
        ("with_text", "random_text"),
    ]
    out = []
    for method in ["Naive", "CQR (joint)", "CQR (per-step)", "Codebook-CQR"]:
        for a, b in contrasts:
            ka, kb = (a, method, alpha), (b, method, alpha)
            if ka not in rows or kb not in rows:
                continue
            la, ha = rows[ka]
            lb, hb = rows[kb]
            d, lo_ci, hi_ci, frac = cl.paired_bootstrap_diff(
                la, ha, lb, hb, test_fut, alpha, stat="width", n_boot=1000
            )
            base = cl.evaluate_intervals(lb, hb, test_fut, alpha)["width"]
            out.append(
                dict(
                    method=method, contrast=f"{a} - {b}", alpha=alpha,
                    width_diff=d, ci_lo=lo_ci, ci_hi=hi_ci,
                    pct_change=100 * d / base,
                    frac_boot_narrower=frac,
                    significant=bool(lo_ci * hi_ci > 0),
                )
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# P5 : sample-budget sweep
# ---------------------------------------------------------------------------

def table_k_sweep(test_fut, alpha=0.10, n_rep=10, seed=0):
    """
    Sample-budget sweep, plus a MATCHED-K comparison across text conditions.

    The matched-K table exists because the cached conditions do NOT share a
    sample budget: no_text / with_text / shuffled_text hold 64 paths per window
    while random_text holds 16. Comparing naive widths across those conditions
    at their native budgets confounds the text effect with a Monte Carlo effect,
    since fewer samples give systematically narrower empirical quantiles. The
    matched rows subsample every condition to the smallest available K so the
    contrast is like-for-like.
    """
    rng = np.random.default_rng(seed)
    budgets = {}
    for cond in CONDITIONS:
        _, test_s = load_condition(cond)
        if test_s is not None:
            budgets[cond] = test_s.shape[1]
    if not budgets:
        return pd.DataFrame(), pd.DataFrame(), {}

    K_common = min(budgets.values())

    out, matched = [], []
    for cond, K_max in budgets.items():
        _, test_s = load_condition(cond)
        grid = sorted({4, 8, 12, K_common, K_max} & set(range(1, K_max + 1)))
        for K in grid:
            for r in range(n_rep):
                idx = rng.choice(K_max, size=K, replace=False)
                lo, hi = cl.sample_quantiles(test_s[:, idx, :], alpha)
                m = cl.evaluate_intervals(lo, hi, test_fut, alpha)
                row = dict(text=cond, K=K, rep=r,
                           cov_perstep=m["coverage_perstep"],
                           cov_joint=m["coverage_joint"],
                           width=m["width"],
                           oracle_finite_K=fk.oracle_coverage(K, alpha),
                           gap_vs_oracle=m["coverage_perstep"]
                           - fk.oracle_coverage(K, alpha))
                out.append(row)
                if K == K_common:
                    matched.append(row)
    return pd.DataFrame(out), pd.DataFrame(matched), budgets


# ---------------------------------------------------------------------------
# P6 : parse failures
# ---------------------------------------------------------------------------

def table_parse_failures():
    """
    ChatTime completions that fail to parse are left as an all-NaN row, and the
    interval is then formed from however many paths survived. A condition with
    more failures gets an interval built from fewer effective samples, which
    biases widths in a way that has nothing to do with what the text said.
    """
    out = []
    for cond in CONDITIONS:
        cal_s, test_s = load_condition(cond)
        if cal_s is None:
            continue
        for split, s in [("cal", cal_s), ("test", test_s)]:
            all_nan = np.isnan(s).all(axis=2)          # (N, K) fully failed paths
            any_nan = np.isnan(s).any(axis=2)
            eff_k = (~all_nan).sum(axis=1)
            out.append(
                dict(
                    text=cond, split=split,
                    frac_paths_all_nan=float(all_nan.mean()),
                    frac_paths_any_nan=float(any_nan.mean()),
                    mean_effective_K=float(eff_k.mean()),
                    min_effective_K=int(eff_k.min()),
                    n_windows_with_lt4_paths=int((eff_k < 4).sum()),
                )
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# P7 : codebook reachability
# ---------------------------------------------------------------------------

def table_codebook_reach(windows):
    """
    Fraction of targets that lie outside the representable codebook range.

    ChatTime scales by MinMax on the context, so the bin grid spans scaled
    values in [-1, 1], i.e. real values in
        [hist_min - 0.5R, hist_min + 1.5R],  R = hist_max - hist_min.
    Anything outside is unreachable regardless of the conformal correction.
    """
    _, _, _, test_ctx, test_fut, _ = windows
    out = []
    n_un = 0
    n_win_un = 0
    for i, ctx in enumerate(test_ctx):
        lo_r, hi_r = float(np.min(ctx)), float(np.max(ctx))
        R = hi_r - lo_r if hi_r > lo_r else 1.0
        reach_lo, reach_hi = lo_r - 0.5 * R, lo_r + 1.5 * R
        un = (test_fut[i] < reach_lo) | (test_fut[i] > reach_hi)
        n_un += int(un.sum())
        n_win_un += int(un.any())
    total = test_fut.size
    out.append(
        dict(
            frac_targets_outside_codebook=n_un / total,
            frac_windows_with_unreachable_target=n_win_un / len(test_ctx),
            max_achievable_perstep_coverage=1 - n_un / total,
        )
    )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    windows = load_windows()
    test_fut = windows[4]
    print(f"cal={len(windows[0])} test={len(windows[3])} windows, H={PRED_LEN}")

    print("\n" + "=" * 78)
    print("[P1/P2/P4] joint vs per-step coverage, all methods x text conditions")
    print("=" * 78)
    df, rows = table_main(windows)
    df.to_csv("rebuttal_ptf_results.csv", index=False)
    print(
        df[df.alpha == 0.10]
        .pivot_table(index="method", columns="text",
                     values=["cov_joint", "cov_perstep", "width", "winkler"])
        .to_string(float_format=lambda v: f"{v:.3f}")
    )
    print("\nWrote rebuttal_ptf_results.csv (all alphas)")

    print("\n" + "=" * 78)
    print("[P3/P8] width contrasts between text conditions, paired bootstrap")
    print("=" * 78)
    tc = table_text_contrasts(rows, test_fut)
    print(tc.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    tc.to_csv("rebuttal_ptf_text_contrasts.csv", index=False)

    print("\n" + "=" * 78)
    print("[P5] sample budget K (alpha=0.10)")
    print("=" * 78)
    ks, matched, budgets = table_k_sweep(test_fut)
    print("Sample budget actually present in each cache:")
    for c, k in sorted(budgets.items()):
        print(f"  {c:<16} K = {k}")
    if len(set(budgets.values())) > 1:
        print("\n  WARNING: the conditions do NOT share a sample budget. Naive")
        print("  width comparisons across conditions at native K confound the")
        print("  text effect with a Monte Carlo effect. Use the matched-K table.")
    print()
    print(ks.groupby(["text", "K"])[["cov_perstep", "oracle_finite_K",
                                     "gap_vs_oracle", "width"]]
          .mean().to_string(float_format=lambda v: f"{v:.4f}"))
    if not matched.empty:
        K_common = int(matched["K"].iloc[0])
        print(f"\nMATCHED-K comparison (all conditions subsampled to K={K_common}):")
        print(matched.groupby("text")[["cov_perstep", "width"]]
              .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4f}"))
        matched.to_csv("rebuttal_ptf_ksweep_matched.csv", index=False)
    ks.to_csv("rebuttal_ptf_ksweep.csv", index=False)

    print("\n" + "=" * 78)
    print("[P6] parse-failure diagnostics")
    print("=" * 78)
    pf = table_parse_failures()
    print(pf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pf.to_csv("rebuttal_ptf_parse_failures.csv", index=False)

    print("\n" + "=" * 78)
    print("[P7] codebook reachability ceiling")
    print("=" * 78)
    cr = table_codebook_reach(windows)
    print(cr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cr.to_csv("rebuttal_ptf_codebook_reach.csv", index=False)


if __name__ == "__main__":
    main()
