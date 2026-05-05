"""
Unified evaluation of all ChatTime methods.

Loads ChatTime + sentence-BERT once, generates samples for every
(window × text-condition) combination once, caches them to disk, then runs
all four methods × {with-text, no-text} × all alphas as cheap post-processing
on the cached samples.

Re-runs skip any (method, text, alpha) combination already in
chattime_results.csv. Use --reset to wipe and start over.

Outputs
-------
  chattime_samples_no_text.npz   } cached raw samples (one ChatTime forward
  chattime_samples_with_text.npz } pass per window per text-condition)
  chattime_results.csv           : coverage and width per method × alpha
  chattime_calibration.png       : empirical coverage vs nominal
  chattime_width.png             : interval width vs nominal

Usage
-----
    python chattime_evaluate.py                # run missing, then plot
    python chattime_evaluate.py --plots-only   # skip experiments, regen plots
    python chattime_evaluate.py --reset        # wipe caches + CSV first
"""

import argparse
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
from chattime_method_naive import (
    predict_samples, compute_metrics,
    MODEL_PATH, CONTEXT_WEEKS, PRED_WEEKS, NUM_SAMPLES, ALPHAS,
    EXAMPLE_STATES, STRIDE,
)
from chattime_method_cqr import cqr_scores, cqr_quantile, apply_correction
from chattime_method_codebook_cqr import (
    fit_scaler, to_bins, codebook_scores,
    apply_codebook_correction, bins_to_values,
)
from chattime_method_text_cqr import (
    embed_texts, text_weights, weighted_quantile, SBERT_MODEL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH         = "chattime_results.csv"
SAMPLES_NO_TEXT  = "chattime_samples_no_text.npz"
SAMPLES_WITH_TXT = "chattime_samples_with_text.npz"

# (method_name, text_condition) pairs we will evaluate
METHODS = [
    ("Naive",         "no_text"),
    ("Naive",         "with_text"),
    ("CQR",           "no_text"),
    ("CQR",           "with_text"),
    ("Codebook-CQR",  "no_text"),
    ("Codebook-CQR",  "with_text"),
    ("Text-CQR",      "with_text"),   # text-weighting only makes sense with text
]


# ---------------------------------------------------------------------------
# CSV bookkeeping
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=["method", "text", "alpha", "nominal",
                                  "coverage", "width"])


def already_done(df, method, text, alpha):
    if df.empty:
        return False
    return (
        (df["method"] == method) & (df["text"] == text) & (df["alpha"] == alpha)
    ).any()


def append_result(method, text, alpha, coverage, width):
    row = pd.DataFrame([{
        "method":   method,
        "text":     text,
        "alpha":    alpha,
        "nominal":  round(1 - alpha, 2),
        "coverage": round(coverage, 4),
        "width":    round(width, 2),
    }])
    write_header = not os.path.exists(CSV_PATH)
    row.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Sample cache (the only expensive thing)
# ---------------------------------------------------------------------------

def get_samples(model, contexts, texts, pred_len, use_text, cache_path):
    """
    Run ChatTime for every window and cache (N, num_samples, pred_len).
    """
    if os.path.exists(cache_path):
        print(f"  loading cached samples: {cache_path}")
        return np.load(cache_path)["samples"]

    print(f"  generating samples ({'with text' if use_text else 'no text'}) ...")
    N = len(contexts)
    out = np.full((N, NUM_SAMPLES, pred_len), np.nan, dtype=np.float32)
    for i, (ctx, txt) in enumerate(zip(contexts, texts)):
        ctx_text = txt if (use_text and txt) else None
        out[i] = predict_samples(model, ctx, pred_len, ctx_text, NUM_SAMPLES)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N}")
    np.savez(cache_path, samples=out)
    print(f"  cached → {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Method runners — all operate on cached samples, no model calls
# ---------------------------------------------------------------------------

def run_naive(cal_s, test_s, cal_fut, test_fut, alpha):
    test_lo = np.nanquantile(test_s, alpha / 2,     axis=1)
    test_hi = np.nanquantile(test_s, 1 - alpha / 2, axis=1)
    return test_lo, test_hi


def run_cqr(cal_s, test_s, cal_fut, test_fut, alpha):
    cal_lo  = np.nanquantile(cal_s, alpha / 2,     axis=1)
    cal_hi  = np.nanquantile(cal_s, 1 - alpha / 2, axis=1)
    scores  = cqr_scores(cal_lo, cal_hi, cal_fut)
    Q_hat   = cqr_quantile(scores, alpha)

    test_lo = np.nanquantile(test_s, alpha / 2,     axis=1)
    test_hi = np.nanquantile(test_s, 1 - alpha / 2, axis=1)
    return apply_correction(test_lo, test_hi, Q_hat)


def run_codebook_cqr(cal_s, test_s, cal_ctx, test_ctx, cal_fut, test_fut,
                     discretizer, alpha):
    """
    Per-window: refit MinMaxScaler on context, convert samples + future to
    bin IDs, compute CQR in bin space, apply, convert back.
    """
    centers    = discretizer.centers
    boundaries = discretizer.boundaries
    n_tokens   = len(centers)

    def per_window_bins(samples_w, ctx, fut):
        scaler   = fit_scaler(ctx)
        # samples may have NaN — replace with the context mean before binning
        s_clean  = np.nan_to_num(samples_w, nan=float(ctx.mean()))
        s_bins   = to_bins(s_clean, scaler, boundaries).astype(np.float32)
        bin_lo   = np.quantile(s_bins, alpha / 2,     axis=0)
        bin_hi   = np.quantile(s_bins, 1 - alpha / 2, axis=0)
        true_b   = to_bins(fut, scaler, boundaries)
        return bin_lo, bin_hi, true_b, scaler

    # Calibration → scores
    cal_bin_lo, cal_bin_hi, cal_true, _ = zip(*[
        per_window_bins(cal_s[i], cal_ctx[i], cal_fut[i])
        for i in range(len(cal_ctx))
    ])
    cal_bin_lo = np.stack(cal_bin_lo)
    cal_bin_hi = np.stack(cal_bin_hi)
    cal_true   = np.stack(cal_true)
    scores     = codebook_scores(cal_bin_lo, cal_bin_hi, cal_true)
    Q_hat_bin  = cqr_quantile(scores, alpha)

    # Test
    lo_arr = np.empty((len(test_ctx), PRED_WEEKS))
    hi_arr = np.empty((len(test_ctx), PRED_WEEKS))
    for i in range(len(test_ctx)):
        bl, bh, _, sc = per_window_bins(test_s[i], test_ctx[i], test_fut[i])
        lo_b, hi_b    = apply_codebook_correction(bl, bh, Q_hat_bin, n_tokens)
        lo_arr[i]     = bins_to_values(lo_b.astype(int), sc, centers)
        hi_arr[i]     = bins_to_values(hi_b.astype(int), sc, centers)
    return lo_arr, hi_arr


def run_text_cqr(cal_s, test_s, cal_fut, test_fut, cal_embs, test_embs, alpha):
    pred_len = cal_fut.shape[1]

    cal_lo  = np.nanquantile(cal_s, alpha / 2,     axis=1)
    cal_hi  = np.nanquantile(cal_s, 1 - alpha / 2, axis=1)
    scores  = cqr_scores(cal_lo, cal_hi, cal_fut)            # (N * pred_len,)

    test_lo = np.nanquantile(test_s, alpha / 2,     axis=1)
    test_hi = np.nanquantile(test_s, 1 - alpha / 2, axis=1)

    N = len(test_lo)
    lo_all = np.empty_like(test_lo)
    hi_all = np.empty_like(test_hi)
    for j in range(N):
        w_window = text_weights(test_embs[j], cal_embs)        # (N_cal,)
        w_step   = np.repeat(w_window, pred_len)               # (N_cal * pred_len,)
        Q_hat_j  = weighted_quantile(scores, w_step, alpha)
        lo_all[j] = test_lo[j] - Q_hat_j
        hi_all[j] = test_hi[j] + Q_hat_j
    return lo_all, hi_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1, label="Perfect calibration")
    for (method, text), grp in df.groupby(["method", "text"]):
        grp = grp.sort_values("nominal")
        label = f"{method} ({text})"
        ax.plot(grp["nominal"], grp["coverage"], marker="o", label=label)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("ChatTime — Calibration curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.55, 1.02); ax.set_ylim(0.55, 1.02)
    fig.tight_layout()
    fig.savefig("chattime_calibration.png", dpi=150)
    plt.close(fig)
    print("Saved chattime_calibration.png")


def plot_width(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (method, text), grp in df.groupby(["method", "text"]):
        grp = grp.sort_values("nominal")
        label = f"{method} ({text})"
        ax.plot(grp["nominal"], grp["width"], marker="o", label=label)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Mean interval width (admissions)")
    ax.set_title("ChatTime — Interval width vs nominal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("chattime_width.png", dpi=150)
    plt.close(fig)
    print("Saved chattime_width.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plots_only: bool, reset: bool):
    if reset:
        for p in [CSV_PATH, SAMPLES_NO_TEXT, SAMPLES_WITH_TXT]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")

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
    print(f"Cal: {len(cal_ctx_np)} windows  |  Test: {len(test_ctx_np)} windows")

    df_existing = load_existing()
    needed_combos = [(m, t, a) for (m, t) in METHODS for a in ALPHAS
                     if not already_done(df_existing, m, t, a)]

    if plots_only:
        print("--plots-only set — skipping experiments")
    elif not needed_combos:
        print("All combinations done — only regenerating plots")
    else:
        print(f"\nLoading ChatTime: {MODEL_PATH}")
        model = ChatTime(
            model_path=MODEL_PATH,
            hist_len=CONTEXT_WEEKS,
            pred_len=PRED_WEEKS,
            num_samples=NUM_SAMPLES,
        )

        # Generate samples for whichever text conditions are needed
        need_no_text   = any(t == "no_text"   for (_, t, _) in needed_combos)
        need_with_text = any(t == "with_text" for (_, t, _) in needed_combos)

        cal_s_nt = test_s_nt = cal_s_wt = test_s_wt = None
        if need_no_text:
            print("\n[samples] no-text condition")
            cal_s_nt  = get_samples(model, cal_ctx_np,  cal_txt,  PRED_WEEKS,
                                    use_text=False, cache_path="cal_"  + SAMPLES_NO_TEXT)
            test_s_nt = get_samples(model, test_ctx_np, test_txt, PRED_WEEKS,
                                    use_text=False, cache_path="test_" + SAMPLES_NO_TEXT)
        if need_with_text:
            print("\n[samples] with-text condition")
            cal_s_wt  = get_samples(model, cal_ctx_np,  cal_txt,  PRED_WEEKS,
                                    use_text=True,  cache_path="cal_"  + SAMPLES_WITH_TXT)
            test_s_wt = get_samples(model, test_ctx_np, test_txt, PRED_WEEKS,
                                    use_text=True,  cache_path="test_" + SAMPLES_WITH_TXT)

        # Sentence-BERT embeddings (only needed for Text-CQR)
        cal_embs = test_embs = None
        if any(m == "Text-CQR" for (m, _, _) in needed_combos):
            print(f"\nLoading sentence-BERT: {SBERT_MODEL}")
            from sentence_transformers import SentenceTransformer
            sbert     = SentenceTransformer(SBERT_MODEL)
            cal_embs  = embed_texts(sbert, cal_txt)
            test_embs = embed_texts(sbert, test_txt)

        # Run every needed (method, text, alpha) combo
        for (method, text, alpha) in needed_combos:
            print(f"\n  [{method}, {text}, α={alpha}]")
            cal_s  = cal_s_wt  if text == "with_text" else cal_s_nt
            test_s = test_s_wt if text == "with_text" else test_s_nt

            if method == "Naive":
                lo, hi = run_naive(cal_s, test_s, cal_fut_np, test_fut_np, alpha)
            elif method == "CQR":
                lo, hi = run_cqr(cal_s, test_s, cal_fut_np, test_fut_np, alpha)
            elif method == "Codebook-CQR":
                lo, hi = run_codebook_cqr(
                    cal_s, test_s, cal_ctx_np, test_ctx_np,
                    cal_fut_np, test_fut_np, model.discretizer, alpha,
                )
            elif method == "Text-CQR":
                lo, hi = run_text_cqr(
                    cal_s, test_s, cal_fut_np, test_fut_np,
                    cal_embs, test_embs, alpha,
                )
            else:
                raise ValueError(method)

            m = compute_metrics(lo, hi, test_fut_np)
            append_result(method, text, alpha, m["coverage"], m["width"])
            print(f"     coverage={m['coverage']:.3f}  width={m['width']:.2f}")

    # Plots
    df = pd.read_csv(CSV_PATH)
    print("\n" + df.sort_values(["method", "text", "alpha"]).to_string(index=False))
    plot_calibration(df)
    plot_width(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--reset",      action="store_true")
    args = p.parse_args()
    main(plots_only=args.plots_only, reset=args.reset)
