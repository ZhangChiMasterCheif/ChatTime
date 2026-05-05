"""
Unified evaluation of all ChatTime methods on the PTF (Paris traffic flow)
dataset, where ChatTime was actually trained for context-guided forecasting.

Mirrors chattime_evaluate.py but uses PTF data and writes to PTF-specific
output files so the existing COVID experiment results stay untouched.

Outputs
-------
  ptf_samples_no_text_{cal,test}.npz   : cached raw samples per condition
  ptf_samples_with_text_{cal,test}.npz
  ptf_results.csv                      : coverage and width per method × alpha
  ptf_calibration.png                  : empirical coverage vs nominal
  ptf_width.png                        : interval width vs nominal

Usage
-----
    python ptf_evaluate.py                # run missing, then plot
    python ptf_evaluate.py --plots-only   # skip experiments, regen plots
    python ptf_evaluate.py --reset        # wipe caches + CSV first
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

from ptf_dataset import load_ptf, split_ptf, to_window_lists, HIST_LEN, PRED_LEN
from chattime_method_naive import predict_samples, compute_metrics
from chattime_method_cqr   import cqr_scores, cqr_quantile, apply_correction
from chattime_method_codebook_cqr import (
    fit_scaler, to_bins, codebook_scores,
    apply_codebook_correction, bins_to_values,
)
from chattime_method_text_cqr import (
    embed_texts, text_weights, weighted_quantile, SBERT_MODEL,
)
import chattime_method_pid as pid_mod

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH    = "ChengsenWang/ChatTime-1-7B-Chat"
NUM_SAMPLES   = 16        # bumped vs the COVID run since we have plenty of data
ALPHAS        = [0.1, 0.2, 0.3]

# Sub-sample windows so the experiment is tractable.
# PTF cal/test each have ~2300 windows; 200 each is plenty for conformal.
N_CAL_WINDOWS  = 200
N_TEST_WINDOWS = 200

CSV_PATH         = "ptf_results.csv"
SAMPLES_NT_CAL   = "ptf_samples_no_text_cal.npz"
SAMPLES_NT_TEST  = "ptf_samples_no_text_test.npz"
SAMPLES_WT_CAL   = "ptf_samples_with_text_cal.npz"
SAMPLES_WT_TEST  = "ptf_samples_with_text_test.npz"

METHODS = [
    ("Naive",         "no_text"),
    ("Naive",         "with_text"),
    ("CQR",           "no_text"),
    ("CQR",           "with_text"),
    ("Codebook-CQR",  "no_text"),
    ("Codebook-CQR",  "with_text"),
    ("PID",           "no_text"),
    ("PID",           "with_text"),
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
# Sample cache
# ---------------------------------------------------------------------------

def get_samples(model, contexts, texts, pred_len, use_text, cache_path):
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
# Method runners (operate on cached samples — no model calls)
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
    centers    = discretizer.centers
    boundaries = discretizer.boundaries
    n_tokens   = len(centers)

    def per_window_bins(samples_w, ctx, fut):
        scaler   = fit_scaler(ctx)
        s_clean  = np.nan_to_num(samples_w, nan=float(ctx.mean()))
        s_bins   = to_bins(s_clean, scaler, boundaries).astype(np.float32)
        bin_lo   = np.quantile(s_bins, alpha / 2,     axis=0)
        bin_hi   = np.quantile(s_bins, 1 - alpha / 2, axis=0)
        true_b   = to_bins(fut, scaler, boundaries)
        return bin_lo, bin_hi, true_b, scaler

    cal_bin_lo, cal_bin_hi, cal_true, _ = zip(*[
        per_window_bins(cal_s[i], cal_ctx[i], cal_fut[i])
        for i in range(len(cal_ctx))
    ])
    cal_bin_lo = np.stack(cal_bin_lo)
    cal_bin_hi = np.stack(cal_bin_hi)
    cal_true   = np.stack(cal_true)
    scores     = codebook_scores(cal_bin_lo, cal_bin_hi, cal_true)
    Q_hat_bin  = cqr_quantile(scores, alpha)

    lo_arr = np.empty((len(test_ctx), PRED_LEN))
    hi_arr = np.empty((len(test_ctx), PRED_LEN))
    for i in range(len(test_ctx)):
        bl, bh, _, sc = per_window_bins(test_s[i], test_ctx[i], test_fut[i])
        lo_b, hi_b    = apply_codebook_correction(bl, bh, Q_hat_bin, n_tokens)
        lo_arr[i]     = bins_to_values(lo_b.astype(int), sc, centers)
        hi_arr[i]     = bins_to_values(hi_b.astype(int), sc, centers)
    return lo_arr, hi_arr


def run_text_cqr(cal_s, test_s, cal_fut, test_fut, cal_embs, test_embs, alpha):
    cal_lo  = np.nanquantile(cal_s, alpha / 2,     axis=1)
    cal_hi  = np.nanquantile(cal_s, 1 - alpha / 2, axis=1)
    scores  = cqr_scores(cal_lo, cal_hi, cal_fut)

    test_lo = np.nanquantile(test_s, alpha / 2,     axis=1)
    test_hi = np.nanquantile(test_s, 1 - alpha / 2, axis=1)

    N = len(test_lo)
    lo_all = np.empty_like(test_lo)
    hi_all = np.empty_like(test_hi)
    for j in range(N):
        w        = text_weights(test_embs[j], cal_embs)
        Q_hat_j  = weighted_quantile(scores, w, alpha)
        lo_all[j] = test_lo[j] - Q_hat_j
        hi_all[j] = test_hi[j] + Q_hat_j
    return lo_all, hi_all


# ---------------------------------------------------------------------------
# Plots
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
    ax.set_title("PTF — Calibration curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.55, 1.02); ax.set_ylim(0.55, 1.02)
    fig.tight_layout()
    fig.savefig("ptf_calibration.png", dpi=150)
    plt.close(fig)
    print("Saved ptf_calibration.png")


def plot_width(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (method, text), grp in df.groupby(["method", "text"]):
        grp = grp.sort_values("nominal")
        label = f"{method} ({text})"
        ax.plot(grp["nominal"], grp["width"], marker="o", label=label)
    ax.set_xlabel("Nominal coverage (1 − α)")
    ax.set_ylabel("Mean interval width (vehicles/hour)")
    ax.set_title("PTF — Interval width vs nominal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("ptf_width.png", dpi=150)
    plt.close(fig)
    print("Saved ptf_width.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plots_only: bool, reset: bool):
    if reset:
        for p in [CSV_PATH, SAMPLES_NT_CAL, SAMPLES_NT_TEST,
                  SAMPLES_WT_CAL, SAMPLES_WT_TEST]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")

    print("Loading PTF data ...")
    df = load_ptf()
    train, cal, test = split_ptf(df)
    print(f"  train: {len(train)}, cal: {len(cal)}, test: {len(test)}")

    cal_ctx, cal_fut, cal_txt   = to_window_lists(cal)
    test_ctx, test_fut, test_txt = to_window_lists(test)

    # subsample to keep cost reasonable, deterministic
    rng = np.random.default_rng(42)
    cal_idx  = rng.choice(len(cal_ctx),  size=min(N_CAL_WINDOWS,  len(cal_ctx)),  replace=False)
    test_idx = rng.choice(len(test_ctx), size=min(N_TEST_WINDOWS, len(test_ctx)), replace=False)
    cal_ctx  = [cal_ctx[i]  for i in cal_idx]
    cal_fut  = [cal_fut[i]  for i in cal_idx]
    cal_txt  = [cal_txt[i]  for i in cal_idx]
    test_ctx = [test_ctx[i] for i in test_idx]
    test_fut = [test_fut[i] for i in test_idx]
    test_txt = [test_txt[i] for i in test_idx]

    cal_fut_arr  = np.stack(cal_fut)
    test_fut_arr = np.stack(test_fut)
    print(f"  Sampled: {len(cal_ctx)} cal windows, {len(test_ctx)} test windows")

    df_existing = load_existing()
    needed = [(m, t, a) for (m, t) in METHODS for a in ALPHAS
              if not already_done(df_existing, m, t, a)]

    if plots_only:
        print("--plots-only set — skipping experiments")
    elif not needed:
        print("All combinations done — only regenerating plots")
    else:
        print(f"\nLoading ChatTime: {MODEL_PATH}  (max_pred_len={PRED_LEN})")
        model = ChatTime(
            model_path=MODEL_PATH,
            hist_len=HIST_LEN,
            pred_len=PRED_LEN,
            max_pred_len=PRED_LEN,        # PTF needs 24-step single-shot
            num_samples=NUM_SAMPLES,
        )

        need_no_text   = any(t == "no_text"   for (_, t, _) in needed)
        need_with_text = any(t == "with_text" for (_, t, _) in needed)

        cal_s_nt = test_s_nt = cal_s_wt = test_s_wt = None
        if need_no_text:
            print("\n[samples] no-text condition")
            cal_s_nt  = get_samples(model, cal_ctx,  cal_txt,  PRED_LEN,
                                    use_text=False, cache_path=SAMPLES_NT_CAL)
            test_s_nt = get_samples(model, test_ctx, test_txt, PRED_LEN,
                                    use_text=False, cache_path=SAMPLES_NT_TEST)
        if need_with_text:
            print("\n[samples] with-text condition")
            cal_s_wt  = get_samples(model, cal_ctx,  cal_txt,  PRED_LEN,
                                    use_text=True,  cache_path=SAMPLES_WT_CAL)
            test_s_wt = get_samples(model, test_ctx, test_txt, PRED_LEN,
                                    use_text=True,  cache_path=SAMPLES_WT_TEST)

        cal_embs = test_embs = None
        if any(m == "Text-CQR" for (m, _, _) in needed):
            print(f"\nLoading sentence-BERT: {SBERT_MODEL}")
            from sentence_transformers import SentenceTransformer
            sbert     = SentenceTransformer(SBERT_MODEL)
            cal_embs  = embed_texts(sbert, cal_txt)
            test_embs = embed_texts(sbert, test_txt)

        for (method, text, alpha) in needed:
            print(f"\n  [{method}, {text}, α={alpha}]")
            cal_s  = cal_s_wt  if text == "with_text" else cal_s_nt
            test_s = test_s_wt if text == "with_text" else test_s_nt

            if method == "Naive":
                lo, hi = run_naive(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "CQR":
                lo, hi = run_cqr(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "Codebook-CQR":
                lo, hi = run_codebook_cqr(
                    cal_s, test_s, cal_ctx, test_ctx,
                    cal_fut_arr, test_fut_arr, model.discretizer, alpha,
                )
            elif method == "Text-CQR":
                lo, hi = run_text_cqr(
                    cal_s, test_s, cal_fut_arr, test_fut_arr,
                    cal_embs, test_embs, alpha,
                )
            elif method == "PID":
                lo, hi = pid_mod.run_pid(
                    cal_s, test_s, cal_fut_arr, test_fut_arr, alpha,
                )
            else:
                raise ValueError(method)

            m = compute_metrics(lo, hi, test_fut_arr)
            append_result(method, text, alpha, m["coverage"], m["width"])
            print(f"     coverage={m['coverage']:.3f}  width={m['width']:.2f}")

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
