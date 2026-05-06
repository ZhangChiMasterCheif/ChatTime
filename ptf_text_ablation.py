"""
Text ablation (Experiment E3) for PTF.

Question: when we say "text helps tighten conformal intervals," is the gain
really from informative text, or is ChatTime just sensitive to *any* text in
the prompt (length, structure, etc.)?

Setup: shuffle the text→window mapping. Each window now sees a description
that belongs to a *different* window. Then re-run ChatTime sampling and the
four conformal methods. Compare to the existing no_text and with_text rows
in ptf_results.csv.

Expected pattern (the safe-deployment story for the paper):
  * Coverage stays valid for shuffled — conformal guarantees don't depend on
    text quality, only on exchangeability of (text, target) pairs.
  * Width with shuffled text ≈ width with no text (no informational gain).
  * Width with real text is meaningfully smaller than both.
  * Text-CQR with shuffled text doesn't tighten beyond plain CQR.

If those four hold, the multimodal advantage is *informational*, and the
method is *safe* — bad text doesn't break coverage, it just stops helping.

Outputs
-------
  cal_ptf_samples_shuffled_text.npz   : cached samples generated with shuffled text
  test_ptf_samples_shuffled_text.npz
  ptf_results.csv (extended with text="shuffled_text" rows)
  ptf_ablation.png : per-method, per-α comparison of three text conditions

Usage
-----
    python ptf_text_ablation.py
    python ptf_text_ablation.py --reset
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
import ptf_evaluate as pe   # reuse run_naive/run_cqr/run_codebook_cqr/run_text_cqr
import chattime_method_pid as pid_mod

# ---------------------------------------------------------------------------
# Config — match ptf_evaluate so window selection is identical
# ---------------------------------------------------------------------------

MODEL_PATH       = pe.MODEL_PATH
NUM_SAMPLES      = pe.NUM_SAMPLES
ALPHAS           = pe.ALPHAS
N_CAL_WINDOWS    = pe.N_CAL_WINDOWS
N_TEST_WINDOWS   = pe.N_TEST_WINDOWS
CSV_PATH         = pe.CSV_PATH
TEXT_LABEL       = "shuffled_text"

SAMPLES_CAL_PATH  = "cal_ptf_samples_shuffled_text.npz"
SAMPLES_TEST_PATH = "test_ptf_samples_shuffled_text.npz"

SHUFFLE_SEED = 17     # deterministic permutation


# ---------------------------------------------------------------------------
# Sample cache (re-uses ptf_evaluate.get_samples logic)
# ---------------------------------------------------------------------------

def get_shuffled_samples(model, contexts, shuffled_texts, cache_path):
    if os.path.exists(cache_path):
        print(f"  loading cached samples: {cache_path}")
        return np.load(cache_path)["samples"]

    N = len(contexts)
    out = np.full((N, NUM_SAMPLES, PRED_LEN), np.nan, dtype=np.float32)
    for i, (ctx, txt) in enumerate(zip(contexts, shuffled_texts)):
        out[i] = predict_samples(model, ctx, PRED_LEN, txt or None, NUM_SAMPLES)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N}")
    np.savez(cache_path, samples=out)
    print(f"  cached → {cache_path}")
    return out


# ---------------------------------------------------------------------------
# CSV helpers — reuse ptf_evaluate's bookkeeping
# ---------------------------------------------------------------------------

def already_done_shuffled(df, method, alpha):
    return pe.already_done(df, method, TEXT_LABEL, alpha)


def append_shuffled(method, alpha, coverage, width):
    pe.append_result(method, TEXT_LABEL, alpha, coverage, width)


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_ablation(df: pd.DataFrame):
    """One panel per method; bars for (no_text, shuffled_text, with_text) at α=0.1."""
    methods = ["Naive", "CQR", "Codebook-CQR", "Text-CQR"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4),
                             sharey=False)

    text_order  = ["no_text", "shuffled_text", "with_text"]
    text_labels = ["No text", "Shuffled text", "Real text"]
    colors      = ["tab:gray", "tab:red", "tab:green"]

    for ax, method in zip(axes, methods):
        # average over alpha for a single bar height per text condition
        widths   = []
        coverages = []
        present  = []
        for t in text_order:
            sub = df[(df["method"] == method) & (df["text"] == t)]
            if sub.empty:
                widths.append(0.0); coverages.append(0.0); present.append(False)
            else:
                widths.append(float(sub["width"].mean()))
                coverages.append(float(sub["coverage"].mean()))
                present.append(True)

        x = np.arange(len(text_order))
        bars = ax.bar(x, widths, color=colors, edgecolor="black")
        for i, (b, cov, ok) in enumerate(zip(bars, coverages, present)):
            if not ok:
                continue
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"cov\n{cov:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(text_labels, fontsize=9)
        ax.set_title(method)
        ax.set_ylabel("Mean interval width")

    fig.suptitle("PTF — text ablation (mean over α∈{0.1,0.2,0.3})", fontsize=11)
    fig.tight_layout()
    fig.savefig("ptf_ablation.png", dpi=150)
    plt.close(fig)
    print("Saved ptf_ablation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reset: bool):
    if reset:
        for p in [SAMPLES_CAL_PATH, SAMPLES_TEST_PATH]:
            if os.path.exists(p):
                os.remove(p); print(f"Removed {p}")
        # also drop only the shuffled rows from the CSV, leaving real-text/no-text alone
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            keep = df[df["text"] != TEXT_LABEL]
            keep.to_csv(CSV_PATH, index=False)
            print(f"Pruned shuffled rows from {CSV_PATH}")

    print("Loading PTF data ...")
    full = load_ptf()
    _, cal, test = split_ptf(full)
    cal_ctx,  cal_fut,  cal_txt  = to_window_lists(cal)
    test_ctx, test_fut, test_txt = to_window_lists(test)

    # SAME deterministic subsample as ptf_evaluate, so windows align with the
    # existing no_text / with_text rows in the CSV.
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

    # --- Shuffle texts deterministically ---
    perm_rng = np.random.default_rng(SHUFFLE_SEED)
    cal_perm  = perm_rng.permutation(len(cal_txt))
    test_perm = perm_rng.permutation(len(test_txt))
    # avoid identity assignment (window getting its own text back)
    cal_perm  = np.where(cal_perm  == np.arange(len(cal_perm)),
                          (cal_perm + 1) % len(cal_perm),  cal_perm)
    test_perm = np.where(test_perm == np.arange(len(test_perm)),
                          (test_perm + 1) % len(test_perm), test_perm)
    cal_txt_shuf  = [cal_txt[i]  for i in cal_perm]
    test_txt_shuf = [test_txt[i] for i in test_perm]
    print(f"Shuffled text mappings (seed {SHUFFLE_SEED})")

    # --- Decide what to run ---
    df_existing = pe.load_existing()
    needed = [(m, a) for m in ["Naive", "CQR", "Codebook-CQR", "Text-CQR", "PID"]
              for a in ALPHAS
              if not already_done_shuffled(df_existing, m, a)]

    if not needed:
        print("All shuffled-text combinations already done — only re-plotting")
    else:
        print(f"\nLoading ChatTime: {MODEL_PATH}")
        model = ChatTime(
            model_path=MODEL_PATH, hist_len=HIST_LEN, pred_len=PRED_LEN,
            max_pred_len=PRED_LEN, num_samples=NUM_SAMPLES,
        )

        print("\n[samples] shuffled-text condition")
        cal_s  = get_shuffled_samples(model, cal_ctx,  cal_txt_shuf,  SAMPLES_CAL_PATH)
        test_s = get_shuffled_samples(model, test_ctx, test_txt_shuf, SAMPLES_TEST_PATH)

        # Sentence-BERT embeddings — under shuffled text we embed the SHUFFLED
        # text the model actually saw, since that's the metadata available at
        # test time. Tests whether weighted CP gracefully degrades when text is
        # uninformative.
        cal_embs = test_embs = None
        if any(m == "Text-CQR" for (m, _) in needed):
            print(f"\nLoading sentence-BERT: {SBERT_MODEL}")
            from sentence_transformers import SentenceTransformer
            sbert     = SentenceTransformer(SBERT_MODEL)
            cal_embs  = embed_texts(sbert, cal_txt_shuf)
            test_embs = embed_texts(sbert, test_txt_shuf)

        for (method, alpha) in needed:
            print(f"\n  [{method}, shuffled, α={alpha}]")
            if method == "Naive":
                lo, hi = pe.run_naive(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "CQR":
                lo, hi = pe.run_cqr(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "Codebook-CQR":
                lo, hi = pe.run_codebook_cqr(
                    cal_s, test_s, cal_ctx, test_ctx,
                    cal_fut_arr, test_fut_arr, model.discretizer, alpha,
                )
            elif method == "Text-CQR":
                lo, hi = pe.run_text_cqr(
                    cal_s, test_s, cal_fut_arr, test_fut_arr,
                    cal_embs, test_embs, alpha,
                )
            elif method == "PID":
                lo, hi = pid_mod.run_pid(
                    cal_s, test_s, cal_fut_arr, test_fut_arr, alpha,
                )
            m = compute_metrics(lo, hi, test_fut_arr)
            append_shuffled(method, alpha, m["coverage"], m["width"])
            print(f"     coverage={m['coverage']:.3f}  width={m['width']:.2f}")

    df = pd.read_csv(CSV_PATH)
    print("\n=== All results ===")
    print(df.sort_values(["method", "text", "alpha"]).to_string(index=False))
    plot_ablation(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    main(reset=args.reset)
