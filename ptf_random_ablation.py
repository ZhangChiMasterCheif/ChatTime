"""
Random-text ablation for PTF.

Tightens the negative finding from ptf_text_ablation.py. Where the shuffled
ablation rotated the (text → window) mapping but kept all texts in-distribution
(weather/calendar descriptions of Paris traffic), this script replaces the
text with a completely unrelated paragraph. If ChatTime's prediction width
still matches the with_text condition, the conclusion is unambiguous:
the model does not extract predictive signal from the textual content.

Three text conditions can now be compared in ptf_results.csv:
  no_text       : empty prompt context
  with_text     : real per-window weather/calendar description
  shuffled_text : real text from a different window (in-distribution noise)
  random_text   : completely off-topic text (out-of-distribution noise)

Output rows append with text="random_text".

Usage
-----
    python ptf_random_ablation.py
    python ptf_random_ablation.py --reset
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.model import ChatTime

from ptf_dataset import load_ptf, split_ptf, to_window_lists, HIST_LEN, PRED_LEN
from chattime_method_naive import predict_samples, compute_metrics
import ptf_evaluate as pe
import chattime_method_pid as pid_mod

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH      = pe.MODEL_PATH
NUM_SAMPLES     = pe.NUM_SAMPLES
ALPHAS          = pe.ALPHAS
N_CAL_WINDOWS   = pe.N_CAL_WINDOWS
N_TEST_WINDOWS  = pe.N_TEST_WINDOWS
CSV_PATH        = pe.CSV_PATH
TEXT_LABEL      = "random_text"

SAMPLES_CAL_PATH  = "cal_ptf_samples_random_text.npz"
SAMPLES_TEST_PATH = "test_ptf_samples_random_text.npz"

# A single off-topic paragraph used for every window. Same length-class as the
# real text descriptions, fluent English, but no mention of weather, dates,
# Paris, traffic, or any feature relevant to highway flow forecasting.
RANDOM_TEXT = (
    "This recipe calls for two cups of all-purpose flour, one cup of whole "
    "milk, three large eggs, and a teaspoon of vanilla extract. Sift the dry "
    "ingredients into a large mixing bowl, then whisk in the wet ingredients "
    "until the batter is smooth and free of lumps. Pour into a greased pan "
    "and bake at 350 degrees Fahrenheit for about 25 minutes, or until a "
    "toothpick inserted into the center comes out clean. Let cool before "
    "slicing and serving with fresh berries."
)


# ---------------------------------------------------------------------------
# Methods to evaluate (drop Text-CQR for the same reasons we did in the main eval)
# ---------------------------------------------------------------------------

EVAL_METHODS = ["Naive", "CQR", "Codebook-CQR", "PID"]


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def get_random_samples(model, contexts, cache_path):
    if os.path.exists(cache_path):
        print(f"  loading cached samples: {cache_path}")
        return np.load(cache_path)["samples"]

    N = len(contexts)
    out = np.full((N, NUM_SAMPLES, PRED_LEN), np.nan, dtype=np.float32)
    for i, ctx in enumerate(contexts):
        out[i] = predict_samples(model, ctx, PRED_LEN, RANDOM_TEXT, NUM_SAMPLES)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N}")
    np.savez(cache_path, samples=out)
    print(f"  cached → {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reset: bool):
    if reset:
        for p in [SAMPLES_CAL_PATH, SAMPLES_TEST_PATH]:
            if os.path.exists(p):
                os.remove(p); print(f"Removed {p}")
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            keep = df[df["text"] != TEXT_LABEL]
            keep.to_csv(CSV_PATH, index=False)
            print(f"Pruned random_text rows from {CSV_PATH}")

    print("Loading PTF data ...")
    full = load_ptf()
    _, cal, test = split_ptf(full)
    cal_ctx,  cal_fut,  _ = to_window_lists(cal)
    test_ctx, test_fut, _ = to_window_lists(test)

    # SAME deterministic subsample as ptf_evaluate.py (seed 42)
    rng = np.random.default_rng(42)
    cal_idx  = rng.choice(len(cal_ctx),  size=min(N_CAL_WINDOWS,  len(cal_ctx)),  replace=False)
    test_idx = rng.choice(len(test_ctx), size=min(N_TEST_WINDOWS, len(test_ctx)), replace=False)
    cal_ctx  = [cal_ctx[i]  for i in cal_idx]
    cal_fut  = [cal_fut[i]  for i in cal_idx]
    test_ctx = [test_ctx[i] for i in test_idx]
    test_fut = [test_fut[i] for i in test_idx]

    cal_fut_arr  = np.stack(cal_fut)
    test_fut_arr = np.stack(test_fut)
    print(f"Cal: {len(cal_ctx)} windows | Test: {len(test_ctx)} windows")
    print(f"Random text: '{RANDOM_TEXT[:80]}...'")

    # decide what to run
    df_existing = pe.load_existing()
    needed = [(m, a) for m in EVAL_METHODS for a in ALPHAS
              if not pe.already_done(df_existing, m, TEXT_LABEL, a)]

    if not needed:
        print("All random_text combinations already done — nothing to do")
    else:
        print(f"\nLoading ChatTime: {MODEL_PATH}")
        model = ChatTime(
            model_path=MODEL_PATH, hist_len=HIST_LEN, pred_len=PRED_LEN,
            max_pred_len=PRED_LEN, num_samples=NUM_SAMPLES,
        )

        print("\n[samples] random-text condition")
        cal_s  = get_random_samples(model, cal_ctx,  SAMPLES_CAL_PATH)
        test_s = get_random_samples(model, test_ctx, SAMPLES_TEST_PATH)

        for (method, alpha) in needed:
            print(f"\n  [{method}, random_text, α={alpha}]")
            if method == "Naive":
                lo, hi = pe.run_naive(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "CQR":
                lo, hi = pe.run_cqr(cal_s, test_s, cal_fut_arr, test_fut_arr, alpha)
            elif method == "Codebook-CQR":
                lo, hi = pe.run_codebook_cqr(
                    cal_s, test_s, cal_ctx, test_ctx,
                    cal_fut_arr, test_fut_arr, model.discretizer, alpha,
                )
            elif method == "PID":
                lo, hi = pid_mod.run_pid(
                    cal_s, test_s, cal_fut_arr, test_fut_arr, alpha,
                )
            m = compute_metrics(lo, hi, test_fut_arr)
            pe.append_result(method, TEXT_LABEL, alpha, m["coverage"], m["width"])
            print(f"     coverage={m['coverage']:.3f}  width={m['width']:.2f}")

    # show the ablation summary across all four text conditions
    df = pd.read_csv(CSV_PATH)
    df = df[df["method"] != "Text-CQR"]
    print("\n=== All ablation results (Text-CQR omitted) ===")
    print(df.sort_values(["method", "alpha", "text"]).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    main(reset=args.reset)
