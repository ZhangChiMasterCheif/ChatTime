"""
Sampling-seed variability for the PTF experiment.

Why
---
Every number in Tables 2 and 3 of the submission comes from a single draw of
16 sample paths on 200 test windows. The headline multimodal claim is a 10-25%
width reduction; with K=16 and N=200 it is not currently possible to tell that
apart from decoding noise. Any reviewer asking "is this within noise?" is
asking for exactly this script.

What it does
------------
Redraws the no-text and with-text sample caches under several torch seeds and
writes one cache per (condition, seed). rebuttal_ptf.py-style analysis then
gives a mean +/- sd for coverage and width, and a per-seed distribution of the
with-text minus no-text width gap.

Cost
----
2 conditions x 4 extra seeds x 400 windows x 16 samples. Roughly 4x the
original PTF run, so budget ~8-12 hours on one A6000. If that is too much,
drop to `--seeds 1 2` -- three total seeds is enough to report a spread.

Usage
-----
    python ptf_seeds.py --seeds 1 2 3 4
    python ptf_seeds.py --analyze-only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import conformal_lib as cl
from ptf_dataset import load_ptf, split_ptf, to_window_lists, HIST_LEN, PRED_LEN

ALPHAS = [0.1, 0.2, 0.3]
NUM_SAMPLES = 16
N_CAL_WINDOWS = 200
N_TEST_WINDOWS = 200
SUBSAMPLE_SEED = 42
MODEL_PATH = "ChengsenWang/ChatTime-1-7B-Chat"
OUT_CSV = "rebuttal_ptf_seeds.csv"

CONDITIONS = ["no_text", "with_text"]

# Seed 0 reuses the caches the original scripts already wrote, so the existing
# run counts as one of the seeds rather than being thrown away.
SEED0_CACHE = {
    ("no_text", "cal"): "ptf_samples_no_text_cal.npz",
    ("no_text", "test"): "ptf_samples_no_text_test.npz",
    ("with_text", "cal"): "ptf_samples_with_text_cal.npz",
    ("with_text", "test"): "ptf_samples_with_text_test.npz",
}


def cache_path(cond, split, seed):
    if seed == 0:
        return SEED0_CACHE[(cond, split)]
    return f"ptf_seed{seed}_samples_{cond}_{split}.npz"


def load_windows():
    df = load_ptf()
    _, cal, test = split_ptf(df)
    cal_ctx, cal_fut, cal_txt = to_window_lists(cal)
    test_ctx, test_fut, test_txt = to_window_lists(test)
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    ci = rng.choice(len(cal_ctx), size=min(N_CAL_WINDOWS, len(cal_ctx)), replace=False)
    ti = rng.choice(len(test_ctx), size=min(N_TEST_WINDOWS, len(test_ctx)), replace=False)
    return dict(
        cal_ctx=[cal_ctx[i] for i in ci],
        cal_fut=np.stack([cal_fut[i] for i in ci]),
        cal_txt=[cal_txt[i] for i in ci],
        test_ctx=[test_ctx[i] for i in ti],
        test_fut=np.stack([test_fut[i] for i in ti]),
        test_txt=[test_txt[i] for i in ti],
    )


def generate(model, contexts, texts, use_text, path, seed):
    if os.path.exists(path):
        print(f"  cached: {path}")
        return
    from chattime_method_naive import predict_samples

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(contexts)
    out = np.full((n, NUM_SAMPLES, PRED_LEN), np.nan, dtype=np.float32)
    for i, (ctx, txt) in enumerate(zip(contexts, texts)):
        out[i] = predict_samples(
            model, ctx, PRED_LEN, (txt if (use_text and txt) else None), NUM_SAMPLES
        )
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n}")
    np.savez_compressed(path, samples=out)
    print(f"  wrote {path}")


def analyze(w, seeds):
    records = []
    for seed in seeds:
        caches = {}
        ok = True
        for cond in CONDITIONS:
            for split in ("cal", "test"):
                p = cache_path(cond, split, seed)
                if not os.path.exists(p):
                    ok = False
                    break
                caches[(cond, split)] = np.load(p)["samples"]
        if not ok:
            print(f"  [skip] incomplete caches for seed {seed}")
            continue

        for cond in CONDITIONS:
            cal_s = caches[(cond, "cal")]
            test_s = caches[(cond, "test")]
            for alpha in ALPHAS:
                for name, (lo, hi) in {
                    "Naive": cl.sample_quantiles(test_s, alpha),
                    "CQR (joint)": cl.cqr_joint(cal_s, w["cal_fut"], test_s, alpha),
                    "CQR (per-step)": cl.cqr_perstep(cal_s, w["cal_fut"], test_s, alpha),
                }.items():
                    m = cl.evaluate_intervals(lo, hi, w["test_fut"], alpha)
                    records.append(
                        dict(seed=seed, text=cond, method=name, alpha=alpha,
                             cov_perstep=m["coverage_perstep"],
                             cov_joint=m["coverage_joint"],
                             width=m["width"], winkler=m["winkler"])
                    )

    df = pd.DataFrame(records)
    if df.empty:
        print("No complete seeds found.")
        return df
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\nMean +/- sd across seeds:")
    agg = (df.groupby(["method", "text", "alpha"])
             .agg(cov_joint_mean=("cov_joint", "mean"),
                  cov_joint_sd=("cov_joint", "std"),
                  width_mean=("width", "mean"),
                  width_sd=("width", "std"),
                  n_seeds=("seed", "nunique"))
             .reset_index())
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nWith-text minus no-text width gap, per seed:")
    piv = df.pivot_table(index=["method", "alpha", "seed"], columns="text",
                         values="width")
    if {"with_text", "no_text"}.issubset(piv.columns):
        piv["gap"] = piv["with_text"] - piv["no_text"]
        piv["pct"] = 100 * piv["gap"] / piv["no_text"]
        print(piv.to_string(float_format=lambda v: f"{v:.2f}"))
        print("\nGap summary (negative = text tightens intervals):")
        print(piv.groupby(level=["method", "alpha"])["pct"]
                 .agg(["mean", "std", "min", "max"])
                 .to_string(float_format=lambda v: f"{v:.2f}"))
    return df


def main(seeds, analyze_only):
    w = load_windows()
    all_seeds = sorted(set([0] + list(seeds)))

    if not analyze_only:
        missing = [
            (c, sp, s)
            for s in seeds
            for c in CONDITIONS
            for sp in ("cal", "test")
            if not os.path.exists(cache_path(c, sp, s))
        ]
        if missing:
            from model.model import ChatTime

            print(f"Loading ChatTime: {MODEL_PATH}")
            model = ChatTime(
                model_path=MODEL_PATH, hist_len=HIST_LEN, pred_len=PRED_LEN,
                max_pred_len=PRED_LEN, num_samples=NUM_SAMPLES,
            )
            for cond, split, seed in missing:
                print(f"\n[seed {seed}] {cond}/{split}")
                generate(
                    model,
                    w[f"{split}_ctx"],
                    w[f"{split}_txt"],
                    cond == "with_text",
                    cache_path(cond, split, seed),
                    seed,
                )

    analyze(w, all_seeds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args()
    main(a.seeds, a.analyze_only)
