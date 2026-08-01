"""
Run the full conformal pipeline on ANY CGTSF subset, not just Paris Traffic Flow.

Addresses r3Md Q5 and sFBd: "only two model-dataset combinations". CGTSF
(ChengsenWang/CGTSF on HuggingFace) is the context-guided forecasting benchmark
released with ChatTime and contains several subsets besides PTF, each with the
same (Hist, Pred, Text) row format. Since the whole pipeline is dataset-agnostic
once the rows are parsed, extending to the other subsets costs only inference
time -- no new code paths.

This script generalises ptf_evaluate.py:
  * discovers the available subsets automatically, or takes explicit CSV paths;
  * caches samples per (dataset, text condition);
  * reports joint AND per-step coverage, the finite-K oracle, Winkler, and the
    repaired codebook variants;
  * writes one tidy CSV covering every dataset, so the cross-dataset table in
    the rebuttal is a single groupby.

The text conditions are no_text and with_text, which is what is needed to test
whether the multimodal width benefit reproduces outside PTF. That is the claim
most exposed by having only one multimodal dataset: the submission's Section 6
already concedes the positive direction ("real and shuffled text behave the
same") is PTF-specific.

Getting the data
----------------
    pip install datasets
    python cgtsf_evaluate.py --list                 # show available subsets
    python cgtsf_evaluate.py --datasets PTF BJPM    # run named subsets

or point at local CSVs with the same columns:

    python cgtsf_evaluate.py --csv dataset/PTF_full.csv dataset/OTHER_full.csv

Cost
----
Roughly the original PTF run per subset (400 windows x 16 samples x 2 text
conditions). Budget ~3 h per subset on one A6000. Resumable: caches are per
(dataset, condition), so an interrupted run resumes where it stopped.
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import conformal_lib as cl
import finite_k

HF_REPO = "ChengsenWang/CGTSF"
MODEL_PATH = "ChengsenWang/ChatTime-1-7B-Chat"

HIST_LEN = 120
PRED_LEN = 24
NUM_SAMPLES = 16
ALPHAS = [0.1, 0.2, 0.3]
N_CAL_WINDOWS = 200
N_TEST_WINDOWS = 200
SUBSAMPLE_SEED = 42
CAL_FRACTION = 0.20
TEST_FRACTION = 0.20

OUT_CSV = "rebuttal_cgtsf_results.csv"
CACHE_DIR = "cache_cgtsf"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_subsets():
    """Ask HuggingFace which CGTSF configs exist."""
    try:
        from datasets import get_dataset_config_names
    except ImportError:
        print("pip install datasets, or use --csv with local files")
        return []
    try:
        return list(get_dataset_config_names(HF_REPO))
    except Exception as e:                       # noqa: BLE001
        print(f"could not query {HF_REPO}: {type(e).__name__}: {e}")
        return []


def load_subset_hf(name):
    from datasets import load_dataset

    ds = load_dataset(HF_REPO, name)
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()


def load_subset_csv(path):
    return pd.read_csv(path)


def normalise(df):
    """
    Coerce a CGTSF-style frame into (contexts, futures, texts).

    Hist/Pred arrive either as stringified lists (CSV) or as real sequences
    (HuggingFace), so handle both.
    """
    def as_array(v):
        if isinstance(v, str):
            v = ast.literal_eval(v)
        return np.asarray(v, dtype=np.float32).ravel()

    if "Idx" in df.columns:
        df = df.sort_values("Idx").reset_index(drop=True)

    text_col = next((c for c in ("Text", "text", "Context") if c in df.columns), None)

    contexts, futures, texts = [], [], []
    for _, row in df.iterrows():
        h = as_array(row["Hist"])[-HIST_LEN:]
        p = as_array(row["Pred"])[:PRED_LEN]
        if len(h) < HIST_LEN or len(p) < PRED_LEN:
            continue
        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(p))):
            continue
        contexts.append(h)
        futures.append(p)
        texts.append(str(row[text_col]) if text_col else "")
    return contexts, futures, texts


def split_and_subsample(contexts, futures, texts):
    n = len(contexts)
    n_test = int(n * TEST_FRACTION)
    n_cal = int(n * CAL_FRACTION)
    n_train = n - n_cal - n_test
    if n_cal < 20 or n_test < 20:
        return None

    idx_cal = np.arange(n_train, n_train + n_cal)
    idx_test = np.arange(n_train + n_cal, n)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    idx_cal = rng.choice(idx_cal, size=min(N_CAL_WINDOWS, len(idx_cal)), replace=False)
    idx_test = rng.choice(idx_test, size=min(N_TEST_WINDOWS, len(idx_test)), replace=False)

    pick = lambda arr, ii: [arr[i] for i in ii]   # noqa: E731
    return dict(
        cal_ctx=pick(contexts, idx_cal),
        cal_fut=np.stack(pick(futures, idx_cal)),
        cal_txt=pick(texts, idx_cal),
        test_ctx=pick(contexts, idx_test),
        test_fut=np.stack(pick(futures, idx_test)),
        test_txt=pick(texts, idx_test),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def cache_path(dataset, cond, split):
    return os.path.join(CACHE_DIR, f"{dataset}_{cond}_{split}.npz")


def get_samples(model, contexts, texts, use_text, path):
    if os.path.exists(path):
        return np.load(path)["samples"]

    from chattime_method_naive import predict_samples

    n = len(contexts)
    out = np.full((n, NUM_SAMPLES, PRED_LEN), np.nan, dtype=np.float32)
    for i, (ctx, txt) in enumerate(zip(contexts, texts)):
        out[i] = predict_samples(
            model, ctx, PRED_LEN, (txt if (use_text and txt) else None), NUM_SAMPLES
        )
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, samples=out)
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(dataset, w, cal_s, test_s, cond, rows):
    oracle = {a: finite_k.oracle_coverage(NUM_SAMPLES, a) for a in ALPHAS}

    for alpha in ALPHAS:
        methods = {"Naive": cl.sample_quantiles(test_s, alpha)}

        lo, hi, beta = finite_k.corrected_interval(test_s, NUM_SAMPLES, alpha)
        methods["Naive (finite-K corrected)"] = (lo, hi)

        methods["CQR (joint)"] = cl.cqr_joint(cal_s, w["cal_fut"], test_s, alpha)
        methods["CQR (per-step)"] = cl.cqr_perstep(cal_s, w["cal_fut"], test_s, alpha)
        methods["ACI"] = cl.aci(cal_s, w["cal_fut"], test_s, w["test_fut"], alpha)
        methods["NexCP"] = cl.nexcp(cal_s, w["cal_fut"], test_s, alpha)
        l, h, _ = cl.pid(cal_s, w["cal_fut"], test_s, w["test_fut"], alpha,
                         gain_frac=0.05)
        methods["PID (scaled gain)"] = (l, h)

        for name, (lo, hi) in methods.items():
            m = cl.evaluate_intervals(lo, hi, w["test_fut"], alpha, nonneg=True)
            rows.append(dict(
                dataset=dataset, text=cond, method=name, alpha=alpha,
                nominal=round(1 - alpha, 2),
                oracle_finite_K=round(oracle[alpha], 4),
                cov_perstep=m["coverage_perstep"], cov_joint=m["coverage_joint"],
                width=m["width"], winkler=m["winkler"],
                n_cal=len(w["cal_ctx"]), n_test=len(w["test_ctx"]),
            ))


def main(args):
    if args.list:
        subs = list_subsets()
        print("CGTSF subsets:" if subs else "no subsets found")
        for s in subs:
            print(f"  {s}")
        return

    sources = []
    if args.csv:
        sources = [(os.path.splitext(os.path.basename(p))[0], ("csv", p))
                   for p in args.csv]
    else:
        names = args.datasets or list_subsets()
        sources = [(n, ("hf", n)) for n in names]
    if not sources:
        print("nothing to run; pass --datasets or --csv, or check --list")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    model = None

    for dataset, (kind, ref) in sources:
        print(f"\n=== {dataset} ===")
        try:
            df = load_subset_csv(ref) if kind == "csv" else load_subset_hf(ref)
        except Exception as e:                    # noqa: BLE001
            print(f"  load failed ({type(e).__name__}: {e}); skipping")
            continue

        contexts, futures, texts = normalise(df)
        print(f"  {len(contexts)} usable windows")
        w = split_and_subsample(contexts, futures, texts)
        if w is None:
            print("  too few windows after splitting; skipping")
            continue
        has_text = any(bool(t.strip()) for t in w["cal_txt"])
        conds = ["no_text", "with_text"] if has_text else ["no_text"]
        if not has_text:
            print("  no text column -- running the unimodal condition only")

        for cond in conds:
            need = [cache_path(dataset, cond, s) for s in ("cal", "test")]
            if not all(os.path.exists(p) for p in need):
                if model is None:
                    from model.model import ChatTime

                    print(f"  loading ChatTime: {MODEL_PATH}")
                    model = ChatTime(
                        model_path=MODEL_PATH, hist_len=HIST_LEN,
                        pred_len=PRED_LEN, max_pred_len=PRED_LEN,
                        num_samples=NUM_SAMPLES,
                    )
                print(f"  [{cond}] sampling")
            cal_s = get_samples(model, w["cal_ctx"], w["cal_txt"],
                                cond == "with_text", need[0])
            test_s = get_samples(model, w["test_ctx"], w["test_txt"],
                                 cond == "with_text", need[1])
            analyse(dataset, w, cal_s, test_s, cond, rows)
            print(f"  [{cond}] done")

    if not rows:
        print("\nno results produced")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\n" + "=" * 84)
    print("Naive coverage vs the finite-K oracle, by dataset (alpha=0.10)")
    print("=" * 84)
    nv = out[(out.method == "Naive") & (out.alpha == 0.10)].copy()
    nv["gap_vs_oracle"] = nv["cov_perstep"] - nv["oracle_finite_K"]
    print(nv[["dataset", "text", "cov_perstep", "oracle_finite_K",
              "nominal", "gap_vs_oracle"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 84)
    print("Joint coverage by dataset and method (alpha=0.10)")
    print("=" * 84)
    print(out[out.alpha == 0.10]
          .pivot_table(index="method", columns=["dataset", "text"],
                       values="cov_joint")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    if out.text.nunique() > 1:
        print("\n" + "=" * 84)
        print("Does the multimodal width benefit reproduce outside PTF?")
        print("(with_text minus no_text width, as a percentage; negative = text helps)")
        print("=" * 84)
        piv = out[out.method == "CQR (joint)"].pivot_table(
            index=["dataset", "alpha"], columns="text", values="width")
        if {"with_text", "no_text"}.issubset(piv.columns):
            piv["pct_change"] = 100 * (piv["with_text"] - piv["no_text"]) / piv["no_text"]
            print(piv.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--csv", nargs="+")
    p.add_argument("--list", action="store_true")
    main(p.parse_args())
