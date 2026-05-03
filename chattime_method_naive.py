"""
Naive ChatTime Sampling — empirical quantile intervals from sample paths.

Same idea as method_naive.py for Chronos: generate N sample trajectories,
take the α/2 and 1-α/2 empirical quantiles. No conformal correction.

ChatTime's built-in `predict()` returns only the median, so we use a custom
`predict_samples()` helper that mirrors its prediction logic but returns the
raw sample matrix.

Key knob:  use_text=True/False. When True, the GPT-generated weekly trend
description is passed as `context` in the LLaMA prompt. When False, the
prompt is purely numeric.

Usage:
    python chattime_method_naive.py
"""

import sys
import numpy as np
import torch
from transformers import pipeline as hf_pipeline

# ChatTime modules live in this folder (we run from /home/czhan168/ChatTime)
sys.path.insert(0, ".")
from model.model import ChatTime
from utils.prompt import getPrompt

from chattime_dataset import (
    load_weekly_hosp, load_text, split_weekly, make_text_windows,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH    = "ChengsenWang/ChatTime-1-7B-Chat"   # HF Hub path
CONTEXT_WEEKS = 12
PRED_WEEKS    = 4
NUM_SAMPLES   = 8       # ChatTime is slow — keep modest
ALPHAS        = [0.1, 0.2, 0.3]

# Subset of states to keep the experiment tractable
EXAMPLE_STATES = [
    "California", "Texas", "Florida", "New York",
    "Illinois", "Pennsylvania", "Ohio", "Georgia",
    "North Carolina", "Michigan",
]
STRIDE = 1


# ---------------------------------------------------------------------------
# Sample generator (replaces ChatTime.predict, but returns all samples)
# ---------------------------------------------------------------------------

def predict_samples(
    model: ChatTime,
    hist_data: np.ndarray,
    pred_len: int,
    context_text: str = None,
    num_samples: int = NUM_SAMPLES,
) -> np.ndarray:
    """
    Generate raw sample trajectories from ChatTime.

    Mirrors ChatTime.predict() but does NOT collapse to the median — returns
    the full (num_samples, pred_len) array. Predictions that fail to parse
    fully are NaN-padded.

    Parameters
    ----------
    model         : a constructed ChatTime instance with hist_len / pred_len set
    hist_data     : 1D numpy float array
    pred_len      : ≤ model.max_pred_len  (single-shot generation)
    context_text  : optional text inserted into the prompt's Instruction
    num_samples   : number of sample paths

    Returns
    -------
    samples : (num_samples, pred_len) float32 array (NaN where parsing failed)
    """
    assert pred_len <= model.max_pred_len, (
        f"pred_len={pred_len} exceeds model.max_pred_len={model.max_pred_len}; "
        "iterative roll-out is not implemented in this helper."
    )

    # 1. discretize + serialize history
    dispersed   = model.discretizer.discretize(hist_data)
    serialized  = model.serializer.serialize(dispersed)
    prompt      = getPrompt(flag="prediction", context=context_text, input=serialized)

    # 2. generate num_samples completions
    pipe = hf_pipeline(
        task="text-generation",
        model=model.model,
        tokenizer=model.tokenizer,
        min_new_tokens=2 * pred_len + 8,
        max_new_tokens=2 * pred_len + 8,
        do_sample=True,
        num_return_sequences=num_samples,
        top_k=model.top_k,
        top_p=model.top_p,
        temperature=model.temperature,
        eos_token_id=model.eos_token_id,
    )
    completions = pipe(prompt)

    # 3. decode each completion back to real values
    out = np.full((num_samples, pred_len), np.nan, dtype=np.float32)
    for i, c in enumerate(completions):
        try:
            response   = c["generated_text"].split("### Response:\n")[1]
            dispersed  = model.serializer.inverse_serialize(response)
            pred       = model.discretizer.inverse_discretize(dispersed)
            n = min(len(pred), pred_len)
            out[i, :n] = pred[:n]
        except (IndexError, ValueError):
            pass   # leave row as NaN
    return out


# ---------------------------------------------------------------------------
# Naive intervals
# ---------------------------------------------------------------------------

def predict_intervals(
    model: ChatTime,
    contexts: list,           # list of 1D numpy arrays
    texts: list,              # list of strings
    pred_len: int,
    alpha: float,
    use_text: bool = False,
    num_samples: int = NUM_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each window, generate samples and return α/2 and 1-α/2 quantiles.

    Returns
    -------
    lo, hi  : (N, pred_len) lower/upper bounds
    samples : (N, num_samples, pred_len) raw samples (handy for downstream
              conformal methods so they can reuse the same draws)
    """
    N = len(contexts)
    samples_all = np.full((N, num_samples, pred_len), np.nan, dtype=np.float32)

    for i, (ctx, txt) in enumerate(zip(contexts, texts)):
        ctx_text = txt if (use_text and txt) else None
        samples_all[i] = predict_samples(model, ctx, pred_len, ctx_text, num_samples)

    lo = np.nanquantile(samples_all, alpha / 2,     axis=1)   # (N, pred_len)
    hi = np.nanquantile(samples_all, 1 - alpha / 2, axis=1)   # (N, pred_len)
    return lo, hi, samples_all


def compute_metrics(lo, hi, futures):
    futures = np.asarray(futures)
    covered = (futures >= lo) & (futures <= hi)
    return {
        "coverage": float(covered.mean()),
        "width":    float((hi - lo).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    weekly  = load_weekly_hosp()
    texts   = load_text()
    _, _, test_weekly = split_weekly(weekly)

    test_ctx, test_fut, test_txt, test_keys = make_text_windows(
        test_weekly, texts, CONTEXT_WEEKS, PRED_WEEKS,
        states=EXAMPLE_STATES, stride=STRIDE,
    )
    test_ctx_np = [c.numpy() for c in test_ctx]
    test_fut_np = np.stack([f.numpy() for f in test_fut])
    print(f"Test windows: {len(test_ctx)}")
    print(f"With text   : {sum(1 for t in test_txt if t)}/{len(test_txt)}")

    print(f"\nLoading ChatTime: {MODEL_PATH}")
    model = ChatTime(
        model_path=MODEL_PATH,
        hist_len=CONTEXT_WEEKS,
        pred_len=PRED_WEEKS,
        num_samples=NUM_SAMPLES,
    )

    print("\n--- Naive ChatTime Sampling ---")
    print(f"{'Method':>20}  {'Alpha':>6}  {'Coverage':>10}  {'Width':>10}")
    print("-" * 55)

    for use_text in (False, True):
        label = "Naive (with text)" if use_text else "Naive (no text)"
        for alpha in ALPHAS:
            lo, hi, _ = predict_intervals(
                model, test_ctx_np, test_txt, PRED_WEEKS, alpha, use_text=use_text
            )
            m = compute_metrics(lo, hi, test_fut_np)
            print(f"{label:>20}  {alpha:>6.2f}  "
                  f"{m['coverage']:>10.3f}  {m['width']:>10.2f}")


if __name__ == "__main__":
    main()
