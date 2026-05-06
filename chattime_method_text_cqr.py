"""
Text-Weighted CQR for ChatTime — locally adaptive conformal prediction
where the kernel is text similarity (cosine on sentence-BERT embeddings).

Algorithm
---------
  1. Embed every calibration text and every test text with sentence-BERT.
  2. For each test point j, compute weights
         w_i ∝ exp(γ · cos(e_test_j, e_cal_i))
     against the calibration embeddings.
  3. Compute test-specific Q̂_j = weighted-quantile(scores, w, α).
  4. Inflate the test interval by Q̂_j.

Calibration scores can come from either the no-text or with-text run; we use
the *with-text* base intervals here so the comparison isolates the
*localization* contribution beyond the global text effect.

Requires:
    pip install sentence-transformers
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer

from model.model import ChatTime
from chattime_dataset import (
    load_weekly_hosp, load_text, split_weekly, make_text_windows,
)
from chattime_method_naive import (
    predict_intervals, compute_metrics,
    MODEL_PATH, CONTEXT_WEEKS, PRED_WEEKS, NUM_SAMPLES, ALPHAS,
    EXAMPLE_STATES, STRIDE,
)
from chattime_method_cqr import cqr_scores, cqr_quantile, apply_correction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 22M params, fast
TEMPERATURE = 1.0   # softmax-style temperature on cosine similarity


# ---------------------------------------------------------------------------
# Text embedding + weighted quantile
# ---------------------------------------------------------------------------

def embed_texts(sbert: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """L2-normalized sentence-BERT embeddings, shape (N, d)."""
    # empty strings get a zero vector — they will produce uniform-ish weights
    embs = sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs)


def text_weights(
    test_emb: np.ndarray,    # (d,)
    cal_embs: np.ndarray,    # (N, d)
    temperature: float = TEMPERATURE,
) -> np.ndarray:
    """
    w_i = exp( T · cos(e_test, e_cal_i) ).  Returns unnormalized (N,).
    Higher T → sharper localization. T → 0 recovers uniform weights.
    """
    cos = cal_embs @ test_emb            # both L2-normalized
    return np.exp(temperature * cos)


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float
                      ) -> float:
    """
    Weighted (1-α)-quantile with a +∞ ghost point at weight 1/(N+1)
    (Tibshirani et al. 2019 finite-sample correction). Falls back to the
    largest calibration score if the ghost is selected.
    """
    N = len(scores)
    w = weights / (weights.sum() + 1e-12)
    w = w * (N / (N + 1))   # leave 1/(N+1) for the ghost

    order      = np.argsort(scores)
    sorted_s   = scores[order]
    sorted_w   = w[order]
    cum_w      = np.cumsum(sorted_w)

    target = 1.0 - alpha
    idx    = np.searchsorted(cum_w, target)
    if idx >= N:
        return float(sorted_s[-1])   # most conservative finite score
    return float(sorted_s[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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

    print(f"Cal: {len(cal_ctx_np)} windows | Test: {len(test_ctx_np)} windows")

    print(f"\nLoading sentence-BERT: {SBERT_MODEL}")
    sbert     = SentenceTransformer(SBERT_MODEL)
    cal_embs  = embed_texts(sbert, cal_txt)
    test_embs = embed_texts(sbert, test_txt)
    print(f"  cal embs : {cal_embs.shape}")
    print(f"  test embs: {test_embs.shape}")

    print(f"\nLoading ChatTime: {MODEL_PATH}")
    model = ChatTime(
        model_path=MODEL_PATH,
        hist_len=CONTEXT_WEEKS,
        pred_len=PRED_WEEKS,
        num_samples=NUM_SAMPLES,
    )

    print("\n--- Text-Weighted CQR for ChatTime ---")
    print(f"{'Alpha':>6}  {'Coverage':>10}  {'Width':>10}  {'Q̂_mean':>10}")
    print("-" * 45)

    for alpha in ALPHAS:
        # Calibration with text: get base quantile intervals and CQR scores
        cal_lo, cal_hi, _ = predict_intervals(
            model, cal_ctx_np, cal_txt, PRED_WEEKS, alpha, use_text=True
        )
        scores = cqr_scores(cal_lo, cal_hi, cal_fut_np)

        # Test: per-window text-weighted Q̂
        test_lo, test_hi, _ = predict_intervals(
            model, test_ctx_np, test_txt, PRED_WEEKS, alpha, use_text=True
        )

        N_test = len(test_ctx_np)
        lo_all = np.empty_like(test_lo)
        hi_all = np.empty_like(test_hi)
        Q_hats = np.empty(N_test)

        for j in range(N_test):
            w        = text_weights(test_embs[j], cal_embs)
            Q_hat_j  = weighted_quantile(scores, w, alpha)
            lo_all[j] = test_lo[j] - Q_hat_j
            hi_all[j] = test_hi[j] + Q_hat_j
            Q_hats[j] = Q_hat_j

        m = compute_metrics(lo_all, hi_all, test_fut_np)
        print(f"{alpha:>6.2f}  {m['coverage']:>10.3f}  {m['width']:>10.2f}  "
              f"{Q_hats.mean():>10.2f}")


if __name__ == "__main__":
    main()
