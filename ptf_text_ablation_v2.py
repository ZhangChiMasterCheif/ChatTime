"""
Text ablation v2 for PTF -- a design that can actually separate prompt STYLE
from prompt CONTENT.

Why v2
------
The submitted ablation compares real / shuffled / random(recipe) / no text and
concludes that style rather than content drives the multimodal width benefit.
That design has two holes a reviewer will find immediately:

1. Every PTF description is the SAME template with different slot values:

     "This sequence records traffic flow at a highway in Paris, France, with a
      collection granularity of 1 hour. The target date for prediction is
      {weekday}, {month} {day}, {year}. It is a {weekday|weekend} with
      {condition} and {wind}. The minimum temperature is {tmin} degrees, and
      the maximum temperature is {tmax} degrees. The sun will rise at {sunrise}
      and set at {sunset}."

   So "shuffled text" differs from "real text" by roughly a dozen tokens out of
   eighty. Finding that the two give the same interval width is close to
   tautological, and it cannot distinguish "the model ignores content" from
   "the two prompts are nearly the same string".

2. The out-of-distribution condition is ONE fixed recipe paragraph, so the
   real-vs-random contrast confounds topic with length, vocabulary and
   token count, all at n=1.

This script adds conditions that vary content while holding style exactly
fixed, and conditions that vary style while holding informativeness fixed.

New conditions
--------------
  weekday_flip     Template untouched; only the day-of-week name and the
                   weekday/weekend word are swapped to their opposite. This is
                   the single most predictive field for traffic volume, so if
                   the model reads content at all, this one field should move
                   the intervals. The cleanest possible test of the paper's
                   claim.
  weather_extreme  Template untouched; condition and wind replaced with an
                   extreme alternative and temperatures shifted by -20 degrees.
                   In-distribution style, out-of-distribution content.
  template_blank   Template untouched; every slot value replaced by "unknown".
                   Style with no content at all.
  ood_multi        Five different off-topic paragraphs, cycled across windows,
                   so the OOD result no longer rests on a single paragraph.
  ood_len_matched  Off-topic text truncated/padded to the same word count as
                   the window's real description, removing the length confound.
  field_shuffle    Every template slot permuted INDEPENDENTLY across windows, so
                   each field stays individually plausible but the description
                   is jointly incoherent (a January date with July temperatures,
                   a Sunday label on a Tuesday). Isolates joint semantic
                   coherence from template, vocabulary, length and field
                   structure, all of which are preserved exactly.
  paraphrase       Same field values, different wording and sentence structure.
                   Separates SEMANTIC CONTENT from TEMPLATE FAMILIARITY, which
                   the submitted ablation cannot do:
                     paraphrase ~ real    -> the model reads the content;
                     paraphrase ~ no text -> the benefit is recognition of the
                                             exact training template, not
                                             information.
                   Caveat: paraphrases run ~15 words shorter than the original
                   template, so this arm is not length-matched; read it together
                   with ood_len_matched, which isolates length.

`field_shuffle` and `paraphrase` are the two controls reviewer tjor explicitly
asked for.

Together with the existing no_text / with_text / shuffled_text / random_text
rows this gives a design where "style, not content" is falsifiable:

  * If STYLE drives the effect: real ~ shuffled ~ weekday_flip ~
    weather_extreme ~ template_blank ~ field_shuffle, all tighter than no_text;
    ood_multi ~ ood_len_matched ~ no_text; and paraphrase ~ no_text.
  * If CONTENT drives it: weekday_flip, weather_extreme and field_shuffle should
    be measurably wider than real, template_blank wider still, and paraphrase
    should stay as tight as real.

Cost
----
7 conditions x (200 cal + 200 test) windows x 16 samples. On one A6000 that is
roughly the original with-text run x7, so budget ~8-11 hours. Samples are cached
per condition, so it is safely resumable -- rerun after an interruption and it
picks up the missing caches only. If time is short, run `weekday_flip` and
`paraphrase` first: between them they test content-sensitivity and
template-familiarity, which is most of the argument.

Usage
-----
    python ptf_text_ablation_v2.py                       # all new conditions
    python ptf_text_ablation_v2.py --conditions weekday_flip
    python ptf_text_ablation_v2.py --analyze-only        # no GPU, reads caches
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ptf_dataset import load_ptf, split_ptf, to_window_lists, HIST_LEN, PRED_LEN

ALPHAS = [0.1, 0.2, 0.3]
NUM_SAMPLES = 16
N_CAL_WINDOWS = 200
N_TEST_WINDOWS = 200
SUBSAMPLE_SEED = 42
MODEL_PATH = "ChengsenWang/ChatTime-1-7B-Chat"
CSV_PATH = "ptf_results_v2.csv"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
WEEKENDS = ["Saturday", "Sunday"]

EXTREME_CONDITIONS = [
    "heavy snow and gale force wind",
    "freezing fog and storm force wind",
    "torrential rain and hurricane force wind",
]

OOD_PARAGRAPHS = [
    "To prepare a traditional risotto, warm the stock in a saucepan and keep it "
    "at a gentle simmer. Toast the rice in olive oil until the grains turn "
    "translucent at the edges, then add the wine and stir until absorbed. Add "
    "the stock one ladle at a time, stirring constantly, and finish with cold "
    "butter and grated cheese off the heat.",
    "The mitochondrion is a double-membrane organelle found in most eukaryotic "
    "cells. Its inner membrane is folded into cristae that increase the surface "
    "area available for the electron transport chain. Oxidative phosphorylation "
    "across this membrane produces the majority of the cell's adenosine "
    "triphosphate under aerobic conditions.",
    "In the third movement the strings introduce a descending chromatic figure "
    "that the woodwinds answer a fourth above. The development section moves "
    "through three related keys before the recapitulation restores the tonic. "
    "Critics at the premiere objected to the abrupt dynamic contrasts, which "
    "later performances have generally softened.",
    "The plaintiff argued that the disputed clause was unenforceable because it "
    "had not been separately negotiated. The court declined to reach that "
    "question, holding instead that the contract had already been discharged by "
    "the defendant's earlier repudiation. Costs were awarded on the standard "
    "basis and the counterclaim was dismissed.",
    "Sedimentary basins accumulate material over geological timescales as "
    "subsidence outpaces deposition. Grain size distributions in the resulting "
    "strata record the energy of the transporting medium, so a fining-upward "
    "sequence typically indicates a waning current. Cross-bedding preserves the "
    "orientation of the ancient flow direction.",
]


# ---------------------------------------------------------------------------
# Text transforms -- each keeps the template exactly where it should
# ---------------------------------------------------------------------------

def flip_weekday(text: str, rng) -> str:
    """
    Swap the day name and the weekday/weekend word to the opposite class,
    leaving every other token untouched.

    Traffic flow at a Paris highway differs sharply between weekdays and
    weekends, so this field carries most of the usable signal in the whole
    description. Everything else about the prompt -- length, register,
    vocabulary, sentence structure -- is byte-identical to the real text.
    """
    is_weekend = "weekend" in text
    if is_weekend:
        new_day = str(rng.choice(WEEKDAYS))
        text = re.sub(r"\bweekend\b", "weekday", text)
    else:
        new_day = str(rng.choice(WEEKENDS))
        text = re.sub(r"\bweekday\b", "weekend", text)

    for day in WEEKDAYS + WEEKENDS:
        if re.search(rf"\b{day}\b", text):
            text = re.sub(rf"\b{day}\b", new_day, text, count=1)
            break
    return text


def extreme_weather(text: str, rng) -> str:
    """Replace the weather clause and shift both temperatures by -20 degrees."""
    text = re.sub(
        r"It is a (weekday|weekend) with .*?\. The minimum",
        lambda m: f"It is a {m.group(1)} with {rng.choice(EXTREME_CONDITIONS)}. The minimum",
        text,
    )

    def shift(m):
        return f"{int(m.group(1)) - 20} degrees"

    return re.sub(r"(-?\d+) degrees", shift, text)


def blank_template(text: str, rng) -> str:
    """
    Keep the sentence frame, delete every informative slot value.

    If this is as tight as the real text, the model is reading the frame and
    not the fields.
    """
    text = re.sub(
        r"The target date for prediction is [^.]*\.",
        "The target date for prediction is unknown.",
        text,
    )
    text = re.sub(
        r"It is a (weekday|weekend) with [^.]*\.",
        "It is a day of unknown type with unknown conditions.",
        text,
    )
    text = re.sub(
        r"The minimum temperature is [^.]*\.",
        "The minimum temperature is unknown, and the maximum temperature is unknown.",
        text,
    )
    text = re.sub(
        r"The sun will rise at [^.]*\.",
        "The sun will rise at an unknown time and set at an unknown time.",
        text,
    )
    return text


def ood_multi(text: str, rng, idx: int = 0) -> str:
    """Cycle through five unrelated paragraphs so OOD is not an n=1 claim."""
    return OOD_PARAGRAPHS[idx % len(OOD_PARAGRAPHS)]


def ood_len_matched(text: str, rng, idx: int = 0) -> str:
    """
    Off-topic paragraph trimmed or repeated to match the real description's
    word count, so real-vs-OOD is not confounded with prompt length.
    """
    target_n = len(text.split())
    src = OOD_PARAGRAPHS[idx % len(OOD_PARAGRAPHS)].split()
    while len(src) < target_n:
        src = src + src
    return " ".join(src[:target_n])


# ---------------------------------------------------------------------------
# Field-level parsing, for the two controls reviewer tjor asked for
# ---------------------------------------------------------------------------

FIELD_RE = re.compile(
    r"The target date for prediction is (?P<day_name>\w+), (?P<date>[^.]+)\. "
    r"It is a (?P<daytype>weekday|weekend) with (?P<weather>[^.]+)\. "
    r"The minimum temperature is (?P<tmin>-?\d+) degrees, and the maximum "
    r"temperature is (?P<tmax>-?\d+) degrees\. "
    r"The sun will rise at (?P<sunrise>[\d:]+) and set at (?P<sunset>[\d:]+)\."
)

PREAMBLE = (
    "This sequence records traffic flow at a highway in Paris, France, with a "
    "collection granularity of 1 hour."
)


def parse_fields(text):
    """Extract the template slots; returns None if the text does not match."""
    m = FIELD_RE.search(text)
    return m.groupdict() if m else None


def render_template(f):
    """Re-render the ORIGINAL template from a field dict."""
    return (
        f"{PREAMBLE} The target date for prediction is {f['day_name']}, "
        f"{f['date']}. It is a {f['daytype']} with {f['weather']}. The minimum "
        f"temperature is {f['tmin']} degrees, and the maximum temperature is "
        f"{f['tmax']} degrees. The sun will rise at {f['sunrise']} and set at "
        f"{f['sunset']}."
    )


def render_paraphrase(f):
    """
    Same information, different wording and sentence structure.

    This is the control that separates SEMANTIC CONTENT from TEMPLATE
    FAMILIARITY, which the submitted ablation cannot do. Every field value is
    preserved exactly; only the surface form changes.

      paraphrase ~ real     -> the model is reading the content
      paraphrase ~ no text  -> the model is keyed to the exact template it saw
                               in training, and the "benefit of text" is
                               template recognition rather than information
    """
    daytype = "an ordinary working day" if f["daytype"] == "weekday" else "a weekend day"
    return (
        "Hourly vehicle counts from a motorway in the Paris area are listed "
        f"below. A forecast is needed for {f['day_name']} {f['date']}, "
        f"{daytype}. Conditions were {f['weather']}. Temperatures ran between "
        f"{f['tmin']} and {f['tmax']} degrees. Daylight lasted from "
        f"{f['sunrise']} until {f['sunset']}."
    )


def build_field_shuffled(texts, rng):
    """
    Permute EACH field independently across windows.

    Whole-string shuffling (the submitted `shuffled_text`) moves a coherent
    description from another day. Field-wise shuffling produces a description
    that is individually plausible in every slot but jointly incoherent -- a
    January date with July temperatures and a Sunday label on a Tuesday. The
    template, vocabulary, length and field structure are all preserved, so this
    isolates joint semantic coherence from everything else.
    """
    parsed = [parse_fields(t) for t in texts]
    ok = [i for i, p in enumerate(parsed) if p is not None]
    if not ok:
        return list(texts)

    keys = ["day_name", "date", "daytype", "weather", "tmin", "tmax",
            "sunrise", "sunset"]
    perms = {k: rng.permutation(ok) for k in keys}

    out = list(texts)
    for pos, i in enumerate(ok):
        mixed = {k: parsed[perms[k][pos]][k] for k in keys}
        out[i] = render_template(mixed)
    return out


def build_paraphrase(texts):
    out = []
    for t in texts:
        f = parse_fields(t)
        out.append(render_paraphrase(f) if f else t)
    return out


TRANSFORMS = {
    "weekday_flip": flip_weekday,
    "weather_extreme": extreme_weather,
    "template_blank": blank_template,
    "ood_multi": ood_multi,
    "ood_len_matched": ood_len_matched,
}
# Conditions that need the whole list at once rather than one text at a time.
LIST_TRANSFORMS = {"field_shuffle", "paraphrase"}
INDEXED = {"ood_multi", "ood_len_matched"}
ALL_CONDITIONS = list(TRANSFORMS) + sorted(LIST_TRANSFORMS)


def build_texts(texts, condition, seed):
    rng = np.random.default_rng(seed)
    if condition == "field_shuffle":
        return build_field_shuffled(texts, rng)
    if condition == "paraphrase":
        return build_paraphrase(texts)
    fn = TRANSFORMS[condition]
    if condition in INDEXED:
        return [fn(t, rng, i) for i, t in enumerate(texts)]
    return [fn(t, rng) for t in texts]


# ---------------------------------------------------------------------------
# Windows -- identical subsample to ptf_evaluate.py so rows are comparable
# ---------------------------------------------------------------------------

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


def cache_path(condition, split):
    return f"ptf_v2_samples_{condition}_{split}.npz"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate(model, contexts, texts, path):
    if os.path.exists(path):
        print(f"  cached: {path}")
        return np.load(path)["samples"]

    from chattime_method_naive import predict_samples

    n = len(contexts)
    out = np.full((n, NUM_SAMPLES, PRED_LEN), np.nan, dtype=np.float32)
    for i, (ctx, txt) in enumerate(zip(contexts, texts)):
        out[i] = predict_samples(model, ctx, PRED_LEN, txt or None, NUM_SAMPLES)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n}")
    np.savez_compressed(path, samples=out)
    print(f"  wrote {path}")
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(w, conditions):
    import conformal_lib as cl

    records = []
    for cond in conditions:
        cp, tp = cache_path(cond, "cal"), cache_path(cond, "test")
        if not (os.path.exists(cp) and os.path.exists(tp)):
            print(f"  [skip] no cache for '{cond}'")
            continue
        cal_s = np.load(cp)["samples"]
        test_s = np.load(tp)["samples"]

        for alpha in ALPHAS:
            methods = {}
            methods["Naive"] = cl.sample_quantiles(test_s, alpha)
            methods["CQR (joint)"] = cl.cqr_joint(cal_s, w["cal_fut"], test_s, alpha)
            methods["CQR (per-step)"] = cl.cqr_perstep(cal_s, w["cal_fut"], test_s, alpha)
            for name, (lo, hi) in methods.items():
                m = cl.evaluate_intervals(lo, hi, w["test_fut"], alpha)
                records.append(
                    dict(text=cond, method=name, alpha=alpha,
                         nominal=round(1 - alpha, 2),
                         cov_perstep=m["coverage_perstep"],
                         cov_joint=m["coverage_joint"],
                         width=m["width"], winkler=m["winkler"])
                )
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(CSV_PATH, index=False)
        print(f"\nWrote {CSV_PATH}")
        print(df[df.alpha == 0.10].to_string(index=False,
                                             float_format=lambda v: f"{v:.3f}"))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(conditions, analyze_only):
    w = load_windows()
    print(f"cal={len(w['cal_ctx'])} test={len(w['test_ctx'])} windows")

    print("\nExample transforms on test window 0:")
    print(f"  real           : {w['test_txt'][0][:150]}")
    for cond in conditions:
        ex = build_texts(w["test_txt"][:1], cond, 1234)[0]
        print(f"  {cond:<15}: {ex[:150]}")

    if not analyze_only:
        from model.model import ChatTime

        missing = [
            c for c in conditions
            if not (os.path.exists(cache_path(c, "cal"))
                    and os.path.exists(cache_path(c, "test")))
        ]
        if missing:
            print(f"\nLoading ChatTime: {MODEL_PATH}")
            model = ChatTime(
                model_path=MODEL_PATH, hist_len=HIST_LEN, pred_len=PRED_LEN,
                max_pred_len=PRED_LEN, num_samples=NUM_SAMPLES,
            )
            for cond in missing:
                print(f"\n[{cond}]")
                # Calibration and test get the SAME transform, so the conformal
                # exchangeability assumption is preserved within a condition.
                generate(model, w["cal_ctx"],
                         build_texts(w["cal_txt"], cond, 1234),
                         cache_path(cond, "cal"))
                generate(model, w["test_ctx"],
                         build_texts(w["test_txt"], cond, 1234),
                         cache_path(cond, "test"))

    analyze(w, conditions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args()
    main(a.conditions, a.analyze_only)
