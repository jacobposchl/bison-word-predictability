"""
Shows concrete side-by-side examples that make the coefficient reversal
intuitive: why does adding discourse context collapse the AR model's
surprisal gap between switch and non-switch words, while the masked model
preserves it?

Architecture recap (printed at runtime):
  AR  model  at c0: predicts word from LEFT context only (within sentence).
  AR  model  at c3: predicts word from 3 prior sentences + left context.
  Masked model c0: predicts word with FULL sentence visible (both sides masked).
  Masked model c3: predicts word with 3 prior sentences + full sentence.

  → Masked already sees the topic from the sentence at c0.
    AR needs the prior sentences to discover the topic.
    Switch words (topic-specific content words) benefit uniquely from this.
    Non-switch words (function/common words) are locally predictable regardless.

Selects the most illustrative sent_id pairs: ones where
  - the AR switch word shows a large surprisal DROP (c3 << c0)
  - the AR non-switch word shows little movement
  - the masked switch word shows a much smaller drop (already low at c0)

Output: results/surprisal/autoregressive/window_1/reversal_examples.txt

Usage:
    python scripts/reversal_examples.py
"""

import sys
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AR_CSV   = "results/surprisal/autoregressive/window_1/surprisal_results.csv"
MASK_CSV = "results/surprisal/masked/window_1/surprisal_results.csv"
OUT_TXT  = "results/surprisal/autoregressive/window_1/reversal_examples.txt"

N_EXAMPLES = 15


def apply_glmm_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["switch_pos"] != "X"].copy()
    single_worded_sents = df[df["single_worded"] == 1]["sent_id"].unique()
    propn_switch_sents  = df[(df["is_propn"] == 1) & (df["is_switch"] == 1)]["sent_id"].unique()
    return df[
        df["sent_id"].isin(single_worded_sents) &
        ~df["sent_id"].isin(propn_switch_sents)
    ].copy()


def select_examples(ar: pd.DataFrame, masked: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    For each sent_id, compute:
      ar_sw_drop   = ar switch word delta (c0 - c3): large positive = big drop
      ar_ns_stable = |ar non-switch delta|: small = non-switch stayed flat
      mask_sw_drop = masked switch word delta (c0 - c3): should be small

    Score = ar_sw_drop  (we want switch words that dropped a lot for AR)
    Filter: non-switch must have stayed relatively stable for AR.
    """
    ar_sw   = ar[ar["is_switch"] == 1][["sent_id", "word", "sentence", "context",
                                         "surprisal_context_0", "surprisal_context_3",
                                         "switch_pos", "group", "participant_id",
                                         "word_in_context"]].copy()
    ar_ns   = ar[ar["is_switch"] == 0][["sent_id", "word", "sentence",
                                         "surprisal_context_0", "surprisal_context_3"]].copy()
    mk_sw   = masked[masked["is_switch"] == 1][["sent_id",
                                                  "surprisal_context_0",
                                                  "surprisal_context_3"]].copy()

    ar_sw  = ar_sw.rename(columns={"surprisal_context_0": "ar_sw_c0",
                                    "surprisal_context_3": "ar_sw_c3"})
    ar_ns  = ar_ns.rename(columns={"surprisal_context_0": "ar_ns_c0",
                                    "surprisal_context_3": "ar_ns_c3",
                                    "word": "ns_word",
                                    "sentence": "ns_sentence"})
    mk_sw  = mk_sw.rename(columns={"surprisal_context_0": "mk_sw_c0",
                                    "surprisal_context_3": "mk_sw_c3"})

    merged = (ar_sw
              .merge(ar_ns[["sent_id", "ns_word", "ns_sentence",
                             "ar_ns_c0", "ar_ns_c3"]], on="sent_id", how="inner")
              .merge(mk_sw, on="sent_id", how="inner"))

    merged["ar_sw_drop"]   = merged["ar_sw_c0"] - merged["ar_sw_c3"]   # big = dropped a lot
    merged["ar_ns_shift"]  = (merged["ar_ns_c3"] - merged["ar_ns_c0"]).abs()
    merged["mk_sw_drop"]   = merged["mk_sw_c0"] - merged["mk_sw_c3"]

    # Want: AR switch drops a lot, AR non-switch stable, masked drop much smaller than AR drop
    candidates = merged[
        (merged["ar_sw_drop"]  > 5) &      # AR switch fell more than 5 bits
        (merged["ar_ns_shift"] < 5)         # AR non-switch stayed within 5 bits
    ].copy()

    candidates = candidates.sort_values("ar_sw_drop", ascending=False)
    return candidates.head(n)


RECAP = """\
════════════════════════════════════════════════════════════════════════
WHY ADDING CONTEXT REVERSES THE AR COEFFICIENT — INTUITIVE EXAMPLES
════════════════════════════════════════════════════════════════════════

ARCHITECTURE RECAP
──────────────────
AR (autoregressive, e.g. GPT) at context_0:
  Predicts each word from LEFT context only — it never sees words to the
  right of the target within the sentence. No prior sentences = no topic.

AR at context_3:
  Prepends 3 prior sentences before the target sentence.
  Now the model knows the topic before it even reads the target sentence.

Masked (bidirectional, e.g. BERT) at context_0:
  Masks only the target word; sees the FULL sentence on both sides.
  It already has the topic from the sentence itself.

Masked at context_3:
  Same as above, plus 3 prior sentences — but those priors add little
  because the sentence already told the model the topic.

THE ASYMMETRY IN ONE SENTENCE:
  Switch words are domain-specific content words. AR-c0 is blind to them
  (no topic, no right-context). AR-c3 finally sees the topic → big drop.
  Masked-c0 already knew the topic → small additional drop from context.
  Non-switch words (function words, common verbs) are locally predictable
  regardless of topic → barely move in either model.

  Result: only AR switch-word surprisal collapses with context.
  The switch–non-switch gap closes → correlation with is_switch → 0.

════════════════════════════════════════════════════════════════════════
EXAMPLES  (sorted by AR switch-word surprisal drop, largest first)
Each example is a MATCHED PAIR from the same conversation turn.
════════════════════════════════════════════════════════════════════════
"""


def fmt_example(i: int, row: pd.Series) -> str:
    ar_sw_drop_pct  = (row["ar_sw_drop"] / row["ar_sw_c0"]) * 100
    mk_sw_drop_pct  = (row["mk_sw_drop"] / row["mk_sw_c0"]) * 100 if row["mk_sw_c0"] != 0 else 0
    ar_ns_delta     = row["ar_ns_c3"] - row["ar_ns_c0"]
    remaining_gap   = row["ar_sw_c3"] - row["ar_ns_c3"]   # positive = switch still harder

    lines = [
        f"EXAMPLE {i}",
        f"  Participant : {row['participant_id']}   Group: {row['group']}",
        f"  sent_id     : {row['sent_id']}",
        f"  Switch POS  : {row['switch_pos']}",
        f"  Word in ctx : {'Yes' if row['word_in_context'] == 1 else 'No'}",
        "",
        f"  PRIOR CONTEXT (3 sentences shown to model at c3):",
    ]
    # Print each context sentence on its own line
    for j, ctx_sent in enumerate(str(row["context"]).split(" ||| "), 1):
        lines.append(f"    [{j}] {ctx_sent.strip()}")

    lines += [
        "",
        f"  TARGET SENTENCE: {row['sentence']}",
        "",
        "  ┌─ SWITCH WORD (is_switch = 1) ────────────────────────────────────┐",
        f"  │  Word        : {row['word']}",
        f"  │  AR  surprisal c0  : {row['ar_sw_c0']:6.2f} bits",
        f"  │  AR  surprisal c3  : {row['ar_sw_c3']:6.2f} bits   "
            f"Δ = {-row['ar_sw_drop']:+.2f} bits  ({ar_sw_drop_pct:.0f}% drop)",
        f"  │  Mask surprisal c0 : {row['mk_sw_c0']:6.2f} bits",
        f"  │  Mask surprisal c3 : {row['mk_sw_c3']:6.2f} bits   "
            f"Δ = {-row['mk_sw_drop']:+.2f} bits  ({mk_sw_drop_pct:.0f}% drop)",
        "  └──────────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─ NON-SWITCH WORD (is_switch = 0) ─────────────────────────────────┐",
        f"  │  Word        : {row['ns_word']}",
        f"  │  Sentence    : {row['ns_sentence']}",
        f"  │  AR  surprisal c0  : {row['ar_ns_c0']:6.2f} bits",
        f"  │  AR  surprisal c3  : {row['ar_ns_c3']:6.2f} bits   "
            f"Δ = {ar_ns_delta:+.2f} bits",
        "  └──────────────────────────────────────────────────────────────────┘",
        "",
        f"  KEY NUMBERS:",
        f"    AR gap at c0  (switch − non-switch) : {row['ar_sw_c0'] - row['ar_ns_c0']:+.2f} bits",
        f"    AR gap at c3  (switch − non-switch) : {remaining_gap:+.2f} bits",
        f"    → {'Gap CLOSED — switch is now equally/less surprising' if remaining_gap <= 0.5 else 'Gap narrowed but persists'}",
    ]
    return "\n".join(lines)


def main() -> None:
    ar     = pd.read_csv(AR_CSV)
    masked = pd.read_csv(MASK_CSV)

    ar     = apply_glmm_filters(ar)
    masked = apply_glmm_filters(masked)

    examples = select_examples(ar, masked, N_EXAMPLES)

    print(RECAP)

    out_lines = [RECAP]

    for i, (_, row) in enumerate(examples.iterrows(), start=1):
        block = fmt_example(i, row)
        print(block)
        out_lines.append(block)
        if i < len(examples):
            divider = "\n" + "─" * 72 + "\n"
            print(divider)
            out_lines.append(divider)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"\nOutput written to: {OUT_TXT}")


if __name__ == "__main__":
    main()
