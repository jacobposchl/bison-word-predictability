"""
Investigates why increasing prior-sentence context for the autoregressive model
reverses the direction of the surprisal → code-switching GLMM coefficient, while
the masked model shows no such reversal.

Four angles of attack:
  1. Word-level mean surprisal trajectory (context 0–3) by switch status
     — shows whether switch words become *more* predictable with context,
       non-switch words less so, or both.
  2. Point-biserial correlation between surprisal and is_switch at each context
     level — directly maps to the GLMM β sign at each level.
  3. word_in_context decomposition — if the switch word appeared verbatim in the
     prior sentences, the AR model will have already seen it and assign low
     surprisal. Tests whether removing word_in_context==1 cases restores the
     positive correlation.
  4. English (ASCII) content of prior context for switch vs. non-switch words —
     tests whether switch-word sentences have more English in their context,
     which would shift an AR model's language distribution toward English and
     make Chinese words more surprising.

Filters applied to match the R GLMM (surprisal_single_no_propn):
  - switch_pos != "X"
  - sent_ids where any row has single_worded == 1
  - sent_ids where no row has is_propn == 1 AND is_switch == 1

Usage:
    python scripts/investigate_coefficient_reversal.py
"""

import re
import sys
import pandas as pd
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AR_CSV   = "results/surprisal/autoregressive/window_1/surprisal_results.csv"
MASK_CSV = "results/surprisal/masked/window_1/surprisal_results.csv"

CONTEXT_LEVELS = [0, 1, 2, 3]


# ── Helpers ───────────────────────────────────────────────────────────────────

def ascii_ratio(text: str) -> float:
    if pd.isna(text):
        return 0.0
    tokens = str(text).split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if re.match(r"^[A-Za-z0-9]+$", t)) / len(tokens)


def pbr(col: pd.Series, binary: pd.Series) -> float:
    """Pearson r between a continuous series and a binary series (= point-biserial r)."""
    combined = pd.DataFrame({"x": col, "y": binary}).dropna()
    if len(combined) < 5:
        return float("nan")
    return float(np.corrcoef(combined["x"], combined["y"])[0, 1])


def apply_glmm_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the R GLMM dataset filters for surprisal_single_no_propn."""
    df = df[df["switch_pos"] != "X"].copy()
    single_worded_sents  = df[df["single_worded"] == 1]["sent_id"].unique()
    propn_switch_sents   = df[(df["is_propn"] == 1) & (df["is_switch"] == 1)]["sent_id"].unique()
    return df[
        df["sent_id"].isin(single_worded_sents) &
        ~df["sent_id"].isin(propn_switch_sents)
    ].copy()


def sep(char="─", width=70):
    print(char * width)


# ── Analysis sections ─────────────────────────────────────────────────────────

def section_trajectory(df: pd.DataFrame) -> None:
    """Mean surprisal at each context level, split by switch status."""
    print("1. MEAN SURPRISAL TRAJECTORY (word-level, by switch status)")
    sep()
    header = f"  {'Context':<12}  {'Switch':>10}  {'Non-switch':>12}  {'Diff (CS−NS)':>14}"
    print(header)
    for i in CONTEXT_LEVELS:
        col = f"surprisal_context_{i}"
        sub = df[[col, "is_switch"]].dropna()
        cs  = sub[sub["is_switch"] == 1][col].mean()
        ns  = sub[sub["is_switch"] == 0][col].mean()
        print(f"  {'context_'+str(i):<12}  {cs:>10.3f}  {ns:>12.3f}  {cs-ns:>+14.3f}")
    print()


def section_correlation(df: pd.DataFrame) -> None:
    """Point-biserial r between surprisal and is_switch at each context level."""
    print("2. POINT-BISERIAL CORRELATION  r(surprisal, is_switch)")
    print("   Positive r = higher surprisal → more likely to be a switch word (matches c0 GLMM β)")
    print("   Negative r = REVERSAL")
    sep()
    print(f"  {'Context':<12}  {'r':>8}  {'n':>6}  Note")
    for i in CONTEXT_LEVELS:
        col = f"surprisal_context_{i}"
        sub = df[[col, "is_switch"]].dropna()
        r   = pbr(sub[col], sub["is_switch"])
        note = ""
        if r < -0.01:
            note = "<-- REVERSED"
        elif abs(r) < 0.01:
            note = "(near zero)"
        print(f"  {'context_'+str(i):<12}  {r:>+8.4f}  {len(sub):>6}  {note}")
    print()


def section_word_in_context(df: pd.DataFrame) -> None:
    """
    Tests whether removing word_in_context==1 cases restores the positive correlation.
    If the switch word appeared verbatim in the preceding sentences, the AR model
    will have already processed it and will assign it low surprisal — which would
    create a spurious negative relationship between surprisal and is_switch.
    """
    if "word_in_context" not in df.columns:
        print("3. WORD-IN-CONTEXT DECOMPOSITION — column not found, skipping.")
        return

    print("3. WORD-IN-CONTEXT DECOMPOSITION")
    print("   Hypothesis: switch words that appear in prior context get low surprisal")
    print("   from the AR model (it already saw them), artificially reversing the correlation.")
    sep()

    n_wic = (df["word_in_context"] == 1).sum()
    n_sw  = (df["is_switch"] == 1).sum()
    print(f"  word_in_context==1 rows : {n_wic} ({n_wic/len(df)*100:.1f}% of dataset)")
    print(f"  is_switch==1 rows       : {n_sw}")
    wic_in_sw = ((df["is_switch"] == 1) & (df["word_in_context"] == 1)).sum()
    print(f"  Switch words in context : {wic_in_sw} ({wic_in_sw/n_sw*100:.1f}% of switch words)")
    print()

    for subset_label, mask in [
        ("All rows",                         pd.Series([True] * len(df), index=df.index)),
        ("word NOT in context (wic==0)",      df["word_in_context"] == 0),
        ("word IS  in context (wic==1)",      df["word_in_context"] == 1),
    ]:
        sub = df[mask]
        print(f"  [{subset_label}, n={len(sub)}]")
        print(f"  {'Context':<12}  {'r':>8}  {'Switch mean surp':>17}  {'NS mean surp':>13}")
        for i in CONTEXT_LEVELS:
            col = f"surprisal_context_{i}"
            s = sub[[col, "is_switch"]].dropna()
            r  = pbr(s[col], s["is_switch"])
            cs = s[s["is_switch"] == 1][col].mean()
            ns = s[s["is_switch"] == 0][col].mean()
            print(f"  {'context_'+str(i):<12}  {r:>+8.4f}  {cs:>17.3f}  {ns:>13.3f}")
        print()


def section_english_context(df: pd.DataFrame) -> None:
    """
    Tests whether switch-word sentences have more English in their prior context.
    If so, the AR model (left-to-right) would shift its language distribution toward
    English after processing that context, making Chinese non-switch words MORE
    surprising and English switch words LESS surprising — flipping the coefficient.
    """
    if "context" not in df.columns:
        print("4. ENGLISH CONTEXT ANALYSIS — context column not found, skipping.")
        return

    print("4. ENGLISH (ASCII) TOKEN RATIO IN PRIOR CONTEXT")
    print("   Hypothesis: switch-word sentences have more English in their context,")
    print("   biasing the AR model toward English and reversing the surprisal ranking.")
    sep()

    df = df.copy()
    df["ascii_ratio_ctx"] = df["context"].apply(ascii_ratio)
    df["delta_c3_c0"]     = df["surprisal_context_3"] - df["surprisal_context_0"]

    # Mean ASCII ratio by switch status
    print(f"  {'Group':<30}  {'mean ascii ratio':>16}  {'median':>8}  n")
    for sw, label in [(1, "Switch words"), (0, "Non-switch words")]:
        sub = df[df["is_switch"] == sw]["ascii_ratio_ctx"]
        print(f"  {label:<30}  {sub.mean():>16.4f}  {sub.median():>8.4f}  {len(sub)}")
    print()

    # Does higher ASCII ratio correlate with the delta (c3−c0)?
    print("  Correlation between ascii_ratio_ctx and delta(c3−c0):")
    print("  (negative r = more English in context → bigger drop in surprisal with context)")
    for sw, label in [(1, "Switch words"), (0, "Non-switch words")]:
        sub = df[df["is_switch"] == sw][["ascii_ratio_ctx", "delta_c3_c0"]].dropna()
        r = pbr(sub["ascii_ratio_ctx"], sub["delta_c3_c0"])
        print(f"    {label:<20}: r = {r:+.4f}  (n={len(sub)})")
    print()

    # Binary split: any English in context vs. none
    print("  Mean delta(c3−c0) by whether context contains ANY English:")
    df["has_english_ctx"] = df["ascii_ratio_ctx"] > 0
    for sw, label in [(1, "switch"), (0, "non-switch")]:
        sub = df[df["is_switch"] == sw]
        tbl = sub.groupby("has_english_ctx")["delta_c3_c0"].agg(["mean", "count"])
        tbl.index = tbl.index.map({False: "no English", True: "has English"})
        print(f"    {label}:")
        for q, row in tbl.iterrows():
            print(f"      {q}: mean delta = {row['mean']:+.3f}  (n={int(row['count'])})")
    print()


def section_synthesis(df: pd.DataFrame) -> None:
    """
    Summarises how much of the surprisal drop for switch words is explained by:
      (a) the switch word shrinking  vs.  (b) non-switch words growing.
    Also reports mean surprisal drop by switch status to show the asymmetry.
    """
    print("5. SYNTHESIS — what drives the convergence?")
    sep()

    df = df.copy()
    df["delta_c3_c0"] = df["surprisal_context_3"] - df["surprisal_context_0"]

    for sw, label in [(1, "Switch words"), (0, "Non-switch words")]:
        sub = df[df["is_switch"] == sw]["delta_c3_c0"].dropna()
        pct_negative = (sub < 0).mean() * 100
        print(f"  {label} (n={len(sub)}):")
        print(f"    Mean delta(c3−c0)   : {sub.mean():+.3f} bits")
        print(f"    Median delta(c3−c0) : {sub.median():+.3f} bits")
        print(f"    % that DECREASED    : {pct_negative:.1f}%")
        print(f"    % that INCREASED    : {100-pct_negative:.1f}%")
        print()

    # How much does each group's movement contribute to the correlation shift?
    c0 = df["surprisal_context_0"]
    c3 = df["surprisal_context_3"]
    sw = df["is_switch"]
    r0 = pbr(c0, sw)
    r3 = pbr(c3.reindex(c0.index), sw)

    # Counterfactual 1: freeze switch words at c0, let non-switch move to c3
    c_freeze_sw = df.apply(
        lambda r: r["surprisal_context_0"] if r["is_switch"] == 1
        else r["surprisal_context_3"], axis=1
    )
    r_freeze_sw = pbr(c_freeze_sw, sw)

    # Counterfactual 2: freeze non-switch words at c0, let switch move to c3
    c_freeze_ns = df.apply(
        lambda r: r["surprisal_context_3"] if r["is_switch"] == 1
        else r["surprisal_context_0"], axis=1
    )
    r_freeze_ns = pbr(c_freeze_ns, sw)

    print("  Counterfactual correlations (to isolate which group drives the shift):")
    print(f"    Actual context_0                         : r = {r0:+.4f}")
    print(f"    Actual context_3                         : r = {r3:+.4f}")
    print(f"    Freeze switch @ c0, move non-switch to c3: r = {r_freeze_sw:+.4f}")
    print(f"      → shows effect of non-switch words moving alone")
    print(f"    Freeze non-switch @ c0, move switch to c3: r = {r_freeze_ns:+.4f}")
    print(f"      → shows effect of switch words moving alone")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(csv_path: str, label: str) -> None:
    df = pd.read_csv(csv_path)
    df = apply_glmm_filters(df)

    print()
    sep("═")
    print(f"  MODEL: {label}  |  n = {len(df)} rows after GLMM filters")
    sep("═")
    print()

    section_trajectory(df)
    section_correlation(df)
    section_word_in_context(df)
    section_english_context(df)
    section_synthesis(df)


if __name__ == "__main__":
    run(AR_CSV,   "Autoregressive / window_1")
    run(MASK_CSV, "Masked / window_1")
