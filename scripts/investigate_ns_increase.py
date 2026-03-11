"""
Investigates why the autoregressive model shows non-switch words becoming MORE
surprising when context is added (Sig Increase) — a pattern absent in the masked model.

Key hypotheses tested:
  1. Words that were very predictable WITHOUT context (low c0 surprisal) are most
     affected — local context made them easy, but discourse context disrupts that.
  2. The context preceding Sig Increase cases contains more English/ASCII tokens,
     shifting the model's language expectations.
  3. Specific POS tags or lexical categories are over-represented.

Usage:
    python scripts/investigate_ns_increase.py
"""

import re
import sys
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUTO_CSV = "results/surprisal/autoregressive/window_1/context_surprisal_diff.csv"
AUTO_FULL = "results/surprisal/autoregressive/window_1/surprisal_results.csv"
OUT_TXT   = "results/surprisal/autoregressive/window_1/ns_increase_examples.txt"


def ascii_token_ratio(text: str) -> float:
    if pd.isna(text):
        return 0.0
    tokens = str(text).split()
    if not tokens:
        return 0.0
    n_ascii = sum(1 for t in tokens if re.match(r"^[A-Za-z0-9]+$", t))
    return n_ascii / len(tokens)


def n_ascii_tokens(text: str) -> int:
    if pd.isna(text):
        return 0
    tokens = str(text).split()
    return sum(1 for t in tokens if re.match(r"^[A-Za-z0-9]+$", t))


def main() -> None:
    flagged = pd.read_csv(AUTO_CSV)
    full    = pd.read_csv(AUTO_FULL)

    # Apply same filters as the main script
    full = full[full["is_propn"] == 0].copy()
    full = full[(full["is_switch"] == 0) | (full["single_worded"] == 1)].copy()

    flagged["n_ascii_ctx"]    = flagged["context"].apply(n_ascii_tokens)
    flagged["ascii_ratio_ctx"] = flagged["context"].apply(ascii_token_ratio)

    ns = flagged[flagged["is_switch"] == 0].copy()
    ns_full = full[full["is_switch"] == 0].copy()
    ns_full["n_ascii_ctx"]     = ns_full["context"].apply(n_ascii_tokens)
    ns_full["ascii_ratio_ctx"] = ns_full["context"].apply(ascii_token_ratio)

    print("=" * 65)
    print("INVESTIGATION: Non-switch Sig Increase in Autoregressive Model")
    print("=" * 65)
    print()

    # ── 1. surprisal_context_0 by group ──────────────────────────────────────
    print("1. surprisal_context_0 distribution by context_effect group (non-switch):")
    for grp, gdf in ns.groupby("context_effect"):
        s = gdf["surprisal_context_0"]
        print(f"   {grp:15s}  n={len(gdf):3d}  "
              f"mean={s.mean():.2f}  median={s.median():.2f}  "
              f"std={s.std():.2f}  min={s.min():.2f}  max={s.max():.2f}")
    # Also baseline: all non-switch in the full filtered dataset
    s_all = ns_full["surprisal_context_0"]
    print(f"   {'(all ns)':15s}  n={len(ns_full):3d}  "
          f"mean={s_all.mean():.2f}  median={s_all.median():.2f}  "
          f"std={s_all.std():.2f}  min={s_all.min():.2f}  max={s_all.max():.2f}")
    print()

    # ── 2. English/ASCII in context ───────────────────────────────────────────
    print("2. English/ASCII tokens in CONTEXT by context_effect group (non-switch):")
    for grp, gdf in ns.groupby("context_effect"):
        r = gdf["ascii_ratio_ctx"]
        n = gdf["n_ascii_ctx"]
        print(f"   {grp:15s}  ascii_ratio: mean={r.mean():.3f}  median={r.median():.3f}  "
              f"n_ascii_tokens: mean={n.mean():.1f}  median={n.median():.1f}")
    r_all = ns_full["ascii_ratio_ctx"]
    n_all = ns_full["n_ascii_ctx"]
    print(f"   {'(all ns)':15s}  ascii_ratio: mean={r_all.mean():.3f}  median={r_all.median():.3f}  "
          f"n_ascii_tokens: mean={n_all.mean():.1f}  median={n_all.median():.1f}")
    print()

    # ── 3. POS breakdown ──────────────────────────────────────────────────────
    if "switch_pos" in ns.columns:
        print("3. Part-of-speech breakdown (non-switch Sig Increase vs Sig Decrease):")
        cross = (
            ns.groupby(["context_effect", "switch_pos"])
            .size()
            .unstack(fill_value=0)
        )
        print(cross.to_string())
        print()

    # ── 4. Scatter: c0 surprisal vs delta for non-switch ─────────────────────
    print("4. Correlation between surprisal_context_0 and delta_c3_c0 (non-switch):")
    if "delta_c3_c0" in ns.columns:
        corr = ns[["surprisal_context_0", "delta_c3_c0"]].corr().iloc[0, 1]
        print(f"   Pearson r (flagged non-switch)       = {corr:.3f}")
    # Compute delta for the full non-switch filtered dataset
    ns_full["delta_c3_c0"] = ns_full["surprisal_context_3"] - ns_full["surprisal_context_0"]
    corr_all = ns_full[["surprisal_context_0", "delta_c3_c0"]].corr().iloc[0, 1]
    print(f"   Pearson r (full non-switch dataset)  = {corr_all:.3f}")
    print()

    # ── 5. Write qualitative examples ────────────────────────────────────────
    ns_inc = ns[ns["context_effect"] == "Sig Increase"].sort_values(
        "delta_c3_c0", ascending=False
    )

    lines = []
    lines.append("NON-SWITCH SIG INCREASE EXAMPLES (Autoregressive / window_1)")
    lines.append("=" * 65)
    lines.append("These are non-switch words whose surprisal INCREASED substantially")
    lines.append("after adding 3 sentences of prior context.")
    lines.append("")

    for i, (_, row) in enumerate(ns_inc.iterrows(), start=1):
        lines.append(f"EXAMPLE {i} of {len(ns_inc)}:")
        lines.append(f"  Word            : {row['word']}")
        if "switch_pos" in row:
            lines.append(f"  POS             : {row['switch_pos']}")
        lines.append(f"  Surprisal c0    : {row['surprisal_context_0']:.3f} bits")
        lines.append(f"  Surprisal c3    : {row['surprisal_context_3']:.3f} bits")
        lines.append(f"  Delta (c3-c0)   : {row['delta_c3_c0']:+.3f} bits")
        lines.append(f"  z-delta         : {row['z_delta']:+.3f}")
        lines.append(f"  ASCII in ctx    : {row['n_ascii_ctx']} tokens ({row['ascii_ratio_ctx']:.1%})")
        lines.append(f"  Sentence        : {row['sentence']}")
        lines.append(f"  Context         : {row['context']}")
        if i < len(ns_inc):
            lines.append("")
            lines.append("─" * 65)
            lines.append("")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Qualitative examples written to: {OUT_TXT}")
    print(f"  ({len(ns_inc)} non-switch Sig Increase cases)")
    print()

    # ── 6. Words with very low c0 surprisal that spiked ──────────────────────
    print("5. Non-switch Sig Increase cases sorted by surprisal_context_0 (lowest first):")
    print("   (low c0 = word was locally predictable but context disrupted that)")
    low_c0 = ns_inc.sort_values("surprisal_context_0")
    show_cols = ["word", "surprisal_context_0", "surprisal_context_3",
                 "delta_c3_c0", "n_ascii_ctx"]
    show_cols = [c for c in show_cols if c in low_c0.columns]
    print(low_c0[show_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
