"""
Identifies rows where surprisal_context_3 differs substantially from
surprisal_context_0 in the autoregressive window_1 surprisal results.

This is useful for understanding the coefficient reversal observed in the
GLMMs: surprisal_context_0 predicts code-switching positively, while
surprisal_context_3 predicts it negatively.

Outputs a CSV of high-delta cases sorted by absolute difference, with
relevant metadata for qualitative inspection.

Usage:
    python scripts/context_surprisal_diff.py
    python scripts/context_surprisal_diff.py --z-threshold 1.5
    python scripts/context_surprisal_diff.py --switch-only
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Allow Unicode output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_CSV  = "results/surprisal/autoregressive/window_1/surprisal_results.csv"
OUTPUT_CSV = "results/surprisal/autoregressive/window_1/context_surprisal_diff.csv"
DEFAULT_Z_THRESHOLD = 2.0

KEEP_COLS = [
    "sent_id", "is_switch", "participant_id", "group",
    "word", "switch_pos", "is_propn", "single_worded",
    "word_length", "normalized_switch_point", "word_in_context",
    "surprisal_context_0", "surprisal_context_1",
    "surprisal_context_2", "surprisal_context_3",
    "entropy_context_0",   "entropy_context_3",
    "sentence", "context", "original_sentence",
]


def main(z_threshold: float, switch_only: bool) -> None:
    df = pd.read_csv(INPUT_CSV)

    # ── Filtering ─────────────────────────────────────────────────────────────
    # Exclude proper nouns
    df = df[df["is_propn"] == 0].copy()

    # For switch words: require single_worded == 1
    # For non-switch words: no single_worded restriction
    df = df[(df["is_switch"] == 0) | (df["single_worded"] == 1)].copy()

    # Optionally restrict to switch words only
    if switch_only:
        df = df[df["is_switch"] == 1].copy()
        print(f"Restricted to is_switch == 1: {len(df)} rows")

    # ── Compute deltas ────────────────────────────────────────────────────────
    df["delta_c3_c0"] = df["surprisal_context_3"] - df["surprisal_context_0"]
    df["abs_delta"]   = df["delta_c3_c0"].abs()

    mean_delta = df["delta_c3_c0"].mean()
    std_delta  = df["delta_c3_c0"].std()
    df["z_delta"] = (df["delta_c3_c0"] - mean_delta) / std_delta

    df["context_effect"] = df["delta_c3_c0"].apply(
        lambda x: "decreased_with_context" if x < 0 else "increased_with_context"
    )

    # ── Filter ────────────────────────────────────────────────────────────────
    flagged = df[df["z_delta"].abs() >= z_threshold].copy()
    flagged = flagged.sort_values("abs_delta", ascending=False)

    # ── Build output ──────────────────────────────────────────────────────────
    computed = ["delta_c3_c0", "abs_delta", "z_delta", "context_effect"]
    out_cols = [c for c in KEEP_COLS if c in flagged.columns] + computed
    flagged[out_cols].to_csv(OUTPUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(df)
    n_flagged = len(flagged)

    print("=" * 60)
    print("CONTEXT SURPRISAL DIFFERENCE ANALYSIS")
    print("=" * 60)
    print(f"  Input            : {INPUT_CSV}")
    print(f"  Total rows       : {total}")
    print(f"  Z-threshold      : |z| >= {z_threshold}")
    print(f"  Mean delta       : {mean_delta:.3f}  bits")
    print(f"  Std  delta       : {std_delta:.3f}  bits")
    print(f"  Flagged rows     : {n_flagged}  ({n_flagged/total*100:.1f}%)")
    print()

    # Breakdown by switch status
    print("Flagged rows by is_switch:")
    switch_counts = flagged["is_switch"].value_counts().rename({0: "non-switch", 1: "switch"})
    print(switch_counts.to_string())
    print()

    # Breakdown by switch × direction
    if not flagged.empty:
        cross = (
            flagged
            .groupby(["is_switch", "context_effect"])
            .size()
            .unstack(fill_value=0)
            .rename(index={0: "non-switch", 1: "switch"})
        )
        print("Flagged rows by switch status × context effect direction:")
        print(cross.to_string())
        print()

        # Top-10 largest absolute deltas
        print("Top 10 largest |delta| cases:")
        top10_cols = [
            "is_switch", "word", "switch_pos", "group", "participant_id",
            "surprisal_context_0", "surprisal_context_3",
            "delta_c3_c0", "z_delta", "context_effect",
        ]
        top10_cols = [c for c in top10_cols if c in flagged.columns]
        print(flagged[top10_cols].head(10).to_string(index=False))
        print()

    # ── Distribution summary of the full filtered dataset ────────────────────
    print("Distribution of delta_c3_c0 (full filtered dataset, before z-threshold):")
    dist = df["delta_c3_c0"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    print(dist.round(3).to_string())
    print()

    print("Distribution split by is_switch:")
    for label, group_df in df.groupby("is_switch"):
        name = "switch" if label == 1 else "non-switch"
        d = group_df["delta_c3_c0"].describe(percentiles=[0.25, 0.50, 0.75])
        print(f"  {name} (n={len(group_df)}):")
        print(d.round(3).rename(index=lambda s: f"    {s}").to_string())
        print()

    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flag rows where surprisal_context_3 differs substantially from surprisal_context_0."
    )
    parser.add_argument(
        "--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD,
        help=f"Flag rows where |z-score of delta| >= this value (default: {DEFAULT_Z_THRESHOLD})"
    )
    parser.add_argument(
        "--switch-only", action="store_true",
        help="Restrict analysis to code-switch words only (is_switch == 1)"
    )
    args = parser.parse_args()
    main(z_threshold=args.z_threshold, switch_only=args.switch_only)
