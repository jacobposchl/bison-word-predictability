"""
Identifies rows where surprisal_context_3 differs substantially from
surprisal_context_0 for both autoregressive and masked window_1 surprisal results.

This is useful for understanding the coefficient reversal observed in the
GLMMs: surprisal_context_0 predicts code-switching positively, while
surprisal_context_3 predicts it negatively.

Outputs a CSV of high-delta cases sorted by absolute difference, with
relevant metadata for qualitative inspection, and a .txt of top-3 examples.

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
DEFAULT_Z_THRESHOLD = 2.0

MODELS = [
    {
        "label":      "Autoregressive / window_1",
        "input_csv":  "results/surprisal/autoregressive/window_1/surprisal_results.csv",
        "output_csv": "results/surprisal/autoregressive/window_1/context_surprisal_diff.csv",
        "output_txt": "results/surprisal/autoregressive/window_1/context_surprisal_examples.txt",
    },
    {
        "label":      "Masked / window_1",
        "input_csv":  "results/surprisal/masked/window_1/surprisal_results.csv",
        "output_csv": "results/surprisal/masked/window_1/context_surprisal_diff.csv",
        "output_txt": "results/surprisal/masked/window_1/context_surprisal_examples.txt",
    },
]


def main(label: str, input_csv: str, output_csv: str, output_txt: str,
         z_threshold: float, switch_only: bool) -> None:
    df = pd.read_csv(input_csv)

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

    df["context_effect"] = df.apply(
        lambda r: (
            "Sig Increase" if r["z_delta"] >= z_threshold
            else "Sig Decrease" if r["z_delta"] <= -z_threshold
            else "No Sig"
        ),
        axis=1,
    )

    # ── Filter ────────────────────────────────────────────────────────────────
    flagged = df[df["context_effect"] != "No Sig"].copy()
    flagged = flagged.sort_values("abs_delta", ascending=False)

    # ── Build output: flagged rows + their sent_id pairs, interleaved ─────────
    paired_df = df[df["sent_id"].isin(flagged["sent_id"].unique())].copy()
    sent_max = flagged.groupby("sent_id")["abs_delta"].max()
    paired_df["_group_rank"] = paired_df["sent_id"].map(sent_max)
    paired_df["_is_flagged"] = (paired_df["context_effect"] != "No Sig").astype(int)
    paired_df = paired_df.sort_values(
        ["_group_rank", "sent_id", "_is_flagged", "abs_delta"],
        ascending=[False, True, False, False],
    ).drop(columns=["_group_rank", "_is_flagged"])

    out_cols = [
        "sent_id", "context_effect", "is_switch", "word", "switch_pos",
        "surprisal_context_0", "surprisal_context_3",
        "sentence", "context", "delta_c3_c0", "z_delta",
    ]
    out_cols = [c for c in out_cols if c in paired_df.columns]
    paired_df[out_cols].to_csv(output_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(df)
    n_flagged = len(flagged)

    print("=" * 60)
    print(f"CONTEXT SURPRISAL DIFFERENCE ANALYSIS  [{label}]")
    print("=" * 60)
    print(f"  Input            : {input_csv}")
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
    for label_i, group_df in df.groupby("is_switch"):
        name = "switch" if label_i == 1 else "non-switch"
        d = group_df["delta_c3_c0"].describe(percentiles=[0.25, 0.50, 0.75])
        print(f"  {name} (n={len(group_df)}):")
        print(d.round(3).rename(index=lambda s: f"    {s}").to_string())
        print()

    # ── Absolute surprisal levels by switch status ────────────────────────────
    print("Absolute surprisal levels by switch status (mean ± std):")
    abs_levels = (
        df.groupby("is_switch")[["surprisal_context_0", "surprisal_context_3"]]
        .agg(["mean", "std"])
    )
    abs_levels.index = abs_levels.index.map({0: "non-switch", 1: "switch"})
    for group_name, row in abs_levels.iterrows():
        c0_mean, c0_std = row[("surprisal_context_0", "mean")], row[("surprisal_context_0", "std")]
        c3_mean, c3_std = row[("surprisal_context_3", "mean")], row[("surprisal_context_3", "std")]
        delta = c3_mean - c0_mean
        print(f"  {group_name}:")
        print(f"    context_0  : {c0_mean:.3f} ± {c0_std:.3f} bits")
        print(f"    context_3  : {c3_mean:.3f} ± {c3_std:.3f} bits")
        print(f"    shift      : {delta:+.3f} bits")
        print()

    print(f"Output written to  : {output_csv}")

    # ── Write top-3 examples to .txt ──────────────────────────────────────────
    topx = flagged[flagged["is_switch"] == 1].head(25)
    lines = []
    for i, (_, cs_row) in enumerate(topx.iterrows(), start=1):
        pair = df[(df["sent_id"] == cs_row["sent_id"]) & (df["is_switch"] == 0)]

        lines.append(f"EXAMPLE {i}:")
        lines.append("")
        lines.append("CS-Sentence")
        lines.append(f"Switch Word: {cs_row['word']}")
        lines.append(f"Code-switched Sentence: {cs_row['sentence']}")
        lines.append(f"Context: {cs_row['context']}")
        lines.append(f"Surprisal w/ Context 0: {cs_row['surprisal_context_0']:.3f}")
        lines.append(f"Surprisal w/ Context 3: {cs_row['surprisal_context_3']:.3f}")
        lines.append("")
        lines.append("Paired-Unilingual-Sentence")
        if not pair.empty:
            uni_row = pair.iloc[0]
            lines.append(f"Switch Word: {uni_row['word']}")
            lines.append(f"Unilingual Sentence: {uni_row['sentence']}")
            lines.append(f"Context: {uni_row['context']}")
            lines.append(f"Surprisal w/ Context 0: {uni_row['surprisal_context_0']:.3f}")
            lines.append(f"Surprisal w/ Context 3: {uni_row['surprisal_context_3']:.3f}")
        else:
            lines.append("  (no paired unilingual row found for this sent_id)")
        if i < len(topx):
            lines.append("")
            lines.append("─" * 60)
            lines.append("")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Examples written to : {output_txt}")
    print()


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

    for model in MODELS:
        main(
            label=model["label"],
            input_csv=model["input_csv"],
            output_csv=model["output_csv"],
            output_txt=model["output_txt"],
            z_threshold=args.z_threshold,
            switch_only=args.switch_only,
        )
