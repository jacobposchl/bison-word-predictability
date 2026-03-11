"""
Explore truncation_context_* columns in surprisal results CSVs.
Finds all rows where any truncation column is not "clean".
"""

import sys
import pandas as pd
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "autoregressive": REPO_ROOT / "results/surprisal/autoregressive/window_1/surprisal_results.csv",
    "masked":         REPO_ROOT / "results/surprisal/masked/window_1/surprisal_results.csv",
}


def explore_truncations(label: str, path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {label.upper()}  |  {path.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    trunc_cols = [c for c in df.columns if c.startswith("truncation_context_")]
    print(f"Truncation columns found: {trunc_cols}")
    print(f"Total rows: {len(df)}")

    # Value counts per column
    print("\n--- Value counts per truncation column ---")
    for col in trunc_cols:
        vc = df[col].value_counts(dropna=False)
        print(f"\n  {col}:")
        for val, count in vc.items():
            marker = "" if val == "clean" else "  <-- NON-CLEAN"
            print(f"    {repr(val):30s}  {count:>6d}{marker}")

    # Rows where ANY truncation column is not "clean"
    non_clean_mask = (df[trunc_cols] != "clean").any(axis=1)
    non_clean = df[non_clean_mask].copy()

    print(f"\n--- Rows with at least one non-'clean' truncation value ---")
    print(f"  Count: {len(non_clean)} / {len(df)}")

    if non_clean.empty:
        print("  (none found)")
        return

    # Show which specific columns are non-clean for each row
    for col in trunc_cols:
        subset = non_clean[non_clean[col] != "clean"]
        if subset.empty:
            continue
        print(f"\n  Non-clean in '{col}'  ({len(subset)} rows):")
        display_cols = ["sent_id", "word", "word_in_context", col]
        # include other trunc cols to show full picture
        display_cols += [c for c in trunc_cols if c != col]
        print(subset[display_cols].to_string(index=True))

    # Summary: unique non-clean values across all truncation columns
    all_non_clean_values = set()
    for col in trunc_cols:
        all_non_clean_values.update(df.loc[non_clean_mask, col].unique())
    all_non_clean_values.discard("clean")
    print(f"\n  Unique non-'clean' values seen: {sorted(all_non_clean_values)}")


def main() -> None:
    for label, path in FILES.items():
        if not path.exists():
            print(f"\n[WARNING] File not found: {path}")
            continue
        explore_truncations(label, path)

    print("\nDone.")


if __name__ == "__main__":
    main()
