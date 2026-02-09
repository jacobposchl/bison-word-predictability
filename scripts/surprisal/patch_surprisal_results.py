"""
Patch existing surprisal results CSV files to:
1. Remove duplicate 'participant' column (keep only 'participant_id')
2. Add 'is_propn' column (1 if switch_pos == 'PROPN', else 0)
3. Add 'single_worded' column (1 if pattern starts with CX-E1, else 0, only for is_switch=1)

Usage:
    python scripts/surprisal/patch_surprisal_results.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import re
from src.core.config import Config


def load_matching_patterns(matching_dataset_path: Path) -> pd.DataFrame:
    """
    Load the matching dataset to get pattern information.
    
    Args:
        matching_dataset_path: Path to analysis_dataset_window_X.csv
        
    Returns:
        DataFrame with cs_sentence and cs_pattern columns
    """
    print(f"  Loading patterns from: {matching_dataset_path.name}")
    df = pd.read_csv(matching_dataset_path)
    
    # We only need cs_sentence and cs_pattern for matching
    if 'cs_sentence' not in df.columns or 'cs_pattern' not in df.columns:
        raise ValueError(f"Required columns 'cs_sentence' and 'cs_pattern' not found in {matching_dataset_path}")
    
    return df[['cs_sentence', 'cs_pattern']].copy()


def is_single_worded_pattern(pattern, is_switch):
    """
    Check if pattern represents a single-worded English code-switch.
    
    Args:
        pattern: Pattern string (e.g., "C3-E1-C2")
        is_switch: Whether this is a code-switched sentence (1 or 0)
        
    Returns:
        1 if pattern starts with CX-E1, 0 if not, NA if is_switch != 1
    """
    if is_switch != 1:
        return pd.NA
    if not pattern or pd.isna(pattern):
        return 0
    # Check if pattern starts with C followed by number, then E1
    match = re.match(r'^C\d+-E1(?:-|$)', str(pattern))
    return 1 if match else 0


def patch_surprisal_results(
    results_path: Path,
    matching_patterns_df: pd.DataFrame,
    backup: bool = True
) -> None:
    """
    Patch a single surprisal_results.csv file.
    
    Args:
        results_path: Path to surprisal_results.csv
        matching_patterns_df: DataFrame with cs_sentence and cs_pattern
        backup: Whether to create a backup before modifying
    """
    print(f"\n  Processing: {results_path}")
    
    # Load results
    df = pd.read_csv(results_path)
    original_rows = len(df)
    print(f"    Loaded {original_rows} rows")
    
    # Create backup if requested
    if backup:
        backup_path = results_path.parent / f"{results_path.stem}_backup.csv"
        df.to_csv(backup_path, index=False, encoding='utf-8-sig')
        print(f"    Created backup: {backup_path.name}")
    
    # 1. Remove 'participant' column if it exists
    if 'participant' in df.columns:
        df = df.drop(columns=['participant'])
        print(f"    ✓ Removed 'participant' column")
    else:
        print(f"    - No 'participant' column to remove")
    
    # 2. Add 'is_propn' column (for all rows)
    if 'switch_pos' in df.columns:
        df['is_propn'] = (df['switch_pos'] == 'PROPN').astype(int)
        n_propn = df['is_propn'].sum()
        print(f"    ✓ Added 'is_propn' column ({int(n_propn)} proper nouns)")
    else:
        df['is_propn'] = 0
        print(f"    - No 'switch_pos' column, setting is_propn=0")
    
    # 3. Add 'single_worded' column
    # First, merge with matching patterns to get cs_pattern for each original_sentence
    # Note: In long format, cs_original_sentence becomes original_sentence
    original_sentence_col = 'original_sentence' if 'original_sentence' in df.columns else 'cs_original_sentence'
    
    if original_sentence_col in df.columns and 'is_switch' in df.columns:
        # Create a mapping from cs_sentence to cs_pattern
        pattern_map = matching_patterns_df.set_index('cs_sentence')['cs_pattern'].to_dict()
        
        # Add pattern column temporarily (only for is_switch=1 rows)
        df['pattern_temp'] = df.apply(
            lambda row: pattern_map.get(row[original_sentence_col], None) if row['is_switch'] == 1 else None,
            axis=1
        )
        
        # Calculate single_worded
        df['single_worded'] = df.apply(
            lambda row: is_single_worded_pattern(row['pattern_temp'], row['is_switch']),
            axis=1
        )
        
        # Drop temporary column
        df = df.drop(columns=['pattern_temp'])
        
        n_single_worded = df[df['is_switch'] == 1]['single_worded'].sum()
        n_cs_sentences = df[df['is_switch'] == 1].shape[0]
        print(f"    ✓ Added 'single_worded' column ({int(n_single_worded)}/{n_cs_sentences} CS sentences are single-worded)")
    else:
        df['single_worded'] = pd.NA
        print(f"    - Required columns not found, setting single_worded=NA")
    
    # Reorder columns to place new columns after switch_pos
    if 'switch_pos' in df.columns:
        # Get current column order
        cols = df.columns.tolist()
        
        # Remove new columns from their current position
        cols = [c for c in cols if c not in ['is_propn', 'single_worded']]
        
        # Find position of switch_pos and insert after it
        switch_pos_idx = cols.index('switch_pos')
        cols.insert(switch_pos_idx + 1, 'is_propn')
        cols.insert(switch_pos_idx + 2, 'single_worded')
        
        df = df[cols]
    
    # Save updated results
    df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"    ✓ Saved updated file ({len(df)} rows)")


def main():
    """Main execution function."""
    print("="*80)
    print("PATCHING SURPRISAL RESULTS FILES")
    print("="*80)
    
    config = Config()
    results_base = Path(config.get_surprisal_results_dir())
    matching_base = Path(config.get_matching_results_dir())
    
    # Model types to process
    model_types = ['masked', 'autoregressive']
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"Processing {model_type.upper()} model results")
        print(f"{'='*80}")
        
        model_dir = results_base / model_type
        
        if not model_dir.exists():
            print(f"  ⚠ Directory not found: {model_dir}")
            continue
        
        # Find all window directories
        window_dirs = sorted(model_dir.glob("window_*"))
        
        if not window_dirs:
            print(f"  ⚠ No window directories found in {model_dir}")
            continue
        
        print(f"  Found {len(window_dirs)} window size dataset(s)")
        
        for window_dir in window_dirs:
            # Extract window size
            window_match = re.search(r'window_(\d+)', window_dir.name)
            if not window_match:
                print(f"  ⚠ Could not extract window size from: {window_dir.name}")
                continue
            
            window_size = int(window_match.group(1))
            print(f"\n  Window size: {window_size}")
            
            # Load matching patterns for this window size
            matching_dataset_path = matching_base / f"analysis_dataset_window_{window_size}.csv"
            
            if not matching_dataset_path.exists():
                print(f"    ⚠ Matching dataset not found: {matching_dataset_path}")
                continue
            
            matching_patterns_df = load_matching_patterns(matching_dataset_path)
            
            # Find surprisal_results.csv (could be in window_dir directly or in with_context/without_context subdirs)
            results_files = list(window_dir.glob("**/surprisal_results.csv"))
            
            if not results_files:
                print(f"    ⚠ No surprisal_results.csv found in {window_dir}")
                continue
            
            # Process each results file
            for results_file in results_files:
                patch_surprisal_results(
                    results_path=results_file,
                    matching_patterns_df=matching_patterns_df,
                    backup=True
                )
    
    print("\n" + "="*80)
    print("PATCHING COMPLETE!")
    print("="*80)
    print("\nBackup files created with '_backup.csv' suffix.")
    print("You can delete the backups once you've verified the changes.")


if __name__ == "__main__":
    main()
