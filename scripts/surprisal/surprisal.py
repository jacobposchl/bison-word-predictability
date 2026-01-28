"""
Surprisal Comparison Analysis

This script compares surprisal values at code-switch points between:
1. Code-switched sentences (translated to full Cantonese)
2. Matched monolingual Cantonese baseline sentences

Supports discourse context by using k previous sentences from the same speaker.

Usage:
    # With context (default)
    python scripts/surprisal/surprisal.py --model masked
    
    # Without context
    python scripts/surprisal/surprisal.py --model autoregressive --no-context
    
    # Compare both modes
    python scripts/surprisal/surprisal.py --model masked --compare-context
    
    Required arguments:
        --model: Type of model - "masked" for BERT-style or "autoregressive" for GPT-style
        
    Optional arguments:
        --sample-size: Number of sentences to process (default: all)
        --no-context: Disable discourse context
        --compare-context: Run both with and without context for comparison
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import torch
from datetime import datetime

from src.core.config import Config
from src.experiments.surprisal_calculator import create_surprisal_calculator
from src.experiments.surprisal_analysis import (
    calculate_surprisal_for_dataset,
    compute_statistics,
    print_statistics_summary
)
# Note: Visualization functions are now called from scripts/plots/figures.py


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run surprisal comparison experiment for code-switching analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['masked', 'autoregressive'],
        help='Type of model - "masked" for BERT-style or "autoregressive" for GPT-style'
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of sentences to process (default: all)"
    )
    parser.add_argument(
        "--no-context",
        action='store_true',
        help="Disable discourse context"
    )
    parser.add_argument(
        "--compare-context",
        action='store_true',
        help="Run both with-context and without-context analyses for comparison"
    )
    
    return parser.parse_args()


def load_analysis_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the pre-computed analysis dataset with matched sentences.
    
    Args:
        dataset_path: Path to the analysis dataset CSV
        
    Returns:
        DataFrame with CS sentences, translations, and matched mono sentences
        
    Raises:
        FileNotFoundError: If analysis dataset doesn't exist
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at path: {dataset_path}\n"
            f"Please run scripts/matching/matching.py first!"
        )
    
    print(f"Loading analysis dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    LOADED")
    
    # Verify required columns exist
    required_columns = [
        'cs_translation', 'matched_mono',
        'switch_index', 'matched_switch_index'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Analysis dataset is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    return df


def initialize_surprisal_calculator(
    model_type: str,
    config: Config
):
    """
    Initialize the surprisal calculator with specified model type.
    
    Args:
        model_type: Type of model - "masked" or "autoregressive"
        config: Configuration object for device settings
        
    Returns:
        Initialized surprisal calculator
    """
    # Get device from config
    device = config.get('experiment.device', 'auto')
    
    print(f"\nInitializing {model_type} surprisal calculator:")
    print(f"  Device: {device}")
    
    # Auto-detect CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Auto-detected device: {device}")
    
    calculator = create_surprisal_calculator(
        model_type=model_type,
        config=config,
        device=device
    )
    
    print(f"  Model loaded successfully")
    
    return calculator


def setup_output_directories(config: Config, model_type: str) -> tuple:
    """
    Create output directories for results.
    
    Args:
        config: Configuration object
        model_type: Type of model - "masked" or "autoregressive"
        
    Returns:
        Tuple of (results_dir, figures_dir) as Path objects
        (figures_dir is kept for compatibility but not used)
    """
    results_base = Path(config.get_surprisal_results_dir())
    base_dir = results_base / model_type
    
    results_dir = base_dir
    figures_dir = Path(config.get_surprisal_figures_dir()) / model_type
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    # Note: figures_dir will be created when generating figures
    
    return results_dir, figures_dir


def main():
    """Main experiment execution."""
    print("="*80)
    print("SURPRISAL COMPARISON EXPERIMENT")
    print("Code-Switched Translation vs. Monolingual Baseline")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = Config()
    
    # Setup base output directories
    base_results_dir, base_figures_dir = setup_output_directories(config, args.model)
    
    # Find all window size datasets
    matching_results_dir = Path(config.get_matching_results_dir())
    window_datasets = sorted(matching_results_dir.glob("analysis_dataset_window_*.csv"))
    
    if not window_datasets:
        raise FileNotFoundError(
            f"No analysis datasets found in {matching_results_dir}. "
            f"Please run scripts/matching/matching.py first!"
        )
    
    print(f"\nFound {len(window_datasets)} window size dataset(s):")
    for ds in window_datasets:
        print(f"  - {ds.name}")
    
    # Initialize surprisal calculator (reused for all window sizes)
    surprisal_calc = initialize_surprisal_calculator(
        model_type=args.model,
        config=config
    )
    
    # Determine context usage mode
    if args.compare_context:
        # Run both with and without context
        context_modes = [False, True]
        mode_names = ['without_context', 'with_context']
    elif args.no_context:
        # No context (when --no-context is specified)
        context_modes = [False]
        mode_names = ['without_context']
    else:
        # Use context (default behavior)
        context_modes = [True]
        mode_names = ['with_context']
    
    # Process each window size dataset
    for dataset_path in window_datasets:
        # Extract window size from filename (e.g., "analysis_dataset_window_2.csv" -> 2)
        import re
        match = re.search(r'window_(\d+)', dataset_path.name)
        if not match:
            print(f"Warning: Could not extract window size from {dataset_path.name}, skipping...")
            continue
        window_size = int(match.group(1))
        
        print(f"\n{'='*80}")
        print(f"Processing window size {window_size}")
        print(f"{'='*80}")
        
        # Load analysis dataset for this window size
        analysis_df = load_analysis_dataset(str(dataset_path))
        
        # Apply sample size limit if specified
        if args.sample_size:
            print(f"\nLimiting to {args.sample_size} sentences (from {len(analysis_df)} total)")
            # Sample unique CS sentences
            unique_cs = analysis_df['cs_translation'].unique()
            sampled_cs = pd.Series(unique_cs).sample(n=min(args.sample_size, len(unique_cs)), random_state=42)
            analysis_df = analysis_df[analysis_df['cs_translation'].isin(sampled_cs)]
            print(f"Selected {len(analysis_df)} comparisons")
        
        # Create window-specific output directories
        window_results_dir = base_results_dir / f"window_{window_size}"
        window_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis for each context mode
        for use_context, mode_name in zip(context_modes, mode_names):
            print("\n" + "="*80)
            print(f"ANALYSIS MODE: {mode_name.upper().replace('_', ' ')}")
            print("="*80)
            
            # Setup mode-specific output directories
            if len(context_modes) > 1:
                mode_results_dir = window_results_dir / mode_name
                mode_results_dir.mkdir(parents=True, exist_ok=True)
            else:
                mode_results_dir = window_results_dir
        
            # Calculate surprisal values
            print("\n" + "-"*80)
            print("CALCULATING SURPRISAL VALUES")
            print("-"*80)
            
            # Get context lengths from config
            if use_context:
                context_lengths = config.get('context.context_lengths', None)
                if context_lengths is None:
                    raise ValueError("context.context_lengths must be specified in config when use_context=True")
                if not isinstance(context_lengths, list) or len(context_lengths) == 0:
                    raise ValueError("context.context_lengths must be a non-empty list")
            else:
                context_lengths = []
            
            results_df = calculate_surprisal_for_dataset(
                analysis_df=analysis_df,
                surprisal_calc=surprisal_calc,
                show_progress=True,
                use_context=use_context,
                context_lengths=context_lengths
            )
            
            # Save results
            results_csv_path = mode_results_dir / "surprisal_results.csv"
            results_df.to_csv(results_csv_path, index=False)
            
            # Compute statistics for each context length
            print("\n" + "-"*80)
            print("COMPUTING STATISTICS")
            print("-"*80)
            
            # Compute statistics for each context length
            all_stats = {}
            for ctx_len in context_lengths:
                print(f"\nComputing statistics for context length {ctx_len}...")
                stats_dict = compute_statistics(results_df, context_length=ctx_len)
                all_stats[ctx_len] = stats_dict
                print_statistics_summary(stats_dict)
            
            # Use first context length for main statistics summary file
            primary_context_length = context_lengths[0] if context_lengths else None
            if primary_context_length:
                stats_dict = all_stats[primary_context_length]
            
            # Save statistics to file
            stats_txt_path = mode_results_dir / "statistics_summary.txt"
            with open(stats_txt_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("SURPRISAL COMPARISON STATISTICS\n")
                if primary_context_length:
                    f.write(f"Context Length: {primary_context_length} sentences\n")
                f.write(f"Mode: {mode_name.replace('_', ' ').upper()}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model type: {args.model}\n\n")
                
                f.write(f"Sample Size:\n")
                f.write(f"  Total comparisons: {stats_dict['n_total']}\n")
                if stats_dict.get('n_filtered', 0) > 0:
                    f.write(f"  Filtered out (failed calculations): {stats_dict['n_filtered']} ({stats_dict['n_filtered']/stats_dict['n_total']:.1%})\n")
                f.write(f"  Valid calculations: {stats_dict['n_valid']}\n")
                f.write(f"  Complete calculations: {stats_dict['n_complete']}\n")
                f.write(f"  Success rate: {stats_dict['success_rate']:.1%}\n")
                f.write(f"  Complete rate: {stats_dict['complete_rate']:.1%}\n")
                
                if 'n_with_context' in stats_dict:
                    f.write(f"\nContext Usage:\n")
                    f.write(f"  With context: {stats_dict['n_with_context']}\n")
                    f.write(f"  Without context: {stats_dict['n_without_context']}\n")
                
                f.write(f"\nCode-Switched Translation Surprisal:\n")
                f.write(f"  Mean:   {stats_dict['cs_surprisal_mean']:.4f}\n")
                f.write(f"  Median: {stats_dict['cs_surprisal_median']:.4f}\n")
                f.write(f"  Std:    {stats_dict['cs_surprisal_std']:.4f}\n\n")
                
                f.write(f"Monolingual Baseline Surprisal:\n")
                f.write(f"  Mean:   {stats_dict['mono_surprisal_mean']:.4f}\n")
                f.write(f"  Median: {stats_dict['mono_surprisal_median']:.4f}\n")
                f.write(f"  Std:    {stats_dict['mono_surprisal_std']:.4f}\n\n")
                
                f.write(f"  Difference (CS - Monolingual):\n")
                f.write(f"  Mean:   {stats_dict['difference_mean']:.4f}\n")
                f.write(f"  Median: {stats_dict['difference_median']:.4f}\n")
                f.write(f"  Std:    {stats_dict['difference_std']:.4f}\n\n")
                
                f.write(f"Paired t-test:\n")
                f.write(f"  t-statistic: {stats_dict['ttest_statistic']:.4f}\n")
                f.write(f"  p-value:     {stats_dict['ttest_pvalue']:.6f}\n\n")
                
                f.write(f"Effect Size:\n")
                f.write(f"  Cohen's d: {stats_dict['cohens_d']:.4f}\n\n")
                
                # Write statistics for all context lengths
                if len(context_lengths) > 1:
                    f.write("\n" + "="*80 + "\n")
                    f.write("STATISTICS FOR ALL CONTEXT LENGTHS\n")
                    f.write("="*80 + "\n\n")
                    for ctx_len in context_lengths:
                        if ctx_len in all_stats:
                            ctx_stats = all_stats[ctx_len]
                            f.write(f"Context Length {ctx_len}:\n")
                            f.write(f"  Complete calculations: {ctx_stats['n_complete']}\n")
                            f.write(f"  CS Mean: {ctx_stats['cs_surprisal_mean']:.4f}\n")
                            f.write(f"  Mono Mean: {ctx_stats['mono_surprisal_mean']:.4f}\n")
                            f.write(f"  Difference Mean: {ctx_stats['difference_mean']:.4f}\n")
                            f.write(f"  p-value: {ctx_stats['ttest_pvalue']:.6f}\n\n")
            
            print(f"\nSaved statistics summary to {stats_txt_path}")
            
            # Note: Visualizations can be generated separately using:
            # python scripts/plots/figures.py --surprisal --model {args.model}
            
            print(f"\nResults saved to: {mode_results_dir}")
            print(f"To generate figures, run: python scripts/plots/figures.py --surprisal --model {args.model}")
        
        print(f"\n{'='*80}")
        print(f"Completed processing window size {window_size}")
        print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("ALL WINDOW SIZES PROCESSED")
    print(f"{'='*80}")
    print(f"\nResults saved to: {base_results_dir}")
    print(f"To generate figures, run: python scripts/plots/figures.py --surprisal --model {args.model}")


if __name__ == "__main__":
    main()