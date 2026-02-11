"""
Surprisal Comparison Analysis

This script calculates surprisal values at switch indices of both:
1. Code-switched sentences (translated to full Cantonese)
2. Matched monolingual Cantonese baseline sentences

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
        --no-context: Disable discourse context
        --compare-context: Run both with and without context for comparison
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import torch

from src.core.config import Config
from src.experiments.surprisal_calculator import create_surprisal_calculator
from src.experiments.surprisal_analysis import (
    calculate_surprisal_for_dataset,
    compute_statistics,
    convert_surprisal_results_to_long
)
from src.plots.surprisal.report_generator import generate_surprisal_statistics_report

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


def initialize_surprisal_calculator( model_type: str, config: Config ):
    """
    Initialize the surprisal calculator with specified model type.
    
    Args:
        model_type: Type of model - "masked" or "autoregressive"
        config: Configuration object for device settings
        
    Returns:
        Initialized surprisal calculator
    """

    device = config.get('experiment.device', 'auto')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_type} model on {device}...")
    
    calculator = create_surprisal_calculator(
        model_type=model_type,
        config=config,
        device=device
    )
    
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
    
    return results_dir, figures_dir


def main():

    """Surprisal analysis experiment execution."""

    print("Starting surprisal analysis...")
    
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
    
    print(f"Found {len(window_datasets)} window size dataset(s)")
    
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
        
        print(f"\nProcessing window size {window_size}...")
        
        # Load analysis dataset for this window size
        analysis_df = load_analysis_dataset(str(dataset_path))
        
        # Create window-specific output directories
        window_results_dir = base_results_dir / f"window_{window_size}"
        window_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis for each context mode
        for use_context, mode_name in zip(context_modes, mode_names):
            mode_label = mode_name.replace('_', ' ')
            print(f"  Running {mode_label} analysis...")
            
            # Setup mode-specific output directories
            if len(context_modes) > 1:
                mode_results_dir = window_results_dir / mode_name
                mode_results_dir.mkdir(parents=True, exist_ok=True)
            else:
                mode_results_dir = window_results_dir
            
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
            
            # Save results (long format)
            results_csv_path = mode_results_dir / "surprisal_results.csv"
            long_results_df = convert_surprisal_results_to_long(results_df)
            long_results_df.to_csv(results_csv_path, index=False)
            
            # Compute statistics for each context length
            all_stats = {}
            for ctx_len in context_lengths:
                stats_dict = compute_statistics(results_df, context_length=ctx_len)
                all_stats[ctx_len] = stats_dict
            
            # Generate and save statistics report
            primary_context_length = context_lengths[0] if context_lengths else None
            if primary_context_length:
                stats_dict = all_stats[primary_context_length]
            
            context_clipped_count = getattr(surprisal_calc, 'context_clipped_count', None)
            
            report_text = generate_surprisal_statistics_report(
                stats_dict=stats_dict,
                all_stats=all_stats,
                context_lengths=context_lengths,
                model_type=args.model,
                mode_name=mode_name,
                primary_context_length=primary_context_length,
                context_clipped_count=context_clipped_count
            )
            
            stats_txt_path = mode_results_dir / "statistics_summary.txt"
            with open(stats_txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"    Report: {stats_txt_path}")
    
    print(f"\nCompleted! Results saved to: {base_results_dir}")


if __name__ == "__main__":
    main()