"""
Main Experiment: Surprisal Comparison Analysis

This script compares surprisal values at code-switch points between:
1. Code-switched sentences (translated to full Cantonese)
2. Matched monolingual Cantonese baseline sentences

Supports discourse context by using k previous sentences from the same speaker.

Usage:
    # Without context (isolated sentences)
    python scripts/main_experiment/run_surprisal_experiment.py --model masked --no-context
    
    # With context (if available in dataset)
    python scripts/main_experiment/run_surprisal_experiment.py --model masked --use-context
    
    # Compare both modes
    python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive --compare-context
    
    Required arguments:
        --model: Type of model - "masked" for BERT-style or "autoregressive" for GPT-style
        
    Optional arguments:
        --dataset: Path to analysis dataset CSV (default: results/exploratory/analysis_dataset.csv)
        --sample-size: Number of sentences to process (default: all)
        --use-context: Use discourse context in calculations (if available)
        --no-context: Disable context even if available
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
from src.experiments.visualization import (
    plot_surprisal_distributions,
    plot_scatter_comparison,
    plot_difference_histogram,
    plot_summary_statistics
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run surprisal comparison experiment for code-switching analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="results/exploratory/analysis_dataset.csv",
        help="Path to analysis dataset CSV (default: results/exploratory/analysis_dataset.csv)"
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
        "--use-context",
        action='store_true',
        help="Use discourse context in surprisal calculations (if available in dataset)"
    )
    parser.add_argument(
        "--no-context",
        action='store_true',
        help="Disable discourse context even if available in dataset"
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
            f"Please run scripts/exploratory/exploratory_analysis.py first!"
        )
    
    print(f"Loading analysis dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} comparisons from dataset")
    
    # Verify a few columns exist
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
    Create output directories for results and figures.
    
    Args:
        config: Configuration object
        model_type: Type of model - "masked" or "autoregressive"
        
    Returns:
        Tuple of (results_dir, figures_dir) as Path objects
    """
    results_base = Path(config.get_results_dir())
    base_dir = results_base / f"main_experiment_{model_type}"
    
    results_dir = base_dir
    figures_dir = Path(config.get_figures_dir()) / f"main_experiment_{model_type}"
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
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
    config_path = "config/config.yaml"
    print(f"Loading configuration from {config_path}")
    config = Config(config_path)
    
    # Setup output directories
    results_dir, figures_dir = setup_output_directories(config, args.model)
    
    # Load analysis dataset
    analysis_df = load_analysis_dataset(args.dataset)
    
    # Apply sample size limit if specified
    if args.sample_size:
        print(f"\nLimiting to {args.sample_size} sentences (from {len(analysis_df)} total)")
        # Sample unique CS sentences
        unique_cs = analysis_df['cs_translation'].unique()
        sampled_cs = pd.Series(unique_cs).sample(n=min(args.sample_size, len(unique_cs)), random_state=42)
        analysis_df = analysis_df[analysis_df['cs_translation'].isin(sampled_cs)]
        print(f"Selected {len(analysis_df)} comparisons")
    
    # Initialize surprisal calculator
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
        context_modes = [False]
        mode_names = ['without_context']
    elif args.use_context or 'cs_context' in analysis_df.columns:
        # Use context if explicitly requested or if available in dataset
        context_modes = [True]
        mode_names = ['with_context']
    else:
        # No context available or requested
        context_modes = [False]
        mode_names = ['without_context']
    
    # Run analysis for each context mode
    for use_context, mode_name in zip(context_modes, mode_names):
        print("\n" + "="*80)
        print(f"ANALYSIS MODE: {mode_name.upper().replace('_', ' ')}")
        print("="*80)
        
        # Setup mode-specific output directories
        if len(context_modes) > 1:
            mode_results_dir = results_dir / mode_name
            mode_figures_dir = figures_dir / mode_name
            mode_results_dir.mkdir(parents=True, exist_ok=True)
            mode_figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            mode_results_dir = results_dir
            mode_figures_dir = figures_dir
        
        # Calculate surprisal values
        print("\n" + "-"*80)
        print("CALCULATING SURPRISAL VALUES")
        print("-"*80)
        
        results_df = calculate_surprisal_for_dataset(
            analysis_df=analysis_df,
            surprisal_calc=surprisal_calc,
            show_progress=True,
            use_context=use_context
        )
        
        # Save detailed results
        results_csv_path = mode_results_dir / "surprisal_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nSaved detailed results to {results_csv_path}")
        
        # Compute statistics
        print("\n" + "-"*80)
        print("COMPUTING STATISTICS")
        print("-"*80)
        
        stats_dict = compute_statistics(results_df)
        print_statistics_summary(stats_dict)
        
        # Save statistics to file
        stats_txt_path = mode_results_dir / "statistics_summary.txt"
        with open(stats_txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SURPRISAL COMPARISON STATISTICS\n")
            f.write(f"Mode: {mode_name.replace('_', ' ').upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model type: {args.model}\n\n")
            
            f.write(f"Sample Size:\n")
            f.write(f"  Total comparisons: {stats_dict['n_total']}\n")
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
        
        print(f"\nSaved statistics summary to {stats_txt_path}")
        
        # Generate visualizations
        print("\n" + "-"*80)
        print("GENERATING VISUALIZATIONS")
        print("-"*80)
        
        # Distribution plots
        plot_surprisal_distributions(
            results_df=results_df,
            output_path=mode_figures_dir / "surprisal_distributions.png"
        )
        
        # Scatter comparison
        plot_scatter_comparison(
            results_df=results_df,
            output_path=mode_figures_dir / "surprisal_scatter.png"
        )
        
        # Difference histogram
        plot_difference_histogram(
            results_df=results_df,
            output_path=mode_figures_dir / "surprisal_differences.png"
        )
        
        # Summary figure
        plot_summary_statistics(
            results_df=results_df,
            output_path=mode_figures_dir / "surprisal_summary.png",
            stats_dict=stats_dict
        )
        
        print("\nAll visualizations generated successfully!")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Detailed results: {results_csv_path}")
    print(f"  - Statistics: {stats_txt_path}")
    print(f"  - Figures: {figures_dir}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()