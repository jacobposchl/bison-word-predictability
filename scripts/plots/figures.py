"""
Centralized figure generation script.

This script generates figures for different stages of the analysis pipeline
by reading from pre-computed CSV files.

Usage:
    python scripts/plots/figures.py --preprocess
    python scripts/plots/figures.py --matching
    python scripts/plots/figures.py --surprisal --model masked
    python scripts/plots/figures.py --regression --model masked
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.plots.preprocessing.plot_preprocessing import (
    plot_matrix_language_distribution,
    plot_pattern_complexity,
    plot_switch_position,
    plot_pattern_type_distribution,
    plot_sentence_length_distribution,
    plot_participant_variation,
    plot_matrix_language_proportions,
    plot_pattern_length_vs_switch_position,
    plot_code_switch_density
)
from src.plots.matching.plot_matching import (
    plot_similarity_distributions_from_csv,
    plot_match_success_rate,
    plot_matches_per_sentence_distribution,
    plot_match_quality_by_group_speaker,
    plot_similarity_vs_characteristics,
    plot_window_size_comparison,
    plot_match_distribution_by_group,
    plot_similarity_threshold_analysis,
    plot_pos_window_alignment_quality
)
from src.experiments.visualization import (
    plot_surprisal_distributions,
    plot_difference_histogram
)
from src.plots.surprisal.plot_surprisal import (
    plot_surprisal_distributions as plot_combined_surprisal_distributions,
    plot_surprisal_distributions_by_context,
    plot_surprisal_differences_by_context,
    plot_surprisal_distributions_matrix
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_preprocessing_figures(config: Config):
    """Generate figures for preprocessing stage."""
    logger.info("="*80)
    logger.info("GENERATING PREPROCESSING FIGURES")
    logger.info("="*80)
    
    # Load all_sentences.csv which has matrix_language column
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    all_sentences_csv = preprocessing_dir / "all_sentences.csv"
    
    if not all_sentences_csv.exists():
        logger.error(f"All sentences CSV not found: {all_sentences_csv}")
        logger.error("Please run preprocessing first: python scripts/preprocess/preprocess.py")
        return False
    
    logger.info(f"Loading data from {all_sentences_csv}")
    all_sentences_df = pd.read_csv(all_sentences_csv)
    
    # Filter to code-switching sentences (those with both 'C' and 'E' in pattern)
    # This matches the filtering done in filter_code_switching_sentences()
    code_switched = all_sentences_df[
        all_sentences_df['pattern'].str.contains('C', na=False) & 
        all_sentences_df['pattern'].str.contains('E', na=False) &
        (all_sentences_df['pattern'] != 'FILLER_ONLY')
    ].copy()
    
    logger.info(f"Found {len(code_switched)} code-switching sentences out of {len(all_sentences_df)} total")
    
    # Generate figures
    figures_dir = config.get_preprocessing_figures_dir()
    
    logger.info("Generating preprocessing figures...")
    
    # 1. Matrix language distribution
    logger.info("  1. Matrix language distribution...")
    plot_matrix_language_distribution(code_switched, figures_dir)
    
    # 2. Pattern complexity (DISABLED)
    # logger.info("  2. Pattern complexity...")
    # plot_pattern_complexity(code_switched, figures_dir)
    
    # 3. Switch position (use cantonese_translated_WITHOUT_fillers.csv for switch_index column)
    logger.info("  2. Switch position analysis...")
    cantonese_translated_csv = preprocessing_dir / "cantonese_translated_WITHOUT_fillers.csv"
    if cantonese_translated_csv.exists():
        logger.info(f"  Loading switch positions from {cantonese_translated_csv}")
        cantonese_translated_df = pd.read_csv(cantonese_translated_csv)
        plot_switch_position(cantonese_translated_df, figures_dir)
    else:
        logger.warning(f"  {cantonese_translated_csv} not found, using all_sentences.csv instead")
        plot_switch_position(code_switched, figures_dir)
    
    # 4. Pattern type distribution (DISABLED)
    # logger.info("  4. Pattern type distribution...")
    # plot_pattern_type_distribution(code_switched, figures_dir)
    
    # 5. Sentence length distribution
    logger.info("  3. Sentence length distribution...")
    plot_sentence_length_distribution(code_switched, figures_dir)
    
    # 6. Participant variation
    logger.info("  4. Participant variation...")
    plot_participant_variation(code_switched, figures_dir)
    
    # 7. Matrix language proportions (DISABLED)
    # logger.info("  7. Matrix language proportions...")
    # plot_matrix_language_proportions(code_switched, figures_dir)
    
    # 8. Pattern length vs switch position (DISABLED)
    # logger.info("  8. Pattern length vs switch position...")
    # plot_pattern_length_vs_switch_position(code_switched, figures_dir)
    
    # 9. Code-switch density
    logger.info("  5. Code-switch density...")
    plot_code_switch_density(code_switched, figures_dir)
    
    logger.info(f"\nAll preprocessing figures saved to: {figures_dir}")
    return True


def generate_matching_figures(config: Config):
    """Generate figures for matching stage."""
    logger.info("="*80)
    logger.info("GENERATING MATCHING FIGURES")
    logger.info("="*80)
    
    matching_dir = Path(config.get_matching_results_dir())
    figures_dir = Path(config.get_matching_figures_dir())
    
    # Check if analysis datasets exist
    window_datasets = sorted(matching_dir.glob("analysis_dataset_window_*.csv"))
    
    if not window_datasets:
        logger.error(f"No analysis datasets found in {matching_dir}")
        logger.error("Please run matching first: python scripts/matching/matching.py")
        return False
    
    logger.info(f"Found {len(window_datasets)} window size dataset(s)")
    for ds in window_datasets:
        logger.info(f"  - {ds.name}")
    
    logger.info("Generating matching figures...")
    
    # 1. Similarity distributions (DISABLED)
    # logger.info("  1. Similarity distributions...")
    # plot_similarity_distributions_from_csv(window_datasets, str(figures_dir))
    
    # 2. Match success rate (DISABLED)
    # logger.info("  1. Match success rate...")
    # plot_match_success_rate(window_datasets, str(figures_dir))
    
    # 3. Matches per sentence distribution
    logger.info("  1. Matches per sentence distribution...")
    plot_matches_per_sentence_distribution(window_datasets, str(figures_dir))
    
    # 4. Match quality by group/speaker (DISABLED)
    # logger.info("  4. Match quality by group/speaker...")
    # plot_match_quality_by_group_speaker(window_datasets, str(figures_dir))
    
    # 5. Similarity vs characteristics (DISABLED)
    # logger.info("  5. Similarity vs characteristics...")
    # plot_similarity_vs_characteristics(window_datasets, str(figures_dir))
    
    # 6. Window size comparison (DISABLED)
    # logger.info("  6. Window size comparison...")
    # plot_window_size_comparison(window_datasets, str(figures_dir))
    
    # 7. Match distribution by group (DISABLED)
    # logger.info("  7. Match distribution by group...")
    # plot_match_distribution_by_group(window_datasets, str(figures_dir))
    
    # 8. Similarity threshold analysis
    logger.info("  2. Similarity threshold analysis...")
    plot_similarity_threshold_analysis(window_datasets, str(figures_dir))
    
    # 9. POS window alignment quality (DISABLED)
    # logger.info("  4. POS window alignment quality...")
    # plot_pos_window_alignment_quality(window_datasets, str(figures_dir))
    
    logger.info(f"\nAll matching figures saved to: {figures_dir}")
    return True


def generate_surprisal_figures(config: Config, model_type: str):
    """Generate figures for surprisal stage."""
    logger.info("="*80)
    logger.info(f"GENERATING SURPRISAL FIGURES (model: {model_type})")
    logger.info("="*80)
    
    results_base = Path(config.get_surprisal_results_dir()) / model_type
    figures_base = Path(config.get_surprisal_figures_dir()) / model_type
    
    # Generate combined surprisal distribution plot (all windows and contexts)
    logger.info("Generating combined surprisal distribution plot...")
    combined_output_path = figures_base / "surprisal_distributions_combined.png"
    plot_combined_surprisal_distributions(
        results_base=results_base,
        output_path=combined_output_path,
        model_type=model_type
    )
    
    # Generate 3x3 matrix plot of all context × window combinations
    logger.info("Generating surprisal distribution matrix plot...")
    matrix_output_path = figures_base / "surprisal_distributions_matrix.png"
    plot_surprisal_distributions_matrix(
        results_base=results_base,
        output_path=matrix_output_path,
        model_type=model_type
    )
    
    # Generate plots grouped by context length (averaging across window sizes)
    logger.info("\nGenerating context-based plots (averaged across window sizes)...")
    for context_length in [1, 2, 3]:
        context_figures_dir = figures_base / f"context_{context_length}"
        context_figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Context length {context_length}...")
        
        # Distribution plot
        plot_surprisal_distributions_by_context(
            results_base=results_base,
            output_path=context_figures_dir / "surprisal_distributions.png",
            context_length=context_length,
            model_type=model_type
        )
        
        # Difference plot
        plot_surprisal_differences_by_context(
            results_base=results_base,
            output_path=context_figures_dir / "surprisal_differences.png",
            context_length=context_length,
            model_type=model_type
        )
    
    # Find all window size directories
    window_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('window_')])
    
    if not window_dirs:
        logger.error(f"No window size directories found in {results_base}")
        logger.error(f"Please run surprisal calculation first: python scripts/surprisal/surprisal.py --model {model_type}")
        return False
    
    # Process each window size
    for window_dir in window_dirs:
        window_size = window_dir.name.replace('window_', '')
        logger.info(f"\nProcessing window size {window_size}...")
        
        # Check for context mode subdirectories or direct results
        mode_dirs = []
        if any(d.is_dir() for d in window_dir.iterdir() if d.name in ['with_context', 'without_context']):
            # Has context mode subdirectories
            for mode_dir in window_dir.iterdir():
                if mode_dir.is_dir() and mode_dir.name in ['with_context', 'without_context']:
                    mode_dirs.append((mode_dir, mode_dir.name))
        else:
            # Direct results in window directory
            mode_dirs.append((window_dir, 'default'))
        
        for mode_dir, mode_name in mode_dirs:
            results_csv = mode_dir / "surprisal_results.csv"
            
            if not results_csv.exists():
                logger.warning(f"Results CSV not found: {results_csv}, skipping...")
                continue
            
            logger.info(f"  Loading results from {results_csv}")
            results_df = pd.read_csv(results_csv)
            
            # Determine context length from columns
            context_cols = [col for col in results_df.columns if 'cs_surprisal_context_' in col]
            if context_cols:
                import re
                match = re.search(r'context_(\d+)', context_cols[0])
                context_length = int(match.group(1)) if match else None
            else:
                context_length = None
            
            # Setup output directory
            if mode_name == 'default':
                mode_figures_dir = figures_base / f"window_{window_size}"
            else:
                mode_figures_dir = figures_base / f"window_{window_size}" / mode_name
            
            mode_figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            logger.info(f"  Generating plots...")
            
            plot_surprisal_distributions(
                results_df=results_df,
                output_path=mode_figures_dir / "surprisal_distributions.png",
                context_length=context_length,
                window_size=int(window_size),
                model_type=model_type
            )
            
            plot_difference_histogram(
                results_df=results_df,
                output_path=mode_figures_dir / "surprisal_differences.png",
                context_length=context_length
            )
            
            logger.info(f"  Figures saved to: {mode_figures_dir}")
    
    logger.info(f"\nAll surprisal figures saved to: {figures_base}")
    return True


def generate_regression_figures(config: Config, model_type: str):
    """Generate figures for regression stage."""
    logger.info("="*80)
    logger.info(f"GENERATING REGRESSION FIGURES (model: {model_type})")
    logger.info("="*80)
    
    regression_base = Path(config.get_results_dir()) / f"regression_{model_type}"
    
    if not regression_base.exists():
        logger.error(f"Regression results directory not found: {regression_base}")
        logger.error(f"Please run regression first: python scripts/regression/regression.py --model {model_type}")
        return False
    
    # Find all window/context combinations
    window_dirs = sorted([d for d in regression_base.iterdir() if d.is_dir() and d.name.startswith('window_')])
    
    if not window_dirs:
        logger.error(f"No window directories found in {regression_base}")
        return False
    
    logger.warning("Regression figures require the full results dictionary with model predictions.")
    logger.warning("These are generated during regression analysis and not saved to CSV.")
    logger.warning("To generate regression figures, please re-run:")
    logger.warning(f"  python scripts/regression/regression.py --model {model_type}")
    logger.warning("")
    logger.warning("Alternatively, you can view the saved CSV files:")
    for window_dir in window_dirs:
        context_dirs = sorted([d for d in window_dir.iterdir() if d.is_dir() and d.name.startswith('context_')])
        for context_dir in context_dirs:
            comparison_csv = context_dir / "model_comparison.csv"
            if comparison_csv.exists():
                logger.info(f"  - {comparison_csv}")
    
    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate figures for analysis pipeline stages"
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Generate preprocessing figures'
    )
    parser.add_argument(
        '--matching',
        action='store_true',
        help='Generate matching figures'
    )
    parser.add_argument(
        '--surprisal',
        action='store_true',
        help='Generate surprisal figures'
    )
    parser.add_argument(
        '--regression',
        action='store_true',
        help='Generate regression figures'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['masked', 'autoregressive'],
        default=None,
        help='Model type for surprisal/regression figures (required for --surprisal or --regression)'
    )
    
    args = parser.parse_args()
    
    # Check that at least one stage is specified
    if not any([args.preprocess, args.matching, args.surprisal, args.regression]):
        parser.error("At least one stage must be specified (--preprocess, --matching, --surprisal, or --regression)")
    
    # Check model argument for surprisal/regression
    if (args.surprisal or args.regression) and not args.model:
        parser.error("--model is required when using --surprisal or --regression")
    
    # Load configuration
    config = Config()
    
    success = True
    
    # Generate figures for requested stages
    if args.preprocess:
        success &= generate_preprocessing_figures(config)
    
    if args.matching:
        success &= generate_matching_figures(config)
    
    if args.surprisal:
        success &= generate_surprisal_figures(config, args.model)
    
    if args.regression:
        success &= generate_regression_figures(config, args.model)
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("FIGURE GENERATION COMPLETE")
        logger.info("="*80)
    else:
        logger.warning("\nSome figures could not be generated. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

