"""
Main script for running exporatory analysis.

This script orchestrates the complete exploratory analysis pipeline:
1. Load CSV data
2. Extract monolingual sentences
3. Analyze POS tagging
4. Test matching algorithm
5. Analyze distributions
6. Generates report
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.data.data_loading import (
    load_dataset,
    load_code_switched_sentences,
    load_monolingual_csvs
)
from src.data.data_export import save_exploratory_outputs
from src.analysis.feasibility import (
    analyze_pos_tagging,
    test_matching_algorithm,
    analyze_distributions,
    generate_report
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for exploratory analysis."""
    parser = argparse.ArgumentParser(
        description='Calvillo methodology feasibility analysis'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of sentences for POS tagging and matching tests (default: process full dataset)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Load configuration
    config = Config()
    
    # Get output and figures directories from config
    output_dir = Path(config.get_exploratory_results_dir())
    figures_dir = Path(config.get_exploratory_figures_dir())
    
    logger.info("=" * 80)
    logger.info("CALVILLO METHODOLOGY ANALYSIS")
    logger.info("=" * 80)
    if args.sample_size is None:
        logger.info("Mode: FULL DATASET (processing all sentences)")
    else:
        logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Figures directory: {figures_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Step 1: Load code-switched sentences (WITHOUT fillers for cleaner matching)
        logger.info("Step 1: Loading code-switched sentences...")
        code_switched = load_code_switched_sentences(config, use_fillers=False)
        
        # Step 2: Load monolingual sentences (WITHOUT fillers for consistent matching)
        logger.info("\nStep 2: Loading monolingual sentences...")
        monolingual = load_monolingual_csvs(config, use_fillers=False)
        
        # Combine for overall stats (use code_switched for general analysis)
        df = code_switched
        
        # Step 3: Analyze POS tagging
        logger.info("\nStep 3: Analyzing POS tagging...")
        # Sample from all sentences for POS analysis (or use full dataset)
        pos_sample_size = 0 if args.sample_size is None else args.sample_size
        pos_results = analyze_pos_tagging(df, sample_size=pos_sample_size)
        
        # Step 4: Test matching algorithm
        logger.info("\nStep 4: Testing matching algorithm...")
        # code_switched is already loaded above
        
        # Determine sample size for matching
        if args.sample_size is None:
            matching_sample_size = 0  # 0 means use all sentences
            logger.info("Processing FULL dataset for matching (this may take a while)...")
        else:
            matching_sample_size = args.sample_size
            logger.info(f"Processing sample of {matching_sample_size} code-switched sentences...\n")
            logger.info("To process the full dataset, omit the --sample-size flag\n")
        
        # Limit monolingual sentences to 500 per language for faster matching
        # (can be adjusted based on dataset size)
        # For full dataset, use all monolingual sentences
        max_mono = None if args.sample_size is None else 500
        matching_results = test_matching_algorithm(
            code_switched,
            monolingual,
            sample_size=matching_sample_size,
            max_monolingual_per_lang=max_mono
        )
        
        # Step 5: Analyze distributions
        logger.info("\nStep 5: Analyzing distributions...")
        distributions = analyze_distributions(code_switched)
        
        # Step 6: Generate report
        logger.info("\nStep 6: Generating report...")
        # Build monolingual dict for report (add code_switched for compatibility)
        monolingual_dict = {
            'cantonese': monolingual['cantonese'],
            'english': monolingual['english'],
            'code_switched': code_switched
        }
        all_results = {
            'monolingual': monolingual_dict,
            'pos_tagging': pos_results,
            'matching': matching_results,
            'distributions': distributions
        }
        report = generate_report(all_results)
        
        # Step 7: Save outputs
        logger.info("\nStep 7: Saving outputs...")
        save_exploratory_outputs(
            output_dir,
            monolingual_dict,
            pos_results,
            matching_results,
            distributions,
            report,
            figures_dir=figures_dir
        )
        
        # Print report to console
        logger.info("\n" + "=" * 80)
        logger.info("REPORT")
        logger.info("=" * 80)
        print(report)
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(f"Please ensure the CSV files exist in {config.get_preprocessing_results_dir()}/ directory")
        logger.error("Run preprocessing first: python -m src.preprocess")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

