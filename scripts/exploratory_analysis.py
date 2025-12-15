"""
Main script for running Calvillo methodology feasibility analysis.

This script orchestrates the complete exploratory analysis pipeline:
1. Load CSV data
2. Extract monolingual sentences
3. Analyze POS tagging
4. Test matching algorithm
5. Analyze distributions
6. Generate feasibility report
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.calvillo_feasibility import (
    extract_monolingual_sentences,
    analyze_pos_tagging,
    test_matching_algorithm,
    analyze_distributions,
    generate_feasibility_report
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


def load_data(dataset: str = 'ALL', config: Config = None) -> pd.DataFrame:
    """
    Load the appropriate CSV dataset.
    
    Args:
        dataset: 'ALL' (all sentences), 'WITH' (code-switched with fillers), 
                 or 'WITHOUT' (code-switched without fillers)
        config: Config object (optional, will create one if not provided)
        
    Returns:
        Loaded DataFrame
    """
    if config is None:
        config = Config()
    
    # Get preprocessing results directory from config
    data_dir = Path(config.get_preprocessing_results_dir())
    
    if dataset.upper() == 'ALL':
        csv_path = data_dir / "all_sentences.csv"
    elif dataset.upper() == 'WITH':
        csv_path = data_dir / "code_switching_WITH_fillers.csv"
    elif dataset.upper() == 'WITHOUT':
        csv_path = data_dir / "code_switching_WITHOUT_fillers.csv"
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'ALL', 'WITH', or 'WITHOUT'")
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Please run preprocessing first: python -m src.preprocess"
        )
    
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} sentences")
    
    return df


def save_outputs(
    output_dir: Path,
    monolingual: dict,
    pos_results: dict,
    matching_results: dict,
    distributions: dict,
    report: str,
    figures_dir: Path = None
):
    """
    Save all output files.
    
    Args:
        output_dir: Directory for CSV and report files
        monolingual: Monolingual sentence data
        pos_results: POS tagging results
        matching_results: Matching algorithm results
        distributions: Distribution analysis results
        report: Feasibility report text
        figures_dir: Directory for figures (if None, uses output_dir/figures)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is None:
        figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving outputs to {output_dir}...")
    
    # Save monolingual sentences
    if 'cantonese' in monolingual:
        monolingual_path = output_dir / "monolingual_sentences.csv"
        # Combine all monolingual sentences
        all_mono = pd.concat([
            monolingual['cantonese'],
            monolingual['english']
        ], ignore_index=True)
        all_mono.to_csv(monolingual_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved monolingual sentences to {monolingual_path}")
    
    # Save POS tagging sample
    if 'sample_results' in pos_results:
        pos_path = output_dir / "pos_tagged_sample.csv"
        pos_results['sample_results'].to_csv(pos_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved POS tagging sample to {pos_path}")
    
    # Save matching results sample
    if 'results' in matching_results:
        match_path = output_dir / "matching_results_sample.csv"
        # Flatten the results for CSV export
        export_results = []
        for _, row in matching_results['results'].iterrows():
            export_row = {
                'sentence': row['sentence'],
                'pattern': row['pattern'],
                'num_matches': row['num_matches'],
                'has_match': row['has_match'],
                'best_similarity': row['best_similarity'],
                'has_c_to_e': row['has_c_to_e'],
                'has_e_to_c': row['has_e_to_c']
            }
            # Add details of top matches if available
            if row['matches_detail']:
                for i, match in enumerate(row['matches_detail'][:3]):
                    export_row[f'match_{i+1}_similarity'] = match.get('similarity', 0)
                    export_row[f'match_{i+1}_language'] = match.get('language', '')
            export_results.append(export_row)
        
        match_df = pd.DataFrame(export_results)
        match_df.to_csv(match_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved matching results to {match_path}")
    
    # Save feasibility report
    report_path = output_dir / "feasibility_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved feasibility report to {report_path}")
    
    logger.info("All outputs saved successfully!")


def main():
    """Main entry point for exploratory analysis."""
    parser = argparse.ArgumentParser(
        description='Calvillo methodology feasibility analysis'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ALL',
        choices=['ALL', 'WITH', 'WITHOUT'],
        help='Which dataset to analyze (ALL=all sentences, WITH/WITHOUT=code-switched only, default: ALL)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of sentences for POS tagging and matching tests (default: 100, use 0 for full dataset)'
    )
    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='Process full dataset instead of sampling (overrides --sample-size)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config file (default: from config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config/config.yaml)'
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
    config = Config(config_path=args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get_exploratory_results_dir())
    
    # Determine figures directory
    figures_dir = Path(config.get_exploratory_figures_dir())
    
    logger.info("=" * 80)
    logger.info("CALVILLO METHODOLOGY FEASIBILITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    if args.full_dataset:
        logger.info(f"Mode: FULL DATASET (processing all sentences)")
    else:
        logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Figures directory: {figures_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        df = load_data(args.dataset, config=config)
        
        # Step 2: Extract monolingual sentences
        logger.info("\nStep 2: Extracting monolingual sentences...")
        monolingual = extract_monolingual_sentences(df)
        
        # Step 3: Analyze POS tagging
        logger.info("\nStep 3: Analyzing POS tagging...")
        # Sample from all sentences for POS analysis (or use full dataset)
        pos_sample_size = 0 if args.full_dataset else args.sample_size
        pos_results = analyze_pos_tagging(df, sample_size=pos_sample_size)
        
        # Step 4: Test matching algorithm
        logger.info("\nStep 4: Testing matching algorithm...")
        code_switched = monolingual['code_switched']
        
        # Determine sample size for matching
        if args.full_dataset:
            matching_sample_size = 0  # 0 means use all sentences
            logger.info("Processing FULL dataset for matching (this may take a while)...")
        else:
            matching_sample_size = args.sample_size
            logger.info(f"Processing sample of {matching_sample_size} code-switched sentences...\n")
            logger.info("If you want to process the full dataset, use the --full-dataset flag\n")
        
        # Limit monolingual sentences to 500 per language for faster matching
        # (can be adjusted based on dataset size)
        # For full dataset, use all monolingual sentences
        max_mono = None if args.full_dataset else 500
        matching_results = test_matching_algorithm(
            code_switched,
            monolingual,
            sample_size=matching_sample_size,
            max_monolingual_per_lang=max_mono
        )
        
        # Step 5: Analyze distributions
        logger.info("\nStep 5: Analyzing distributions...")
        distributions = analyze_distributions(code_switched)
        
        # Step 6: Generate feasibility report
        logger.info("\nStep 6: Generating feasibility report...")
        all_results = {
            'monolingual': monolingual,
            'pos_tagging': pos_results,
            'matching': matching_results,
            'distributions': distributions
        }
        report = generate_feasibility_report(all_results)
        
        # Step 7: Save outputs
        logger.info("\nStep 7: Saving outputs...")
        save_outputs(
            output_dir,
            monolingual,
            pos_results,
            matching_results,
            distributions,
            report,
            figures_dir=figures_dir
        )
        
        # Print report to console
        logger.info("\n" + "=" * 80)
        logger.info("FEASIBILITY REPORT")
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

