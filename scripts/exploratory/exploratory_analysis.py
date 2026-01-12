"""
Main script for running exploratory analysis.

This script orchestrates the exploratory analysis pipeline:
1. Load translated code-switched sentences
2. Load monolingual Cantonese sentences
3. Analyze POS window matching
4. Create final analysis dataset
5. Save results
"""

import argparse
import logging
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.data.analysis_dataset import create_analysis_dataset
from src.analysis.feasibility import (
    plot_similarity_distributions,
    generate_window_matching_report
)
from src.analysis.matching_algorithm import analyze_window_matching

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
        description='POS window matching analysis for code-switching'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of sentences to process (default: process all sentences)'
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
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    
    logger.info("=" * 80)
    logger.info("POS WINDOW MATCHING ANALYSIS")
    logger.info("=" * 80)
    if args.sample_size is None:
        logger.info("Mode: FULL DATASET (processing all sentences)")
    else:
        logger.info(f"Sample size: {args.sample_size} sentences")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Figures directory: {figures_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Step 1: Load translated code-switched sentences
        logger.info("Step 1: Loading translated code-switched sentences...")
        translated_csv = preprocessing_dir / "cantonese_translated_WITHOUT_fillers.csv"
        
        if not translated_csv.exists():
            raise FileNotFoundError(f"Translated sentences CSV not found: {translated_csv}")
        
        translated_df = pd.read_csv(translated_csv)
        logger.info(f"Loaded {len(translated_df)} translated sentences")
        
        # Step 2: Load monolingual Cantonese sentences
        logger.info("\nStep 2: Loading monolingual Cantonese sentences...")
        monolingual_csv = preprocessing_dir / "cantonese_monolingual_WITHOUT_fillers.csv"
        
        if not monolingual_csv.exists():
            raise FileNotFoundError(f"Monolingual CSV not found: {monolingual_csv}")
        
        monolingual_df = pd.read_csv(monolingual_csv)
        logger.info(f"Loaded {len(monolingual_df)} monolingual Cantonese sentences")
        
        # Step 3: Run POS window matching analysis
        logger.info("\nStep 3: Running POS window matching analysis...")
        
        # Get parameters from config
        window_size = config.get_analysis_window_size()
        similarity_threshold = config.get_analysis_similarity_threshold()
        
        logger.info(f"Window size: {window_size}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        
        # Filter to sentences with valid switch indices
        translated_sentences = [s for s in translated_df.to_dict('records') if s.get('switch_index', -1) >= 0]
        monolingual_sentences = monolingual_df.to_dict('records')
        
        # Apply sample size if specified
        if args.sample_size is not None:
            translated_sentences = translated_sentences[:args.sample_size]
            logger.info(f"Limited to {len(translated_sentences)} sentences for testing")
        
        logger.info(f"Analyzing {len(translated_sentences)} sentences with valid switch points")
        
        # Run window matching
        window_results = analyze_window_matching(
            translated_sentences=translated_sentences,
            monolingual_sentences=monolingual_sentences,
            window_sizes=[window_size],
            similarity_threshold=similarity_threshold,
            top_k=5
        )
        
        # Step 4: Create analysis dataset
        logger.info("\nStep 4: Creating final analysis dataset...")
        analysis_df = create_analysis_dataset(config, translated_df, monolingual_df)
        
        # Step 5: Save outputs
        logger.info("\nStep 5: Saving outputs...")
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis dataset
        analysis_csv_path = output_dir / "analysis_dataset.csv"
        analysis_df.to_csv(analysis_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved analysis dataset: {analysis_csv_path} ({len(analysis_df)} rows)")
        
        # Generate and save similarity distribution plot
        logger.info("\nGenerating similarity distribution plots...")
        plot_path = plot_similarity_distributions(window_results, str(figures_dir))
        
        # Generate window matching report
        logger.info("\nGenerating window matching report...")
        window_report = generate_window_matching_report(window_results, similarity_threshold=similarity_threshold)
        
        # Save window matching results
        window_key = f'window_{window_size}'
        if window_key in window_results:
            results = window_results[window_key]
            
            # Save summary CSV
            summary_data = [{
                'window_size': results['window_size'],
                'total_sentences': results['total_sentences'],
                'sentences_with_matches': results['sentences_with_matches'],
                'match_rate': results['match_rate'],
                'total_matches': results['total_matches'],
                'avg_matches_per_sentence': results['avg_matches_per_sentence'],
                'avg_similarity': results['avg_similarity'],
                'similarity_min': min(results['similarity_scores']) if results['similarity_scores'] else 0.0,
                'similarity_max': max(results['similarity_scores']) if results['similarity_scores'] else 0.0,
                'similarity_median': pd.Series(results['similarity_scores']).median() if results['similarity_scores'] else 0.0
            }]
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = output_dir / "window_matching_summary.csv"
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            logger.info(f"Saved window matching summary: {summary_csv}")
            
            # Save detailed CSV
            detailed_data = results['detailed_matches']
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_csv = output_dir / "window_matching_detailed.csv"
                detailed_df.to_csv(detailed_csv, index=False, encoding='utf-8-sig')
                logger.info(f"Saved window matching details: {detailed_csv} ({len(detailed_df)} matches)")
            
            # Save window matching report
            window_report_path = output_dir / "window_matching_report.txt"
            with open(window_report_path, 'w', encoding='utf-8') as f:
                f.write(window_report)
            logger.info(f"Saved window matching report: {window_report_path}")
        
        # Print window matching report
        logger.info("\n" + "=" * 80)
        logger.info("WINDOW MATCHING REPORT")
        logger.info("=" * 80)
        print(window_report)
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure preprocessing has been run first")
        logger.error("Run: python scripts/preprocessing/preprocess.py")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

