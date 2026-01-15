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
from src.experiments.nllb_translator import NLLBTranslator
from src.plots.exploratory.plot_functions import plot_similarity_distributions
from src.plots.exploratory.report_generator import generate_window_matching_report
from src.analysis.matching_algorithm import analyze_window_matching

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG,
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config()
    
    # Get output and figures directories from config
    output_dir = Path(config.get_exploratory_results_dir())
    figures_dir = Path(config.get_exploratory_figures_dir())
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    
    logger.info("=" * 80)
    logger.info("POS WINDOW MATCHING IN PROGRESS...")
    logger.info("=" * 80)

    if args.sample_size is not None:
        logger.info(f"Selected sample size of: {args.sample_size} sentences")

    logger.info(f"Output directory:  {output_dir}")
    logger.info(f"Figures directory: {figures_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        #Load translated code-switched sentences
        logger.info("Loading translated code-switched sentences...")
        translated_csv = preprocessing_dir / "cantonese_translated_WITHOUT_fillers.csv"
        
        if not translated_csv.exists():
            raise FileNotFoundError(f"Translated sentences CSV not found: {translated_csv}")
        
        translated_df = pd.read_csv(translated_csv)
        logger.info("   WORKED")

        #Load monolingual Cantonese sentences
        logger.info("\nLoading monolingual Cantonese sentences...")
        monolingual_csv = preprocessing_dir / "cantonese_monolingual_WITHOUT_fillers.csv"
        
        if not monolingual_csv.exists():
            raise FileNotFoundError(f"Monolingual CSV not found: {monolingual_csv}")
        
        monolingual_df = pd.read_csv(monolingual_csv)
        logger.info("   WORKED")

        #Load all sentences for discourse context
        logger.info("\nLoading all sentences dataset for context...")
        all_sentences_csv = preprocessing_dir / config.get('output.csv_all_sentences', 'all_sentences.csv')
        
        all_sentences_df = None
        translator = None
        
        if not all_sentences_csv.exists():
            raise FileNotFoundError(f"All sentences CSV not found: {all_sentences_csv}")
        
        all_sentences_df = pd.read_csv(all_sentences_csv)
        logger.info("   WORKED")
        
        #Initialize translator for context translation
        logger.info("Initializing translator for context processing...")
        translator = NLLBTranslator(
            model_name=config.get_translation_model(),
            device=config.get_translation_device(),
            show_progress=False
        )
        logger.info("   WORKED")
        
        #Run POS window matching analysis
        logger.info("Running POS window matching analysis...")
        
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
        
        logger.info(f"Analyzing {len(translated_sentences)} sentences with valid switch points")
        
        # Run window matching
        window_results = analyze_window_matching(
            translated_sentences=translated_sentences,
            monolingual_sentences=monolingual_sentences,
            window_sizes=[window_size],
            similarity_threshold=similarity_threshold,
            top_k=5
        )
        
        logger.info("   WORKED")

        #Create analysis dataset
        logger.info("\nCreating final analysis dataset...")
        analysis_df = create_analysis_dataset(
            config,
            translated_df,
            monolingual_df,
            all_sentences_df,
            translator
        )
        
        logger.info("   WORKED")

        #Save outputs
        logger.info("\nSaving outputs...")
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save unified analysis dataset (includes all CS sentence info, matched mono info, and statistics)
        analysis_csv_path = output_dir / "analysis_dataset.csv"
        analysis_df.to_csv(analysis_csv_path, index=False, encoding='utf-8-sig')
        
        # Generate and save similarity distribution plot
        plot_path = plot_similarity_distributions(window_results, str(figures_dir))
        
        # Generate window matching report
        window_report = generate_window_matching_report(window_results, similarity_threshold=similarity_threshold)
        
        # Log summary of excluded sentences
        window_key = f'window_{window_size}'
        if window_key in window_results:
            # Save window matching report
            window_report_path = output_dir / "window_matching_report.txt"
            with open(window_report_path, 'w', encoding='utf-8') as f:
                f.write(window_report)

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

