"""
Main script for running matching analysis.

This script orchestrates the matching analysis pipeline:
1. Load translated code-switched sentences
2. Load monolingual Cantonese sentences
3. Analyze POS window matching
4. Create final analysis dataset for each window size
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
from src.analysis.pos_tagging import parse_pattern_segments

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main entry point for matching analysis."""
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
        '--batch-size',
        type=int,
        default=None,
        help='Number of sentences per batch (default: from config or process all at once)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for checkpoint files (default: from config or no checkpoints)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resuming from checkpoints (default: resume enabled)'
    )
    parser.add_argument(
        '--num-workers-free',
        type=int,
        default=None,
        help='Number of CPU cores to leave free (default: from config or use all cores)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config()
    
    # Get output and figures directories from config
    output_dir = Path(config.get_matching_results_dir())
    figures_dir = Path(config.get_matching_figures_dir())
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    
    logger.info("Starting POS window matching analysis...")
    if args.sample_size is not None:
        logger.info(f"Sample size: {args.sample_size} sentences")
    
    try:
        #Load datasets
        translated_csv = preprocessing_dir / "cantonese_translated_WITHOUT_fillers.csv"
        
        if not translated_csv.exists():
            raise FileNotFoundError(f"Translated sentences CSV not found: {translated_csv}")
        
        translated_df = pd.read_csv(translated_csv)

        monolingual_csv = preprocessing_dir / "cantonese_monolingual_WITHOUT_fillers.csv"
        
        if not monolingual_csv.exists():
            raise FileNotFoundError(f"Monolingual CSV not found: {monolingual_csv}")
        
        monolingual_df = pd.read_csv(monolingual_csv)

        all_sentences_csv = preprocessing_dir / config.get('output.csv_all_sentences', 'all_sentences.csv')
        
        if not all_sentences_csv.exists():
            raise FileNotFoundError(f"All sentences CSV not found: {all_sentences_csv}")
        
        all_sentences_df = pd.read_csv(all_sentences_csv)
        
        #Initialize translator
        translator = NLLBTranslator(
            model_name=config.get_translation_model(),
            device=config.get_translation_device(),
            show_progress=False
        )
        
        # Get parameters from config
        window_sizes = config.get_analysis_window_sizes()
        similarity_threshold = config.get_analysis_similarity_threshold()
        min_cantonese = config.get_analysis_min_cantonese_words()
        
        # Get optimization parameters (CLI args override config)
        # num_workers means "cores to leave free" - None means use all cores
        num_workers = args.num_workers_free if args.num_workers_free is not None else config.get_analysis_num_workers()
        batch_size = args.batch_size if args.batch_size is not None else config.get_analysis_batch_size()
        checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else config.get_analysis_checkpoint_dir()
        resume = not args.no_resume
        
        # Create checkpoint directory if specified
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory: {checkpoint_path}")
            if resume:
                logger.info("Resume enabled: will load existing checkpoints")
            else:
                logger.info("Resume disabled: will reprocess all batches")
        
        # Filter to sentences with valid switch indices
        translated_sentences = [s for s in translated_df.to_dict('records') if s.get('switch_index', -1) >= 0]
        
        # Apply additional filtering for sentences matching pattern criteria:
        # Pattern must start with C >= min_cantonese, followed by E
        filtered_translated_sentences = []
        for sent in translated_sentences:
            pattern = sent.get('pattern', '')
            segments = parse_pattern_segments(pattern)
            
            # Check criteria: starts with C >= min_cantonese, followed by E
            if len(segments) >= 2:
                first_lang, first_count = segments[0]
                second_lang, _ = segments[1]
                
                if first_lang == 'C' and first_count >= min_cantonese and second_lang == 'E':
                    filtered_translated_sentences.append(sent)
        
        monolingual_sentences = monolingual_df.to_dict('records')
        
        # Apply sample size if specified
        if args.sample_size is not None:
            filtered_translated_sentences = filtered_translated_sentences[:args.sample_size]
        
        logger.info(f"Analyzing {len(filtered_translated_sentences)} sentences...")
        logger.info(f"Window sizes to process: {window_sizes}")
        
        # Run window matching for all window sizes
        window_results = analyze_window_matching(
            translated_sentences=filtered_translated_sentences,
            monolingual_sentences=monolingual_sentences,
            window_sizes=window_sizes,
            similarity_threshold=similarity_threshold,
            top_k=5,
            num_workers=num_workers,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            resume=resume
        )
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and save analysis dataset for each window size
        logger.info("\nCreating analysis datasets for each window size...")
        for window_size in window_sizes:
            logger.info(f"\nProcessing window size {window_size}...")
            
            # Create analysis dataset for this window size
            analysis_df = create_analysis_dataset(
                config,
                filtered_translated_sentences,
                window_results,
                all_sentences_df,
                translator,
                window_size=window_size
            )
            
            if len(analysis_df) == 0:
                logger.warning(f"No data for window size {window_size}, skipping...")
                continue
            
            # Save dataset with window size in filename
            analysis_csv_path = output_dir / f"analysis_dataset_window_{window_size}.csv"
            analysis_df.to_csv(analysis_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved analysis dataset: {analysis_csv_path} ({len(analysis_df)} rows)")
        
        # Generate and save similarity distribution plot (for all window sizes)
        plot_path = plot_similarity_distributions(window_results, str(figures_dir))
        
        # Generate window matching report (for all window sizes)
        window_report = generate_window_matching_report(window_results, similarity_threshold=similarity_threshold)
        
        # Save window matching report
        window_report_path = output_dir / "window_matching_report.txt"
        with open(window_report_path, 'w', encoding='utf-8') as f:
            f.write(window_report)
        
        logger.info(f"\nAnalysis complete! Results saved to: {output_dir}")
        logger.info(f"Created {len(window_sizes)} analysis datasets (one per window size)")
        
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

