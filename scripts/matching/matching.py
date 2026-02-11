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
from src.plots.matching.report_generator import generate_window_matching_report
from src.analysis.matching_algorithm import analyze_window_matching

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
    
    # Get output directories from config
    output_dir = Path(config.get_matching_results_dir())
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    
    logger.info("Starting POS window matching analysis...")
    
    try:
        # Load datasets (using filenames from config)
        translated_csv = preprocessing_dir / config.get('output.csv_cantonese_translated')
        
        if not translated_csv.exists():
            raise FileNotFoundError(f"Translated sentences CSV not found: {translated_csv}")
        
        translated_df = pd.read_csv(translated_csv)

        monolingual_csv = preprocessing_dir / config.get('output.csv_cantonese_mono_without_fillers')
        
        if not monolingual_csv.exists():
            raise FileNotFoundError(f"Monolingual CSV not found: {monolingual_csv}")
        
        monolingual_df = pd.read_csv(monolingual_csv)

        all_sentences_csv = preprocessing_dir / config.get('output.csv_all_sentences')
        
        if not all_sentences_csv.exists():
            raise FileNotFoundError(f"All sentences CSV not found: {all_sentences_csv}")
        
        all_sentences_df = pd.read_csv(all_sentences_csv)
        

        translator = NLLBTranslator(
            model_name=config.get_translation_model(),
            device=config.get_translation_device(),
            show_progress=False
        )
        

        window_sizes = config.get_analysis_window_sizes()
        similarity_threshold = config.get_analysis_similarity_threshold()

        num_workers = args.num_workers_free if args.num_workers_free is not None else config.get_analysis_num_workers()
        
        # Filter to sentences with valid switch indices
        # Note: Pattern filtering (C >= min_cantonese followed by E) is already done in preprocessing
        filtered_translated_sentences = [s for s in translated_df.to_dict('records') if s.get('switch_index', -1) >= 0]
        
        monolingual_sentences = monolingual_df.to_dict('records')
        
        logger.info(f"Analyzing {len(filtered_translated_sentences)} sentences...")
        logger.info(f"Window sizes to process: {window_sizes}")
        
        # Run window matching for all window sizes
        window_results = analyze_window_matching(
            translated_sentences=filtered_translated_sentences,
            monolingual_sentences=monolingual_sentences,
            window_sizes=window_sizes,
            similarity_threshold=similarity_threshold,
            top_k=5,
            num_workers=num_workers
        )
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track counts for filtering report
        num_cs_sentences = len(filtered_translated_sentences)
        num_mono_sentences = len(monolingual_sentences)
        
        # Create and save analysis dataset for each window size
        logger.info("\nCreating analysis datasets for each window size...")
        analysis_datasets = {}  # Store datasets for report generation
        context_stats_by_window = {}  # Store context quality stats
        
        for window_size in window_sizes:
            logger.info(f"\nProcessing window size {window_size}...")
            
            # Create analysis dataset for this window size
            analysis_df, context_stats = create_analysis_dataset(
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
            
            # Store for report generation
            analysis_datasets[window_size] = analysis_df
            # context_stats should always exist when we have data (only None on errors)
            if context_stats is not None:
                context_stats_by_window[window_size] = context_stats
        
        # Generate window matching CSV reports (for all window sizes)
        logger.info("\nGenerating matching analysis CSV reports...")
        report_message = generate_window_matching_report(
            window_results,
            similarity_threshold=similarity_threshold,
            output_dir=str(output_dir),
            analysis_datasets=analysis_datasets,
            num_cs_sentences=num_cs_sentences,
            num_mono_sentences=num_mono_sentences,
            context_stats_by_window=context_stats_by_window
        )
        logger.info(report_message)
        logger.info(f"\nAnalysis complete! Results saved to: {output_dir}")
        logger.info(f"Created {len(window_sizes)} analysis datasets (one per window size)")
        logger.info(f"To generate figures, run: python scripts/plots/figures.py --matching")
        
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

