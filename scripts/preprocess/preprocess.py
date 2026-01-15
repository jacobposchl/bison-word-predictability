"""
Preprocessor for code-switching predictability analysis.

This script preprocesses raw EAF annotation files into processed data:
1. Load EAF files from raw data directory
2. Process all files (clean, tokenize, extract patterns)
3. Build code-switching patterns (with and without fillers)
4. Export processed data to CSV files
5. Generate visualizations
6. Print analysis summaries
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.analysis.pattern_analysis import process_all_files
from src.data.data_export import (
    export_to_csv,
    filter_code_switching_sentences,
    export_all_sentences_to_csv,
    export_monolingual_sentences,
    export_translated_sentences
)
from src.plots.preprocessing.plot_preprocessing import (
    print_analysis_summary,
    plot_matrix_language_distribution,
    plot_equal_matrix_cases,
    plot_filler_impact
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """
    Main entry point for the code-switching data preprocessing pipeline
    """
    parser = argparse.ArgumentParser(
        description='Preprocess raw EAF files into processed code-switching data'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    parser.add_argument(
        '--no-translation',
        action='store_true',
        help='Skip translation process'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config()
        
        data_path = config.get_data_path()
        min_sentence_words = config.get_min_sentence_words()
        time_gap_threshold_ms = config.get_time_gap_threshold_ms()
        csv_with_fillers_path = config.get_csv_with_fillers_path()
        csv_without_fillers_path = config.get_csv_without_fillers_path()
        csv_all_sentences_path = config.get_csv_all_sentences_path()
        figures_dir = config.get_preprocessing_figures_dir()
        
        logger.info("="*80)
        logger.info("Code-Switching Data Preprocessing")
        logger.info("="*80)
        logger.info(f"Raw data path: {data_path}")
        logger.info(f"Min sentence words: {min_sentence_words}")
        logger.info(f"Time gap threshold: {time_gap_threshold_ms}ms")
        logger.info(f"Processed data output: {config.get_preprocessing_results_dir()}/")
        logger.info(f"Figures output: {figures_dir}")
        logger.info("="*80)
        
        # Step 1: Process all EAF files
        logger.info("\nStep 1: Processing EAF files...")
        all_sentences = process_all_files(
            data_path=data_path,
            min_sentence_words=min_sentence_words,
            time_gap_threshold_ms=time_gap_threshold_ms
        )
        
        if not all_sentences:
            logger.error("No sentences were processed. Check your data path and file format.")
            sys.exit(1)
        
        logger.info(f"Processed {len(all_sentences)} total sentences")
        
        # Step 2: Filter code-switching sentences
        logger.info("\nStep 2: Filtering code-switching sentences...")
        with_fillers = filter_code_switching_sentences(all_sentences, include_fillers=True)
        without_fillers = filter_code_switching_sentences(all_sentences, include_fillers=False)
        
        logger.info(f"Code-switching sentences WITH fillers: {len(with_fillers)}")
        logger.info(f"Code-switching sentences WITHOUT fillers: {len(without_fillers)}")
        
        # Step 3: Export to CSV
        logger.info("\nStep 3: Exporting processed data to CSV...")
        csv_with_fillers_df, csv_without_fillers_df = export_to_csv(
            all_sentences,
            csv_with_fillers_path,
            csv_without_fillers_path
        )
        
        # Also export ALL sentences (monolingual + code-switched) for exploratory analysis
        logger.info("\nStep 3b: Exporting ALL sentences (monolingual + code-switched)...")
        csv_all_sentences_df = export_all_sentences_to_csv(
            all_sentences,
            csv_all_sentences_path
        )
        
        # Export monolingual sentences (Cantonese and English, with/without fillers)
        logger.info("\nStep 3c: Exporting monolingual sentences...")
        cant_with, cant_without, eng_with, eng_without = export_monolingual_sentences(
            all_sentences,
            config
        )
        
        # Export translated code-switched sentences
        translated_df = None
        if not args.no_translation:
            logger.info("\nStep 3d: Exporting and translating sentences...")
            translated_df = export_translated_sentences(
                all_sentences,
                config,
                do_translation=True
            )
        else:
            logger.info("\nStep 3d: Skipping translation process (--no-translation flag set)")
            # Still create the structure but without translation
            translated_df = export_translated_sentences(
                all_sentences,
                config,
                do_translation=False
            )
        
        # Step 4: Generate visualizations
        if not args.no_plots:
            logger.info("\nStep 4: Generating visualizations...")
            plot_matrix_language_distribution(with_fillers, without_fillers, figures_dir)
            plot_equal_matrix_cases(with_fillers, without_fillers, figures_dir)
            plot_filler_impact(with_fillers, without_fillers, figures_dir)
            logger.info("Visualizations saved to " + figures_dir)
        else:
            logger.info("\nStep 4: Skipping visualizations (--no-plots flag set)")
        
        # Step 5: Print analysis summary
        logger.info("\nStep 5: Analysis Summary")
        logger.info("="*80)
        print_analysis_summary(with_fillers, without_fillers)
        
        logger.info("\n" + "="*80)
        logger.info("Preprocessing complete!")
        logger.info("="*80)
        logger.info(f"Processed data saved to {config.get_preprocessing_results_dir()}/")
        logger.info("\nCode-switching datasets:")
        logger.info(f"  - {csv_with_fillers_path}")
        logger.info(f"  - {csv_without_fillers_path}")
        logger.info("\nAll sentences:")
        logger.info(f"  - {csv_all_sentences_path}")
        logger.info("\nMonolingual sentences:")
        logger.info(f"  - {config.get_csv_cantonese_mono_with_fillers_path()} ({len(cant_with)} sentences)")
        logger.info(f"  - {config.get_csv_cantonese_mono_without_fillers_path()} ({len(cant_without)} sentences)")
        logger.info(f"  - {config.get_csv_english_mono_with_fillers_path()} ({len(eng_with)} sentences)")
        logger.info(f"  - {config.get_csv_english_mono_without_fillers_path()} ({len(eng_without)} sentences)")
        if not args.no_translation:
            logger.info("\nTranslated sentences:")
            logger.info(f"  - {config.get_csv_cantonese_translated_path()} ({len(translated_df)} sentences)")
        if not args.no_plots:
            logger.info(f"\nFigures saved to: {figures_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

