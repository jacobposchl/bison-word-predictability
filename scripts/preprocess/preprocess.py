"""
Preprocessor for code-switching predictability analysis.

This script preprocesses raw EAF annotation files into processed data:
1. Load EAF files from raw data directory
2. Process all files (clean, tokenize, extract patterns)
3. Build code-switching patterns (without fillers)
4. Export processed data to CSV files (only datasets needed for downstream analysis)
5. Generate visualizations
6. Print analysis summaries

Exports only:
- all_sentences.csv: All sentences for context retrieval
- cantonese_monolingual_WITHOUT_fillers.csv: Monolingual Cantonese for matching
- cantonese_translated_WITHOUT_fillers.csv: Translated code-switched sentences
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
    filter_code_switching_sentences,
    export_all_sentences_to_csv,
    export_cantonese_monolingual,
    export_translated_sentences
)
from src.plots.preprocessing.plot_preprocessing_simple import (
    print_analysis_summary_simple,
    plot_matrix_language_distribution_simple
)


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
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
        '--no-translation',
        action='store_true',
        help='Skip translation process'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config()
        
        data_path = config.get_data_path()
        min_sentence_words = config.get_min_sentence_words()
        csv_all_sentences_path = config.get_csv_all_sentences_path()
        figures_dir = config.get_preprocessing_figures_dir()
        
        logger.info("="*80)
        logger.info("Code-Switching Data Preprocessing")
        logger.info("="*80)
        logger.info(f"Raw data path: {data_path}")
        logger.info(f"Min sentence words: {min_sentence_words}")
        logger.info(f"Processed data output: {config.get_preprocessing_results_dir()}/")
        logger.info(f"Figures output: {figures_dir}")
        logger.info("="*80)
        
        # Process all EAF files
        logger.info("\nProcessing EAF files...")
        all_sentences = process_all_files(
            data_path=data_path
        )
        
        if not all_sentences:
            logger.error("No sentences were processed. Check your data path and file format.")
            sys.exit(1)
        
        logger.info(f"Processed {len(all_sentences)} total sentences")
        
        # Filter code-switching sentences (for visualization only)
        logger.info("\nFiltering code-switching sentences...")
        without_fillers = filter_code_switching_sentences(all_sentences, include_fillers=False)
        
        logger.info(f"Code-switching sentences WITHOUT fillers: {len(without_fillers)}")
        
        # Export ALL sentences (monolingual + code-switched) for context retrieval
        logger.info("\nExporting ALL sentences (monolingual + code-switched)...")
        csv_all_sentences_df = export_all_sentences_to_csv(
            all_sentences,
            csv_all_sentences_path,
            min_sentence_words=min_sentence_words
        )
        
        # Export only Cantonese monolingual sentences WITHOUT fillers (needed for matching)
        logger.info("\nExporting Cantonese monolingual sentences (WITHOUT fillers)...")
        cant_without = export_cantonese_monolingual(
            all_sentences,
            config,
            min_sentence_words=min_sentence_words
        )
        
        # Export translated code-switched sentences
        translated_df = None
        if not args.no_translation:
            logger.info("\nExporting and translating sentences...")
            translated_df = export_translated_sentences(
                all_sentences,
                config,
                do_translation=True,
                min_sentence_words=min_sentence_words
            )
        else:
            logger.info("\nSkipping translation process (--no-translation flag set)")
            # Still create the structure but without translation
            translated_df = export_translated_sentences(
                all_sentences,
                config,
                do_translation=False,
                min_sentence_words=min_sentence_words
            )
        
        # Generate visualizations (using only WITHOUT fillers data)
        logger.info("\nGenerating visualizations...")
        plot_matrix_language_distribution_simple(without_fillers, figures_dir)
        logger.info("Visualizations saved to " + figures_dir)
        
        # Print analysis summary
        logger.info("\nAnalysis Summary")
        logger.info("="*80)
        print_analysis_summary_simple(without_fillers)
        
        logger.info("\n" + "="*80)
        logger.info("Preprocessing complete!")
        logger.info("="*80)
        logger.info(f"Processed data saved to {config.get_preprocessing_results_dir()}/")
        logger.info("\nAll sentences:")
        logger.info(f"  - {csv_all_sentences_path}")
        logger.info("\nMonolingual sentences:")
        logger.info(f"  - {config.get_csv_cantonese_mono_without_fillers_path()} ({len(cant_without)} sentences)")
        if not args.no_translation:
            logger.info("\nTranslated sentences:")
            logger.info(f"  - {config.get_csv_cantonese_translated_path()} ({len(translated_df)} sentences)")
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

