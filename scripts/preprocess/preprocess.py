"""
Preprocessor for code-switching predictability analysis.

This script preprocesses raw EAF annotation files into processed data:
1. Load EAF files from raw data directory
2. Process all files (clean, tokenize, extract patterns)
3. Build code-switching patterns
4. Export processed data to CSV files

Exports:
- all_sentences.csv: All sentences for context retrieval
- cantonese_monolingual_WITHOUT_fillers.csv: Monolingual Cantonese for matching
- cantonese_translated_WITHOUT_fillers.csv: Translated code-switched sentences

Note: For analysis summaries and visualizations, run: python scripts/plots/figures.py --preprocess
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.analysis.pattern_analysis import process_all_files
from src.data.data_export import (
    export_all_sentences_to_csv,
    export_cantonese_monolingual,
    export_translated_sentences,
    export_interviewer_sentences,
    generate_preprocessing_report
)


def setup_logging() -> None:
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
        csv_interviewer_path = config.get_csv_interviewer_path()
        
        logger.info("Processing EAF files...")
        all_sentences, interviewer_sentences = process_all_files(data_path=data_path)
        
        if not all_sentences:
            logger.error("No sentences were processed. Check your data path and file format.")
            sys.exit(1)
        
        logger.info(f"Processed {len(all_sentences)} total participant sentences")
        logger.info(f"Processed {len(interviewer_sentences)} total interviewer sentences")
        
        # Export ALL sentences (monolingual + code-switched) for context retrieval
        logger.info("Exporting all sentences...")
        csv_all_sentences_df, preprocessing_stats = export_all_sentences_to_csv(
            all_sentences,
            csv_all_sentences_path,
            min_sentence_words=min_sentence_words
        )
        
        # Export interviewer sentences (IR tier)
        logger.info("Exporting interviewer sentences...")
        interviewer_df, interviewer_stats = export_interviewer_sentences(
            interviewer_sentences,
            csv_interviewer_path,
            min_sentence_words=min_sentence_words
        )
        
        # Export only Cantonese monolingual sentences WITHOUT fillers (needed for matching)
        logger.info("Exporting Cantonese monolingual sentences...")
        cant_without, monolingual_stats = export_cantonese_monolingual(
            all_sentences,
            config,
            min_sentence_words=min_sentence_words
        )
        
        # Export translated code-switched sentences
        if not args.no_translation:
            logger.info("Exporting and translating sentences...")
            translated_df, translation_stats = export_translated_sentences(
                all_sentences,
                config,
                do_translation=True,
                min_sentence_words=min_sentence_words
            )
        else:
            logger.info("Skipping translation (--no-translation flag set)")
            translated_df, translation_stats = export_translated_sentences(
                all_sentences,
                config,
                do_translation=False,
                min_sentence_words=min_sentence_words
            )
        
        # Generate CSV report
        report_path = os.path.join(config.get_preprocessing_results_dir(), 'preprocessing_report.csv')
        generate_preprocessing_report(
            preprocessing_stats,
            monolingual_stats,
            translation_stats,
            report_path,
            interviewer_stats=interviewer_stats
        )
        
        logger.info("Preprocessing complete!")
        logger.info(f"Output directory: {config.get_preprocessing_results_dir()}/")
        logger.info(f"To generate figures: python scripts/plots/figures.py --preprocess")
        
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

