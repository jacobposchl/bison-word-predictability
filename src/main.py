"""
Main entry point for code-switching predictability analysis.

This script orchestrates the full pipeline:
1. Load EAF files
2. Process all files
3. Build patterns (with and without fillers)
4. Export to CSV
5. Generate visualizations
6. Print analysis summaries
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .pattern_analysis import process_all_files
from .data_export import export_to_csv, filter_code_switching_sentences
from .visualization import (
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
    """Main entry point for the code-switching analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Code-switching predictability analysis from EAF files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override data path from config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config file'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
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
        config = Config(config_path=args.config)
        
        # Override config with command-line arguments if provided
        if args.data_path:
            config._config['data']['path'] = args.data_path
        
        data_path = config.get_data_path()
        buffer_ms = config.get_buffer_ms()
        min_sentence_words = config.get_min_sentence_words()
        csv_with_fillers_path = config.get_csv_with_fillers_path()
        csv_without_fillers_path = config.get_csv_without_fillers_path()
        figures_dir = args.output_dir or config.get_figures_dir()
        
        logger.info("="*80)
        logger.info("Code-Switching Predictability Analysis")
        logger.info("="*80)
        logger.info(f"Data path: {data_path}")
        logger.info(f"Buffer: {buffer_ms*1000:.0f} ms")
        logger.info(f"Min sentence words: {min_sentence_words}")
        logger.info(f"Output directory: {figures_dir}")
        logger.info("="*80)
        
        # Step 1: Process all EAF files
        logger.info("\nStep 1: Processing EAF files...")
        all_sentences = process_all_files(
            data_path=data_path,
            buffer_ms=buffer_ms,
            min_sentence_words=min_sentence_words
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
        logger.info("\nStep 3: Exporting to CSV...")
        csv_with_fillers_df, csv_without_fillers_df = export_to_csv(
            all_sentences,
            csv_with_fillers_path,
            csv_without_fillers_path
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
        logger.info("Analysis complete!")
        logger.info("="*80)
        logger.info(f"CSV files saved:")
        logger.info(f"  - {csv_with_fillers_path}")
        logger.info(f"  - {csv_without_fillers_path}")
        if not args.no_plots:
            logger.info(f"Figures saved to: {figures_dir}")
        
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

