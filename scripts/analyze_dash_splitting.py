"""
Script to analyze dash splitting/removal in English and Cantonese annotations.

This script searches through the dataset to find cases where tokens contain dashes
and are split by the split_on_internal_dashes function. It reports statistics
and examples for both English and Cantonese cases.
"""

import argparse
import logging
import os
import sys
import csv
from pathlib import Path
import pycantonese

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.eaf_processor import (
    get_all_eaf_files,
    load_eaf_file,
    get_main_tier,
    extract_tiers
)
from src.text_cleaning import (
    split_on_internal_dashes,
    normalize_dashes,
    clean_word,
    has_content,
    UNICODE_DASHES,
    DASH_RE
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def has_dash(text: str) -> bool:
    """
    Check if text contains any dash characters (ASCII or Unicode).
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains any dash characters
    """
    if not text:
        return False
    return bool(DASH_RE.search(text))


def get_cleaned_annotation_text(annotation_text: str, lang: str) -> str:
    """
    Apply the full cleaning pipeline to annotation text and return cleaned version.
    
    This mimics the tokenization process: tokenize, split on dashes, clean words,
    filter non-content, then join back together.
    
    Args:
        annotation_text: Original annotation text
        lang: Language code ('C' for Cantonese, 'E' for English)
        
    Returns:
        Cleaned annotation text as a space-separated string
    """
    # Split based on language
    if lang == 'C':  # Cantonese
        raw_tokens = pycantonese.segment(annotation_text)
    else:  # English
        raw_tokens = annotation_text.split()
    
    cleaned_tokens = []
    
    for w in raw_tokens:
        # Normalize and split on internal dashes
        subparts = split_on_internal_dashes(w)
        if not subparts:  # either pure dash or empty after split
            continue
        for piece in subparts:
            cw = clean_word(piece)
            if cw and has_content(cw):  # keep only real content
                cleaned_tokens.append(cw)
    
    return ' '.join(cleaned_tokens)


def get_token_context(token: str, annotation_text: str, context_chars: int = 30) -> str:
    """
    Get the token with surrounding context from the annotation text.
    
    Args:
        token: Token to find
        annotation_text: Full annotation text
        context_chars: Number of characters to include before and after
        
    Returns:
        String with token and surrounding context
    """
    # Find the token in the text (handle multiple occurrences by using the first)
    token_pos = annotation_text.find(token)
    if token_pos == -1:
        # If exact match not found, return just the token
        return token
    
    # Get context before and after
    start = max(0, token_pos - context_chars)
    end = min(len(annotation_text), token_pos + len(token) + context_chars)
    
    context = annotation_text[start:end]
    return context


def get_cleaned_context(token: str, annotation_text: str, cleaned_text: str, lang: str, context_chars: int = 30) -> str:
    """
    Get the cleaned version of the token with surrounding context.
    
    Args:
        token: Original token
        annotation_text: Original annotation text
        cleaned_text: Cleaned annotation text
        lang: Language code
        context_chars: Number of characters to include before and after
        
    Returns:
        String with cleaned token parts and surrounding context
    """
    # Find the token position in original text
    token_pos = annotation_text.find(token)
    if token_pos == -1:
        return ""
    
    # Get the context area in original text (same as original context)
    orig_start = max(0, token_pos - context_chars)
    orig_end = min(len(annotation_text), token_pos + len(token) + context_chars)
    orig_context = annotation_text[orig_start:orig_end]
    
    # Clean this context area to show how it would appear after cleaning
    cleaned_context = get_cleaned_annotation_text(orig_context, lang)
    
    return cleaned_context


def analyze_token_dash_splitting(token: str, lang: str) -> dict:
    """
    Analyze a single token for dash splitting behavior.
    
    Args:
        token: Original token text
        lang: Language code ('C' for Cantonese, 'E' for English)
        
    Returns:
        Dictionary with analysis results, or None if no dashes found
    """
    if not has_dash(token):
        return None
    
    return {
        'token': token  # Store token for later context extraction
    }


def analyze_annotation(annotation_text: str, lang: str) -> list:
    """
    Analyze an annotation text for dash splitting cases.
    
    Args:
        annotation_text: Full annotation text
        lang: Language code ('C' for Cantonese, 'E' for English)
        
    Returns:
        List of dictionaries with dash splitting analysis results
    """
    results = []
    
    # Get cleaned version of annotation text
    cleaned_text = get_cleaned_annotation_text(annotation_text, lang)
    
    # Split based on language
    if lang == 'C':  # Cantonese
        raw_tokens = pycantonese.segment(annotation_text)
    else:  # English
        raw_tokens = annotation_text.split()
    
    # Analyze each token
    for token in raw_tokens:
        analysis = analyze_token_dash_splitting(token, lang)
        if analysis:
            # Get context around the token
            original_context = get_token_context(token, annotation_text)
            cleaned_context = get_cleaned_context(token, annotation_text, cleaned_text, lang)
            
            analysis['original'] = original_context
            analysis['cleaned'] = cleaned_context
            analysis['annotation_text_before'] = annotation_text
            analysis['annotation_text_after'] = cleaned_text
            # Remove the temporary 'token' key
            del analysis['token']
            results.append(analysis)
    
    return results


def process_eaf_file(file_path: str) -> dict:
    """
    Process a single EAF file to find dash splitting cases.
    
    Args:
        file_path: Path to EAF file
        
    Returns:
        Dictionary with statistics and examples for English and Cantonese
    """
    results = {
        'english': [],
        'cantonese': [],
        'file_name': os.path.basename(file_path),
        'participant_id': None
    }
    
    try:
        eaf = load_eaf_file(file_path)
        main_tier = get_main_tier(eaf)
        
        if not main_tier:
            logging.warning(f"No participant tier found in {file_path}")
            return results
        
        results['participant_id'] = main_tier
        
        # Extract tier annotations
        cant_annotations, eng_annotations = extract_tiers(eaf, main_tier)
        
        # Analyze Cantonese annotations
        for start_time, end_time, text in cant_annotations:
            analysis_list = analyze_annotation(text, 'C')
            results['cantonese'].extend(analysis_list)
        
        # Analyze English annotations
        for start_time, end_time, text in eng_annotations:
            analysis_list = analyze_annotation(text, 'E')
            results['english'].extend(analysis_list)
                
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
    
    return results


def generate_statistics(all_results: list) -> dict:
    """
    Generate statistics from all analysis results.
    
    Args:
        all_results: List of results dictionaries from process_eaf_file
        
    Returns:
        Dictionary with aggregated statistics
    """
    all_english = []
    all_cantonese = []
    
    # Collect all cases
    for file_results in all_results:
        all_english.extend(file_results['english'])
        all_cantonese.extend(file_results['cantonese'])
    
    stats = {
        'total_files': len(all_results),
        'english': {
            'total_cases': len(all_english),
            'all_cases': all_english,
            'examples': all_english[:20]  # First 20 examples for report
        },
        'cantonese': {
            'total_cases': len(all_cantonese),
            'all_cases': all_cantonese,
            'examples': all_cantonese[:20]  # First 20 examples for report
        }
    }
    
    return stats


def generate_report(stats: dict) -> str:
    """
    Generate a formatted text report from statistics.
    
    Args:
        stats: Statistics dictionary from generate_statistics
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DASH SPLITTING ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Total EAF files processed: {stats['total_files']}")
    report_lines.append("")
    
    # English statistics
    report_lines.append("-" * 80)
    report_lines.append("ENGLISH ANNOTATIONS")
    report_lines.append("-" * 80)
    eng = stats['english']
    report_lines.append(f"Total tokens with dashes: {eng['total_cases']}")
    report_lines.append("")
    
    if eng['examples']:
        report_lines.append("Examples of English tokens with dashes:")
        report_lines.append("")
        for i, example in enumerate(eng['examples'][:10], 1):
            report_lines.append(f"  {i}. Original context: '{example['original'][:80]}...'")
            report_lines.append(f"     Cleaned context: '{example['cleaned'][:80]}...'")
            report_lines.append("")
    
    # Cantonese statistics
    report_lines.append("-" * 80)
    report_lines.append("CANTONESE ANNOTATIONS")
    report_lines.append("-" * 80)
    cant = stats['cantonese']
    report_lines.append(f"Total tokens with dashes: {cant['total_cases']}")
    report_lines.append("")
    
    if cant['examples']:
        report_lines.append("Examples of Cantonese tokens with dashes:")
        report_lines.append("")
        for i, example in enumerate(cant['examples'][:10], 1):
            report_lines.append(f"  {i}. Original context: '{example['original'][:80]}...'")
            report_lines.append(f"     Cleaned context: '{example['cleaned'][:80]}...'")
            report_lines.append("")
    
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def export_to_csv(stats: dict, output_dir: Path):
    """
    Export analysis results to CSV files.
    
    Args:
        stats: Statistics dictionary
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the columns we want to export (original and cleaned next to each other)
    fieldnames = ['original', 'cleaned', 'annotation_text_before', 'annotation_text_after']
    
    # Export all English cases
    if stats['english']['all_cases']:
        eng_path = output_dir / "dash_splitting_english.csv"
        with open(eng_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in stats['english']['all_cases']:
                # Only write the fields we want
                row = {field: case.get(field, '') for field in fieldnames}
                writer.writerow(row)
        logging.info(f"Exported {len(stats['english']['all_cases'])} English cases to {eng_path}")
    
    # Export all Cantonese cases
    if stats['cantonese']['all_cases']:
        cant_path = output_dir / "dash_splitting_cantonese.csv"
        with open(cant_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in stats['cantonese']['all_cases']:
                # Only write the fields we want
                row = {field: case.get(field, '') for field in fieldnames}
                writer.writerow(row)
        logging.info(f"Exported {len(stats['cantonese']['all_cases'])} Cantonese cases to {cant_path}")


def main():
    """Main entry point for dash splitting analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze dash splitting/removal in English and Cantonese annotations'
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
        help='Override output directory from config file (default: from config)'
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
        # Use config default if output-dir not specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(config.get_dash_analysis_results_dir())
        
        logger.info("=" * 80)
        logger.info("DASH SPLITTING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Raw data path: {data_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        logger.info("")
        
        # Get all EAF files
        eaf_files = get_all_eaf_files(data_path)
        logger.info(f"Found {len(eaf_files)} EAF files to process")
        logger.info("")
        
        # Process each file
        all_results = []
        for file_idx, eaf_file in enumerate(eaf_files, 1):
            file_path = os.path.join(data_path, eaf_file)
            logger.info(f"[{file_idx}/{len(eaf_files)}] Processing {eaf_file}...")
            
            file_results = process_eaf_file(file_path)
            all_results.append(file_results)
            
            eng_count = len(file_results['english'])
            cant_count = len(file_results['cantonese'])
            if eng_count > 0 or cant_count > 0:
                logger.info(f"  Found {eng_count} English and {cant_count} Cantonese dash cases")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Generating statistics and report...")
        logger.info("=" * 80)
        
        # Generate statistics
        stats = generate_statistics(all_results)
        
        # Generate report
        report = generate_report(stats)
        
        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dash_splitting_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved report to {report_path}")
        
        # Export CSV files
        logger.info("Exporting results to CSV...")
        export_to_csv(stats, output_dir)
        
        # Print report to console
        logger.info("")
        print(report)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

