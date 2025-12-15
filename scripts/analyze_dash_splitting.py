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
    get_main_tier
)
from src.text_cleaning import has_content
from src.text_cleaning import (
    split_on_internal_dashes,
    normalize_dashes,
    clean_word,
    has_content,
    UNICODE_DASHES,
    DASH_RE
)


def extract_tiers_non_spaced(eaf, main_tier):
    """
    Extract Cantonese and English tier annotations using the non-spaced Cantonese tier.
    
    This is a custom version for the dash splitting analysis script that uses
    the regular Cantonese tier (not the Spaced version).
    
    Args:
        eaf: EAF object
        main_tier: Name of the main participant tier
        
    Returns:
        Tuple of (cantonese_annotations, english_annotations)
        Each is a list of (start_time, end_time, text) tuples
    """
    # Get annotations from the language-specific tiers
    # Use non-spaced Cantonese tier
    cant_tier_name = f"{main_tier}-Cantonese"
    eng_tier_name = f"{main_tier}-English"
    
    try:
        cant_c = eaf.get_annotation_data_for_tier(cant_tier_name)
        eng_c = eaf.get_annotation_data_for_tier(eng_tier_name)
    except KeyError as e:
        raise ValueError(f"Required tier not found: {e}")
    
    # Remove duplicates
    cant_c = list(set(cant_c))
    eng_c = list(set(eng_c))
    
    # Filter out punctuation-only and annotation marker annotations
    cant_c = [item for item in cant_c if has_content(item[2])]
    eng_c = [item for item in eng_c if has_content(item[2])]
    
    return cant_c, eng_c


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


def get_token_context(token: str, annotation_text: str, search_start: int = 0, context_chars: int = 5) -> str:
    """
    Get the token with surrounding context from the annotation text.
    Always tries to include characters before and after when available.
    
    Args:
        token: Token to find
        annotation_text: Full annotation text
        search_start: Position in text to start searching from (for finding specific occurrence)
        context_chars: Target number of characters to include before and after
        
    Returns:
        String with token and surrounding context
    """
    # Find the token starting from search_start (to handle multiple occurrences)
    token_pos = annotation_text.find(token, search_start)
    if token_pos == -1:
        # Fallback: try finding it from the beginning
        token_pos = annotation_text.find(token)
        if token_pos == -1:
            return token
    
    # Calculate available space on each side
    available_before = token_pos
    available_after = len(annotation_text) - (token_pos + len(token))
    
    # Try to get context_chars on each side, but use what's available
    # If we're near the start, get more after; if near the end, get more before
    if available_before < context_chars:
        # Near the start - get as much before as possible, more after
        start = 0
        end = min(len(annotation_text), token_pos + len(token) + context_chars + (context_chars - available_before))
    elif available_after < context_chars:
        # Near the end - get as much after as possible, more before
        start = max(0, token_pos - context_chars - (context_chars - available_after))
        end = len(annotation_text)
    else:
        # Enough space on both sides - get context_chars on each side
        start = token_pos - context_chars
        end = token_pos + len(token) + context_chars
    
    context = annotation_text[start:end]
    return context


def get_cleaned_context(token: str, annotation_text: str, cleaned_text: str, lang: str, search_start: int = 0, context_chars: int = 5) -> str:
    """
    Get the cleaned version of the token with surrounding context.
    
    Args:
        token: Original token
        annotation_text: Original annotation text
        cleaned_text: Cleaned annotation text
        lang: Language code
        search_start: Position in text to start searching from (for finding specific occurrence)
        context_chars: Number of characters to include before and after
        
    Returns:
        String with cleaned token parts and surrounding context
    """
    # Find the token position (same logic as get_token_context)
    token_pos = annotation_text.find(token, search_start)
    if token_pos == -1:
        token_pos = annotation_text.find(token)
        if token_pos == -1:
            return ""
    
    # Calculate available space on each side (same logic as get_token_context)
    available_before = token_pos
    available_after = len(annotation_text) - (token_pos + len(token))
    
    if available_before < context_chars:
        orig_start = 0
        orig_end = min(len(annotation_text), token_pos + len(token) + context_chars + (context_chars - available_before))
    elif available_after < context_chars:
        orig_start = max(0, token_pos - context_chars - (context_chars - available_after))
        orig_end = len(annotation_text)
    else:
        orig_start = token_pos - context_chars
        orig_end = token_pos + len(token) + context_chars
    
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
    
    # Analyze each token, tracking position to handle multiple occurrences
    search_start = 0
    for token in raw_tokens:
        analysis = analyze_token_dash_splitting(token, lang)
        if analysis:
            # Get context around the token (pass search_start to find this specific occurrence)
            original_context = get_token_context(token, annotation_text, search_start)
            cleaned_context = get_cleaned_context(token, annotation_text, cleaned_text, lang, search_start)
            
            # Update search_start to after this token for next iteration
            token_pos = annotation_text.find(token, search_start)
            if token_pos != -1:
                search_start = token_pos + len(token)
            else:
                # If not found, increment search_start to avoid infinite loop
                search_start += 1
            
            analysis['original'] = original_context
            analysis['cleaned'] = cleaned_context
            analysis['annotation_text_before'] = annotation_text
            analysis['annotation_text_after'] = cleaned_text
            # Remove the temporary 'token' key
            del analysis['token']
            results.append(analysis)
        else:
            # Even if no dash, update search_start for accurate positioning
            token_pos = annotation_text.find(token, search_start)
            if token_pos != -1:
                search_start = token_pos + len(token)
            else:
                search_start += 1
    
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
        
        # Extract tier annotations using non-spaced Cantonese tier
        cant_annotations, eng_annotations = extract_tiers_non_spaced(eaf, main_tier)
        
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


def is_important_case(case: dict) -> bool:
    """
    Check if a case is "important" - meaning it should be included when filtering.
    
    A case is important if:
    1. The sentence has content beyond just dashes
    2. The dash token is not only at the beginning or end of the sentence
    
    Args:
        case: Case dictionary with annotation_text_before and original fields
        
    Returns:
        True if the case is important, False otherwise
    """
    annotation_text = case.get('annotation_text_before', '')
    original_context = case.get('original', '')
    
    if not annotation_text or not original_context:
        return False
    
    # Check 1: Sentence should have content beyond just dashes
    # Remove all dashes and whitespace, check if anything remains
    text_without_dashes = DASH_RE.sub('', annotation_text)
    text_without_dashes = text_without_dashes.replace(' ', '').replace('\t', '').replace('\n', '')
    if not text_without_dashes or len(text_without_dashes.strip()) == 0:
        # Sentence is only dashes
        return False
    
    # Check 2: Dash should not be only at the beginning or end
    # Find dash characters in the original context
    dash_match = DASH_RE.search(original_context)
    if not dash_match:
        return False
    
    dash_start = dash_match.start()
    dash_end = dash_match.end()
    
    # Get content before and after the dash in the context
    before_dash = original_context[:dash_start].strip()
    after_dash = original_context[dash_end:].strip()
    
    # Remove dashes from before/after to see if there's real content
    before_no_dashes = DASH_RE.sub('', before_dash).strip()
    after_no_dashes = DASH_RE.sub('', after_dash).strip()
    
    # If there's no content on either side (after removing dashes), it's at an edge
    if not before_no_dashes and not after_no_dashes:
        return False
    
    # Also check if the annotation text starts or ends with only dashes
    annotation_stripped = annotation_text.strip()
    if annotation_stripped:
        # Check if the annotation, when stripped of leading/trailing dashes, still has the dash in middle
        # Find the dash position in the full annotation
        annotation_dash_match = DASH_RE.search(annotation_stripped)
        if annotation_dash_match:
            dash_pos = annotation_dash_match.start()
            # Check if there's non-dash content before and after in the full annotation
            before_in_annotation = annotation_stripped[:dash_pos].strip()
            after_in_annotation = annotation_stripped[annotation_dash_match.end():].strip()
            
            before_clean = DASH_RE.sub('', before_in_annotation).strip()
            after_clean = DASH_RE.sub('', after_in_annotation).strip()
            
            # If no content on either side in the full annotation, it's at edge
            if not before_clean and not after_clean:
                return False
    
    # If we got here, the sentence has content and the dash has content on at least one side
    return True


def export_to_csv(stats: dict, output_dir: Path, show_important_only: bool = False):
    """
    Export analysis results to CSV files.
    
    Args:
        stats: Statistics dictionary
        output_dir: Output directory path
        show_important_only: If True, filter out cases where dashes are only at beginning/end
                            or where sentence contains only dashes
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the columns we want to export (original and cleaned next to each other)
    fieldnames = ['original', 'cleaned', 'annotation_text_before', 'annotation_text_after']
    
    # Export all English cases
    if stats['english']['all_cases']:
        eng_cases = stats['english']['all_cases']
        if show_important_only:
            eng_cases = [case for case in eng_cases if is_important_case(case)]
            logging.info(f"Filtered to {len(eng_cases)} important English cases (from {len(stats['english']['all_cases'])} total)")
        
        eng_path = output_dir / "dash_splitting_english.csv"
        with open(eng_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in eng_cases:
                # Only write the fields we want
                row = {field: case.get(field, '') for field in fieldnames}
                writer.writerow(row)
        logging.info(f"Exported {len(eng_cases)} English cases to {eng_path}")
    
    # Export all Cantonese cases
    if stats['cantonese']['all_cases']:
        cant_cases = stats['cantonese']['all_cases']
        if show_important_only:
            cant_cases = [case for case in cant_cases if is_important_case(case)]
            logging.info(f"Filtered to {len(cant_cases)} important Cantonese cases (from {len(stats['cantonese']['all_cases'])} total)")
        
        cant_path = output_dir / "dash_splitting_cantonese.csv"
        with open(cant_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in cant_cases:
                # Only write the fields we want
                row = {field: case.get(field, '') for field in fieldnames}
                writer.writerow(row)
        logging.info(f"Exported {len(cant_cases)} Cantonese cases to {cant_path}")


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
    parser.add_argument(
        '--show-important',
        action='store_true',
        help='Filter out cases where dashes are only at beginning/end of sentence or sentence contains only dashes'
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
        export_to_csv(stats, output_dir, show_important_only=args.show_important)
        
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

