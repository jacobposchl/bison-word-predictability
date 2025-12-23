"""
Data export functionality for code-switching analysis.

This module handles exporting processed data to CSV files with filtering
for code-switching sentences.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


def filter_code_switching_sentences(
    sentences: List[Dict],
    include_fillers: bool = True
) -> List[Dict]:
    """
    Filter sentences to only keep those with actual code-switching.
    
    Code-switching is defined as sentences containing both 'C' and 'E'
    in their pattern.
    
    Args:
        sentences: List of sentence data dictionaries
        include_fillers: If True, use pattern_with_fillers; else use pattern_content_only
        
    Returns:
        Filtered list of sentences with code-switching
    """
    filtered = []
    
    for sentence in sentences:
        if include_fillers:
            pattern = sentence.get('pattern_with_fillers', '')
        else:
            pattern = sentence.get('pattern_content_only', '')
        
        # Skip sentences that became monolingual or filler-only after removing fillers
        if 'C' in pattern and 'E' in pattern and pattern != 'FILLER_ONLY':
            filtered.append(sentence)
    
    return filtered


def export_to_csv(
    all_sentences: List[Dict],
    csv_with_fillers_path: str,
    csv_without_fillers_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export processed sentences to CSV files.
    
    Creates two datasets:
    1. WITH fillers: Includes filler words in pattern analysis
    2. WITHOUT fillers: Excludes filler words from pattern analysis
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        csv_with_fillers_path: Output path for CSV with fillers
        csv_without_fillers_path: Output path for CSV without fillers
        
    Returns:
        Tuple of (dataframe_with_fillers, dataframe_without_fillers)
    """
    # type: ignore
    # Filter both datasets to only keep sentences with actual code-switching
    with_fillers = filter_code_switching_sentences(all_sentences, include_fillers=True)
    without_fillers = filter_code_switching_sentences(all_sentences, include_fillers=False)
    
    logger.info(f"Dataset WITH fillers: {len(with_fillers)} code-switching sentences")
    logger.info(f"Dataset WITHOUT fillers: {len(without_fillers)} code-switching sentences")
    logger.info(
        f"Difference: {len(with_fillers) - len(without_fillers)} sentences "
        f"reclassified as non-code-switching"
    )
    
    # Create the first CSV - WITH fillers
    csv_with_fillers = pd.DataFrame({
        'reconstructed_sentence': [s['reconstructed_text'] for s in with_fillers],
        'sentence_original': [s['text'] for s in with_fillers],
        'pattern': [s['pattern_with_fillers'] for s in with_fillers],
        'matrix_language': [s['matrix_language'] for s in with_fillers],
        'group_code': [s['group_code'] for s in with_fillers],
        'group': [s['group'] for s in with_fillers],
        'participant_id': [s['participant_id'] for s in with_fillers],
        'filler_count': [s['filler_count'] for s in with_fillers],
        'has_fillers': [s['has_fillers'] for s in with_fillers]
    })
    
    # Create the second CSV - WITHOUT fillers
    csv_without_fillers = pd.DataFrame({
        'reconstructed_sentence': [s['reconstructed_text'] for s in without_fillers],
        'sentence_original': [s['text'] for s in without_fillers],
        'pattern': [s['pattern_content_only'] for s in without_fillers],
        'matrix_language': [s['matrix_language'] for s in without_fillers],
        'group_code': [s['group_code'] for s in without_fillers],
        'group': [s['group'] for s in without_fillers],
        'participant_id': [s['participant_id'] for s in without_fillers],
        'filler_count': [s['filler_count'] for s in without_fillers],
        'has_fillers': [s['has_fillers'] for s in without_fillers]
    })
    
    # Create output directory if it doesn't exist
    csv_dir = os.path.dirname(csv_with_fillers_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
        logger.info(f"Created output directory: {csv_dir}")
    
    # Save both datasets
    csv_with_fillers.to_csv(csv_with_fillers_path, index=False, encoding='utf-8-sig')
    csv_without_fillers.to_csv(csv_without_fillers_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved two datasets:")
    logger.info(f"  '{csv_with_fillers_path}' - {len(csv_with_fillers)} sentences")
    logger.info(f"  '{csv_without_fillers_path}' - {len(csv_without_fillers)} sentences")
    
    return csv_with_fillers, csv_without_fillers


def export_all_sentences_to_csv(
    all_sentences: List[Dict],
    csv_all_sentences_path: str
) -> pd.DataFrame:
    """
    Export ALL sentences (both monolingual and code-switched) to CSV.
    
    This includes sentences that were filtered out in the code-switching analysis.
    Useful for exploratory analysis that needs monolingual sentences.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        csv_all_sentences_path: Output path for CSV with all sentences
        
    Returns:
        DataFrame with all sentences
    """
    logger.info(f"Exporting ALL sentences (monolingual + code-switched) to CSV...")
    
    # Create DataFrame with all sentences
    # Use pattern_with_fillers for consistency
    csv_all = pd.DataFrame({
        'reconstructed_sentence': [s['reconstructed_text'] for s in all_sentences],
        'sentence_original': [s['text'] for s in all_sentences],
        'pattern': [s.get('pattern_with_fillers', s.get('pattern', '')) for s in all_sentences],
        'matrix_language': [s.get('matrix_language', 'Unknown') for s in all_sentences],
        'group_code': [s.get('group_code', '') for s in all_sentences],
        'group': [s.get('group', '') for s in all_sentences],
        'participant_id': [s.get('participant_id', '') for s in all_sentences],
        'filler_count': [s.get('filler_count', 0) for s in all_sentences],
        'has_fillers': [s.get('has_fillers', False) for s in all_sentences]
    })
    
    # Create output directory if it doesn't exist
    csv_dir = os.path.dirname(csv_all_sentences_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
        logger.info(f"Created output directory: {csv_dir}")
    
    # Save CSV
    csv_all.to_csv(csv_all_sentences_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved all sentences dataset:")
    logger.info(f"  '{csv_all_sentences_path}' - {len(csv_all)} sentences")
    
    return csv_all


def save_exploratory_outputs(
    output_dir: Path,
    monolingual: dict,
    pos_results: dict,
    matching_results: dict,
    distributions: dict,
    report: str,
    figures_dir: Optional[Path] = None
) -> None:
    """
    Save all exploratory analysis output files.
    
    Args:
        output_dir: Directory for CSV and report files
        monolingual: Monolingual sentence data
        pos_results: POS tagging results
        matching_results: Matching algorithm results
        distributions: Distribution analysis results
        report: Feasibility report text
        figures_dir: Directory for figures (if None, uses output_dir/figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if figures_dir is None:
        figures_dir = output_dir / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving outputs to {output_dir}...")
    
    # Save monolingual sentences
    if 'cantonese' in monolingual:
        monolingual_path = output_dir / "monolingual_sentences.csv"
        # Combine all monolingual sentences
        all_mono = pd.concat([
            monolingual['cantonese'],
            monolingual['english']
        ], ignore_index=True)
        all_mono.to_csv(monolingual_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved monolingual sentences to {monolingual_path}")
    
    # Save POS tagging sample
    if 'sample_results' in pos_results:
        pos_path = output_dir / "pos_tagged_sample.csv"
        pos_results['sample_results'].to_csv(pos_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved POS tagging sample to {pos_path}")
    
    # Save matching results sample
    if 'results' in matching_results:
        match_path = output_dir / "matching_results_sample.csv"
        # Flatten the results for CSV export
        export_results = []
        for _, row in matching_results['results'].iterrows():
            export_row = {
                'sentence': row['sentence'],
                'pattern': row['pattern'],
                'num_matches': row['num_matches'],
                'has_match': row['has_match'],
                'best_similarity': row['best_similarity'],
                'has_c_to_e': row['has_c_to_e'],
                'has_e_to_c': row['has_e_to_c']
            }
            # Add details of top matches if available
            if row['matches_detail']:
                for i, match in enumerate(row['matches_detail'][:3]):
                    export_row[f'match_{i+1}_similarity'] = match.get('similarity', 0)
                    export_row[f'match_{i+1}_language'] = match.get('language', '')
            export_results.append(export_row)
        
        match_df = pd.DataFrame(export_results)
        match_df.to_csv(match_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved matching results to {match_path}")
    
    # Save feasibility report
    report_path = output_dir / "feasibility_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved feasibility report to {report_path}")
    
    logger.info("All outputs saved successfully!")
