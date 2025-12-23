"""
Data export functionality for code-switching analysis.

This module handles exporting processed data to CSV files with filtering
for code-switching sentences.
"""

import pandas as pd
from typing import List, Dict, Tuple
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
