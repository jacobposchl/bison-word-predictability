"""
Functions for creating analysis dataset from translated code-switched sentences.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict
from src.analysis.pos_tagging import parse_pattern_segments
from src.analysis.matching_algorithm import analyze_window_matching

logger = logging.getLogger(__name__)


def extract_code_switched_segment(
    sentence: str,
    pattern: str,
    min_cantonese_words: int
) -> Optional[str]:
    """
    Extract code-switched segment from sentence if it matches criteria.
    
    Criteria:
    - Pattern must start with Cantonese (C) with at least min_cantonese_words
    - Immediately followed by English (E) segment
    - Returns: Cantonese segment + English segment
    
    Args:
        sentence: Space-separated sentence
        pattern: Pattern like "C5-E3-C2"
        min_cantonese_words: Minimum Cantonese words required at start
        
    Returns:
        Code-switched segment (Cantonese + English) or None if doesn't match
    """
    if not sentence or not pattern:
        return None
    
    # Parse pattern
    segments = parse_pattern_segments(pattern)
    
    if len(segments) < 2:
        return None
    
    # Check if first segment is Cantonese with enough words
    first_lang, first_count = segments[0]
    if first_lang != 'C' or first_count < min_cantonese_words:
        return None
    
    # Check if second segment is English
    second_lang, second_count = segments[1]
    if second_lang != 'E':
        return None
    
    # Extract words
    words = sentence.split()
    
    # Extract Cantonese segment (first segment)
    cantonese_end = first_count
    if cantonese_end > len(words):
        return None
    
    # Extract English segment (second segment)
    english_start = cantonese_end
    english_end = english_start + second_count
    if english_end > len(words):
        return None
    
    # Combine Cantonese + English segments
    cantonese_words = words[:cantonese_end]
    english_words = words[english_start:english_end]
    
    code_switched_segment = ' '.join(cantonese_words + english_words)
    
    return code_switched_segment


def create_analysis_dataset(config, translated_df: pd.DataFrame, monolingual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analysis dataset using window matching to find matched monolingual sentences.
    
    Filters sentences that start with at least x Cantonese words followed by
    English words, then finds the top-1 matched monolingual sentence using POS window matching.
    
    Args:
        config: Config object
        translated_df: DataFrame with translated code-switched sentences
        monolingual_df: DataFrame with monolingual Cantonese sentences
        
    Returns:
        DataFrame with columns:
        - translated_cs: Full Cantonese translation
        - matched_mono: Top-1 matched monolingual sentence
        - context: Placeholder for future use
        - surprisal: Placeholder for future use
        - switch_index: Index where code-switch occurs
    """
    logger.info("Creating analysis dataset with window matching...")
    
    # Get parameters from config
    min_cantonese = config.get_analysis_min_cantonese_words()
    window_size = config.get_analysis_window_size()
    similarity_threshold = config.get_analysis_similarity_threshold()
    
    logger.info(f"Filtering for sentences with at least {min_cantonese} Cantonese words at start, followed by English")
    logger.info(f"Using window size: {window_size}, similarity threshold: {similarity_threshold}")
    
    # Filter translated sentences to those matching criteria
    filtered_sentences = []
    for idx, row in translated_df.iterrows():
        pattern = row.get('pattern', '')
        switch_index = row.get('switch_index', -1)
        
        # Parse pattern
        segments = parse_pattern_segments(pattern)
        
        # Check criteria: starts with C >= min_cantonese, followed by E
        if len(segments) >= 2:
            first_lang, first_count = segments[0]
            second_lang, _ = segments[1]
            
            if first_lang == 'C' and first_count >= min_cantonese and second_lang == 'E' and switch_index >= 0:
                filtered_sentences.append(row.to_dict())
    
    logger.info(f"Found {len(filtered_sentences)} sentences matching criteria")
    
    if not filtered_sentences:
        logger.warning("No sentences matched the criteria!")
        return pd.DataFrame(columns=['translated_cs', 'matched_mono', 'context', 'surprisal', 'switch_index'])
    
    # Convert monolingual DataFrame to list of dicts
    monolingual_sentences = monolingual_df.to_dict('records')
    
    # Run window matching analysis
    logger.info("Running window matching analysis...")
    window_results = analyze_window_matching(
        translated_sentences=filtered_sentences,
        monolingual_sentences=monolingual_sentences,
        window_sizes=[window_size],  # Use single window size from config
        similarity_threshold=similarity_threshold,
        top_k=5  # Get top 5 matches to calculate proper statistics
    )
    
    # Extract results
    window_key = f'window_{window_size}'
    if window_key not in window_results:
        logger.error(f"No results for window size {window_size}")
        return pd.DataFrame()
    
    detailed_matches = window_results[window_key]['detailed_matches']
    
    # Build analysis dataset: one row per code-switched sentence with best match
    # Group matches by CS sentence to collect statistics
    sentence_data = {}
    
    for match in detailed_matches:
        cs_translation = match['cs_translation']
        
        if cs_translation not in sentence_data:
            # Initialize with CS sentence data and match counts
            sentence_data[cs_translation] = {
                'cs_sentence': match['cs_sentence'],
                'cs_translation': cs_translation,
                'cs_pattern': match['cs_pattern'],
                'cs_group': match['cs_group'],
                'cs_participant': match['cs_participant'],
                'cs_start_time': match['cs_start_time'],
                'switch_index': match['switch_index'],
                'pos_window': match['pos_window'],
                'best_match': None,
                'all_matches': []
            }
        
        # Collect all matches for statistics
        sentence_data[cs_translation]['all_matches'].append(match)
        
        # Store best match (rank 1)
        if match['rank'] == 1:
            sentence_data[cs_translation]['best_match'] = match
    
    # Build analysis rows with statistics
    analysis_rows = []
    sentences_without_matches = 0
    
    for cs_translation, data in sentence_data.items():
        best_match = data['best_match']
        all_matches = data['all_matches']
        
        # Only include sentences that have at least one match
        if best_match:
            analysis_rows.append({
                # Code-switched sentence info
                'cs_sentence': data['cs_sentence'],
                'cs_translation': cs_translation,
                'cs_pattern': data['cs_pattern'],
                'cs_group': data['cs_group'],
                'cs_participant': data['cs_participant'],
                'cs_start_time': data['cs_start_time'],
                'switch_index': data['switch_index'],
                'pos_window': data['pos_window'],
                
                # Best matched monolingual sentence info
                'matched_mono': best_match['matched_sentence'],
                'matched_group': best_match['matched_group'],
                'matched_participant': best_match['matched_participant'],
                'matched_start_time': best_match['matched_start_time'],
                'matched_pos': best_match['matched_pos'],
                'matched_switch_index': best_match['matched_switch_index'],  # Center of matched POS window
                'similarity': best_match['similarity'],
                
                # Match statistics (from ALL matches, not just those in all_matches list)
                'total_matches_above_threshold': best_match.get('total_matches_above_threshold', len(all_matches)),
                'matches_same_group': best_match.get('all_matches_same_group', sum(1 for m in all_matches if m['same_group'])),
                'matches_same_speaker': best_match.get('all_matches_same_speaker', sum(1 for m in all_matches if m['same_speaker'])),
                'selected_match_time_distance_sec': best_match['time_distance'] / 1000.0,  # Convert ms to seconds
                'same_group': best_match['same_group'],
                'same_speaker': best_match['same_speaker']
            })
    
    # Count sentences without matches
    for sent in filtered_sentences:
        cs_translation = sent.get('cantonese_translation', '')
        if cs_translation not in sentence_data:
            sentences_without_matches += 1
    
    logger.info(f"Created analysis dataset with {len(analysis_rows)} rows (sentences with matches)")
    logger.info(f"  - Sentences with matches: {len(analysis_rows)}")
    logger.info(f"  - Sentences without matches (excluded): {sentences_without_matches}")
    
    # Create DataFrame
    analysis_df = pd.DataFrame(analysis_rows)
    
    return analysis_df

