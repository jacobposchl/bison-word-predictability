"""
Functions for creating analysis dataset from translated code-switched sentences.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional
from src.analysis.pos_tagging import parse_pattern_segments

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


def create_analysis_dataset(config) -> pd.DataFrame:
    """
    Create analysis dataset from cantonese_translated_WITHOUT_fillers.csv.
    
    Filters sentences that start with at least x Cantonese words followed by
    English words, and extracts the code-switched segment.
    
    Args:
        config: Config object
        
    Returns:
        DataFrame with columns:
        - original_cs: Code-switched segment (Cantonese start + English)
        - translated_cs: Corresponding translation
        - matched_mono: Blank column
        - context: Blank column
        - switch_index: Blank column
    """
    logger.info("Creating analysis dataset...")
    
    # Load translated sentences
    csv_path = config.get_csv_cantonese_translated_path()
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    
    # Get minimum Cantonese words from config
    min_cantonese = config.get_analysis_min_cantonese_words()
    logger.info(f"Filtering for sentences with at least {min_cantonese} Cantonese words at start, followed by English")
    
    # Filter and extract code-switched segments
    analysis_rows = []
    
    for idx, row in df.iterrows():
        sentence = row.get('code_switch_original', '')
        pattern = row.get('pattern', '')
        translation = row.get('cantonese_translation', '')
        
        # Extract code-switched segment
        cs_segment = extract_code_switched_segment(
            str(sentence),
            str(pattern),
            min_cantonese
        )
        
        if cs_segment:
            # Use the full translation for now
            # (Can be improved later to extract just the corresponding segment)
            translated_segment = str(translation) if pd.notna(translation) and translation else ''
            
            analysis_rows.append({
                'original_cs': cs_segment,
                'translated_cs': translated_segment,
                'matched_mono': '',
                'context': '',
                'switch_index': ''
            })
    
    logger.info(f"Found {len(analysis_rows)} sentences matching criteria")
    
    # Create DataFrame
    analysis_df = pd.DataFrame(analysis_rows)
    
    return analysis_df

