"""
Data helper utilities for code-switching analysis.

This module provides functions for data processing, filtering, and sorting.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional


def parse_pattern_segments(pattern: str) -> List[Tuple[str, int]]:
    """
    Parse pattern string into language segments.
    
    Args:
        pattern: Pattern like "C5-E3-C2"
        
    Returns:
        List of (language, count) tuples
    """
    if not pattern:
        return []
    
    matches = re.findall(r'([CE])(\d+)', pattern)
    return [(lang, int(count)) for lang, count in matches]


def is_monolingual(pattern: str) -> Optional[str]:
    """
    Determine if a pattern represents a monolingual sentence.
    
    Args:
        pattern: Pattern string like "C10", "E8", or "C5-E3"
        
    Returns:
        'Cantonese' if pure Cantonese, 'English' if pure English,
        None if code-switched
    """
    segments = parse_pattern_segments(pattern)
    languages = {lang for lang, _ in segments}
    
    if len(languages) == 0:
        return None
    elif len(languages) == 1:
        lang = languages.pop()
        return 'Cantonese' if lang == 'C' else 'English'
    else:
        return None


def split_sentence_by_pattern(sentence: str, pattern: str) -> List[Tuple[str, str]]:
    """
    Split a mixed-language sentence according to its pattern.
    
    Args:
        sentence: Space-separated sentence text
        pattern: Pattern like "C5-E3-C2"
        
    Returns:
        List of (text_segment, language) tuples
    """
    words = sentence.split()
    segments = parse_pattern_segments(pattern)
    
    if not segments:
        return []
    
    result = []
    word_idx = 0
    
    for lang, count in segments:
        if word_idx >= len(words):
            break
        
        # Extract the segment
        segment_words = words[word_idx:word_idx + count]
        segment_text = ' '.join(segment_words)
        result.append((segment_text, lang))
        word_idx += count
    
    return result


def count_words_from_pattern(pattern: str) -> int:
    """
    Count total words from a pattern string.
    
    Args:
        pattern: Pattern like "C3-E2-C1" or "C5"
        
    Returns:
        Total word count (sum of all numbers in pattern)
    """
    if not pattern or pattern == 'FILLER_ONLY':
        return 0
    
    # Extract all numbers from pattern (e.g., "C3-E2-C1" -> [3, 2, 1])
    numbers = re.findall(r'\d+', pattern)
    return sum(int(n) for n in numbers)


def count_words_from_text(text: str) -> int:
    """
    Count words in a text string by splitting on whitespace.
    
    Args:
        text: Text string
        
    Returns:
        Number of words
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def filter_by_min_words(sentences: List[Dict], min_words: int) -> List[Dict]:
    """
    Filter sentences to only keep those with at least min_words.
    
    Counts words from the pattern field.
    
    Args:
        sentences: List of sentence dictionaries
        min_words: Minimum number of words required
        
    Returns:
        Filtered list of sentences
    """
    filtered = []
    for s in sentences:
        pattern = s.get('pattern', '')
        word_count = count_words_from_pattern(pattern)
        
        if word_count >= min_words:
            filtered.append(s)
    
    return filtered


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame by group, participant_id, then start_time.
    
    Args:
        df: DataFrame to sort
        
    Returns:
        Sorted DataFrame
    """
    if len(df) == 0:
        return df
    
    return df.sort_values(by=['group', 'participant_id', 'start_time'], na_position='last').reset_index(drop=True)


def find_switch_points(pattern: str) -> List[int]:
    """
    Find positions where language switches occur in a pattern.
    
    Args:
        pattern: Pattern string like "C5-E3-C2"
        
    Returns:
        List of word indices where switches occur
        
    Example:
        >>> find_switch_points("C5-E3-C2")
        [5, 8]  # Switch at word 5 (C->E) and word 8 (E->C)
    """
    segments = parse_pattern_segments(pattern)
    switch_points = []
    word_idx = 0
    
    for i in range(len(segments) - 1):
        _, count = segments[i]
        word_idx += count
        switch_points.append(word_idx)
    
    return switch_points
