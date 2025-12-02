"""
Matching algorithm for code-switching analysis.

This module implements the Levenshtein similarity-based matching algorithm
for finding similar monolingual sentences to code-switched sentences,
following the methodology of Calvillo et al. (2020).
"""

import logging
from typing import List, Tuple, Dict, Optional
from Levenshtein import distance as levenshtein_distance
from functools import lru_cache
from tqdm import tqdm

from .pos_tagging import (
    pos_tag_cantonese,
    pos_tag_english,
    pos_tag_mixed_sentence,
    extract_pos_sequence,
    parse_pattern_segments
)

logger = logging.getLogger(__name__)

# Global cache for monolingual POS sequences
_monolingual_pos_cache = {}


def _sequence_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate edit distance between two sequences (not strings).
    
    Uses dynamic programming to compute Levenshtein distance on sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Edit distance (minimum number of insertions, deletions, substitutions)
    """
    m, n = len(seq1), len(seq2)
    
    # Create a matrix to store distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                # Characters match, no cost
                dp[i][j] = dp[i-1][j-1]
            else:
                # Take minimum of insert, delete, or substitute
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[m][n]


def levenshtein_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    Calculate normalized Levenshtein similarity between two sequences.
    
    Similarity is calculated as:
        similarity = 1 - (edit_distance / max_length)
    
    This uses proper sequence-level edit distance, not string-level.
    
    Args:
        seq1: First sequence (list of strings)
        seq2: Second sequence (list of strings)
        
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = completely different)
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    # Calculate edit distance on sequences (not strings)
    edit_dist = _sequence_edit_distance(seq1, seq2)
    
    # Normalize by maximum length
    max_len = max(len(seq1), len(seq2))
    similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    return max(0.0, similarity)  # Ensure non-negative


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


def extract_pos_window(
    pos_sequence: List[str],
    switch_index: int,
    window_size: int = 3
) -> List[str]:
    """
    Extract POS tags around a switch point.
    
    Args:
        pos_sequence: Full POS tag sequence
        switch_index: Word index where switch occurs
        window_size: Number of words before and after switch to include
        
    Returns:
        List of POS tags in the window
    """
    start = max(0, switch_index - window_size)
    end = min(len(pos_sequence), switch_index + window_size + 1)
    
    return pos_sequence[start:end]


def _get_monolingual_pos_sequence(mono_sent: Dict, lang_code: str) -> Optional[List[str]]:
    """
    Get POS sequence for a monolingual sentence, using cache.
    
    Args:
        mono_sent: Monolingual sentence dictionary
        lang_code: Language code ('C' or 'E')
        
    Returns:
        POS sequence or None if tagging fails
    """
    # Use sentence text as cache key
    sentence = mono_sent.get('reconstructed_sentence', '')
    cache_key = (sentence, lang_code)
    
    if cache_key in _monolingual_pos_cache:
        return _monolingual_pos_cache[cache_key]
    
    try:
        if lang_code == 'C':
            mono_tagged = pos_tag_cantonese(sentence)
        else:
            mono_tagged = pos_tag_english(sentence)
        
        mono_pos_seq = extract_pos_sequence(mono_tagged)
        _monolingual_pos_cache[cache_key] = mono_pos_seq
        return mono_pos_seq
    except Exception as e:
        logger.debug(f"Error tagging monolingual sentence: {e}")
        _monolingual_pos_cache[cache_key] = None
        return None


def precompute_monolingual_pos_sequences(
    monolingual_sentences: Dict[str, List[Dict]],
    max_per_language: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Pre-compute and cache POS sequences for monolingual sentences.
    
    This significantly speeds up matching by avoiding repeated POS tagging.
    
    Args:
        monolingual_sentences: Dict with 'cantonese' and 'english' keys
        max_per_language: Optional limit on number of sentences per language to process
        
    Returns:
        Filtered monolingual_sentences dict (if max_per_language was used)
    """
    logger.info("Pre-computing POS sequences for monolingual sentences...")
    
    total = 0
    filtered = {}
    
    for lang_key, lang_code in [('cantonese', 'C'), ('english', 'E')]:
        if lang_key in monolingual_sentences:
            mono_list = monolingual_sentences[lang_key]
            
            # Optionally limit the number of sentences
            if max_per_language and len(mono_list) > max_per_language:
                logger.info(f"  Limiting {lang_key} to {max_per_language} sentences (from {len(mono_list)})")
                mono_list = mono_list[:max_per_language]
                filtered[lang_key] = mono_list
            else:
                filtered[lang_key] = mono_list
            
            # Pre-compute POS sequences with progress bar
            for mono_sent in tqdm(filtered[lang_key], desc=f"Tagging {lang_key}", leave=False):
                _get_monolingual_pos_sequence(mono_sent, lang_code)
                total += 1
    
    logger.info(f"Cached POS sequences for {total} monolingual sentences")
    return filtered


def find_matches(
    code_switched_sentence: Dict,
    monolingual_sentences: Dict[str, List[Dict]],
    similarity_threshold: float = 0.4,
    window_size: int = 3,
    max_matches_per_switch: int = 10
) -> List[Dict]:
    """
    Find matching monolingual sentences for a code-switched sentence.
    
    For each switch point in the code-switched sentence:
    1. Extract 3-word POS window around switch
    2. Calculate similarity to all monolingual sentences in the same language
    3. Keep matches above threshold
    
    Args:
        code_switched_sentence: Dictionary with 'reconstructed_sentence', 'pattern'
        monolingual_sentences: Dict with 'cantonese' and 'english' keys, each
                              containing list of sentence dicts
        similarity_threshold: Minimum similarity score (default 0.4 = 40%)
        window_size: Number of words around switch point (default 3)
        
    Returns:
        List of match dictionaries with:
        - 'match_sentence': The matched monolingual sentence
        - 'similarity': Similarity score
        - 'switch_point': Which switch point this match is for
        - 'language': Language of the matched segment
    """
    sentence = code_switched_sentence.get('reconstructed_sentence', '')
    pattern = code_switched_sentence.get('pattern', '')
    
    if not sentence or not pattern:
        return []
    
    # Tag the code-switched sentence
    try:
        tagged = pos_tag_mixed_sentence(sentence, pattern)
        pos_seq = extract_pos_sequence(tagged)
    except Exception as e:
        logger.warning(f"Error tagging code-switched sentence: {e}")
        return []
    
    if not pos_seq:
        return []
    
    # Find switch points
    switch_points = find_switch_points(pattern)
    if not switch_points:
        return []
    
    # Parse pattern to get language at each segment
    segments = parse_pattern_segments(pattern)
    matches = []
    
    for switch_idx, switch_point in enumerate(switch_points):
        # Determine which language we're matching for
        # We want to match the segment AFTER the switch
        if switch_idx < len(segments) - 1:
            target_lang_code = segments[switch_idx + 1][0]  # Language after switch
            target_lang = 'cantonese' if target_lang_code == 'C' else 'english'
        else:
            continue
        
        # Extract POS window around switch point
        pos_window = extract_pos_window(pos_seq, switch_point, window_size)
        
        if not pos_window:
            continue
        
        # Get monolingual sentences in target language
        monolingual_list = monolingual_sentences.get(target_lang, [])
        
        if not monolingual_list:
            continue
        
        # Collect matches for this switch point
        switch_matches = []
        
        # Calculate similarity to each monolingual sentence
        for mono_sent in monolingual_list:
            # Get cached POS sequence (or compute if not cached)
            mono_pos_seq = _get_monolingual_pos_sequence(mono_sent, target_lang_code)
            
            if not mono_pos_seq:
                continue
            
            # Calculate similarity
            # Compare window to all possible windows in monolingual sentence
            best_similarity = 0.0
            best_window = []
            
            # Try all possible windows of same size in monolingual sentence
            window_len = len(pos_window)
            for i in range(len(mono_pos_seq) - window_len + 1):
                mono_window = mono_pos_seq[i:i + window_len]
                similarity = levenshtein_similarity(pos_window, mono_window)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_window = mono_window
                    
                    # Early termination: if we find a perfect match, stop searching
                    if similarity >= 1.0:
                        break
            
            # If no exact window size match, compare to full sequence
            if len(mono_pos_seq) < window_len:
                similarity = levenshtein_similarity(pos_window, mono_pos_seq)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_window = mono_pos_seq
            
            # Keep if above threshold
            if best_similarity >= similarity_threshold:
                switch_matches.append({
                    'match_sentence': mono_sent,
                    'similarity': best_similarity,
                    'switch_point': switch_point,
                    'switch_index': switch_idx,
                    'language': target_lang,
                    'pos_window': ' '.join(pos_window),
                    'matched_window': ' '.join(best_window)
                })
        
        # Sort matches for this switch by similarity and keep only top N
        switch_matches.sort(key=lambda x: x['similarity'], reverse=True)
        matches.extend(switch_matches[:max_matches_per_switch])
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return matches

