"""
Matching algorithm for code-switching analysis.

This module implements the Levenshtein similarity-based matching algorithm
for finding similar monolingual sentences to code-switched sentences,
following the methodology of Calvillo et al. (2020).
"""

import logging
import multiprocessing
import os
from pathlib import Path
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


def build_monolingual_pos_cache(monolingual_sentences: List[Dict]) -> Dict[int, List[str]]:
    """
    Pre-compute POS sequences for all monolingual sentences.
    
    This avoids repeated string splitting during matching, providing a significant
    performance improvement when processing many sentences.
    
    Args:
        monolingual_sentences: List of monolingual sentence dictionaries
        
    Returns:
        Dictionary mapping sentence index to pre-computed POS sequence (list of strings)
    """

    cache = {}
    for idx, mono_sent in enumerate(monolingual_sentences):
        mono_pos_str = mono_sent.get('pos', '')
        if mono_pos_str:
            cache[idx] = mono_pos_str.split()
        else:
            cache[idx] = []
    return cache


def _sequence_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate edit distance between two sequences.
    

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


def rank_matches_by_context( matches: List[Dict], source_sentence: Dict ) -> List[Dict]:
    """
    Rank matches by contextual relevance: same speaker > same group > time proximity (only for same speaker).
    
    Args:
        matches: List of match dictionaries from find_window_matches
        source_sentence: The code-switched sentence being matched
        
    Returns:
        Sorted list of matches
    """

    source_group = source_sentence.get('group', '')
    source_speaker = source_sentence.get('participant_id', '')
    source_time = source_sentence.get('start_time', 0.0)
    
    def sort_key(match: Dict) -> Tuple[int, int, float]:
        """
        Create composite sort key for match ranking.
        
        Returns tuple of (same_speaker_priority, same_group_priority, time_distance)
        where lower values are better (sorted ascending).
        Time distance is only meaningful for same-speaker matches.
        """
        
        match_sent = match.get('match_sentence', {})
        match_group = match_sent.get('group', '')
        match_speaker = match_sent.get('participant_id', '')
        match_time = match_sent.get('start_time', 0.0)
        
        # Priority 1: Same speaker (0 if same, 1 if different)
        same_speaker_priority = 0 if match_speaker == source_speaker else 1
        
        # Priority 2: Same group (0 if same, 1 if different)
        same_group_priority = 0 if match_group == source_group else 1
        
        # Priority 3: Time proximity (only meaningful for same speaker)
        if match_speaker == source_speaker:
            # Use actual time distance for same speaker
            time_distance = abs(match_time - source_time)
        else:
            # Use large constant for different speakers (time proximity doesn't matter)
            time_distance = float('inf')
        
        return (same_speaker_priority, same_group_priority, time_distance)
    
    # Sort by the composite key
    ranked_matches = sorted(matches, key=sort_key)
    
    return ranked_matches


def find_window_matches(
    code_switched_sentence: Dict,
    monolingual_sentences: List[Dict],
    window_size: int = 1,
    similarity_threshold: float = 0.4,
    mono_pos_cache: Optional[Dict[int, List[str]]] = None
) -> List[Dict]:
    """
    Find monolingual sentences matching POS window around switch point.
    
    This function:
    1. Identifies the switch point in the code-switched sentence
    2. Extracts a window of n words before and after the switch
    3. Finds all monolingual sentences with similar POS sequences
    4. Returns matches with similarity >= threshold
    
    Args:
        code_switched_sentence: Dict with 'cantonese_translation', 'translated_pos',
                               'switch_index', 'pattern', 'group', 'participant_id', etc.
        monolingual_sentences: List of Cantonese monolingual sentence dicts
        window_size: Number of words before/after switch point (n)
        similarity_threshold: Minimum Levenshtein similarity (0-1)
        mono_pos_cache: Optional pre-computed cache mapping sentence index to POS sequence.
                       If provided, avoids repeated string splitting for better performance.
        
    Returns:
        List of match dictionaries with:
        - 'match_sentence': The matched monolingual sentence
        - 'similarity': Similarity score (0-1)
        - 'window_size': The window size used
        - 'pos_window': POS sequence from code-switched sentence
        - 'matched_pos': POS sequence from monolingual sentence
        - 'matched_window_start': Starting index of match in monolingual sentence
    """
    # Extract switch point information
    switch_index = code_switched_sentence.get('switch_index', -1)
    translated_pos = code_switched_sentence.get('translated_pos', '')
    pattern = code_switched_sentence.get('pattern', '')
    
    if switch_index < 0 or not translated_pos:
        return []
    
    # Parse POS sequence
    pos_sequence = translated_pos.split()
    
    if not pos_sequence:
        return []
    
    # Validate switch_index is within bounds
    # This should never happen if translation preserves Cantonese segments correctly.
    # If it does, it indicates a data quality issue (translation or POS tagging problem).
    if switch_index >= len(pos_sequence):
        logger.error(
            f"switch_index ({switch_index}) is out of bounds for POS sequence length ({len(pos_sequence)}) "
            f"for pattern '{pattern}'. This indicates a data quality issue - the translation or POS tagging "
            f"may have failed. The first {switch_index} Cantonese words should be preserved exactly. "
            f"Skipping this sentence."
        )
        return []
    
    # Extract POS window around switch point
    window_start = max(0, switch_index - window_size)
    window_end = min(len(pos_sequence), switch_index + window_size + 1)
    pos_window = pos_sequence[window_start:window_end]
    
    if not pos_window:
        return []
    
    # Calculate the position of switch_index within the window
    # This allows us to directly map the switch position to matched sentences
    switch_index_in_window = switch_index - window_start
    
    matches = []
    window_len = len(pos_window)
    
    # Search through all monolingual sentences
    for idx, mono_sent in enumerate(monolingual_sentences):
        # Use cache if available, otherwise parse on the fly
        if mono_pos_cache is not None:
            mono_pos_seq = mono_pos_cache.get(idx, [])
        else:
            mono_pos_str = mono_sent.get('pos', '')
            if not mono_pos_str:
                continue
            mono_pos_seq = mono_pos_str.split()
        
        if not mono_pos_seq:
            continue
        
        # Try all possible windows in the monolingual sentence
        best_similarity = 0.0
        best_window = []
        best_start_idx = -1
        
        # Sliding window
        for i in range(len(mono_pos_seq) - window_len + 1):
            mono_window = mono_pos_seq[i:i + window_len]
            similarity = levenshtein_similarity(pos_window, mono_window)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_window = mono_window
                best_start_idx = i

        # Also try comparing to full sequence if monolingual is shorter than window
        if len(mono_pos_seq) < window_len:
            similarity = levenshtein_similarity(pos_window, mono_pos_seq)
            if similarity > best_similarity:
                best_similarity = similarity
                best_window = mono_pos_seq
                best_start_idx = 0
        
        # Keep if above threshold AND we found a valid window position
        # best_start_idx will be -1 if no window was ever found (all similarities were 0.0)
        if best_similarity >= similarity_threshold and best_start_idx >= 0:
            # Direct mapping: switch_index is at position switch_index_in_window within the CS window
            # When we find a match starting at best_start_idx, the equivalent switch_index
            # in the matched sentence is at best_start_idx + switch_index_in_window
            matched_switch_index = best_start_idx + switch_index_in_window
            # Ensure index is within bounds of monolingual sentence
            matched_switch_index = min(matched_switch_index, len(mono_pos_seq) - 1)
            
            # Extract POS tags at switch positions
            cs_switch_pos = pos_sequence[switch_index] if switch_index < len(pos_sequence) else 'UNKNOWN'
            mono_switch_pos = mono_pos_seq[matched_switch_index] if matched_switch_index < len(mono_pos_seq) else 'UNKNOWN'
            
            matches.append({
                'match_sentence': mono_sent,
                'similarity': best_similarity,
                'window_size': window_size,
                'pos_window': ' '.join(pos_window),
                'matched_pos': ' '.join(best_window),
                'matched_window_start': best_start_idx,
                'matched_switch_index': matched_switch_index,
                'cs_switch_pos': cs_switch_pos,
                'mono_switch_pos': mono_switch_pos
            })
    
    return matches


def _process_single_cs_sentence( args: Tuple[Dict, List[Dict], int, float, Dict[int, List[str]], int] ) -> Tuple[Dict, List[Dict], int, List[float]]:
    """
    Worker function to process a single code-switched sentence.
    
    This function is designed to be used with multiprocessing.Pool.
    
    Args:
        args: Tuple containing:
            - cs_sent: Code-switched sentence dictionary
            - monolingual_sentences: List of monolingual sentence dictionaries
            - window_size: Window size for matching
            - similarity_threshold: Similarity threshold
            - mono_pos_cache: Pre-computed POS cache
            - top_k: Number of top matches to keep
            
    Returns:
        Tuple of (cs_sent, detailed_matches, has_matches, similarity_scores) where:
        - cs_sent: The original code-switched sentence
        - detailed_matches: List of detailed match dictionaries (top_k matches)
        - has_matches: 1 if matches found, 0 otherwise
        - similarity_scores: List of similarity scores from ALL matches (not just top_k)
    """
    cs_sent, monolingual_sentences, window_size, similarity_threshold, mono_pos_cache, top_k = args
    
    # Find all matches for this sentence
    matches = find_window_matches(
        cs_sent,
        monolingual_sentences,
        window_size=window_size,
        similarity_threshold=similarity_threshold,
        mono_pos_cache=mono_pos_cache
    )
    
    detailed_matches = []
    has_matches = 0
    similarity_scores = []
    
    if matches:
        has_matches = 1
        
        # Collect similarity scores from ALL matches (before truncation)
        similarity_scores = [m['similarity'] for m in matches]
        
        # Calculate stats from ALL matches (before truncation)
        total_matches_count = len(matches)
        all_matches_same_group = sum(1 for m in matches 
                                    if m['match_sentence'].get('group', '') == cs_sent.get('group', ''))
        all_matches_same_speaker = sum(1 for m in matches 
                                      if m['match_sentence'].get('participant_id', '') == cs_sent.get('participant_id', ''))
        
        # Rank matches by context
        ranked_matches = rank_matches_by_context(matches, cs_sent)
        
        # Keep only top-k matches for storage
        top_matches = ranked_matches[:top_k]
        
        # Store detailed results
        for rank, match in enumerate(top_matches, 1):
            detailed_match = {
                'cs_sentence': cs_sent.get('code_switch_original', ''),
                'cs_translation': cs_sent.get('cantonese_translation', ''),
                'cs_pattern': cs_sent.get('pattern', ''),
                'cs_group': cs_sent.get('group', ''),
                'cs_participant': cs_sent.get('participant_id', ''),
                'cs_start_time': cs_sent.get('start_time', 0.0),
                'switch_index': cs_sent.get('switch_index', -1),
                'rank': rank,
                'similarity': match['similarity'],
                'pos_window': match['pos_window'],
                'matched_sentence': match['match_sentence'].get('reconstructed_sentence', ''),
                'matched_pos': match['matched_pos'],
                'matched_window_start': match['matched_window_start'],
                'matched_switch_index': match['matched_switch_index'],
                'cs_switch_pos': match['cs_switch_pos'],
                'mono_switch_pos': match['mono_switch_pos'],
                'matched_group': match['match_sentence'].get('group', ''),
                'matched_participant': match['match_sentence'].get('participant_id', ''),
                'matched_start_time': match['match_sentence'].get('start_time', 0.0),
                'same_group': cs_sent.get('group', '') == match['match_sentence'].get('group', ''),
                'same_speaker': cs_sent.get('participant_id', '') == match['match_sentence'].get('participant_id', ''),
                'time_distance': abs(cs_sent.get('start_time', 0.0) - match['match_sentence'].get('start_time', 0.0)),
                # Statistics from ALL matches
                'total_matches_above_threshold': total_matches_count,
                'all_matches_same_group': all_matches_same_group,
                'all_matches_same_speaker': all_matches_same_speaker
            }
            detailed_matches.append(detailed_match)
    
    return (cs_sent, detailed_matches, has_matches, similarity_scores)


def analyze_window_matching(
    translated_sentences: List[Dict],
    monolingual_sentences: List[Dict],
    window_sizes: List[int] = [1, 2, 3],
    similarity_threshold: float = 0.4,
    top_k: int = 5,
    num_workers: Optional[int] = None
) -> Dict:
    """
    Analyze POS window matching across multiple window sizes.
    
    For each window size:
    1. Extract POS windows around switch points
    2. Find matching monolingual sentences
    3. Rank matches by group/speaker/time proximity
    4. Collect statistics and examples
    
    Args:
        translated_sentences: List of code-switched sentences with Cantonese translations
        monolingual_sentences: List of Cantonese monolingual sentences
        window_sizes: List of window sizes to analyze (default: [1, 2, 3])
        similarity_threshold: Minimum similarity threshold (default: 0.4)
        top_k: Number of top matches to keep per sentence (default: 5)
        num_workers: Number of CPU cores to leave free. If None, uses all available cores (leaves 0 free).
                     If set to a number, that many cores will be left free.
                     Example: On an 8-core system, num_workers=2 means use 6 workers, leaving 2 free.
                     Default: None (uses all cores).
        
    Returns:
        Dictionary with results for each window size:
        {
            'window_1': {
                'total_sentences': int,
                'sentences_with_matches': int,
                'match_rate': float,
                'total_matches': int,
                'avg_matches_per_sentence': float,
                'avg_similarity': float,
                'similarity_scores': List[float],
                'detailed_matches': List[Dict],
                'example_matches': List[Dict]  # Top 3-5 examples with their top 5 matches
            },
            'window_2': {...},
            'window_3': {...}
        }
    """
    
    results = {}
    
    # Build POS cache once for all window sizes
    logger.info("Building monolingual POS cache...")
    mono_pos_cache = build_monolingual_pos_cache(monolingual_sentences)
    logger.info(f"Cached POS sequences for {len(mono_pos_cache)} monolingual sentences")
    
    # Determine number of workers
    # num_workers parameter now means "cores to leave free"
    total_cores = os.cpu_count() or 1
    
    if num_workers is None:
        # Default: use all cores (leave 0 free)
        actual_workers = total_cores
        logger.info(f"Using all available CPU cores ({actual_workers} workers, leaving 0 free)")
    else:
        # Calculate actual workers: total_cores - cores_to_leave_free
        # Ensure at least 1 worker is used
        actual_workers = max(1, total_cores - num_workers)
        if actual_workers == 1:
            logger.info(f"Leaving {num_workers} cores free, using sequential processing (1 worker)")
        else:
            logger.info(f"Leaving {num_workers} cores free, using {actual_workers} parallel workers (out of {total_cores} total)")
    
    num_workers = actual_workers  # Update num_workers to actual workers for rest of function
    
    for window_size in window_sizes:
        logger.info(f"\nAnalyzing window size n={window_size}...")
        
        window_key = f'window_{window_size}'
        
        # Process all sentences
        detailed_matches = []
        similarity_scores = []
        sentences_with_matches = 0
        
        # Prepare arguments for workers
        worker_args = [
            (cs_sent, monolingual_sentences, window_size, similarity_threshold, mono_pos_cache, top_k)
            for cs_sent in translated_sentences
        ]
        
        # Process sentences in parallel or sequentially
        if num_workers == 1:
            # Sequential processing with progress bar
            processing_results = []
            for args in tqdm(worker_args, desc=f"Window n={window_size}", leave=False):
                result = _process_single_cs_sentence(args)
                processing_results.append(result)
        else:
            # Parallel processing
            with multiprocessing.Pool(processes=num_workers) as pool:
                processing_results = list(tqdm(
                    pool.imap(_process_single_cs_sentence, worker_args),
                    total=len(worker_args),
                    desc=f"Window n={window_size}",
                    leave=False
                ))
        
        # Process results
        for cs_sent, sent_detailed_matches, has_matches, sent_similarity_scores in processing_results:
            if has_matches:
                sentences_with_matches += 1
                detailed_matches.extend(sent_detailed_matches)
                # Collect similarity scores from ALL matches (not just top_k)
                similarity_scores.extend(sent_similarity_scores)
        
        # Calculate statistics
        total_sentences = len(translated_sentences)
        match_rate = sentences_with_matches / total_sentences if total_sentences > 0 else 0.0
        total_matches = len(detailed_matches)
        avg_matches = total_matches / total_sentences if total_sentences > 0 else 0.0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        # Select example matches (sentences with highest average similarity in top-5)
        sentence_avg_similarities = {}
        for match in detailed_matches:
            key = (match['cs_sentence'], match['cs_translation'])
            if key not in sentence_avg_similarities:
                sentence_avg_similarities[key] = []
            sentence_avg_similarities[key].append(match['similarity'])
        
        # Calculate average similarity per sentence and sort
        sentence_rankings = []
        for (cs_sent, cs_trans), sims in sentence_avg_similarities.items():
            avg_sim = sum(sims) / len(sims)
            sentence_rankings.append((cs_sent, cs_trans, avg_sim))
        
        sentence_rankings.sort(key=lambda x: x[2], reverse=True)
        
        # Get top example sentences
        num_examples = min(5, len(sentence_rankings))
        example_sentences = sentence_rankings[:num_examples]
        
        example_matches = []
        for cs_sent, cs_trans, _ in example_sentences:
            # Get all matches for this sentence
            sent_matches = [m for m in detailed_matches 
                           if m['cs_sentence'] == cs_sent and m['cs_translation'] == cs_trans]
            sent_matches = sorted(sent_matches, key=lambda x: x['rank'])[:top_k]
            
            example_matches.append({
                'cs_sentence': cs_sent,
                'cs_translation': cs_trans,
                'matches': sent_matches
            })
        
        # Store results for this window size
        results[window_key] = {
            'window_size': window_size,
            'total_sentences': total_sentences,
            'sentences_with_matches': sentences_with_matches,
            'match_rate': match_rate,
            'total_matches': total_matches,
            'avg_matches_per_sentence': avg_matches,
            'avg_similarity': avg_similarity,
            'similarity_scores': similarity_scores,
            'detailed_matches': detailed_matches,
            'example_matches': example_matches
        }
    
    return results
