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


def rank_matches_by_context(
    matches: List[Dict],
    source_sentence: Dict
) -> List[Dict]:
    """
    Rank matches by contextual relevance: same group > same speaker > time proximity.
    
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
        """
        match_sent = match.get('match_sentence', {})
        match_group = match_sent.get('group', '')
        match_speaker = match_sent.get('participant_id', '')
        match_time = match_sent.get('start_time', 0.0)
        
        # Priority 1: Same speaker (0 if same, 1 if different)
        same_speaker_priority = 0 if match_speaker == source_speaker else 1
        
        # Priority 2: Same group (0 if same, 1 if different)
        same_group_priority = 0 if match_group == source_group else 1
        
        # Priority 3: Time proximity (smaller difference is better)
        time_distance = abs(match_time - source_time)
        
        return (same_speaker_priority, same_group_priority, time_distance)
    
    # Sort by the composite key
    ranked_matches = sorted(matches, key=sort_key)
    
    return ranked_matches


def find_window_matches(
    code_switched_sentence: Dict,
    monolingual_sentences: List[Dict],
    window_size: int = 1,
    similarity_threshold: float = 0.4
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
    
    if switch_index < 0 or not translated_pos:
        return []
    
    # Parse POS sequence
    pos_sequence = translated_pos.split()
    
    if not pos_sequence:
        return []
    
    # Extract POS window around switch point
    pos_window = extract_pos_window(pos_sequence, switch_index, window_size)
    
    if not pos_window:
        return []
    
    matches = []
    window_len = len(pos_window)
    
    # Search through all monolingual sentences
    for mono_sent in monolingual_sentences:
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
        
        # Sliding window approach
        for i in range(len(mono_pos_seq) - window_len + 1):
            mono_window = mono_pos_seq[i:i + window_len]
            similarity = levenshtein_similarity(pos_window, mono_window)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_window = mono_window
                best_start_idx = i
                
                # Early termination for perfect match
                if similarity >= 1.0:
                    break
        
        # Also try comparing to full sequence if monolingual is shorter than window
        if len(mono_pos_seq) < window_len:
            similarity = levenshtein_similarity(pos_window, mono_pos_seq)
            if similarity > best_similarity:
                best_similarity = similarity
                best_window = mono_pos_seq
                best_start_idx = 0
        
        # Keep if above threshold
        if best_similarity >= similarity_threshold:
            # Calculate actual center of matched window
            # For normal window: center = start + window_size
            # For short sentences: center = middle of the actual sentence
            matched_window_length = len(best_window)
            matched_center_idx = best_start_idx + (matched_window_length // 2)
            
            matches.append({
                'match_sentence': mono_sent,
                'similarity': best_similarity,
                'window_size': window_size,
                'pos_window': ' '.join(pos_window),
                'matched_pos': ' '.join(best_window),
                'matched_window_start': best_start_idx,
                'matched_window_center': matched_center_idx  # Actual center of matched window
            })
    
    return matches


def analyze_window_matching(
    translated_sentences: List[Dict],
    monolingual_sentences: List[Dict],
    window_sizes: List[int] = [1, 2, 3],
    similarity_threshold: float = 0.4,
    top_k: int = 5
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
    logger.info(f"Starting window matching analysis for {len(translated_sentences)} sentences")
    logger.info(f"Window sizes: {window_sizes}, Similarity threshold: {similarity_threshold}")
    
    results = {}
    
    for window_size in window_sizes:
        logger.info(f"\nAnalyzing window size n={window_size}...")
        
        window_key = f'window_{window_size}'
        detailed_matches = []
        similarity_scores = []
        sentences_with_matches = 0
        
        # Process each code-switched sentence
        for cs_sent in tqdm(translated_sentences, desc=f"Window n={window_size}", leave=False):
            # Find all matches for this sentence
            matches = find_window_matches(
                cs_sent,
                monolingual_sentences,
                window_size=window_size,
                similarity_threshold=similarity_threshold
            )
            
            if matches:
                sentences_with_matches += 1
                
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
                
                # Collect similarity scores for distribution analysis
                similarity_scores.extend([m['similarity'] for m in matches])
                
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
                        'matched_switch_index': match['matched_window_center'],  # Use actual center instead of start + window_size
                        'matched_group': match['match_sentence'].get('group', ''),
                        'matched_participant': match['match_sentence'].get('participant_id', ''),
                        'matched_start_time': match['match_sentence'].get('start_time', 0.0),
                        'same_group': cs_sent.get('group', '') == match['match_sentence'].get('group', ''),
                        'same_speaker': cs_sent.get('participant_id', '') == match['match_sentence'].get('participant_id', ''),
                        'time_distance': abs(cs_sent.get('start_time', 0.0) - match['match_sentence'].get('start_time', 0.0)),
                        # Statistics from ALL matches (not just top-k)
                        'total_matches_above_threshold': total_matches_count,
                        'all_matches_same_group': all_matches_same_group,
                        'all_matches_same_speaker': all_matches_same_speaker
                    }
                    detailed_matches.append(detailed_match)
        
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
        
        # Get top 3-5 example sentences
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
        
        logger.info(f"  Sentences with matches: {sentences_with_matches}/{total_sentences} ({match_rate*100:.1f}%)")
        logger.info(f"  Total matches found: {total_matches}")
        logger.info(f"  Average similarity: {avg_similarity:.3f}")
    
    return results
