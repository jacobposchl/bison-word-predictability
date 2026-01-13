"""
Functions for creating analysis dataset from translated code-switched sentences.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from src.analysis.pos_tagging import parse_pattern_segments
from src.analysis.matching_algorithm import analyze_window_matching

logger = logging.getLogger(__name__)


def is_english_word(word: str) -> bool:
    """
    Check if word is primarily English characters.
    
    Args:
        word: Word to check
        
    Returns:
        True if word contains primarily ASCII characters
    """
    if not word:
        return False
    alpha_chars = [c for c in word if c.isalpha()]
    if not alpha_chars:
        return False
    return all(ord(c) < 128 for c in alpha_chars)


def get_previous_sentences(
    all_sentences_df: pd.DataFrame,
    participant: str,
    start_time: float,
    num_previous: int = 3
) -> List[str]:
    """
    Get k previous sentences from same speaker's conversation.
    
    Args:
        all_sentences_df: DataFrame with all sentences (from all_sentences.csv)
        participant: Speaker participant ID
        start_time: Start time of current sentence (milliseconds)
        num_previous: Number of previous sentences to retrieve
        
    Returns:
        List of previous sentences (may be < num_previous if not enough exist)
    """
    # Filter to same participant, before current sentence
    previous = all_sentences_df[
        (all_sentences_df['participant'] == participant) &
        (all_sentences_df['start_time'] < start_time)
    ].sort_values('start_time').tail(num_previous)
    
    # Return reconstructed sentences
    return previous['reconstructed_sentence'].tolist()


def translate_context_sentences(
    context_sentences: List[str],
    translator,
    min_quality: float = 0.3
) -> Dict:
    """
    Translate context sentences and assess quality.
    
    Removes untranslatable words (UNKNOWN tokens, English words) from translations.
    
    Args:
        context_sentences: List of sentences to translate
        translator: NLLBTranslator instance
        min_quality: Minimum ratio of successfully translated words
        
    Returns:
        Dict with:
            - translated_context: Cleaned Cantonese context (space-separated)
            - quality_score: Ratio of translated words to original words
            - is_valid: Whether quality meets minimum threshold
            - num_context_sentences: Number of context sentences
    """
    if not context_sentences:
        return {
            'translated_context': '',
            'quality_score': 0.0,
            'is_valid': False,
            'num_context_sentences': 0
        }
    
    translations = []
    total_original_words = 0
    total_translated_words = 0
    
    for sent in context_sentences:
        # Translate sentence
        cantonese = translator.translate_english_to_cantonese(sent)
        
        # Count original words
        original_words = sent.split()
        total_original_words += len(original_words)
        
        # Clean translation - remove UNKNOWN/untranslatable tokens
        translated_words = [
            w for w in cantonese.split()
            if w not in ['UNKNOWN', 'UNK', ''] and not is_english_word(w)
        ]
        
        total_translated_words += len(translated_words)
        
        # Keep cleaned translation
        if translated_words:
            translations.append(' '.join(translated_words))
    
    # Join all context sentences
    full_context = ' '.join(translations)
    
    # Calculate quality score
    quality_score = total_translated_words / max(total_original_words, 1)
    is_valid = quality_score >= min_quality
    
    return {
        'translated_context': full_context,
        'quality_score': quality_score,
        'is_valid': is_valid,
        'num_context_sentences': len(context_sentences)
    }


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


def create_analysis_dataset(
    config,
    translated_df: pd.DataFrame,
    monolingual_df: pd.DataFrame,
    all_sentences_df: pd.DataFrame = None,
    translator = None
) -> pd.DataFrame:
    """
    Create analysis dataset using window matching to find matched monolingual sentences.
    
    Optionally adds discourse context from previous sentences in the same conversation.
    
    Filters sentences that start with at least x Cantonese words followed by
    English words, then finds the top-1 matched monolingual sentence using POS window matching.
    
    Args:
        config: Config object
        translated_df: DataFrame with translated code-switched sentences
        monolingual_df: DataFrame with monolingual Cantonese sentences
        all_sentences_df: Optional DataFrame with all sentences for context retrieval
        translator: Optional NLLBTranslator for translating CS context
        
    Returns:
        DataFrame with columns:
        - translated_cs: Full Cantonese translation
        - matched_mono: Top-1 matched monolingual sentence
        - cs_context, mono_context: Discourse context (if provided)
        - context quality metrics
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
    
    # Add discourse context if all_sentences_df and translator are provided
    if all_sentences_df is not None and translator is not None:
        logger.info("\nAdding discourse context from previous sentences...")
        analysis_df = add_context_to_analysis_dataset(
            analysis_df,
            all_sentences_df,
            translator,
            config
        )
    
    return analysis_df


def add_context_to_analysis_dataset(
    analysis_df: pd.DataFrame,
    all_sentences_df: pd.DataFrame,
    translator,
    config
) -> pd.DataFrame:
    """
    Add discourse context columns to analysis dataset.
    
    Retrieves k previous sentences from same speaker and translates CS context.
    
    Args:
        analysis_df: Analysis dataset with CS and mono sentence info
        all_sentences_df: Full DataFrame with all sentences
        translator: NLLBTranslator instance
        config: Config object
        
    Returns:
        DataFrame with added context columns:
            - cs_context: Translated context for CS sentence
            - mono_context: Context for mono sentence (already Cantonese)
            - cs_context_quality: Translation quality score for CS context
            - cs_context_valid: Whether CS context meets quality threshold
            - mono_context_valid: Whether mono context has enough sentences
            - cs_context_count: Number of CS context sentences
            - mono_context_count: Number of mono context sentences
    """
    num_context = config.get('context.num_previous_sentences', 3)
    min_required = config.get('context.min_required_sentences', 2)
    min_quality = config.get('context.min_translation_quality', 0.3)
    
    logger.info(f"Context settings: {num_context} previous sentences, min {min_required} required, min quality {min_quality}")
    
    results = []
    
    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Adding context"):
        row_data = row.to_dict()
        
        # Get context for CS sentence
        cs_context_sents = get_previous_sentences(
            all_sentences_df,
            row['cs_participant'],
            row['cs_start_time'],
            num_context
        )
        
        # Get context for matched mono sentence
        mono_context_sents = get_previous_sentences(
            all_sentences_df,
            row['matched_participant'],
            row['matched_start_time'],
            num_context
        )
        
        # Translate CS context
        if cs_context_sents:
            cs_context_result = translate_context_sentences(
                cs_context_sents,
                translator,
                min_quality
            )
        else:
            cs_context_result = {
                'translated_context': '',
                'quality_score': 0.0,
                'is_valid': False,
                'num_context_sentences': 0
            }
        
        # Mono context doesn't need translation (already Cantonese)
        # Just join with spaces for raw text
        mono_context_result = {
            'translated_context': ' '.join(mono_context_sents),
            'quality_score': 1.0,  # Perfect quality (no translation)
            'is_valid': len(mono_context_sents) >= min_required,
            'num_context_sentences': len(mono_context_sents)
        }
        
        # Add context data to row
        row_data.update({
            'cs_context': cs_context_result['translated_context'],
            'cs_context_quality': cs_context_result['quality_score'],
            'cs_context_valid': cs_context_result['is_valid'],
            'cs_context_count': cs_context_result['num_context_sentences'],
            
            'mono_context': mono_context_result['translated_context'],
            'mono_context_quality': mono_context_result['quality_score'],
            'mono_context_valid': mono_context_result['is_valid'],
            'mono_context_count': mono_context_result['num_context_sentences']
        })
        
        results.append(row_data)
    
    result_df = pd.DataFrame(results)
    
    # Log context statistics
    logger.info(f"\nContext statistics:")
    logger.info(f"  CS context valid: {result_df['cs_context_valid'].sum()} / {len(result_df)}")
    logger.info(f"  Mono context valid: {result_df['mono_context_valid'].sum()} / {len(result_df)}")
    logger.info(f"  Average CS context quality: {result_df['cs_context_quality'].mean():.2f}")
    logger.info(f"  Average CS context count: {result_df['cs_context_count'].mean():.1f}")
    logger.info(f"  Average mono context count: {result_df['mono_context_count'].mean():.1f}")
    
    return result_df

