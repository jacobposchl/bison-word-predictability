"""
Functions for creating analysis dataset from translated code-switched sentences.
"""

import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from collections import defaultdict
from src.analysis.pos_tagging import parse_pattern_segments
from src.analysis.matching_algorithm import analyze_window_matching

logger = logging.getLogger(__name__)

# Delimiter for joining context sentences (preserves sentence boundaries)
CONTEXT_SENTENCE_DELIMITER = ' ||| '


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



def _build_participant_index(all_sentences_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build an index of sentences grouped by participant for faster lookups.
    
    Args:
        all_sentences_df: DataFrame with all sentences
        
    Returns:
        Dictionary mapping participant_id to sorted DataFrame of their sentences
    """

    index = {}
    for participant_id, group_df in all_sentences_df.groupby('participant_id'):
        # Sort by start_time for efficient lookups
        index[participant_id] = group_df.sort_values('start_time').reset_index(drop=True)
    return index


def _get_previous_sentences_indexed(
    participant_index: Dict[str, pd.DataFrame],
    participant: str,
    start_time: float,
    num_previous: int = 3
) -> List[Dict]:
    """
    Get k previous sentences using pre-built index (optimized version).
    
    Args:
        participant_index: Pre-built index from _build_participant_index
        participant: Speaker participant ID
        start_time: Start time of current sentence (milliseconds)
        num_previous: Number of previous sentences to retrieve
        
    Returns:
        List of sentence dictionaries with 'sentence' and 'pattern' keys
    """

    if participant not in participant_index:
        return []
    
    participant_df = participant_index[participant]
    
    # Find all rows with start_time < current start_time
    mask = participant_df['start_time'] < start_time
    previous = participant_df[mask].tail(num_previous)
    
    # Return as list of dicts
    result = []
    for _, row in previous.iterrows():
        result.append({
            'sentence': row.get('reconstructed_sentence', ''),
            'pattern': row.get('pattern', '')
        })
    
    return result


def _assemble_context_from_cache(
    context_sentences: List[Dict],
    translation_cache: Dict[Tuple[str, str], str],
    quality_stats: Dict[Tuple[str, str], Tuple[int, int]],
    min_quality: float = 0.3
) -> Dict:
    """
    Assemble context from cache and calculate word-level quality.
    
    Takes a list of context sentences, looks them up in the translation cache,
    joins them with delimiter, and calculates quality score based on word counts.
    
    Args:
        context_sentences: List of sentence dicts with 'sentence' and 'pattern' keys
        translation_cache: Cache dict mapping (sentence, pattern) -> translated text
        quality_stats: Dict mapping (sentence, pattern) -> (attempted_words, completed_words)
        min_quality: Minimum ratio of successfully completed word translations
        
    Returns:
        Dict with:
            - translated_context: Joined Cantonese context (delimiter: ' ||| ')
            - quality_score: Ratio of completed words to attempted words
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
    total_attempted_words = 0
    total_completed_words = 0
    
    for sent_dict in context_sentences:
        sentence = sent_dict.get('sentence', '')
        pattern = sent_dict.get('pattern', '')
        
        if not sentence:
            continue
        
        cache_key = (sentence, pattern)
        if cache_key not in translation_cache:
            logger.warning(f"Sentence not found in translation cache: {sentence[:50]}... (pattern: {pattern})")
            continue
        
        # Get translation and word-level quality stats
        cantonese = translation_cache[cache_key]
        attempted_words, completed_words = quality_stats.get(cache_key, (0, 0))
        
        # Accumulate word counts
        total_attempted_words += attempted_words
        total_completed_words += completed_words
        
        if cantonese:  # Empty string means all words were filtered out
            translations.append(cantonese)
    
    # Join all context sentences with delimiter
    full_context = CONTEXT_SENTENCE_DELIMITER.join(translations)
    
    # Calculate word-level quality score: completed words / attempted words
    quality_score = total_completed_words / max(total_attempted_words, 1)
    is_valid = quality_score >= min_quality
    
    return {
        'translated_context': full_context,
        'quality_score': quality_score,
        'is_valid': is_valid,
        'num_context_sentences': len(context_sentences)
    }


def _batch_translate_context_sentences(
    all_context_sentences: List[Dict],
    translator,
    min_quality: float = 0.3
) -> tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], Tuple[int, int]]]:
    """
    Batch translate all context sentences and return a cache with quality stats.
    
    This function collects all unique context sentences, translates them in batches,
    and returns a cache mapping (sentence, pattern) -> translated text along with
    word-level quality statistics.
    
    Args:
        all_context_sentences: List of all context sentence dicts from all rows
        translator: NLLBTranslator instance
        min_quality: Minimum quality threshold (not used here, but kept for consistency)
        
    Returns:
        Tuple of:
            - Dictionary mapping (sentence, pattern) to translated Cantonese text
            - Dictionary mapping (sentence, pattern) to (attempted_words, completed_words)
    """

    from src.core.text_cleaning import remove_fillers_from_text
    
    # Deduplicate context sentences
    unique_sentences = {}
    for sent_dict in all_context_sentences:
        sentence = sent_dict.get('sentence', '')
        pattern = sent_dict.get('pattern', '')
        if sentence:
            unique_sentences[(sentence, pattern)] = sent_dict
    
    logger.info(f"Translating {len(unique_sentences)} unique context sentences...")
    
    translation_cache = {}
    
    # Group sentences by type for batch processing
    cs_sentences = []
    cs_patterns = []
    cs_words_list = []
    english_sentences_with_patterns = []  # List of (sentence, pattern) tuples
    
    for (sentence, pattern), sent_dict in unique_sentences.items():
        if not pattern:
            continue
        
        segments = parse_pattern_segments(pattern)
        languages = {lang for lang, _ in segments}
        is_code_switched = 'C' in languages and 'E' in languages
        is_pure_cantonese = languages == {'C'}
        is_pure_english = languages == {'E'}
        
        if is_code_switched:
            cs_sentences.append(sentence)
            cs_patterns.append(pattern)
            cs_words_list.append(sentence.split())
        elif is_pure_english:
            english_sentences_with_patterns.append((sentence, pattern))
        elif is_pure_cantonese:
            # Pure Cantonese - no translation needed
            translation_cache[(sentence, pattern)] = sentence
    
    # Batch translate code-switched sentences
    if cs_sentences:
        logger.info(f"Batch translating {len(cs_sentences)} code-switched context sentences...")
        cs_results = translator.translate_batch(cs_sentences, cs_patterns, cs_words_list)
        for (sentence, pattern), result in zip(zip(cs_sentences, cs_patterns), cs_results):
            cantonese = result.get('translated_sentence', '')
            translation_cache[(sentence, pattern)] = cantonese
    
    # Batch translate English sentences
    if english_sentences_with_patterns:
        logger.info(f"Translating {len(english_sentences_with_patterns)} English context sentences...")
        for sentence, pattern in tqdm(english_sentences_with_patterns, desc="Translating English", leave=False):
            try:
                cantonese = translator.translate_english_to_cantonese(sentence)
                translation_cache[(sentence, pattern)] = cantonese
            except Exception as e:
                logger.debug(f"Translation failed for English sentence: {sentence[:50]}... Error: {e}")
                continue
    
    # Clean all translations (remove fillers, UNKNOWN tokens, etc.)
    # Track word-level quality: words before vs after cleaning
    cleaned_cache = {}
    quality_stats = {}  # Maps (sentence, pattern) -> (attempted_words, completed_words)
    
    for (sentence, pattern), cantonese in translation_cache.items():
        # Remove fillers
        cantonese = remove_fillers_from_text(cantonese, lang=None)
        
        # Count words before cleaning (after filler removal)
        words_before_cleaning = [w for w in cantonese.split() if w]
        attempted_words = len(words_before_cleaning)
        
        # Clean translation - remove UNKNOWN/untranslatable tokens and English words
        translated_words = [
            w for w in words_before_cleaning
            if w not in ['UNKNOWN', 'UNK', ''] and not is_english_word(w)
        ]
        completed_words = len(translated_words)
        
        # Store quality stats (word counts)
        quality_stats[(sentence, pattern)] = (attempted_words, completed_words)
        
        if translated_words:
            cleaned_cache[(sentence, pattern)] = ' '.join(translated_words)
        else:
            cleaned_cache[(sentence, pattern)] = ''
    
    logger.info(f"Translation cache built with {len(cleaned_cache)} entries")
    return cleaned_cache, quality_stats


def create_analysis_dataset(
    config,
    filtered_translated_sentences: List[Dict],
    window_results: Dict,
    all_sentences_df: pd.DataFrame = None,
    translator = None,
    window_size: int = None
) -> pd.DataFrame:
    """
    Create analysis dataset from pre-computed window matching results.
    
    
    Takes filtered code-switched sentences and their pre-computed window matching results,
    then builds the final analysis dataset with the top matched monolingual sentence
    for each code-switched sentence.
    
    Args:
        config: Config object
        filtered_translated_sentences: List of filtered CS sentence dicts (already filtered
            for sentences that start with >= min_cantonese words followed by English)
        window_results: Pre-computed results from analyze_window_matching() containing
            detailed matches for each window size
        all_sentences_df: DataFrame with all sentences for context retrieval
        translator: NLLBTranslator for translating CS context
        window_size: Window size to use (if None, uses first window size from config)
        
    Returns:
        DataFrame with columns:
        - translated_cs: Full Cantonese translation
        - matched_mono: Top-1 matched monolingual sentence
        - cs_context, mono_context: Discourse context (if provided)
        - context quality metrics
        - switch_index: Index where code-switch occurs
    """

    logger.info("Creating analysis dataset from pre-computed window matching results...")
    
    # Get window size
    if window_size is None:
        raise ValueError("window_size must be provided to create_analysis_dataset()")
    
    logger.info(f"Using window size: {window_size}")
    logger.info(f"Processing {len(filtered_translated_sentences)} filtered sentences")
    
    if not filtered_translated_sentences:
        logger.warning("No sentences provided!")
        return pd.DataFrame(columns=['translated_cs', 'matched_mono', 'context', 'surprisal', 'switch_index'])
    
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
                'cs_start_time': data['cs_start_time'],
                'switch_index': data['switch_index'],
                'pos_window': data['pos_window'],
                # Keep cs_participant temporarily for context retrieval (removed later)
                'cs_participant': data['cs_participant'],
                
                # Best matched monolingual sentence info
                'matched_mono': best_match['matched_sentence'],
                'matched_group': best_match['matched_group'],
                'matched_start_time': best_match['matched_start_time'],
                'matched_pos': best_match['matched_pos'],
                'matched_switch_index': best_match['matched_switch_index'],  # Center of matched POS window
                'cs_switch_pos': best_match.get('cs_switch_pos', 'UNKNOWN'),
                'mono_switch_pos': best_match.get('mono_switch_pos', 'UNKNOWN'),
                'similarity': best_match['similarity'],
                # Keep matched_participant temporarily for context retrieval (removed later)
                'matched_participant': best_match['matched_participant'],
                
                # Match statistics (from ALL matches, not just those in all_matches list)
                'total_matches_above_threshold': best_match.get('total_matches_above_threshold', len(all_matches)),
                'matches_same_group': best_match.get('all_matches_same_group', sum(1 for m in all_matches if m['same_group'])),
                'matches_same_speaker': best_match.get('all_matches_same_speaker', sum(1 for m in all_matches if m['same_speaker']))
            })
    
    # Count sentences without matches
    for sent in filtered_translated_sentences:
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
    
    columns_to_remove = ['cs_participant', 'matched_participant']
    existing_columns_to_remove = [col for col in columns_to_remove if col in analysis_df.columns]
    if existing_columns_to_remove:
        analysis_df = analysis_df.drop(columns=existing_columns_to_remove)
        logger.info(f"Removed temporary columns used for context retrieval: {existing_columns_to_remove}")
    
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
            - cs_context_valid: Whether CS context meets quality threshold
            - mono_context: Context for mono sentence (already Cantonese)
            - mono_context_valid: Whether mono context meets quality threshold
            
    """

    # Calculate max context length from context_lengths list
    context_lengths = config.get('context.context_lengths')
    if isinstance(context_lengths, list) and len(context_lengths) > 0:
        num_context = max(context_lengths)

    min_quality = config.get('context.min_translation_quality')
    
    logger.info(f"Context settings: {num_context} previous sentences, min quality {min_quality}")
    
    # Build participant index for faster lookups
    logger.info("Building participant index for faster context retrieval...")
    participant_index = _build_participant_index(all_sentences_df)
    logger.info(f"Indexed sentences for {len(participant_index)} participants")
    
    # Collect all context sentences first (before translation)
    logger.info("Collecting all context sentences...")
    all_cs_context_sents = []
    all_mono_context_sents = []
    
    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Collecting context"):
        # Get context for CS sentence (using indexed lookup)
        cs_context_sents = _get_previous_sentences_indexed(
            participant_index,
            row['cs_participant'],
            row['cs_start_time'],
            num_context
        )
        all_cs_context_sents.extend(cs_context_sents)
        
        # Get context for matched mono sentence
        mono_context_sents = _get_previous_sentences_indexed(
            participant_index,
            row['matched_participant'],
            row['matched_start_time'],
            num_context
        )
        all_mono_context_sents.extend(mono_context_sents)
    
    # Batch translate all context sentences
    logger.info("Batch translating context sentences...")
    cs_translation_cache, cs_quality_stats = _batch_translate_context_sentences(all_cs_context_sents, translator, min_quality)
    mono_translation_cache, mono_quality_stats = _batch_translate_context_sentences(all_mono_context_sents, translator, min_quality)
    
    # Process rows and use cached translations
    logger.info("Processing rows with cached translations...")
    results = []
    
    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Adding context"):
        row_data = row.to_dict()
        
        # Get context for CS sentence (using indexed lookup)
        cs_context_sents = _get_previous_sentences_indexed(
            participant_index,
            row['cs_participant'],
            row['cs_start_time'],
            num_context
        )
        
        # Get context for matched mono sentence
        mono_context_sents = _get_previous_sentences_indexed(
            participant_index,
            row['matched_participant'],
            row['matched_start_time'],
            num_context
        )
        
        # Assemble CS context from cache
        cs_context_result = _assemble_context_from_cache(
            cs_context_sents,
            cs_translation_cache,
            cs_quality_stats,
            min_quality
        )
        
        # Assemble mono context from cache
        mono_context_result = _assemble_context_from_cache(
            mono_context_sents,
            mono_translation_cache,
            mono_quality_stats,
            min_quality
        )
        
        # Add context data to row
        row_data.update({
            'cs_context': cs_context_result['translated_context'],
            'cs_context_valid': cs_context_result['is_valid'],
            
            'mono_context': mono_context_result['translated_context'],
            'mono_context_valid': mono_context_result['is_valid']
        })
        
        results.append(row_data)
    
    result_df = pd.DataFrame(results)
    
    # Log context statistics
    logger.info(f"\nContext statistics:")
    logger.info(f"  CS context valid: {result_df['cs_context_valid'].sum()} / {len(result_df)}")
    logger.info(f"  Mono context valid: {result_df['mono_context_valid'].sum()} / {len(result_df)}")
    logger.info(f"  Both contexts valid: {((result_df['cs_context_valid']) & (result_df['mono_context_valid'])).sum()} / {len(result_df)}")
    
    return result_df

