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


def regenerate_pattern_from_sentence(sentence: str) -> str:
    """
    Regenerate pattern from actual sentence by detecting language of each word.
    
    This ensures the pattern matches the actual sentence structure (without fillers),
    rather than using a pattern from a different version of the sentence.
    
    Args:
        sentence: Space-separated sentence
        
    Returns:
        Pattern string like "C5-E2-C3" representing language segments
    """
    if not sentence or not sentence.strip():
        return ""
    
    words = sentence.split()
    if not words:
        return ""
    
    segments = []
    current_lang = None
    current_count = 0
    
    for word in words:
        # Determine language of current word
        is_english = is_english_word(word)
        word_lang = 'E' if is_english else 'C'
        
        if word_lang == current_lang:
            # Continue current segment
            current_count += 1
        else:
            # Start new segment
            if current_lang is not None:
                segments.append(f"{current_lang}{current_count}")
            current_lang = word_lang
            current_count = 1
    
    # Add final segment
    if current_lang is not None:
        segments.append(f"{current_lang}{current_count}")
    
    return '-'.join(segments) if segments else ""


def get_previous_sentences(
    all_sentences_df: pd.DataFrame,
    participant: str,
    start_time: float,
    num_previous: int = 3
) -> List[Dict]:
    """
    Get k previous sentences from same speaker's conversation.
    
    Args:
        all_sentences_df: DataFrame with all sentences (from all_sentences.csv)
        participant: Speaker participant ID
        start_time: Start time of current sentence (milliseconds)
        num_previous: Number of previous sentences to retrieve
        
    Returns:
        List of sentence dictionaries with 'sentence' and 'pattern' keys
        (may be < num_previous if not enough exist)
    """
    # Filter to same participant, before current sentence
    previous = all_sentences_df[
        (all_sentences_df['participant_id'] == participant) &
        (all_sentences_df['start_time'] < start_time)
    ].sort_values('start_time').tail(num_previous)
    
    # Return as list of dicts with sentence and pattern
    result = []
    for _, row in previous.iterrows():
        result.append({
            'sentence': row.get('reconstructed_sentence', ''),
            'pattern': row.get('pattern', '')
        })
    
    return result


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
    
    # Use binary search to find position (much faster than filtering)
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


def translate_context_sentences(
    context_sentences: List[Dict],
    translator,
    min_quality: float = 0.3,
    translation_cache: Optional[Dict[Tuple[str, str], str]] = None
) -> Dict:
    """
    Translate context sentences and assess quality.
    
    Translates sentences (code-switched or English), removes fillers, and filters
    untranslatable words (UNKNOWN tokens, English words) from translations.
    
    Args:
        context_sentences: List of sentence dicts with 'sentence' and 'pattern' keys
        translator: NLLBTranslator instance
        min_quality: Minimum ratio of successfully completed translations
        translation_cache: Optional cache dict mapping (sentence, pattern) -> translated text
        
    Returns:
        Dict with:
            - translated_context: Cleaned Cantonese context (space-separated, fillers removed)
            - quality_score: Ratio of completed translations to attempted translations
            - is_valid: Whether quality meets minimum threshold
            - num_context_sentences: Number of context sentences
    """
    from src.core.text_cleaning import remove_fillers_from_text
    
    if not context_sentences:
        return {
            'translated_context': '',
            'quality_score': 0.0,
            'is_valid': False,
            'num_context_sentences': 0
        }
    
    if translation_cache is None:
        translation_cache = {}
    
    translations = []
    attempted_translations = len(context_sentences)
    completed_translations = 0
    
    for sent_dict in context_sentences:
        sentence = sent_dict.get('sentence', '')
        pattern = sent_dict.get('pattern', '')
        
        if not sentence:
            continue
        
        # Check cache first
        cache_key = (sentence, pattern)
        if cache_key in translation_cache:
            cantonese = translation_cache[cache_key]
        else:
            try:
                # Determine sentence type based on pattern
                is_code_switched = False
                is_pure_cantonese = False
                is_pure_english = False
                
                if pattern:
                    segments = parse_pattern_segments(pattern)
                    # Check what languages are in the pattern
                    languages = {lang for lang, _ in segments}
                    is_code_switched = 'C' in languages and 'E' in languages
                    is_pure_cantonese = languages == {'C'}
                    is_pure_english = languages == {'E'}
                
                # Translate based on type
                if is_code_switched:
                    # Use code-switched translation method
                    words = sentence.split()
                    translation_result = translator.translate_code_switched_sentence(
                        sentence=sentence,
                        pattern=pattern,
                        words=words
                    )
                    cantonese = translation_result.get('translated_sentence', '')
                elif is_pure_cantonese:
                    # Pure Cantonese - use as-is (no translation needed)
                    cantonese = sentence
                elif is_pure_english:
                    # Pure English - translate to Cantonese
                    cantonese = translator.translate_english_to_cantonese(sentence)
                else:
                    # No pattern or unknown pattern - skip for safety
                    logger.debug(f"Skipping sentence with invalid or missing pattern: {pattern}")
                    continue
                
                # Cache the translation
                translation_cache[cache_key] = cantonese
            
            except Exception as e:
                # Translation failed for this sentence, skip it
                logger.debug(f"Translation failed for sentence: {sentence[:50]}... Error: {e}")
                continue
        
        # Remove fillers from translation
        cantonese = remove_fillers_from_text(cantonese, lang=None)  # Remove both C and E fillers
        
        # Clean translation - remove UNKNOWN/untranslatable tokens and English words
        translated_words = [
            w for w in cantonese.split()
            if w not in ['UNKNOWN', 'UNK', ''] and not is_english_word(w)
        ]
        
        # Keep cleaned translation if it has content
        if translated_words:
            translations.append(' '.join(translated_words))
            completed_translations += 1
    
    # Join all context sentences with a delimiter to preserve sentence boundaries
    # This allows extracting different context lengths later
    # Use ||| as delimiter (unlikely to appear in Cantonese text)
    CONTEXT_SENTENCE_DELIMITER = ' ||| '
    full_context = CONTEXT_SENTENCE_DELIMITER.join(translations)
    
    # Calculate quality score: completed translations / attempted translations
    quality_score = completed_translations / max(attempted_translations, 1)
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
) -> Dict[Tuple[str, str], str]:
    """
    Batch translate all context sentences and return a cache.
    
    This function collects all unique context sentences, translates them in batches,
    and returns a cache mapping (sentence, pattern) -> translated text.
    
    Args:
        all_context_sentences: List of all context sentence dicts from all rows
        translator: NLLBTranslator instance
        min_quality: Minimum quality threshold (not used here, but kept for consistency)
        
    Returns:
        Dictionary mapping (sentence, pattern) tuple to translated Cantonese text
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
    cantonese_sentences = []
    
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
    
    # Batch translate English sentences (one at a time, but more efficient than before)
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
    cleaned_cache = {}
    for (sentence, pattern), cantonese in translation_cache.items():
        # Remove fillers
        cantonese = remove_fillers_from_text(cantonese, lang=None)
        
        # Clean translation - remove UNKNOWN/untranslatable tokens and English words
        translated_words = [
            w for w in cantonese.split()
            if w not in ['UNKNOWN', 'UNK', ''] and not is_english_word(w)
        ]
        
        if translated_words:
            cleaned_cache[(sentence, pattern)] = ' '.join(translated_words)
        else:
            cleaned_cache[(sentence, pattern)] = ''
    
    logger.info(f"Translation cache built with {len(cleaned_cache)} entries")
    return cleaned_cache


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
    filtered_translated_sentences: List[Dict],
    window_results: Dict,
    all_sentences_df: pd.DataFrame = None,
    translator = None,
    window_size: int = None
) -> pd.DataFrame:
    """
    Create analysis dataset from pre-computed window matching results.
    
    Optionally adds discourse context from previous sentences in the same conversation.
    
    Takes filtered code-switched sentences and their pre-computed window matching results,
    then builds the final analysis dataset with the top-1 matched monolingual sentence
    for each code-switched sentence.
    
    Args:
        config: Config object
        filtered_translated_sentences: List of filtered CS sentence dicts (already filtered
            for sentences that start with >= min_cantonese words followed by English)
        window_results: Pre-computed results from analyze_window_matching() containing
            detailed matches for each window size
        all_sentences_df: Optional DataFrame with all sentences for context retrieval
        translator: Optional NLLBTranslator for translating CS context
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
    
    # Get window size (must be provided)
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
    
    # Remove temporary columns used only for context retrieval
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
    Add discourse context columns to analysis dataset (optimized version).
    
    Retrieves k previous sentences from same speaker and translates CS context.
    Uses batch translation and indexing for better performance.
    
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
            
    Note: Rows with low context quality (below min_quality threshold) are NOT automatically
    removed. They are kept in the dataset but marked with cs_context_valid=False or
    mono_context_valid=False. Context will only be used in surprisal calculation if both
    cs_context_valid and mono_context_valid are True.
    """
    # Calculate max context length from context_lengths list
    # This ensures we store enough context for all context_lengths in surprisal analysis
    context_lengths = config.get('context.context_lengths', [3])
    if isinstance(context_lengths, list) and len(context_lengths) > 0:
        num_context = max(context_lengths)
    else:
        # Fallback if context_lengths is not a valid list
        num_context = 3
    min_required = config.get('context.min_required_sentences', 2)
    min_quality = config.get('context.min_translation_quality', 0.3)
    
    logger.info(f"Context settings: {num_context} previous sentences, min {min_required} required, min quality {min_quality}")
    
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
    cs_translation_cache = _batch_translate_context_sentences(all_cs_context_sents, translator, min_quality)
    mono_translation_cache = _batch_translate_context_sentences(all_mono_context_sents, translator, min_quality)
    
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
        
        # Translate CS context using cache
        if cs_context_sents:
            cs_context_result = translate_context_sentences(
                cs_context_sents,
                translator,
                min_quality,
                translation_cache=cs_translation_cache
            )
        else:
            cs_context_result = {
                'translated_context': '',
                'quality_score': 0.0,
                'is_valid': False,
                'num_context_sentences': 0
            }
        
        # Mono context also needs translation and filler removal (same as CS context)
        if mono_context_sents:
            mono_context_result = translate_context_sentences(
                mono_context_sents,
                translator,
                min_quality,
                translation_cache=mono_translation_cache
            )
        else:
            mono_context_result = {
                'translated_context': '',
                'quality_score': 0.0,
                'is_valid': False,
                'num_context_sentences': 0
            }
        
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

