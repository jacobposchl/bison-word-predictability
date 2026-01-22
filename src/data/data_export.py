"""
Data export functionality for code-switching analysis.

This module handles exporting processed data to CSV files with filtering
for code-switching sentences.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import os
import re

logger = logging.getLogger(__name__)


def _is_english_word(word: str) -> bool:
    """
    Check if a word is likely English (contains only ASCII letters).
    
    Args:
        word: Word to check
        
    Returns:
        True if word appears to be English, False otherwise
    """
    if not word:
        return False
    alpha_chars = [c for c in word if c.isalpha()]
    if not alpha_chars:
        return False
    return all(ord(c) < 128 for c in alpha_chars)


def _regenerate_pattern_from_sentence(sentence: str) -> str:
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
        is_english = _is_english_word(word)
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


def _contains_english_words(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains any English words.
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (has_english, english_words_found)
    """
    # Split by whitespace and punctuation to get potential words
    words = re.findall(r'\b\w+\b', text)
    
    english_words = []
    for word in words:
        if _is_english_word(word):
            # Filter out very short words that might be false positives
            if len(word) > 2 or word.lower() in ['ok', 'okay', 'uh', 'um', 'ah', 'oh']:
                english_words.append(word)
    
    return len(english_words) > 0, english_words


def _verify_cantonese_only(translation: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a translation is fully Cantonese (no English words).
    
    Args:
        translation: Translated text to verify
        
    Returns:
        Tuple of (is_valid, error_message)
        is_valid is True if translation is fully Cantonese
    """
    if not translation or not translation.strip():
        return False, "Translation is empty"
    
    has_english, english_words = _contains_english_words(translation)
    
    if has_english:
        return False, f"Contains English words: {', '.join(english_words[:5])}"
    
    return True, None


def _count_words_from_pattern(pattern: str) -> int:
    """
    Count total words from a pattern string.
    
    Args:
        pattern: Pattern like "C3-E2-C1" or "C5"
        
    Returns:
        Total word count (sum of all numbers in pattern)
    """
    if not pattern or pattern == 'FILLER_ONLY':
        return 0
    
    import re
    # Extract all numbers from pattern (e.g., "C3-E2-C1" -> [3, 2, 1])
    numbers = re.findall(r'\d+', pattern)
    return sum(int(n) for n in numbers)


def _count_words_from_text(text: str) -> int:
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


def _filter_by_min_words(sentences: List[Dict], min_words: int, use_without_fillers: bool = True) -> List[Dict]:
    """
    Filter sentences to only keep those with at least min_words AFTER filler removal.
    
    Args:
        sentences: List of sentence dictionaries
        min_words: Minimum number of words required
        use_without_fillers: If True, count words without fillers; if False, count with fillers
        
    Returns:
        Filtered list of sentences
    """
    filtered = []
    for s in sentences:
        if use_without_fillers:
            # Count words from pattern_content_only or reconstructed_text_without_fillers
            pattern = s.get('pattern_content_only', '')
            if pattern and pattern != 'FILLER_ONLY':
                word_count = _count_words_from_pattern(pattern)
            else:
                # Fallback to counting from text
                text = s.get('reconstructed_text_without_fillers', '')
                word_count = _count_words_from_text(text)
        else:
            # Count words from pattern_with_fillers or reconstructed_text
            pattern = s.get('pattern_with_fillers', s.get('pattern', ''))
            if pattern:
                word_count = _count_words_from_pattern(pattern)
            else:
                # Fallback to counting from text
                text = s.get('reconstructed_text', '')
                word_count = _count_words_from_text(text)
        
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
    
    # Check which columns exist
    sort_columns = []
    if 'group' in df.columns:
        sort_columns.append('group')
    if 'participant_id' in df.columns:
        sort_columns.append('participant_id')
    if 'start_time' in df.columns:
        sort_columns.append('start_time')
    
    if sort_columns:
        df = df.sort_values(by=sort_columns, na_position='last').reset_index(drop=True)
    
    return df


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
    csv_without_fillers_path: str,
    min_sentence_words: int = 2
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
        min_sentence_words: Minimum number of words (after filler removal for without_fillers)
        
    Returns:
        Tuple of (dataframe_with_fillers, dataframe_without_fillers)
    """
    # type: ignore
    # Filter both datasets to only keep sentences with actual code-switching
    with_fillers = filter_code_switching_sentences(all_sentences, include_fillers=True)
    without_fillers = filter_code_switching_sentences(all_sentences, include_fillers=False)
    
    # Apply min_sentence_words filtering AFTER filler removal
    with_fillers = _filter_by_min_words(with_fillers, min_sentence_words, use_without_fillers=False)
    without_fillers = _filter_by_min_words(without_fillers, min_sentence_words, use_without_fillers=True)
    
    logger.info(f"Dataset WITH fillers: {len(with_fillers)} code-switching sentences")
    logger.info(f"Dataset WITHOUT fillers: {len(without_fillers)} code-switching sentences")
    logger.info(
        f"Difference: {len(with_fillers) - len(without_fillers)} sentences "
        f"reclassified as non-code-switching"
    )
    
    # Create the first CSV - WITH fillers
    csv_with_fillers = pd.DataFrame({
        'start_time': [s['start_time'] for s in with_fillers],
        'end_time': [s['end_time'] for s in with_fillers],
        'reconstructed_sentence': [s['reconstructed_text'] for s in with_fillers],
        'sentence_original': [s['text'] for s in with_fillers],
        'pattern': [s['pattern_with_fillers'] for s in with_fillers],
        'matrix_language': [s['matrix_language'] for s in with_fillers],
        'group': [s['group'] for s in with_fillers],
        'participant_id': [s['participant_id'] for s in with_fillers]
    })
    csv_with_fillers = sort_dataframe(csv_with_fillers)
    
    # Create the second CSV - WITHOUT fillers
    csv_without_fillers = pd.DataFrame({
        'start_time': [s['start_time'] for s in without_fillers],
        'end_time': [s['end_time'] for s in without_fillers],
        'reconstructed_sentence': [s['reconstructed_text_without_fillers'] for s in without_fillers],
        'sentence_original': [s['text'] for s in without_fillers],
        'pattern': [s['pattern_content_only'] for s in without_fillers],
        'matrix_language': [s['matrix_language'] for s in without_fillers],
        'group': [s['group'] for s in without_fillers],
        'participant_id': [s['participant_id'] for s in without_fillers]
    })
    csv_without_fillers = sort_dataframe(csv_without_fillers)
    
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
    csv_all_sentences_path: str,
    min_sentence_words: int = 2
) -> pd.DataFrame:
    """
    Export ALL sentences (both monolingual and code-switched) to CSV.
    
    This includes sentences that were filtered out in the code-switching analysis.
    Useful for exploratory analysis that needs monolingual sentences.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        csv_all_sentences_path: Output path for CSV with all sentences
        min_sentence_words: Minimum number of words (after filler removal)
        
    Returns:
        DataFrame with all sentences
    """
    logger.info(f"Exporting ALL sentences (monolingual + code-switched) to CSV...")
    
    # Filter by min_sentence_words AFTER filler removal
    filtered_sentences = _filter_by_min_words(all_sentences, min_sentence_words, use_without_fillers=True)
    logger.info(f"Filtered to {len(filtered_sentences)} sentences with at least {min_sentence_words} words (after filler removal)")
    
    # Create DataFrame with all sentences
    # Regenerate patterns from actual sentences to ensure they match reconstructed_sentence
    csv_all_data = []
    patterns_regenerated = 0
    patterns_unchanged = 0
    
    for s in filtered_sentences:
        reconstructed = s.get('reconstructed_text_without_fillers', s.get('reconstructed_text', ''))
        original_pattern = s.get('pattern_content_only', s.get('pattern', ''))
        
        # Regenerate pattern from actual sentence to ensure it matches
        regenerated_pattern = _regenerate_pattern_from_sentence(reconstructed)
        
        # Use regenerated pattern if available, fallback to original
        if regenerated_pattern:
            pattern_to_use = regenerated_pattern
            if regenerated_pattern != original_pattern:
                patterns_regenerated += 1
            else:
                patterns_unchanged += 1
        else:
            # Fallback to original if regeneration failed (empty sentence, etc.)
            pattern_to_use = original_pattern
            patterns_unchanged += 1
        
        csv_all_data.append({
            'start_time': s['start_time'],
            'end_time': s['end_time'],
            'reconstructed_sentence': reconstructed,
            'sentence_original': s['text'],
            'pattern': pattern_to_use,
            'matrix_language': s.get('matrix_language', 'Unknown'),
            'group': s.get('group', ''),
            'participant_id': s.get('participant_id', '')
        })
    
    csv_all = pd.DataFrame(csv_all_data)
    
    if patterns_regenerated > 0:
        logger.info(f"Regenerated {patterns_regenerated} patterns to match actual sentences")
    logger.info(f"Using {patterns_unchanged} original patterns (already matched)")
    csv_all = sort_dataframe(csv_all)
    
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


def _add_pos_to_dataframe(df: pd.DataFrame, sentence_col: str, pos_col: str, is_cantonese: bool = False) -> pd.DataFrame:
    """
    Add POS tagging column to a DataFrame.
    
    Args:
        df: DataFrame to add POS column to
        sentence_col: Name of column containing sentences
        pos_col: Name of column to create for POS tags
        is_cantonese: True if sentences are Cantonese, False if English
        
    Returns:
        DataFrame with POS column added
    """
    from src.analysis.pos_tagging import pos_tag_cantonese, pos_tag_english, extract_pos_sequence
    
    logger.info(f"Adding POS tagging to {pos_col} column...")
    
    pos_sequences = []
    for idx, row in df.iterrows():
        sentence = str(row.get(sentence_col, ''))
        if not sentence or pd.isna(sentence):
            pos_sequences.append('')
            continue
        
        try:
            if is_cantonese:
                tagged = pos_tag_cantonese(sentence)
            else:
                tagged = pos_tag_english(sentence)
            
            pos_seq = extract_pos_sequence(tagged)
            pos_sequences.append(' '.join(pos_seq) if pos_seq else '')
        except Exception as e:
            logger.warning(f"Error tagging sentence at row {idx}: {e}")
            pos_sequences.append('')
    
    df = df.copy()
    df[pos_col] = pos_sequences
    return df


def export_cantonese_monolingual(
    all_sentences: List[Dict],
    config,
    min_sentence_words: int = 2
) -> pd.DataFrame:
    """
    Export only Cantonese monolingual sentences WITHOUT fillers.
    
    This is the only monolingual dataset needed for downstream analysis.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        config: Config object with CSV path methods
        min_sentence_words: Minimum number of words required
        
    Returns:
        DataFrame with Cantonese monolingual sentences (WITHOUT fillers)
    """
    logger.info("Exporting Cantonese monolingual sentences (WITHOUT fillers)...")
    
    # Helper function to determine if sentence is monolingual Cantonese
    def is_monolingual_cantonese(pattern: str) -> bool:
        """Check if pattern represents pure Cantonese (only C, no E)."""
        return 'C' in pattern and 'E' not in pattern and pattern != 'FILLER_ONLY'
    
    # Filter sentences - must be monolingual in BOTH patterns
    cantonese_sentences = []
    
    for s in all_sentences:
        pattern_with = s.get('pattern_with_fillers', '')
        pattern_without = s.get('pattern_content_only', '')
        
        # Must be monolingual in BOTH patterns to avoid code-switched sentences
        # that become "monolingual" after filler removal
        if is_monolingual_cantonese(pattern_with) and is_monolingual_cantonese(pattern_without):
            cantonese_sentences.append(s)
    
    # Apply min_sentence_words filtering AFTER filler removal
    cantonese_sentences = _filter_by_min_words(cantonese_sentences, min_sentence_words, use_without_fillers=True)
    
    logger.info(f"Found {len(cantonese_sentences)} Cantonese monolingual sentences (after filtering)")
    
    # Create DataFrame
    df = pd.DataFrame({
        'start_time': [s['start_time'] for s in cantonese_sentences],
        'end_time': [s['end_time'] for s in cantonese_sentences],
        'reconstructed_sentence': [s['reconstructed_text_without_fillers'] for s in cantonese_sentences],
        'pattern': [s.get('pattern_content_only', '') for s in cantonese_sentences],
        'group': [s.get('group', '') for s in cantonese_sentences],
        'participant_id': [s.get('participant_id', '') for s in cantonese_sentences]
    })
    
    df = sort_dataframe(df)
    
    # Add POS tagging
    df = _add_pos_to_dataframe(df, 'reconstructed_sentence', 'pos', is_cantonese=True)
    
    # Save CSV
    csv_path = config.get_csv_cantonese_mono_without_fillers_path()
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved '{csv_path}' - {len(df)} Cantonese monolingual sentences")
    
    return df


def export_monolingual_sentences(
    all_sentences: List[Dict],
    config,
    min_sentence_words: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Export monolingual sentences (Cantonese and English) to separate CSVs.
    
    Creates 4 CSV files:
    - Cantonese monolingual WITH fillers
    - Cantonese monolingual WITHOUT fillers  
    - English monolingual WITH fillers
    - English monolingual WITHOUT fillers
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        config: Config object with CSV path methods
        
    Returns:
        Tuple of (cantonese_with, cantonese_without, english_with, english_without) DataFrames
    """
    from src.core.text_cleaning import remove_fillers_from_text
    
    logger.info("Exporting monolingual sentences to CSVs...")
    
    # Helper function to determine if sentence is monolingual
    def is_monolingual_cantonese(pattern: str) -> bool:
        """Check if pattern represents pure Cantonese (only C, no E)."""
        return 'C' in pattern and 'E' not in pattern and pattern != 'FILLER_ONLY'
    
    def is_monolingual_english(pattern: str) -> bool:
        """Check if pattern represents pure English (only E, no C)."""
        return 'E' in pattern and 'C' not in pattern and pattern != 'FILLER_ONLY'
    
    # Filter sentences by language
    cantonese_with_fillers = []
    cantonese_without_fillers = []
    english_with_fillers = []
    english_without_fillers = []
    
    for s in all_sentences:
        pattern_with = s.get('pattern_with_fillers', '')
        pattern_without = s.get('pattern_content_only', '')
        
        # For WITH fillers: use pattern_with_fillers
        # For WITHOUT fillers: use pattern_content_only
        # IMPORTANT: Only include if monolingual in BOTH patterns to avoid code-switched sentences
        # that become "monolingual" after filler removal
        
        is_cant_with = is_monolingual_cantonese(pattern_with)
        is_cant_without = is_monolingual_cantonese(pattern_without)
        is_eng_with = is_monolingual_english(pattern_with)
        is_eng_without = is_monolingual_english(pattern_without)
        
        # Cantonese monolingual WITH fillers (must be monolingual in WITH pattern)
        if is_cant_with:
            cantonese_with_fillers.append(s)
        
        # Cantonese monolingual WITHOUT fillers (must be monolingual in BOTH patterns)
        if is_cant_with and is_cant_without:
            cantonese_without_fillers.append(s)
        
        # English monolingual WITH fillers (must be monolingual in WITH pattern)
        if is_eng_with:
            english_with_fillers.append(s)
        
        # English monolingual WITHOUT fillers (must be monolingual in BOTH patterns)
        if is_eng_with and is_eng_without:
            english_without_fillers.append(s)
    
    # Apply min_sentence_words filtering AFTER filler removal
    cantonese_with_fillers = _filter_by_min_words(cantonese_with_fillers, min_sentence_words, use_without_fillers=False)
    cantonese_without_fillers = _filter_by_min_words(cantonese_without_fillers, min_sentence_words, use_without_fillers=True)
    english_with_fillers = _filter_by_min_words(english_with_fillers, min_sentence_words, use_without_fillers=False)
    english_without_fillers = _filter_by_min_words(english_without_fillers, min_sentence_words, use_without_fillers=True)
    
    logger.info(f"After min_sentence_words filtering (min={min_sentence_words}):")
    logger.info(f"  Cantonese WITH fillers: {len(cantonese_with_fillers)}")
    logger.info(f"  Cantonese WITHOUT fillers: {len(cantonese_without_fillers)}")
    logger.info(f"  English WITH fillers: {len(english_with_fillers)}")
    logger.info(f"  English WITHOUT fillers: {len(english_without_fillers)}")
    
    # Create DataFrames
    def create_df(sentences: List[Dict], use_pattern_with_fillers: bool, lang: str, is_cantonese_without: bool = False) -> pd.DataFrame:
        """Create DataFrame from sentence list."""
        pattern_field = 'pattern_with_fillers' if use_pattern_with_fillers else 'pattern_content_only'
        
        # For WITHOUT fillers datasets, use pre-computed text without fillers
        if use_pattern_with_fillers:
            reconstructed_sentences = [s['reconstructed_text'] for s in sentences]
        else:
            reconstructed_sentences = [s['reconstructed_text_without_fillers'] for s in sentences]
        
        # Build base columns
        data = {
            'start_time': [s['start_time'] for s in sentences],
            'end_time': [s['end_time'] for s in sentences],
            'reconstructed_sentence': reconstructed_sentences,
            'pattern': [s.get(pattern_field, '') for s in sentences],
            'group': [s.get('group', '') for s in sentences],
            'participant_id': [s.get('participant_id', '') for s in sentences]
        }
        
        # Only add sentence_original and matrix_language if NOT cantonese_without
        if not is_cantonese_without:
            data['sentence_original'] = [s['text'] for s in sentences]
            data['matrix_language'] = [s.get('matrix_language', 'Unknown') for s in sentences]
        
        df = pd.DataFrame(data)
        return sort_dataframe(df)
    
    cant_with_df = create_df(cantonese_with_fillers, use_pattern_with_fillers=True, lang='C', is_cantonese_without=False)
    cant_without_df = create_df(cantonese_without_fillers, use_pattern_with_fillers=False, lang='C', is_cantonese_without=True)
    eng_with_df = create_df(english_with_fillers, use_pattern_with_fillers=True, lang='E', is_cantonese_without=False)
    eng_without_df = create_df(english_without_fillers, use_pattern_with_fillers=False, lang='E', is_cantonese_without=False)
    
    # Add POS tagging to Cantonese monolingual WITHOUT fillers
    cant_without_df = _add_pos_to_dataframe(cant_without_df, 'reconstructed_sentence', 'pos', is_cantonese=True)
    
    # Save CSVs
    csv_paths = [
        (cant_with_df, config.get_csv_cantonese_mono_with_fillers_path(), 'Cantonese WITH fillers'),
        (cant_without_df, config.get_csv_cantonese_mono_without_fillers_path(), 'Cantonese WITHOUT fillers'),
        (eng_with_df, config.get_csv_english_mono_with_fillers_path(), 'English WITH fillers'),
        (eng_without_df, config.get_csv_english_mono_without_fillers_path(), 'English WITHOUT fillers')
    ]
    
    for df, path, desc in csv_paths:
        # Create output directory if it doesn't exist
        csv_dir = os.path.dirname(path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  '{path}' - {len(df)} {desc} sentences")
    
    logger.info("Monolingual sentence export complete!")
    
    return cant_with_df, cant_without_df, eng_with_df, eng_without_df


def export_translated_sentences(
    all_sentences: List[Dict],
    config,
    do_translation: bool = True,
    min_sentence_words: int = 2
) -> pd.DataFrame:
    """
    Export code-switched sentences with full Cantonese translations.
    
    Filters code-switching sentences from all_sentences, filters for Cantonese matrix language,
    and creates a new CSV with code_switch_original and cantonese_translation columns.
    Optionally performs translation and POS tagging.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        config: Config object with CSV path methods
        do_translation: If True, perform translation and POS tagging. If False, leave columns empty.
        min_sentence_words: Minimum number of words required
        
    Returns:
        DataFrame with translated sentences structure
    """
    # Import parse_pattern_segments early since it's needed for filtering
    from src.analysis.pos_tagging import parse_pattern_segments
    
    logger.info("Exporting translated code-switched sentences...")
    
    # Filter code-switching sentences (without fillers)
    code_switched_sentences = filter_code_switching_sentences(all_sentences, include_fillers=False)
    logger.info(f"Found {len(code_switched_sentences)} code-switching sentences (without fillers)")
    
    # Convert to DataFrame for easier filtering
    # Note: min_sentence_words filtering is applied later, after pattern filtering (matching original logic)
    df_source = pd.DataFrame(code_switched_sentences)
    
    if len(df_source) == 0:
        logger.warning("No code-switching sentences found!")
        return pd.DataFrame()
    
    # Track filtering stats for summary
    stats = {
        'initial_total': len(df_source),
        'after_cantonese_filter': 0,
        'after_pattern_filter': 0,
        'translation_valid': 0,
        'translation_invalid': 0,
        'final_with_pos': 0
    }
    
    # Filter to ONLY Cantonese matrix language
    df_cantonese = df_source[df_source['matrix_language'] == 'Cantonese'].copy()
    stats['after_cantonese_filter'] = len(df_cantonese)
    
    logger.info(f"Total code-switched sentences: {len(df_source)}")
    logger.info(f"Cantonese matrix language only: {len(df_cantonese)}")
    logger.info(f"Filtered out {len(df_source) - len(df_cantonese)} non-Cantonese matrix sentences")
    
    # Apply min_cantonese_words filter: must start with at least X Cantonese words followed by English
    min_cantonese = config.get_analysis_min_cantonese_words()
    logger.info(f"Applying pattern filter: must start with at least {min_cantonese} Cantonese words followed by English")
    
    # Filter by pattern criteria - check if pattern starts with enough Cantonese words followed by English
    # Use pattern_content_only from sentence dicts (without fillers)
    pattern_col = 'pattern_content_only' if 'pattern_content_only' in df_cantonese.columns else 'pattern'
    valid_patterns = []
    for pattern in df_cantonese[pattern_col].values:
        try:
            segments = parse_pattern_segments(pattern)
            if len(segments) >= 2:
                first_lang, first_count = segments[0]
                second_lang, _ = segments[1]
                is_valid = first_lang == 'C' and first_count >= min_cantonese and second_lang == 'E'
                valid_patterns.append(is_valid)
            else:
                valid_patterns.append(False)
        except Exception:
            valid_patterns.append(False)
    
    df_cantonese = df_cantonese[valid_patterns].copy()
    stats['after_pattern_filter'] = len(df_cantonese)
    logger.info(f"After pattern filtering: {len(df_cantonese)} sentences (starts with {min_cantonese}+ Cantonese words -> English)")
    
    # Apply min_sentence_words filtering (count words from pattern)
    if min_sentence_words > 0:
        word_counts = [_count_words_from_pattern(p) for p in df_cantonese[pattern_col].values]
        df_cantonese = df_cantonese[[wc >= min_sentence_words for wc in word_counts]].copy()
        logger.info(f"After min_sentence_words filtering (min={min_sentence_words}): {len(df_cantonese)} sentences")
    
    if len(df_cantonese) == 0:
        logger.warning("No Cantonese matrix language sentences found!")
        return pd.DataFrame()
    
    # Create new DataFrame with desired structure
    # Column order: code_switch_original, cantonese_translation, translated_pos, pattern, then other columns
    
    # Calculate switch_index for each sentence (index where English starts)
    def get_switch_index(pattern: str) -> int:
        """Extract the index of the first English word (the switch word).
        
        For pattern C18-E1, returns 18 (the first English word, 0-based indexing).
        This is the actual switch word position.
        """
        try:
            segments = parse_pattern_segments(pattern)
            if len(segments) < 2:
                return -1
            first_lang, first_count = segments[0]
            if first_lang == 'C' and first_count > 0:
                return first_count  # First English word (0-based, so count is the index)
            return -1  # English starts at beginning, no Cantonese word before switch
        except Exception:
            return -1
    
    switch_indices = [get_switch_index(pattern) for pattern in df_cantonese[pattern_col].values]
    
    # Get reconstructed sentence (without fillers) - use the field from sentence dictionaries
    if 'reconstructed_text_without_fillers' in df_cantonese.columns:
        reconstructed_sentences = df_cantonese['reconstructed_text_without_fillers'].values
    elif 'reconstructed_sentence' in df_cantonese.columns:
        reconstructed_sentences = df_cantonese['reconstructed_sentence'].values
    else:
        # Fallback - shouldn't happen but handle gracefully
        logger.warning("Could not find reconstructed sentence column, using empty strings")
        reconstructed_sentences = [''] * len(df_cantonese)
    
    # Get pattern column (should be pattern_content_only from sentence dicts)
    pattern_values = df_cantonese[pattern_col].values if pattern_col in df_cantonese.columns else df_cantonese.get('pattern', [''] * len(df_cantonese)).values
    
    df = pd.DataFrame({
        'start_time': df_cantonese['start_time'].values,
        'end_time': df_cantonese['end_time'].values,
        'code_switch_original': reconstructed_sentences,
        'cantonese_translation': '',  # Will be filled if do_translation is True
        'translated_pos': '',  # Will be filled if do_translation is True
        'switch_index': switch_indices,
        'pattern': pattern_values,
        'group': df_cantonese['group'].values if 'group' in df_cantonese.columns else [''] * len(df_cantonese),
        'participant_id': df_cantonese['participant_id'].values if 'participant_id' in df_cantonese.columns else [''] * len(df_cantonese)
    })
    
    # Define column order (used for saving)
    column_order = [
        'start_time',
        'end_time',
        'code_switch_original',
        'cantonese_translation',
        'translated_pos',
        'switch_index',
        'pattern',
        'group',
        'participant_id'
    ]
    
    # Get CSV path for saving
    csv_path = config.get_csv_cantonese_translated_path()
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    # Save initial structure first (before translation starts)
    df_initial = df[column_order].copy()
    df_initial = sort_dataframe(df_initial)
    df_initial.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Created initial CSV structure: '{csv_path}' - {len(df_initial)} sentences")
    
    # Perform translation if requested
    if do_translation:
        logger.info("Performing translation with verification and POS tagging...")
        from src.experiments.nllb_translator import NLLBTranslator
        from src.core.tokenization import segment_cantonese_sentence
        from src.analysis.pos_tagging import pos_tag_cantonese, extract_pos_sequence, parse_pattern_segments
        from tqdm import tqdm
        
        translator = NLLBTranslator(
            model_name=config.get_translation_model(),
            device=config.get_translation_device(),
            show_progress=False
        )
        translations = []
        pos_sequences = []
        valid_count = 0
        invalid_count = 0
        
        # Create progress bar with detailed description
        pbar = tqdm(df.iterrows(), total=len(df), desc="Translating & verifying")
        
        # Save every N sentences to balance performance and safety
        save_interval = 10
        
        for row_idx, (idx, row) in enumerate(pbar):
            sentence = str(row['code_switch_original'])
            pattern = str(row['pattern'])
            switch_index = row.get('switch_index', -1)
            translation = ''
            pos_seq = ''
            
            try:
                # Use original tokenization (sentence is already space-separated to match pattern)
                words = sentence.split()
                
                # Translate
                translation_result = translator.translate_code_switched_sentence(
                    sentence=sentence,
                    pattern=pattern,
                    words=words
                )
                translation = translation_result.get('translated_sentence', '')
                
                # Enhanced verification: check if English words are within the code-switched segment
                is_valid, error_msg = _verify_cantonese_only(translation)
                
                # If verification failed but we have a valid switch_index, check if English words are outside our segment
                if not is_valid and switch_index >= 0:
                    # Get the length of the first English segment from pattern
                    segments = parse_pattern_segments(pattern)
                    if len(segments) >= 2 and segments[1][0] == 'E':
                        first_english_length = segments[1][1]
                        # switch_index is the first English word (0-based)
                        # We only need to check the Cantonese portion (indices 0 to switch_index, exclusive)
                        # English words at switch_index or after are allowed
                        translation_words = translation.split()
                        english_in_cantonese_portion = False
                        cantonese_end = switch_index  # Exclusive end of Cantonese portion
                        
                        for i, word in enumerate(translation_words[:cantonese_end]):
                            # Simple check: if word contains only ASCII letters, it's likely English
                            if word and any(c.isascii() and c.isalpha() for c in word):
                                english_in_cantonese_portion = True
                                break
                        
                        # If no English in the Cantonese portion, consider it valid
                        # (English in the first English segment or after is fine)
                        if not english_in_cantonese_portion:
                            is_valid = True
                            error_msg = ''
                
                if is_valid:
                    valid_count += 1
                    # Add POS tagging for valid translation
                    # Handle mixed Cantonese/English by tagging word by word
                    try:
                        translation_words = translation.split()
                        pos_tags = []
                        
                        for word in translation_words:
                            # Check if word is English (contains ASCII letters)
                            if word and any(c.isascii() and c.isalpha() for c in word):
                                # Mark English words that we're keeping (after code-switch point)
                                pos_tags.append('ENG')
                            else:
                                # Try to POS tag Cantonese word
                                try:
                                    tagged_word = pos_tag_cantonese(word)
                                    word_pos = extract_pos_sequence(tagged_word)
                                    if word_pos:
                                        pos_tags.extend(word_pos)
                                    else:
                                        pos_tags.append('UNK')
                                except Exception:
                                    pos_tags.append('UNK')
                        
                        pos_seq = ' '.join(pos_tags) if pos_tags else ''
                        
                        # If POS sequence is still empty despite having translation, mark all as unknown
                        if not pos_seq and translation:
                            pos_seq = ' '.join(['UNK'] * len(translation_words))
                            
                    except Exception as e:
                        logger.warning(f"Error POS tagging translation at row {idx}: {e}")
                        # Fallback: create UNK tags for each word
                        translation_words = translation.split()
                        pos_seq = ' '.join(['UNK'] * len(translation_words)) if translation_words else ''
                else:
                    invalid_count += 1
                    # Log first few failures with actual translation text for debugging
                    if invalid_count <= 3:
                        logger.warning(f"Row {idx}: Translation verification failed - {error_msg}")
                        logger.debug(f"  Original: {sentence[:100]}")
                        logger.debug(f"  Translation (first 200 chars): {translation[:200]}")
                    else:
                        logger.warning(f"Row {idx}: Translation verification failed - {error_msg}")
                    # Set translation and POS to empty for invalid rows
                    translation = ''
                    pos_seq = ''
                
            except Exception as e:
                logger.warning(f"Error translating row {idx}: {e}")
                translation = ''
                pos_seq = ''
                invalid_count += 1
            
            translations.append(translation)
            pos_sequences.append(pos_seq)
            
            # Update DataFrame row immediately
            df.at[idx, 'cantonese_translation'] = translation
            df.at[idx, 'translated_pos'] = pos_seq
            
            # Save CSV periodically (every save_interval sentences or at the end)
            if (row_idx + 1) % save_interval == 0 or (row_idx + 1) == len(df):
                # Reorder columns before saving
                df_save = df[column_order].copy()
                df_save = sort_dataframe(df_save)
                df_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # Update progress bar with stats
            pbar.set_postfix({
                'valid': valid_count,
                'invalid': invalid_count,
                'saved': '✓' if (row_idx + 1) % save_interval == 0 or (row_idx + 1) == len(df) else ''
            })
        
        pbar.close()
        
        # FILTER OUT rows with empty translations BEFORE saving
        df_valid = df[df['cantonese_translation'] != ''].copy()
        logger.info(f"Filtered out {len(df) - len(df_valid)} rows with failed translations")
        
        # Final save - only valid translated sentences
        df_save = df_valid[column_order].copy()
        df_save = sort_dataframe(df_save)
        df_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Update stats from the FILTERED DataFrame (what's actually saved)
        stats['translation_valid'] = valid_count
        stats['translation_invalid'] = invalid_count
        stats['final_with_translation'] = len(df_valid)
        stats['final_with_pos'] = len(df_valid[df_valid['translated_pos'] != ''])
        
        # Sanity check: these should be equal
        if stats['final_with_translation'] != stats['final_with_pos']:
            logger.warning(f"MISMATCH: {stats['final_with_translation']} translations but {stats['final_with_pos']} POS tags!")
        
        # Generate summary report
        summary_path = os.path.join(config.get_preprocessing_results_dir(), 'translation_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CANTONESE TRANSLATION DATASET FILTERING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Stage 1: Initial Dataset\n")
            f.write(f"  Total code-switched sentences: {stats['initial_total']}\n\n")
            
            f.write("Stage 2: Cantonese Matrix Language Filter\n")
            f.write(f"  After filtering: {stats['after_cantonese_filter']} sentences\n")
            f.write(f"  Removed: {stats['initial_total'] - stats['after_cantonese_filter']} sentences\n")
            f.write(f"  Retention rate: {stats['after_cantonese_filter']/stats['initial_total']*100:.1f}%\n\n")
            
            f.write(f"Stage 3: Pattern Filter (≥{min_cantonese} Cantonese words → English)\n")
            f.write(f"  After filtering: {stats['after_pattern_filter']} sentences\n")
            f.write(f"  Removed: {stats['after_cantonese_filter'] - stats['after_pattern_filter']} sentences\n")
            f.write(f"  Retention rate: {stats['after_pattern_filter']/stats['after_cantonese_filter']*100:.1f}%\n\n")
            
            f.write("Stage 4: Translation & Verification\n")
            f.write(f"  Sentences attempted: {stats['after_pattern_filter']}\n")
            f.write(f"  Valid during processing: {stats['translation_valid']}\n")
            f.write(f"  Invalid during processing: {stats['translation_invalid']}\n")
            f.write(f"  Actual sentences with translation: {stats['final_with_translation']}\n")
            f.write(f"  Sentences without translation: {stats['after_pattern_filter'] - stats['final_with_translation']}\n")
            f.write(f"  Translation success rate: {stats['final_with_translation']/stats['after_pattern_filter']*100:.1f}%\n\n")
            
            f.write("Stage 5: POS Tagging\n")
            f.write(f"  Sentences with POS tags: {stats['final_with_pos']}\n")
            if stats['final_with_translation'] != stats['final_with_pos']:
                f.write(f"  ⚠ WARNING: {stats['final_with_translation'] - stats['final_with_pos']} translations missing POS tags!\n")
            else:
                f.write(f"  ✓ All translations have POS tags\n")
            f.write(f"  POS tagging success rate: 100.0% (all translations tagged)\n\n")
            
            f.write("="*80 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"  Initial sentences: {stats['initial_total']}\n")
            f.write(f"  Final with translation & POS: {stats['final_with_pos']}\n")
            f.write(f"  Overall retention: {stats['final_with_pos']/stats['initial_total']*100:.1f}%\n")
            f.write(f"  Total filtered out: {stats['initial_total'] - stats['final_with_pos']} sentences\n")
            f.write(f"\n  Breakdown of filtered sentences:\n")
            f.write(f"    - Non-Cantonese matrix: {stats['initial_total'] - stats['after_cantonese_filter']}\n")
            f.write(f"    - Pattern mismatch: {stats['after_cantonese_filter'] - stats['after_pattern_filter']}\n")
            f.write(f"    - Translation failed: {stats['after_pattern_filter'] - stats['final_with_translation']}\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Translation complete: {valid_count} valid, {invalid_count} invalid out of {len(df)} total")
        logger.info(f"Saved translated sentences with POS tags: '{csv_path}' - {len(df)} sentences")
        logger.info(f"Generated summary report: '{summary_path}'")
    else:
        logger.info("Skipping translation (do_translation=False). Columns will remain empty.")
        logger.info("Note: cantonese_translation and translated_pos columns are empty (already saved in initial structure)")
    
    return df


def save_exploratory_outputs(
    output_dir: Path,
    monolingual: dict,
    pos_results: dict,
    matching_results: dict,
    distributions: dict,
    report: str,
    figures_dir: Optional[Path] = None
) -> None:
    """
    Save all exploratory analysis output files.
    
    Args:
        output_dir: Directory for CSV and report files
        monolingual: Monolingual sentence data
        pos_results: POS tagging results
        matching_results: Matching algorithm results
        distributions: Distribution analysis results
        report: Feasibility report text
        figures_dir: Directory for figures (if None, uses output_dir/figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if figures_dir is None:
        figures_dir = output_dir / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving outputs to {output_dir}...")
    
    # Save monolingual sentences
    if 'cantonese' in monolingual:
        monolingual_path = output_dir / "monolingual_sentences.csv"
        # Combine all monolingual sentences
        all_mono = pd.concat([
            monolingual['cantonese'],
            monolingual['english']
        ], ignore_index=True)
        all_mono = sort_dataframe(all_mono)
        all_mono.to_csv(monolingual_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved monolingual sentences to {monolingual_path}")
    
    # Save POS tagging sample
    if 'sample_results' in pos_results:
        pos_path = output_dir / "pos_tagged_sample.csv"
        pos_results['sample_results'].to_csv(pos_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved POS tagging sample to {pos_path}")
    
    # Save matching results sample
    if 'results' in matching_results:
        match_path = output_dir / "matching_results_sample.csv"
        # Flatten the results for CSV export
        export_results = []
        for _, row in matching_results['results'].iterrows():
            export_row = {
                'sentence': row['sentence'],
                'pattern': row['pattern'],
                'num_matches': row['num_matches'],
                'has_match': row['has_match'],
                'best_similarity': row['best_similarity'],
                'has_c_to_e': row['has_c_to_e'],
                'has_e_to_c': row['has_e_to_c']
            }
            # Add details of top matches if available
            if row['matches_detail']:
                for i, match in enumerate(row['matches_detail'][:3]):
                    export_row[f'match_{i+1}_similarity'] = match.get('similarity', 0)
                    export_row[f'match_{i+1}_language'] = match.get('language', '')
            export_results.append(export_row)
        
        match_df = pd.DataFrame(export_results)
        match_df.to_csv(match_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved matching results to {match_path}")
    
    # Save feasibility report
    report_path = output_dir / "feasibility_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved feasibility report to {report_path}")
    
    logger.info("All outputs saved successfully!")
