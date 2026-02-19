"""
Data export functionality for code-switching analysis.

This module handles exporting processed data to CSV files with filtering
for code-switching sentences.
"""

import pandas as pd
from typing import List, Dict, Tuple
import logging
import os

from src.utils.text_validation import verify_cantonese_only, contains_english_words, is_english_word
from src.utils.data_helpers import count_words_from_pattern, filter_by_min_words, sort_dataframe, find_switch_points

logger = logging.getLogger(__name__)


def filter_code_switching_sentences(sentences: List[Dict]) -> List[Dict]:
    """
    Filter sentences to only keep those with actual code-switching.
    
    Args:
        sentences: List of sentence data dictionaries
        
    Returns:
        Filtered list of sentences with code-switching
    """
    
    return [
        s for s in sentences
        if (pattern := s.get('pattern', '')) and 'C' in pattern and 'E' in pattern and pattern != 'FILLER_ONLY'
    ]


def export_all_sentences_to_csv(
    all_sentences: List[Dict],
    csv_all_sentences_path: str,
    min_sentence_words: int = 2
) -> Tuple[pd.DataFrame, Dict]:
    """
    Export ALL sentences (both monolingual and code-switched) to CSV.
    
    This includes sentences that were filtered out in the code-switching analysis.
    Useful for exploratory analysis that needs monolingual sentences.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        csv_all_sentences_path: Output path for CSV with all sentences
        min_sentence_words: Minimum number of words (after filler removal)
        
    Returns:
        Tuple of (DataFrame with all sentences, statistics dictionary)
    """

    logger.info(f"Exporting ALL sentences (monolingual + code-switched) to CSV...")
    
    # First, remove FILLER_ONLY sentences (sentences with no content after filler removal)
    total_sentences = len(all_sentences)
    sentences_without_filler_only = [s for s in all_sentences if s.get('pattern', '') != 'FILLER_ONLY']
    filler_only_count = total_sentences - len(sentences_without_filler_only)
    
    logger.info(f"Removed {filler_only_count} filler-only sentences (no content)")
    
    # Calculate total filler words removed across all sentences (excluding FILLER_ONLY)
    total_filler_words = sum(s.get('filler_count', 0) for s in sentences_without_filler_only)
    sentences_with_fillers = sum(1 for s in sentences_without_filler_only if s.get('has_fillers', False))
    
    logger.info(f"Filler removal: {total_filler_words} filler words removed from {sentences_with_fillers} sentences")
    
    # Filter by min_sentence_words AFTER filler removal
    filtered_sentences = filter_by_min_words(sentences_without_filler_only, min_sentence_words)
    logger.info(f"Filtered to {len(filtered_sentences)} sentences with at least {min_sentence_words} words (after filler removal)")
    
    # Create DataFrame with all sentences
    csv_all_data = []
    
    for s in filtered_sentences:
        reconstructed = s.get('reconstructed_text_without_fillers', s.get('reconstructed_text', ''))
        
        csv_all_data.append({
            'start_time': s['start_time'],
            'end_time': s['end_time'],
            'reconstructed_sentence': reconstructed,
            'sentence_original': s['text'],
            'pattern': s.get('pattern', ''),
            'matrix_language': s.get('matrix_language', 'Unknown'),
            'group': s.get('group', ''),
            'participant_id': s.get('participant_id', '')
        })
    
    csv_all = pd.DataFrame(csv_all_data)
    csv_all = sort_dataframe(csv_all)
    
    # Create output directory if it doesn't exist
    csv_dir = os.path.dirname(csv_all_sentences_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    # Save CSV
    csv_all.to_csv(csv_all_sentences_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved all sentences dataset:")
    logger.info(f"  '{csv_all_sentences_path}' - {len(csv_all)} sentences")
    
    # Calculate statistics
    stats = {
        'total_processed': len(all_sentences),
        'filler_only_removed': filler_only_count,
        'after_filler_only_removal': len(sentences_without_filler_only),
        'total_filler_words_removed': total_filler_words,
        'sentences_with_fillers': sentences_with_fillers,
        'after_min_words_filter': len(filtered_sentences),
        'filtered_out': len(sentences_without_filler_only) - len(filtered_sentences),
        'code_switched': sum(1 for s in filtered_sentences if 'C' in s.get('pattern', '') and 'E' in s.get('pattern', '')),
        'monolingual_cantonese': sum(1 for s in filtered_sentences if 'C' in s.get('pattern', '') and 'E' not in s.get('pattern', '')),
        'monolingual_english': sum(1 for s in filtered_sentences if 'E' in s.get('pattern', '') and 'C' not in s.get('pattern', '')),
        'cantonese_matrix': sum(1 for s in filtered_sentences if s.get('matrix_language') == 'Cantonese'),
        'english_matrix': sum(1 for s in filtered_sentences if s.get('matrix_language') == 'English'),
        'equal_matrix': sum(1 for s in filtered_sentences if s.get('matrix_language') == 'Equal')
    }
    
    return csv_all, stats


def _add_pos_to_dataframe(df: pd.DataFrame, sentence_col: str, pos_col: str) -> pd.DataFrame:
    """
    Add Cantonese POS tagging column to a DataFrame.
    
    Args:
        df: DataFrame to add POS column to
        sentence_col: Name of column containing sentences
        pos_col: Name of column to create for POS tags
        
    Returns:
        DataFrame with POS column added
    """
    from src.analysis.pos_tagging import pos_tag_cantonese, extract_pos_sequence
    
    logger.info(f"Adding POS tagging to {pos_col} column...")
    
    pos_sequences = []
    for idx, row in df.iterrows():
        sentence = str(row.get(sentence_col, ''))
        if not sentence or pd.isna(sentence):
            pos_sequences.append('')
            continue
        
        try:
            tagged = pos_tag_cantonese(sentence)
            pos_seq = extract_pos_sequence(tagged)
            pos_sequences.append(' '.join(pos_seq) if pos_seq else '')
        except Exception as e:
            logger.warning(f"Error tagging sentence at row {idx}: {e}")
            pos_sequences.append('')
    
    df = df.copy()
    df[pos_col] = pos_sequences
    return df


def export_interviewer_sentences(
    interviewer_sentences: List[Dict],
    csv_interviewer_path: str,
    min_sentence_words: int = 2
) -> Tuple[pd.DataFrame, Dict]:
    """
    Export interviewer (IR tier) sentences to CSV.
    
    This function processes interviewer sentences extracted from the IR tier,
    applying the same cleaning and filtering as participant sentences.
    
    Args:
        interviewer_sentences: List of interviewer sentence data dictionaries
        csv_interviewer_path: Output path for interviewer CSV
        min_sentence_words: Minimum number of words (after filler removal)
        
    Returns:
        Tuple of (DataFrame with interviewer sentences, statistics dictionary)
    """
    
    logger.info(f"Exporting interviewer sentences to CSV...")
    
    # Remove FILLER_ONLY sentences
    total_sentences = len(interviewer_sentences)
    sentences_without_filler_only = [s for s in interviewer_sentences if s.get('pattern', '') != 'FILLER_ONLY']
    filler_only_count = total_sentences - len(sentences_without_filler_only)
    
    logger.info(f"Removed {filler_only_count} filler-only interviewer sentences")
    
    # Calculate total filler words removed
    total_filler_words = sum(s.get('filler_count', 0) for s in sentences_without_filler_only)
    sentences_with_fillers = sum(1 for s in sentences_without_filler_only if s.get('has_fillers', False))
    
    logger.info(f"Filler removal: {total_filler_words} filler words removed from {sentences_with_fillers} interviewer sentences")
    
    # Filter by min_sentence_words AFTER filler removal
    filtered_sentences = filter_by_min_words(sentences_without_filler_only, min_sentence_words)
    logger.info(f"Filtered to {len(filtered_sentences)} interviewer sentences with at least {min_sentence_words} words")
    
    # Create DataFrame with interviewer sentences
    csv_data = []
    
    for s in filtered_sentences:
        reconstructed = s.get('reconstructed_text_without_fillers', s.get('reconstructed_text', ''))
        
        csv_data.append({
            'start_time': s['start_time'],
            'end_time': s['end_time'],
            'reconstructed_sentence': reconstructed,
            'sentence_original': s['text'],
            'pattern': s.get('pattern', ''),
            'participant_id': s.get('participant_id', '')
        })
    
    df = pd.DataFrame(csv_data)
    df = sort_dataframe(df)
    
    # Create output directory if it doesn't exist
    csv_dir = os.path.dirname(csv_interviewer_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    # Save CSV
    df.to_csv(csv_interviewer_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved interviewer dataset:")
    logger.info(f"  '{csv_interviewer_path}' - {len(df)} sentences")
    
    # Calculate statistics
    stats = {
        'total_processed': len(interviewer_sentences),
        'filler_only_removed': filler_only_count,
        'after_filler_only_removal': len(sentences_without_filler_only),
        'total_filler_words_removed': total_filler_words,
        'sentences_with_fillers': sentences_with_fillers,
        'after_min_words_filter': len(filtered_sentences),
        'filtered_out': len(sentences_without_filler_only) - len(filtered_sentences),
    }
    
    return df, stats


def export_cantonese_monolingual(
    all_sentences: List[Dict],
    config,
    min_sentence_words: int = 2
) -> Tuple[pd.DataFrame, Dict]:
    """
    Export only Cantonese monolingual sentences WITHOUT fillers.
    
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        config: Config object with CSV path methods
        min_sentence_words: Minimum number of words required
        
    Returns:
        Tuple of (DataFrame with Cantonese monolingual sentences, statistics dictionary)
    """

    logger.info("Exporting Cantonese monolingual sentences (WITHOUT fillers)...")
    
    # Helper function to determine if sentence is monolingual Cantonese
    def is_monolingual_cantonese(pattern: str) -> bool:
        """Check if pattern represents pure Cantonese (only C, no E)."""
        return 'C' in pattern and 'E' not in pattern and pattern != 'FILLER_ONLY'
    
    # Filter sentences - must be monolingual in BOTH patterns
    cantonese_sentences = []
    
    for s in all_sentences:
        pattern = s.get('pattern', '')
        
        # Must be monolingual Cantonese (pattern is already without fillers)
        if is_monolingual_cantonese(pattern):
            cantonese_sentences.append(s)
    
    # Apply min_sentence_words filtering AFTER filler removal
    cantonese_sentences = filter_by_min_words(cantonese_sentences, min_sentence_words)
    
    logger.info(f"Found {len(cantonese_sentences)} Cantonese monolingual sentences (after filtering)")
    
    # Create DataFrame
    df = pd.DataFrame({
        'start_time': [s['start_time'] for s in cantonese_sentences],
        'end_time': [s['end_time'] for s in cantonese_sentences],
        'reconstructed_sentence': [s['reconstructed_text_without_fillers'] for s in cantonese_sentences],
        'pattern': [s.get('pattern', '') for s in cantonese_sentences],
        'group': [s.get('group', '') for s in cantonese_sentences],
        'participant_id': [s.get('participant_id', '') for s in cantonese_sentences]
    })
    
    df = sort_dataframe(df)
    
    # Add POS tagging
    df = _add_pos_to_dataframe(df, 'reconstructed_sentence', 'pos')
    
    # Save CSV
    csv_path = config.get_csv_cantonese_mono_without_fillers_path()
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved '{csv_path}' - {len(df)} Cantonese monolingual sentences")
    
    # Calculate statistics
    stats = {
        'total_cantonese_monolingual_before_filter': len([s for s in all_sentences if 'C' in s.get('pattern', '') and 'E' not in s.get('pattern', '') and s.get('pattern', '') != 'FILLER_ONLY']),
        'after_min_words_filter': len(cantonese_sentences),
        'filtered_out': len([s for s in all_sentences if 'C' in s.get('pattern', '') and 'E' not in s.get('pattern', '') and s.get('pattern', '') != 'FILLER_ONLY']) - len(cantonese_sentences)
    }
    
    return df, stats


def export_translated_sentences(
    all_sentences: List[Dict],
    config,
    do_translation: bool = True,
    min_sentence_words: int = 2
) -> Tuple[pd.DataFrame, Dict]:
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
        Tuple of (DataFrame with translated sentences structure, statistics dictionary)
    """

    from src.utils.data_helpers import parse_pattern_segments
    
    logger.info("Exporting translated code-switched sentences...")
    
    # Filter code-switching sentences
    code_switched_sentences = filter_code_switching_sentences(all_sentences)
    logger.info(f"Found {len(code_switched_sentences)} code-switching sentences")
    
    df_source = pd.DataFrame(code_switched_sentences)
    
    assert len(df_source) != 0, "No code switched sentences found..."
    
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
    pattern_col = 'pattern'
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
        word_counts = [count_words_from_pattern(p) for p in df_cantonese[pattern_col].values]
        df_cantonese = df_cantonese[[wc >= min_sentence_words for wc in word_counts]].copy()
        logger.info(f"After min_sentence_words filtering (min={min_sentence_words}): {len(df_cantonese)} sentences")
    
    if len(df_cantonese) == 0:
        logger.warning("No Cantonese matrix language sentences found!")
        return pd.DataFrame(), stats
    
    # Create new DataFrame with
    # Column order: code_switch_original, cantonese_translation, translated_pos, pattern, then other columns
    
    # Get switch indices using find_switch_points (first switch point for each pattern)
    switch_indices = []
    for pattern in df_cantonese[pattern_col].values:
        switch_points = find_switch_points(pattern)
        switch_indices.append(switch_points[0] if switch_points else -1)
    
    # Get reconstructed sentence (without fillers) - use the field from sentence dictionaries
    if 'reconstructed_text_without_fillers' in df_cantonese.columns:
        reconstructed_sentences = df_cantonese['reconstructed_text_without_fillers'].values
    else:
        raise ValueError("No reconstructed sentences without fillers?")
    
    # Get pattern column (pattern is always without fillers now)
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
        from src.analysis.pos_tagging import pos_tag_cantonese, extract_pos_sequence
        from src.utils.data_helpers import parse_pattern_segments
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
        invalid_records = []  # Track invalid translations for separate CSV
        pos_success_count = 0  # Track successful POS tagging
        invalid_pos_records = []  # Track sentences with fallback POS tags
        
        # Word-level statistics
        total_words = 0
        words_with_real_pos = 0
        words_with_fallback_pos = 0
        
        # Create progress bar with detailed description
        pbar = tqdm(df.iterrows(), total=len(df), desc="Translating & verifying")
        
        # Save every N sentences
        save_interval = 10
        
        for row_idx, (idx, row) in enumerate(pbar):
            sentence = str(row['code_switch_original'])
            pattern = str(row['pattern'])
            switch_index = row.get('switch_index', -1)
            translation = ''
            pos_seq = ''
            
            try:
                # Sentence is already space-separated
                words = sentence.split()
                
                # Translate
                translation_result = translator.translate_code_switched_sentence(sentence=sentence, pattern=pattern, words=words)
                translation = translation_result.get('translated_sentence', '')
                
                # Enhanced verification: check if English words are within the code-switched segment
                is_valid, error_msg = verify_cantonese_only(translation)
                
                # If verification failed check if failures are outside our segment
                if not is_valid and switch_index >= 0:
                    # Get the length of the first English segment from pattern
                    segments = parse_pattern_segments(pattern)
                    if len(segments) >= 2 and segments[1][0] == 'E':
                        first_english_length = segments[1][1]
                        # switch_index is the first English word (0-based)
                        # We only need to check the Cantonese portion (indices 0 to switch_index, exclusive)
                        # English words at switch_index or after are allowed
                        translation_words = translation.split()
                        cantonese_end = switch_index  # Exclusive end of Cantonese portion
                        
                        # Check if any english words left in "should-be" cantonese section (0 until first english word - exclusive)
                        cantonese_portion = ' '.join(translation_words[:cantonese_end])
                        has_english, english_words = contains_english_words(cantonese_portion)
                        
                        # If no English in the Cantonese portion, consider it valid
                        # (English in the first English segment or after is fine)
                        if not has_english:
                            is_valid = True
                            error_msg = ''
                
                if is_valid:
                    # Truncate translation at first error after switch_index
                    translation_words = translation.split()
                    
                    # Calculate where the code-switch segment ends (switch_index + length of first English segment)
                    segments = parse_pattern_segments(pattern)
                    first_english_length = 0
                    if len(segments) >= 2 and segments[1][0] == 'E':
                        first_english_length = segments[1][1]
                    
                    # Expected end of code-switch segment (exclusive)
                    cs_segment_end = switch_index + first_english_length
                    
                    # Find first problematic word after switch_index
                    truncate_at = None
                    for i in range(switch_index, len(translation_words)):
                        word = translation_words[i]
                        is_unknown = word in ['UNKNOWN', 'UNK', '']
                        is_english = is_english_word(word)
                        
                        if is_unknown or is_english:
                            truncate_at = i
                            break
                    
                    # Check if truncation would cut into the critical code-switch segment
                    if truncate_at is not None and truncate_at < cs_segment_end:
                        # Critical segment would be truncated - reject this sentence
                        is_valid = False
                        error_msg = f"Translation has errors in critical code-switch segment (truncate at word {truncate_at}, need {cs_segment_end} words). Switch point or code-switch segment corrupted."
                        # Note: invalid_count increment and invalid_records append handled in else block below
                    elif truncate_at is not None:
                        # Truncation is safe (after code-switch segment) - apply it
                        translation_words = translation_words[:truncate_at]
                        translation = ' '.join(translation_words)
                        logger.info(f"Truncated translation at word {truncate_at} (after code-switch segment ends at {cs_segment_end})")
                    
                    # Only continue validation if still valid after truncation check
                    if is_valid:
                        # Validate switch word itself is valid Cantonese
                        if switch_index >= len(translation.split()):
                            is_valid = False
                            error_msg = f"Switch index ({switch_index}) out of bounds after truncation (translation has {len(translation.split())} words)"


                if is_valid:
                    valid_count += 1
                    
                    # Add POS tagging for translation
                    try:
                        # POS tag the entire translation at once
                        tagged_translation = pos_tag_cantonese(translation)
                        pos_tags = extract_pos_sequence(tagged_translation)
                        pos_seq = ' '.join(pos_tags) if pos_tags else ''
                        
                        # Get segmented words from tagged output
                        segmented_words = [word for word, _ in tagged_translation]
                        
                        # Track word count for this sentence
                        total_words += len(segmented_words)
                        
                        # Check for fallback tags (X, UNK, ENG)
                        fallback_tags = [(word, tag) for word, tag in tagged_translation if tag in ['X', 'UNK', 'ENG']]
                        has_fallback = len(fallback_tags) > 0
                        
                        # Update word-level statistics
                        if has_fallback:
                            words_with_fallback_pos += len(fallback_tags)
                            words_with_real_pos += len(pos_tags) - len(fallback_tags)
                        else:
                            words_with_real_pos += len(pos_tags)
                        
                        # Track sentence-level POS success
                        if has_fallback:
                            invalid_pos_records.append({
                                'original_sentence': sentence,
                                'cantonese_translation': translation,
                                'pattern': pattern,
                                'fallback_words': ', '.join([f"{word} ({tag})" for word, tag in fallback_tags]),
                                'pos_sequence': pos_seq,
                                'group': row.get('group', ''),
                                'participant_id': row.get('participant_id', '')
                            })
                        else:
                            pos_success_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error POS tagging translation at row {idx}: {e}")
                        # Fallback: create UNK tags for each word
                        translation_words = translation.split()
                        num_words = len(translation_words)
                        pos_seq = ' '.join(['UNK'] * num_words) if translation_words else ''
                        
                        # Track all words as fallback
                        total_words += num_words
                        words_with_fallback_pos += num_words
                        
                        # Add to invalid POS records
                        if translation_words:
                            invalid_pos_records.append({
                                'original_sentence': sentence,
                                'cantonese_translation': translation,
                                'pattern': pattern,
                                'fallback_words': ', '.join([f"{word} (UNK)" for word in translation_words]),
                                'pos_sequence': pos_seq,
                                'group': row.get('group', ''),
                                'participant_id': row.get('participant_id', '')
                            })
                else:
                    invalid_count += 1
                    # Track invalid translation for reporting
                    invalid_records.append({
                        'original_sentence': sentence,
                        'pattern': pattern,
                        'switch_index': switch_index,
                        'attempted_translation': translation if translation else '(failed)',
                        'error_reason': error_msg if error_msg else 'Validation failed',
                        'group': row.get('group', ''),
                        'participant_id': row.get('participant_id', '')
                    })
                    # Set translation and POS to empty for invalid rows
                    translation = ''
                    pos_seq = ''
                
            except Exception as e:
                logger.warning(f"Error translating row {idx}: {e}")
                # Track exception-based failures
                invalid_records.append({
                    'original_sentence': sentence,
                    'pattern': pattern,
                    'switch_index': switch_index,
                    'attempted_translation': '(exception)',
                    'error_reason': f'Translation exception: {str(e)}',
                    'group': row.get('group', ''),
                    'participant_id': row.get('participant_id', '')
                })
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
        
        # Save invalid translations to separate CSV
        if invalid_records:
            invalid_df = pd.DataFrame(invalid_records)
            invalid_csv_path = os.path.join(config.get_preprocessing_results_dir(), 'invalid_translations.csv')
            invalid_df.to_csv(invalid_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved invalid translations report: '{invalid_csv_path}' - {len(invalid_records)} failed translations")
        
        # Save invalid POS tags to separate CSV
        if invalid_pos_records:
            invalid_pos_df = pd.DataFrame(invalid_pos_records)
            invalid_pos_csv_path = os.path.join(config.get_preprocessing_results_dir(), 'invalid_pos.csv')
            invalid_pos_df.to_csv(invalid_pos_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved invalid POS tags report: '{invalid_pos_csv_path}' - {len(invalid_pos_records)} sentences with fallback tags")
        
        # FILTER OUT rows with empty translations BEFORE saving
        df_valid = df[df['cantonese_translation'] != ''].copy()
        logger.info(f"Filtered out {len(df) - len(df_valid)} rows with failed translations")
        
        # Final save - only valid translated sentences
        df_save = df_valid[column_order].copy()
        df_save = sort_dataframe(df_save)
        df_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Update stats from the FILTERED DataFrame
        stats['translation_valid'] = valid_count
        stats['translation_invalid'] = invalid_count
        stats['final_with_translation'] = len(df_valid)
        stats['final_with_pos'] = pos_success_count  # Only count real POS tags
        stats['final_with_any_pos'] = len(df_valid[df_valid['translated_pos'] != ''])  # Includes UNK/ENG fallbacks
        
        # Word-level statistics
        stats['total_words'] = total_words
        stats['words_with_real_pos'] = words_with_real_pos
        stats['words_with_fallback_pos'] = words_with_fallback_pos
        
        if stats['final_with_translation'] != stats['final_with_pos']:
            word_fallback_pct = (stats['words_with_fallback_pos'] / stats['total_words'] * 100) if stats['total_words'] > 0 else 0
            logger.warning(f"POS tagging (sentence-level): {stats['final_with_pos']}/{stats['final_with_translation']} got real tags, "
                          f"{stats['final_with_any_pos'] - stats['final_with_pos']} used fallbacks (X/UNK/ENG)")
            logger.warning(f"POS tagging (word-level): {stats['words_with_fallback_pos']}/{stats['total_words']} words "
                          f"with fallback tags ({word_fallback_pct:.1f}%)")
        
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
            f.write(f"  Sentences with real POS tags: {stats['final_with_pos']}\n")
            f.write(f"  Sentences with fallback tags (X/UNK/ENG): {stats.get('final_with_any_pos', 0) - stats['final_with_pos']}\n")
            f.write(f"  Total with any POS data: {stats.get('final_with_any_pos', 0)}\n")
            if stats['final_with_translation'] != stats['final_with_pos']:
                f.write(f"  ⚠ NOTE: {stats['final_with_translation'] - stats['final_with_pos']} translations have fallback/unknown POS tags\n")
            else:
                f.write(f"  ✓ All translations have real POS tags\n")
            
            pos_success_rate = (stats['final_with_pos'] / stats['final_with_translation'] * 100) if stats['final_with_translation'] > 0 else 0
            f.write(f"  Real POS tagging success rate (sentence-level): {pos_success_rate:.1f}%\n\n")
            
            f.write("Stage 5b: Word-Level POS Tagging Statistics\n")
            f.write(f"  Total words in translations: {stats.get('total_words', 0)}\n")
            f.write(f"  Words with real POS tags: {stats.get('words_with_real_pos', 0)}\n")
            f.write(f"  Words with fallback POS tags (X/UNK/ENG): {stats.get('words_with_fallback_pos', 0)}\n")
            
            word_pos_success_rate = (stats.get('words_with_real_pos', 0) / stats.get('total_words', 1) * 100) if stats.get('total_words', 0) > 0 else 0
            f.write(f"  Word-level POS success rate: {word_pos_success_rate:.1f}%\n\n")
            
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
        logger.info(f"POS tagging (sentence-level): {pos_success_count} with real tags, {stats.get('final_with_any_pos', 0) - pos_success_count} with fallbacks")
        total_words = stats.get('total_words', 0)
        if total_words > 0:
            logger.info(f"POS tagging (word-level): {stats.get('words_with_real_pos', 0)}/{total_words} words with real tags ({(stats.get('words_with_real_pos', 0)/total_words*100):.1f}%)")
        else:
            logger.info(f"POS tagging (word-level): No words processed")
        logger.info(f"Saved translated sentences: '{csv_path}' - {len(df_valid)} sentences")
        logger.info(f"Generated summary report: '{summary_path}'")
        
        # Add final stats
        stats['final_with_translation'] = len(df_valid)
        stats['final_with_pos'] = stats['final_with_pos']
        stats['min_cantonese_words'] = min_cantonese
    else:
        logger.info("Skipping translation (do_translation=False). Columns will remain empty.")
        logger.info("Note: cantonese_translation and translated_pos columns are empty (already saved in initial structure)")
        # Return empty stats if translation was skipped
        stats = {
            'initial_total': 0,
            'after_cantonese_filter': 0,
            'after_pattern_filter': 0,
            'translation_valid': 0,
            'translation_invalid': 0,
            'final_with_translation': 0,
            'final_with_pos': 0,
            'final_with_any_pos': 0,
            'total_words': 0,
            'words_with_real_pos': 0,
            'words_with_fallback_pos': 0,
            'min_cantonese_words': config.get_analysis_min_cantonese_words() if hasattr(config, 'get_analysis_min_cantonese_words') else 0
        }
    
    return df, stats


def generate_preprocessing_report(
    preprocessing_stats: Dict,
    monolingual_stats: Dict,
    translation_stats: Dict,
    output_path: str,
    interviewer_stats: Dict = None
) -> None:
    """
    Generate a CSV report summarizing preprocessing and translation statistics.
    
    Args:
        preprocessing_stats: Statistics from export_all_sentences_to_csv
        monolingual_stats: Statistics from export_cantonese_monolingual
        translation_stats: Statistics from export_translated_sentences
        output_path: Path to save the CSV report
        interviewer_stats: Statistics from export_interviewer_sentences (optional)
    """
    report_data = []
    
    # Section 1: Preprocessing Statistics
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Total sentences processed',
        'value': preprocessing_stats.get('total_processed', 0),
        'details': 'From all EAF files'
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Filler-only sentences removed',
        'value': preprocessing_stats.get('filler_only_removed', 0),
        'details': 'Sentences with no content after filler removal'
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'After filler-only removal',
        'value': preprocessing_stats.get('after_filler_only_removal', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Total filler words removed',
        'value': preprocessing_stats.get('total_filler_words_removed', 0),
        'details': f"From {preprocessing_stats.get('sentences_with_fillers', 0)} sentences"
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'After min_words filter',
        'value': preprocessing_stats.get('after_min_words_filter', 0),
        'details': f"Filtered out: {preprocessing_stats.get('filtered_out', 0)}"
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Code-switched sentences',
        'value': preprocessing_stats.get('code_switched', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Monolingual Cantonese',
        'value': preprocessing_stats.get('monolingual_cantonese', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Monolingual English',
        'value': preprocessing_stats.get('monolingual_english', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Cantonese matrix language',
        'value': preprocessing_stats.get('cantonese_matrix', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'English matrix language',
        'value': preprocessing_stats.get('english_matrix', 0),
        'details': ''
    })
    
    report_data.append({
        'section': 'Preprocessing',
        'metric': 'Equal matrix language',
        'value': preprocessing_stats.get('equal_matrix', 0),
        'details': ''
    })
    
    if interviewer_stats:
        report_data.append({
            'section': 'Interviewer',
            'metric': 'Total interviewer sentences processed',
            'value': interviewer_stats.get('total_processed', 0),
            'details': 'From IR tier'
        })
        
        report_data.append({
            'section': 'Interviewer',
            'metric': 'Filler-only sentences removed',
            'value': interviewer_stats.get('filler_only_removed', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Interviewer',
            'metric': 'After filler-only removal',
            'value': interviewer_stats.get('after_filler_only_removal', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Interviewer',
            'metric': 'Total filler words removed',
            'value': interviewer_stats.get('total_filler_words_removed', 0),
            'details': f"From {interviewer_stats.get('sentences_with_fillers', 0)} sentences"
        })
        
        report_data.append({
            'section': 'Interviewer',
            'metric': 'After min_words filter',
            'value': interviewer_stats.get('after_min_words_filter', 0),
            'details': f"Filtered out: {interviewer_stats.get('filtered_out', 0)}"
        })
    
    # Section 2: Translation Statistics
    if translation_stats.get('initial_total', 0) > 0:
        report_data.append({
            'section': 'Translation',
            'metric': 'Initial code-switched sentences',
            'value': translation_stats.get('initial_total', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'After Cantonese matrix filter',
            'value': translation_stats.get('after_cantonese_filter', 0),
            'details': f"Removed: {translation_stats.get('initial_total', 0) - translation_stats.get('after_cantonese_filter', 0)}"
        })
        
        min_cantonese = translation_stats.get('min_cantonese_words', 0)
        report_data.append({
            'section': 'Translation',
            'metric': f'After pattern filter (≥{min_cantonese} C words → E)',
            'value': translation_stats.get('after_pattern_filter', 0),
            'details': f"Removed: {translation_stats.get('after_cantonese_filter', 0) - translation_stats.get('after_pattern_filter', 0)}"
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Translation attempts',
            'value': translation_stats.get('after_pattern_filter', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Valid translations',
            'value': translation_stats.get('translation_valid', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Invalid translations',
            'value': translation_stats.get('translation_invalid', 0),
            'details': ''
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Final with translation',
            'value': translation_stats.get('final_with_translation', 0),
            'details': ''
        })
        
        # Calculate rates
        initial = translation_stats.get('initial_total', 0)
        after_cant = translation_stats.get('after_cantonese_filter', 0)
        after_pattern = translation_stats.get('after_pattern_filter', 0)
        final = translation_stats.get('final_with_translation', 0)
        
        if initial > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Cantonese matrix retention rate (%)',
                'value': round(after_cant / initial * 100, 2) if initial > 0 else 0,
                'details': ''
            })
        
        if after_cant > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Pattern filter retention rate (%)',
                'value': round(after_pattern / after_cant * 100, 2) if after_cant > 0 else 0,
                'details': ''
            })
        
        if after_pattern > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Translation success rate (%)',
                'value': round(final / after_pattern * 100, 2) if after_pattern > 0 else 0,
                'details': ''
            })
        
        if initial > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Overall retention rate (%)',
                'value': round(final / initial * 100, 2) if initial > 0 else 0,
                'details': ''
            })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Sentences with real POS tags',
            'value': translation_stats.get('final_with_pos', 0),
            'details': 'Excludes UNK/ENG fallbacks'
        })
        
        report_data.append({
            'section': 'Translation',
            'metric': 'Sentences with fallback POS tags',
            'value': translation_stats.get('final_with_any_pos', 0) - translation_stats.get('final_with_pos', 0),
            'details': 'X, UNK, or ENG tags only'
        })
        
        pos_coverage = translation_stats.get('final_with_pos', 0)
        final_trans = translation_stats.get('final_with_translation', 0)
        if final_trans > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Real POS tagging success rate - sentence-level (%)',
                'value': round(pos_coverage / final_trans * 100, 2) if final_trans > 0 else 0,
                'details': 'Percentage of sentences with all real POS tags'
            })
        
        # Word-level POS statistics
        total_words = translation_stats.get('total_words', 0)
        if total_words > 0:
            report_data.append({
                'section': 'Translation',
                'metric': 'Total words in translations',
                'value': total_words,
                'details': ''
            })
            
            report_data.append({
                'section': 'Translation',
                'metric': 'Words with real POS tags',
                'value': translation_stats.get('words_with_real_pos', 0),
                'details': 'Excludes X/UNK/ENG'
            })
            
            report_data.append({
                'section': 'Translation',
                'metric': 'Words with fallback POS tags',
                'value': translation_stats.get('words_with_fallback_pos', 0),
                'details': 'X, UNK, or ENG tags'
            })
            
            word_success_rate = (translation_stats.get('words_with_real_pos', 0) / total_words * 100) if total_words > 0 else 0
            report_data.append({
                'section': 'Translation',
                'metric': 'Real POS tagging success rate - word-level (%)',
                'value': round(word_success_rate, 2),
                'details': 'Percentage of words with real POS tags'
            })
    else:
        report_data.append({
            'section': 'Translation',
            'metric': 'Translation skipped',
            'value': 0,
            'details': 'Translation was not performed (--no-translation flag)'
        })
    
    # Create DataFrame and save
    df_report = pd.DataFrame(report_data)
    
    # Create output directory if it doesn't exist
    csv_dir = os.path.dirname(output_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    df_report.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Generated preprocessing report: '{output_path}'")
