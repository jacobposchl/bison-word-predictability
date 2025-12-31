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
    # Remove punctuation and whitespace
    cleaned = re.sub(r'[^\w]', '', word)
    
    # If empty after cleaning, it's not an English word
    if not cleaned:
        return False
    
    # Check if all characters are ASCII letters (a-z, A-Z)
    # This is a simple heuristic: English words use ASCII, Cantonese uses Unicode
    return all(ord(c) < 128 and c.isalpha() for c in cleaned)


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
    csv_without_fillers_path: str
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
        
    Returns:
        Tuple of (dataframe_with_fillers, dataframe_without_fillers)
    """
    # type: ignore
    # Filter both datasets to only keep sentences with actual code-switching
    with_fillers = filter_code_switching_sentences(all_sentences, include_fillers=True)
    without_fillers = filter_code_switching_sentences(all_sentences, include_fillers=False)
    
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
    csv_all_sentences_path: str
) -> pd.DataFrame:
    """
    Export ALL sentences (both monolingual and code-switched) to CSV.
    
    This includes sentences that were filtered out in the code-switching analysis.
    Useful for exploratory analysis that needs monolingual sentences.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        csv_all_sentences_path: Output path for CSV with all sentences
        
    Returns:
        DataFrame with all sentences
    """
    logger.info(f"Exporting ALL sentences (monolingual + code-switched) to CSV...")
    
    # Create DataFrame with all sentences
    # Use pattern_with_fillers for consistency
    csv_all = pd.DataFrame({
        'start_time': [s['start_time'] for s in all_sentences],
        'end_time': [s['end_time'] for s in all_sentences],
        'reconstructed_sentence': [s['reconstructed_text'] for s in all_sentences],
        'sentence_original': [s['text'] for s in all_sentences],
        'pattern': [s.get('pattern_with_fillers', s.get('pattern', '')) for s in all_sentences],
        'matrix_language': [s.get('matrix_language', 'Unknown') for s in all_sentences],
        'group': [s.get('group', '') for s in all_sentences],
        'participant_id': [s.get('participant_id', '') for s in all_sentences]
    })
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


def export_monolingual_sentences(
    all_sentences: List[Dict],
    config
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
    
    # Create DataFrames
    def create_df(sentences: List[Dict], use_pattern_with_fillers: bool, lang: str) -> pd.DataFrame:
        """Create DataFrame from sentence list."""
        pattern_field = 'pattern_with_fillers' if use_pattern_with_fillers else 'pattern_content_only'
        
        # For WITHOUT fillers datasets, use pre-computed text without fillers
        if use_pattern_with_fillers:
            reconstructed_sentences = [s['reconstructed_text'] for s in sentences]
        else:
            reconstructed_sentences = [s['reconstructed_text_without_fillers'] for s in sentences]
        
        df = pd.DataFrame({
            'start_time': [s['start_time'] for s in sentences],
            'end_time': [s['end_time'] for s in sentences],
            'reconstructed_sentence': reconstructed_sentences,
            'sentence_original': [s['text'] for s in sentences],
            'pattern': [s.get(pattern_field, '') for s in sentences],
            'matrix_language': [s.get('matrix_language', 'Unknown') for s in sentences],
            'group': [s.get('group', '') for s in sentences],
            'participant_id': [s.get('participant_id', '') for s in sentences]
        })
        return sort_dataframe(df)
    
    cant_with_df = create_df(cantonese_with_fillers, use_pattern_with_fillers=True, lang='C')
    cant_without_df = create_df(cantonese_without_fillers, use_pattern_with_fillers=False, lang='C')
    eng_with_df = create_df(english_with_fillers, use_pattern_with_fillers=True, lang='E')
    eng_without_df = create_df(english_without_fillers, use_pattern_with_fillers=False, lang='E')
    
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
    do_translation: bool = True
) -> pd.DataFrame:
    """
    Export code-switched sentences with full Cantonese translations.
    
    Reads from code_switching_WITHOUT_fillers.csv, filters for Cantonese matrix language,
    and creates a new CSV with code_switch_original and cantonese_translation columns.
    Optionally performs translation and POS tagging.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries (not used, kept for compatibility)
        config: Config object with CSV path methods
        do_translation: If True, perform translation and POS tagging. If False, leave columns empty.
        
    Returns:
        DataFrame with translated sentences structure
    """
    logger.info("Exporting translated code-switched sentences...")
    
    # Read from code_switching_WITHOUT_fillers.csv
    csv_without_fillers_path = config.get_csv_without_fillers_path()
    
    if not os.path.exists(csv_without_fillers_path):
        logger.error(f"Source CSV not found: {csv_without_fillers_path}")
        logger.error("Please run preprocessing first to generate code_switching_WITHOUT_fillers.csv")
        return pd.DataFrame()
    
    logger.info(f"Reading from: {csv_without_fillers_path}")
    df_source = pd.read_csv(csv_without_fillers_path)
    logger.info(f"Loaded {len(df_source)} rows from source CSV")
    
    # Filter to ONLY Cantonese matrix language
    df_cantonese = df_source[df_source['matrix_language'] == 'Cantonese'].copy()
    
    logger.info(f"Total code-switched sentences: {len(df_source)}")
    logger.info(f"Cantonese matrix language only: {len(df_cantonese)}")
    logger.info(f"Filtered out {len(df_source) - len(df_cantonese)} non-Cantonese matrix sentences")
    
    if len(df_cantonese) == 0:
        logger.warning("No Cantonese matrix language sentences found!")
        return pd.DataFrame()
    
    # Create new DataFrame with desired structure
    # Column order: code_switch_original, cantonese_translation, translated_pos, pattern, then other columns
    df = pd.DataFrame({
        'start_time': df_cantonese['start_time'].values,
        'end_time': df_cantonese['end_time'].values,
        'code_switch_original': df_cantonese['reconstructed_sentence'].values,
        'cantonese_translation': '',  # Will be filled if do_translation is True
        'translated_pos': '',  # Will be filled if do_translation is True
        'pattern': df_cantonese['pattern'].values,
        'matrix_language': df_cantonese['matrix_language'].values,
        'group': df_cantonese['group'].values,
        'participant_id': df_cantonese['participant_id'].values
    })
    
    # Define column order (used for saving)
    column_order = [
        'start_time',
        'end_time',
        'code_switch_original',
        'cantonese_translation',
        'translated_pos',
        'pattern',
        'matrix_language',
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
        from src.analysis.pos_tagging import pos_tag_cantonese, extract_pos_sequence
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
            translation = ''
            pos_seq = ''
            
            try:
                # Segment the sentence
                words = segment_cantonese_sentence(sentence)
                
                # Translate
                translation_result = translator.translate_code_switched_sentence(
                    sentence=sentence,
                    pattern=pattern,
                    words=words
                )
                translation = translation_result.get('translated_sentence', '')
                
                # Verify translation is full Cantonese
                is_valid, error_msg = _verify_cantonese_only(translation)
                
                if is_valid:
                    valid_count += 1
                    # Add POS tagging for valid translation
                    try:
                        tagged = pos_tag_cantonese(translation)
                        pos_seq_list = extract_pos_sequence(tagged)
                        pos_seq = ' '.join(pos_seq_list) if pos_seq_list else ''
                    except Exception as e:
                        logger.warning(f"Error POS tagging translation at row {idx}: {e}")
                        pos_seq = ''
                else:
                    invalid_count += 1
                    # Log first few failures with actual translation text for debugging
                    if invalid_count <= 3:
                        logger.warning(f"Row {idx}: Translation verification failed - {error_msg}")
                        logger.debug(f"  Original: {sentence[:100]}")
                        logger.debug(f"  Translation (first 200 chars): {translation[:200]}")
                    else:
                        logger.warning(f"Row {idx}: Translation verification failed - {error_msg}")
                    # Still add empty POS for consistency
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
        
        # Final save to ensure everything is saved
        df = df[column_order].copy()
        df = sort_dataframe(df)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Translation complete: {valid_count} valid, {invalid_count} invalid out of {len(df)} total")
        logger.info(f"Saved translated sentences with POS tags: '{csv_path}' - {len(df)} sentences")
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
