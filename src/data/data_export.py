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

logger = logging.getLogger(__name__)


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
    
    # Create the second CSV - WITHOUT fillers
    csv_without_fillers = pd.DataFrame({
        'start_time': [s['start_time'] for s in without_fillers],
        'end_time': [s['end_time'] for s in without_fillers],
        'reconstructed_sentence': [s['reconstructed_text'] for s in without_fillers],
        'sentence_original': [s['text'] for s in without_fillers],
        'pattern': [s['pattern_content_only'] for s in without_fillers],
        'matrix_language': [s['matrix_language'] for s in without_fillers],
        'group': [s['group'] for s in without_fillers],
        'participant_id': [s['participant_id'] for s in without_fillers]
    })
    
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
        
        # For WITHOUT fillers datasets, remove fillers from text
        if use_pattern_with_fillers:
            reconstructed_sentences = [s['reconstructed_text'] for s in sentences]
        else:
            reconstructed_sentences = [
                remove_fillers_from_text(s['reconstructed_text'], lang=lang) 
                for s in sentences
            ]
        
        return pd.DataFrame({
            'start_time': [s['start_time'] for s in sentences],
            'end_time': [s['end_time'] for s in sentences],
            'reconstructed_sentence': reconstructed_sentences,
            'sentence_original': [s['text'] for s in sentences],
            'pattern': [s.get(pattern_field, '') for s in sentences],
            'matrix_language': [s.get('matrix_language', 'Unknown') for s in sentences],
            'group': [s.get('group', '') for s in sentences],
            'participant_id': [s.get('participant_id', '') for s in sentences]
        })
    
    cant_with_df = create_df(cantonese_with_fillers, use_pattern_with_fillers=True, lang='C')
    cant_without_df = create_df(cantonese_without_fillers, use_pattern_with_fillers=False, lang='C')
    eng_with_df = create_df(english_with_fillers, use_pattern_with_fillers=True, lang='E')
    eng_without_df = create_df(english_without_fillers, use_pattern_with_fillers=False, lang='E')
    
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
    api_key: Optional[str] = None,
    skip_translation: bool = False
) -> pd.DataFrame:
    """
    Export code-switched sentences with full Cantonese translations.
    
    Translates English portions to Cantonese and exports to CSV.
    Only processes sentences WITHOUT fillers for cleaner translation.
    
    Args:
        all_sentences: List of all processed sentence data dictionaries
        config: Config object with CSV path methods
        api_key: OpenAI API key for translation (optional if skip_translation=True)
        skip_translation: If True, skip translation and just add empty column
        
    Returns:
        DataFrame with translated sentences
    """
    logger.info("Exporting translated code-switched sentences...")
    
    # Filter to code-switched sentences WITHOUT fillers
    code_switched = filter_code_switching_sentences(all_sentences, include_fillers=False)
    
    if not code_switched:
        logger.warning("No code-switched sentences found!")
        return pd.DataFrame()
    
    # Create base DataFrame
    df = pd.DataFrame({
        'start_time': [s['start_time'] for s in code_switched],
        'end_time': [s['end_time'] for s in code_switched],
        'reconstructed_sentence': [s['reconstructed_text'] for s in code_switched],
        'sentence_original': [s['text'] for s in code_switched],
        'pattern': [s.get('pattern_content_only', '') for s in code_switched],
        'matrix_language': [s.get('matrix_language', 'Unknown') for s in code_switched],
        'group': [s.get('group', '') for s in code_switched],
        'participant_id': [s.get('participant_id', '') for s in code_switched]
    })
    
    # Add translation column
    if skip_translation or api_key is None:
        logger.warning("Skipping translation (no API key provided or skip_translation=True)")
        df['cantonese_translation'] = ''
    else:
        # Import translation service
        from src.translation.translator import TranslationService
        
        logger.info(f"Translating {len(code_switched)} sentences to Cantonese...")
        translator = TranslationService(
            api_key=api_key,
            model=config.get_translation_model(),
            use_cache=config.get_translation_use_cache(),
            cache_dir=config.get_translation_cache_dir()
        )
        
        # Batch translate
        translations = translator.batch_translate_to_cantonese(
            [s['reconstructed_text'] for s in code_switched]
        )
        
        df['cantonese_translation'] = translations
        logger.info("Translation complete!")
    
    # Save CSV
    csv_path = config.get_csv_cantonese_translated_path()
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Saved translated sentences: '{csv_path}' - {len(df)} sentences")
    
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
