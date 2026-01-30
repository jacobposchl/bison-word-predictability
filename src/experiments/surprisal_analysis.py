"""
Surprisal analysis for code-switching predictability experiment.

This module contains functions for calculating and comparing surprisal values
between code-switched translations and matched monolingual baseline sentences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from scipy import stats
from tqdm import tqdm
import pycantonese
import logging
import torch

from src.experiments.surprisal_calculator import MaskedLMSurprisalCalculator, AutoregressiveLMSurprisalCalculator

logger = logging.getLogger(__name__)

NEUTRAL_CONTEXTS = ['係', '的', '我', '你', '在', '有', '是', '了']


def get_word_frequency(word: str, model, tokenizer, device) -> float:
    word = str(word).strip()
    if not word or pd.isna(word):
        return np.nan
    
    try:
        log_probs = []
        
        for context in NEUTRAL_CONTEXTS:
            text = f"{context} {word}"
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            input_ids = inputs.input_ids[0]
            word_token_start = len(tokenizer(context, add_special_tokens=False).input_ids) + 1
            
            token_log_probs = []
            for i in range(word_token_start, len(input_ids)):
                token_logits = logits[0, i-1, :]
                log_prob_dist = torch.log_softmax(token_logits, dim=-1)
                token_log_prob = log_prob_dist[input_ids[i]].item()
                token_log_probs.append(token_log_prob)
            
            if token_log_probs:
                log_probs.append(np.mean(token_log_probs))
        
        return np.mean(log_probs) if log_probs else np.nan
        
    except Exception as e:
        logger.warning(f"Failed to calculate model-based frequency for '{word}': {e}")
        return np.nan


def calculate_surprisal_for_dataset(
    analysis_df: pd.DataFrame,
    surprisal_calc: Union[MaskedLMSurprisalCalculator, AutoregressiveLMSurprisalCalculator],
    show_progress: bool = True,
    use_context: bool = True,
    context_lengths: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculate surprisal values for both CS translations and matched monolingual sentences.
    
    Args:
        analysis_df: DataFrame from analysis_dataset.csv with columns:
            - cs_sentence: Original code-switched sentence
            - cs_translation: Cantonese translation
            - matched_mono: Matched monolingual sentence
            - switch_index: Position of switch word (first English word) in CS translation
            - matched_switch_index: Corresponding position in matched sentence
            - similarity: Similarity score
            - pattern: Code-switch pattern
            - cs_participant: Participant ID
            - cs_context, mono_context: Optional discourse context
        surprisal_calc: Initialized surprisal calculator (masked or autoregressive)
        show_progress: Whether to show progress bar
        use_context: Whether to use discourse context in calculations
        context_lengths: List of context lengths (number of sentences) to calculate.
                        If None and use_context=True, uses single context length.
                        Creates separate columns for each length (e.g., cs_surprisal_context_1, cs_surprisal_context_2)
        
    Returns:
        DataFrame with added columns (only rows with valid surprisal calculations are returned):
            - cs_surprisal_context_{N}: Total surprisal at switch point in CS translation (for each context length N)
            - cs_entropy_context_{N}: Entropy of probability distribution at CS word position
            - mono_surprisal_context_{N}: Total surprisal at matched position in mono sentence
            - mono_entropy_context_{N}: Entropy of probability distribution at mono word position
            - surprisal_difference_context_{N}: cs_surprisal - mono_surprisal
            - cs_word: Word at switch point in CS translation
            - cs_num_tokens: Number of subword tokens for CS word
            - cs_word_frequency: Word frequency from pycantonese corpus
            - mono_word: Word at matched position in mono sentence
            - mono_num_tokens: Number of subword tokens for mono word
            - mono_word_frequency: Word frequency from pycantonese corpus
            
        Note: Rows with failed calculations (NaN surprisal values) are automatically filtered out.
    """

    results = []
    
    iterator = tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Calculating surprisal") \
        if show_progress else analysis_df.iterrows()
    
    # Context sentence delimiter
    CONTEXT_SENTENCE_DELIMITER = ' ||| '
    
    if context_lengths is None:
        if use_context:
            raise ValueError("context_lengths must be provided when use_context=True")
        else:
            context_lengths = []
    
    # Columns to exclude from input dataset (matching-related, not needed for surprisal results)
    COLUMNS_TO_EXCLUDE = [
        'cs_pattern', 'cs_start_time', 'pos_window', 'matched_start_time', 'matched_pos',
        'total_matches_above_threshold', 'matches_same_group', 'matches_same_speaker',
        'cs_context_valid', 'mono_context_valid'
    ]
    
    for idx, row in iterator:
        result = row.to_dict()
        
        for col in COLUMNS_TO_EXCLUDE:
            result.pop(col, None)
        
        # Extract context sentences if available
        cs_context_sentences = []
        mono_context_sentences = []
        
        if use_context:

            cs_context_valid = True
            mono_context_valid = True
            
            if 'cs_context_valid' in row:
                cs_context_valid = row.get('cs_context_valid', False)
            if 'mono_context_valid' in row:
                mono_context_valid = row.get('mono_context_valid', False)
            
            if 'cs_context' in row and cs_context_valid:
                cs_context_full = row['cs_context']

                if pd.notna(cs_context_full) and cs_context_full and str(cs_context_full).strip() != '':
                    cs_context_sentences = [s.strip() for s in str(cs_context_full).split(CONTEXT_SENTENCE_DELIMITER) if s.strip()]
                    result['cs_context'] = cs_context_full
                else:
                    cs_context_sentences = []
                    result['cs_context'] = 'N/A'
            else:
                cs_context_sentences = []
                result['cs_context'] = 'N/A'
                
            if 'mono_context' in row and mono_context_valid:
                mono_context_full = row['mono_context']

                if pd.notna(mono_context_full) and mono_context_full and str(mono_context_full).strip() != '':
                    mono_context_sentences = [s.strip() for s in str(mono_context_full).split(CONTEXT_SENTENCE_DELIMITER) if s.strip()]
                    result['mono_context'] = mono_context_full
                else:
                    mono_context_sentences = []
                    result['mono_context'] = 'N/A'
            else:
                mono_context_sentences = []
                result['mono_context'] = 'N/A'
        else:
            result['cs_context'] = 'N/A'
            result['mono_context'] = 'N/A'
        
        
        cs_words = row['cs_translation'].split()
        
        
        # switch_index is already 0-based and points to the switch word
        switch_token_idx = int(row.get('switch_index', 0))
        
        if switch_token_idx >= len(cs_words):
            raise ValueError(f"Switch token index {switch_token_idx} out of bounds for sentence with {len(cs_words)} words")
        

        mono_words = pycantonese.segment(row['matched_mono'])
        matched_switch_idx = int(row['matched_switch_index'])
        
        if matched_switch_idx >= len(mono_words):
            raise ValueError(f"Matched switch index {matched_switch_idx} out of bounds for sentence with {len(mono_words)} words. Cannot measure surprisal at switch word position (matched_switch_index={row['matched_switch_index']}, sentence_length={len(mono_words)}).")
        

        cs_word_info = surprisal_calc.calculate_surprisal(
            word_index=switch_token_idx,
            words=cs_words,
            context=None
        )
        mono_word_info = surprisal_calc.calculate_surprisal(
            word_index=matched_switch_idx,
            words=mono_words,
            context=None
        )
        
        # Store word-level metadata (same for all context lengths)
        result['cs_word'] = cs_word_info['word']
        result['cs_num_tokens'] = cs_word_info['num_tokens']
        result['cs_word_frequency'] = get_word_frequency(
            cs_word_info['word'], 
            surprisal_calc.model, 
            surprisal_calc.tokenizer,
            surprisal_calc.device
        ) if pd.notna(cs_word_info['word']) else np.nan
        
        result['mono_word'] = mono_word_info['word']
        result['mono_num_tokens'] = mono_word_info['num_tokens']
        result['mono_word_frequency'] = get_word_frequency(
            mono_word_info['word'],
            surprisal_calc.model,
            surprisal_calc.tokenizer,
            surprisal_calc.device
        ) if pd.notna(mono_word_info['word']) else np.nan
        
        # Calculate surprisal for each context length
        if context_lengths:
            for context_len in context_lengths:
                # Extract last N sentences for this context length (most recent)
                cs_context = None
                mono_context = None
                
                if len(cs_context_sentences) >= context_len:
                    cs_context = ' '.join(cs_context_sentences[-context_len:])
                if len(mono_context_sentences) >= context_len:
                    mono_context = ' '.join(mono_context_sentences[-context_len:])
                
                if cs_context and mono_context:
                    cs_result = surprisal_calc.calculate_surprisal(
                        word_index=switch_token_idx,
                        words=cs_words,
                        context=cs_context
                    )
                    
                    mono_result = surprisal_calc.calculate_surprisal(
                        word_index=matched_switch_idx,
                        words=mono_words,
                        context=mono_context
                    )
                    
                    result[f'cs_surprisal_context_{context_len}'] = cs_result['surprisal']
                    result[f'cs_entropy_context_{context_len}'] = cs_result['entropy']
                    result[f'mono_surprisal_context_{context_len}'] = mono_result['surprisal']
                    result[f'mono_entropy_context_{context_len}'] = mono_result['entropy']
                    result[f'surprisal_difference_context_{context_len}'] = cs_result['surprisal'] - mono_result['surprisal']
                else:
                    result[f'cs_surprisal_context_{context_len}'] = np.nan
                    result[f'cs_entropy_context_{context_len}'] = np.nan
                    result[f'mono_surprisal_context_{context_len}'] = np.nan
                    result[f'mono_entropy_context_{context_len}'] = np.nan
                    result[f'surprisal_difference_context_{context_len}'] = np.nan
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Log context clipping statistics if any occurred
    if hasattr(surprisal_calc, 'context_clipped_count') and surprisal_calc.context_clipped_count > 0:
        logger.info(f"Context clipped for {surprisal_calc.context_clipped_count} sentences due to max_length constraint")
    
    return results_df


def compute_statistics(results_df: pd.DataFrame, context_length: int) -> Dict:
    """
    Compute statistical comparisons between CS and monolingual surprisal.
    
    Args:
        results_df: DataFrame from calculate_surprisal_for_dataset()
        context_length: Context length to compute statistics for (required)
        
    Returns:
        Dictionary containing:
            - n_total: Total number of comparisons (before filtering)
            - n_valid: Number of successful calculations (should equal n_total if pre-filtered)
            - n_complete: Number of complete calculations (all tokens valid)
            - n_filtered: Number of rows filtered out (if pre-filtering was done)
            - cs_mean: Mean CS surprisal
            - cs_std: Std dev CS surprisal
            - mono_mean: Mean mono surprisal
            - mono_std: Std dev mono surprisal
            - difference_mean: Mean difference (CS - mono)
            - difference_std: Std dev of difference
            - paired_ttest: Results of paired t-test
            - cohens_d: Cohen's d effect size
    """
    
    cs_surprisal_col = f'cs_surprisal_context_{context_length}'
    mono_surprisal_col = f'mono_surprisal_context_{context_length}'
    difference_col = f'surprisal_difference_context_{context_length}'
    
    if cs_surprisal_col not in results_df.columns or mono_surprisal_col not in results_df.columns:
        raise ValueError(f"Required columns not found: {cs_surprisal_col}, {mono_surprisal_col}")
    

    has_context_col = 'cs_context' in results_df.columns and 'mono_context' in results_df.columns
    
    if has_context_col:
        # Only include rows where context is not "N/A"
        valid_context_mask = (
            (results_df['cs_context'] != 'N/A') & 
            (results_df['mono_context'] != 'N/A') &
            results_df['cs_context'].notna() &
            results_df['mono_context'].notna()
        )
        complete_df = results_df[
            valid_context_mask &
            pd.notna(results_df[cs_surprisal_col]) &
            pd.notna(results_df[mono_surprisal_col])
        ].copy()
    else:
        complete_df = results_df[
            pd.notna(results_df[cs_surprisal_col]) &
            pd.notna(results_df[mono_surprisal_col])
        ].copy()
    
    n_filtered = len(results_df) - len(complete_df)
    
    stats_dict = {
        'n_total': len(results_df),
        'n_valid': len(complete_df),
        'n_complete': len(complete_df),
        'n_filtered': n_filtered,
        'success_rate': len(complete_df) / len(results_df) if len(results_df) > 0 else 0,
        'complete_rate': len(complete_df) / len(results_df) if len(results_df) > 0 else 0,
        'context_length': context_length
    }
    
    if has_context_col:
        stats_dict['n_with_context'] = len(complete_df)
        stats_dict['n_without_context'] = len(results_df) - len(complete_df)
    elif context_length is not None:
        stats_dict['n_with_context'] = len(complete_df)
        stats_dict['n_without_context'] = 0
    
    if len(complete_df) == 0:
        return stats_dict
    
    # Use complete_df for all statistics (only fully valid token calculations)
    # Basic statistics
    stats_dict['cs_surprisal_mean'] = complete_df[cs_surprisal_col].mean()
    stats_dict['cs_surprisal_std'] = complete_df[cs_surprisal_col].std()
    stats_dict['cs_surprisal_median'] = complete_df[cs_surprisal_col].median()
    
    stats_dict['mono_surprisal_mean'] = complete_df[mono_surprisal_col].mean()
    stats_dict['mono_surprisal_std'] = complete_df[mono_surprisal_col].std()
    stats_dict['mono_surprisal_median'] = complete_df[mono_surprisal_col].median()
    
    if difference_col in complete_df.columns:
        stats_dict['difference_mean'] = complete_df[difference_col].mean()
        stats_dict['difference_std'] = complete_df[difference_col].std()
        stats_dict['difference_median'] = complete_df[difference_col].median()
    else:
        # Calculate difference if column doesn't exist
        diff = complete_df[cs_surprisal_col] - complete_df[mono_surprisal_col]
        stats_dict['difference_mean'] = diff.mean()
        stats_dict['difference_std'] = diff.std()
        stats_dict['difference_median'] = diff.median()
    
    # Paired t-test (filter out any remaining infinite values)
    valid_pairs = complete_df[
        np.isfinite(complete_df[cs_surprisal_col]) & 
        np.isfinite(complete_df[mono_surprisal_col])
    ]
    
    if len(valid_pairs) >= 2:
        t_stat, p_value = stats.ttest_rel(
            valid_pairs[cs_surprisal_col],
            valid_pairs[mono_surprisal_col]
        )
    else:
        t_stat, p_value = np.nan, np.nan
        
    stats_dict['ttest_statistic'] = t_stat
    stats_dict['ttest_pvalue'] = p_value
    
    # Cohen's d effect size
    diff = complete_df[cs_surprisal_col] - complete_df[mono_surprisal_col]
    if diff.std() > 0:
        cohens_d = diff.mean() / diff.std()
    else:
        cohens_d = np.nan
    stats_dict['cohens_d'] = cohens_d
    
    return stats_dict


def print_statistics_summary(stats_dict: Dict):
    """
    Print a formatted summary of statistical results.
    
    Args:
        stats_dict: Dictionary from compute_statistics()
    """
    print("\n" + "="*80)
    print("SURPRISAL COMPARISON STATISTICS")
    print("="*80)
    
    print(f"\nSample Size:")
    print(f"  Total comparisons: {stats_dict['n_total']}")
    if stats_dict.get('n_filtered', 0) > 0:
        print(f"  Filtered out (failed calculations): {stats_dict['n_filtered']} ({stats_dict['n_filtered']/stats_dict['n_total']:.1%})")
    print(f"  Valid calculations: {stats_dict['n_valid']}")
    print(f"  Complete calculations: {stats_dict['n_complete']} (all tokens valid)")
    print(f"  Success rate: {stats_dict['success_rate']:.1%}")
    print(f"  Complete rate: {stats_dict['complete_rate']:.1%}")
    
    # Show context usage if available
    if 'n_with_context' in stats_dict:
        print(f"\nContext Usage:")
        print(f"  Calculations with context: {stats_dict['n_with_context']}")
        print(f"  Calculations without context: {stats_dict['n_without_context']}")
    
    if stats_dict['n_complete'] == 0:
        print("\nNo complete calculations to report.")
        print("(Complete = both CS and mono words have all tokens successfully calculated)")
        return
    
    print(f"\nCode-Switched Translation Surprisal:")
    print(f"  Mean:   {stats_dict['cs_surprisal_mean']:.4f}")
    print(f"  Median: {stats_dict['cs_surprisal_median']:.4f}")
    print(f"  Std:    {stats_dict['cs_surprisal_std']:.4f}")
    
    print(f"\nMonolingual Baseline Surprisal:")
    print(f"  Mean:   {stats_dict['mono_surprisal_mean']:.4f}")
    print(f"  Median: {stats_dict['mono_surprisal_median']:.4f}")
    print(f"  Std:    {stats_dict['mono_surprisal_std']:.4f}")
    
    print(f"\nDifference (CS - Monolingual):")
    print(f"  Mean:   {stats_dict['difference_mean']:.4f}")
    print(f"  Median: {stats_dict['difference_median']:.4f}")
    print(f"  Std:    {stats_dict['difference_std']:.4f}")
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {stats_dict['ttest_statistic']:.4f}")
    print(f"  p-value:     {stats_dict['ttest_pvalue']:.6f}")
    sig_marker = "***" if stats_dict['ttest_pvalue'] < 0.001 else \
                 "**" if stats_dict['ttest_pvalue'] < 0.01 else \
                 "*" if stats_dict['ttest_pvalue'] < 0.05 else "ns"
    print(f"  Significance: {sig_marker}")
    
    print(f"\nEffect Size:")
    print(f"  Cohen's d: {stats_dict['cohens_d']:.4f}")
    
    print("\n" + "="*80)
