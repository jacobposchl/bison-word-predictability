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

from src.experiments.surprisal_calculator import MaskedLMSurprisalCalculator, AutoregressiveLMSurprisalCalculator
from src.analysis.pos_tagging import parse_pattern_segments


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
        DataFrame with added columns (only rows with calculation_success == True are returned):
            - cs_surprisal_total: Total surprisal at switch point in CS translation
            - cs_surprisal_per_token: Per-token average surprisal for CS
            - cs_probability: Probability of CS word given context
            - cs_entropy: Entropy of probability distribution at CS word position
            - cs_word: Word at switch point in CS translation
            - cs_num_tokens: Number of subword tokens for CS word
            - cs_num_valid_tokens: Number of valid tokens for CS word
            - mono_surprisal_total: Total surprisal at matched position in mono sentence
            - mono_surprisal_per_token: Per-token average surprisal for mono
            - mono_probability: Probability of mono word given context
            - mono_entropy: Entropy of probability distribution at mono word position
            - mono_word: Word at matched position in mono sentence
            - mono_num_tokens: Number of subword tokens for mono word
            - mono_num_valid_tokens: Number of valid tokens for mono word
            - surprisal_difference: cs_surprisal_total - mono_surprisal_total
            - calculation_success: Boolean indicating successful calculation (always True in returned DataFrame)
            - used_context: Whether context was actually used for this calculation
            
        Note: Rows with calculation_success == False are automatically filtered out.
    """
    results = []
    
    iterator = tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Calculating surprisal") \
        if show_progress else analysis_df.iterrows()
    
    # Context sentence delimiter (must match what's used in analysis_dataset.py)
    CONTEXT_SENTENCE_DELIMITER = ' ||| '
    
    # If context_lengths not specified, raise error (must be explicitly provided)
    if context_lengths is None:
        if use_context:
            raise ValueError("context_lengths must be provided when use_context=True")
        else:
            context_lengths = []
    
    for idx, row in iterator:
        result = row.to_dict()
        
        # Extract context sentences if available
        cs_context_sentences = []
        mono_context_sentences = []
        
        if use_context:
            # Check if context columns exist and are valid
            if 'cs_context' in row and row.get('cs_context_valid', False):
                cs_context_full = row['cs_context']
                # Split by delimiter to get individual sentences
                cs_context_sentences = [s.strip() for s in cs_context_full.split(CONTEXT_SENTENCE_DELIMITER) if s.strip()]
            if 'mono_context' in row and row.get('mono_context_valid', False):
                mono_context_full = row['mono_context']
                # Split by delimiter to get individual sentences
                mono_context_sentences = [s.strip() for s in mono_context_full.split(CONTEXT_SENTENCE_DELIMITER) if s.strip()]
        
        try:
            # Calculate CS translation surprisal
            # Use space-splitting to match how switch_index was calculated
            # (translations are already properly segmented and space-joined)
            cs_words = row['cs_translation'].split()
            
            # Get switch word index directly from pattern
            # Pattern C8-E1 means 8 Cantonese words, then switch word at index 8 (0-based)
            # This is simpler and more direct than using switch_index + 1
            pattern = row.get('cs_pattern', row.get('pattern', ''))
            segments = parse_pattern_segments(pattern)
            # First segment is always Cantonese for our data, so switch word is at first_count (0-based)
            switch_token_idx = segments[0][1]
            
            if switch_token_idx >= len(cs_words):
                raise ValueError(f"Switch token index {switch_token_idx} out of bounds for sentence with {len(cs_words)} words (pattern: {pattern})")
            
            # Calculate matched monolingual surprisal
            # Use PyCantonese segmentation because matched_switch_index is based on POS sequence,
            # which was created by re-segmenting with PyCantonese
            mono_words = pycantonese.segment(row['matched_mono'])
            # matched_switch_index is the direct mapping of switch_index from CS sentence to mono sentence
            # switch_index now points to the first English word (the switch word)
            # So matched_switch_index also points to the equivalent switch word position
            matched_switch_idx = int(row['matched_switch_index'])
            
            # Check if the switch word position is available in the matched sentence
            # If not, we can't measure surprisal at the correct position, so skip this row
            if matched_switch_idx >= len(mono_words):
                raise ValueError(f"Matched switch index {matched_switch_idx} out of bounds for sentence with {len(mono_words)} words. Cannot measure surprisal at switch word position (matched_switch_index={row['matched_switch_index']}, sentence_length={len(mono_words)}).")
            
            # Calculate surprisal for each context length
            first_successful_cs_result = None
            first_successful_mono_result = None
            
            for context_len in context_lengths:
                # Extract first N sentences for this context length
                cs_context = None
                mono_context = None
                
                if len(cs_context_sentences) >= context_len:
                    cs_context = ' '.join(cs_context_sentences[:context_len])
                if len(mono_context_sentences) >= context_len:
                    mono_context = ' '.join(mono_context_sentences[:context_len])
                
                # Only calculate if we have enough context for this length
                if cs_context and mono_context:
                    # Calculate CS surprisal with this context length
                    cs_result = surprisal_calc.calculate_surprisal(
                        word_index=switch_token_idx,
                        words=cs_words,
                        context=cs_context
                    )
                    
                    # Calculate mono surprisal with this context length
                    mono_result = surprisal_calc.calculate_surprisal(
                        word_index=matched_switch_idx,
                        words=mono_words,
                        context=mono_context
                    )
                    
                    # Store word info from first successful calculation (same for all context lengths)
                    if first_successful_cs_result is None:
                        first_successful_cs_result = cs_result
                        first_successful_mono_result = mono_result
                    
                    # Store results with context length suffix
                    result[f'cs_surprisal_context_{context_len}'] = cs_result['surprisal']
                    result[f'cs_surprisal_per_token_context_{context_len}'] = cs_result['surprisal'] / cs_result['num_tokens'] if cs_result['num_tokens'] > 0 else np.nan
                    result[f'cs_probability_context_{context_len}'] = cs_result['probability']
                    result[f'cs_entropy_context_{context_len}'] = cs_result['entropy']
                    result[f'mono_surprisal_context_{context_len}'] = mono_result['surprisal']
                    result[f'mono_surprisal_per_token_context_{context_len}'] = mono_result['surprisal'] / mono_result['num_tokens'] if mono_result['num_tokens'] > 0 else np.nan
                    result[f'mono_probability_context_{context_len}'] = mono_result['probability']
                    result[f'mono_entropy_context_{context_len}'] = mono_result['entropy']
                    result[f'surprisal_difference_context_{context_len}'] = cs_result['surprisal'] - mono_result['surprisal']
                    result[f'used_context_{context_len}'] = True
                else:
                    # Not enough context - set to NaN
                    result[f'cs_surprisal_context_{context_len}'] = np.nan
                    result[f'cs_surprisal_per_token_context_{context_len}'] = np.nan
                    result[f'cs_probability_context_{context_len}'] = np.nan
                    result[f'cs_entropy_context_{context_len}'] = np.nan
                    result[f'mono_surprisal_context_{context_len}'] = np.nan
                    result[f'mono_surprisal_per_token_context_{context_len}'] = np.nan
                    result[f'mono_probability_context_{context_len}'] = np.nan
                    result[f'mono_entropy_context_{context_len}'] = np.nan
                    result[f'surprisal_difference_context_{context_len}'] = np.nan
                    result[f'used_context_{context_len}'] = False
            
            # Store word info (same for all context lengths, use first successful calculation)
            if first_successful_cs_result is not None:
                result['cs_word'] = first_successful_cs_result['word']
                result['cs_num_tokens'] = first_successful_cs_result['num_tokens']
                result['cs_num_valid_tokens'] = first_successful_cs_result['num_valid_tokens']
                result['mono_word'] = first_successful_mono_result['word']
                result['mono_num_tokens'] = first_successful_mono_result['num_tokens']
                result['mono_num_valid_tokens'] = first_successful_mono_result['num_valid_tokens']
            elif not context_lengths:
                # No context lengths specified - calculate without context
                cs_result = surprisal_calc.calculate_surprisal(
                    word_index=switch_token_idx,
                    words=cs_words,
                    context=None
                )
                mono_result = surprisal_calc.calculate_surprisal(
                    word_index=matched_switch_idx,
                    words=mono_words,
                    context=None
                )
                result['cs_word'] = cs_result['word']
                result['cs_num_tokens'] = cs_result['num_tokens']
                result['cs_num_valid_tokens'] = cs_result['num_valid_tokens']
                result['mono_word'] = mono_result['word']
                result['mono_num_tokens'] = mono_result['num_tokens']
                result['mono_num_valid_tokens'] = mono_result['num_valid_tokens']
            else:
                # No successful calculations
                result['cs_word'] = None
                result['cs_num_tokens'] = np.nan
                result['cs_num_valid_tokens'] = np.nan
                result['mono_word'] = None
                result['mono_num_tokens'] = np.nan
                result['mono_num_valid_tokens'] = np.nan
            
            result['calculation_success'] = True
            
        except Exception as e:
            # Handle errors gracefully - set all context length columns to NaN
            for context_len in context_lengths:
                result[f'cs_surprisal_context_{context_len}'] = np.nan
                result[f'cs_surprisal_per_token_context_{context_len}'] = np.nan
                result[f'cs_probability_context_{context_len}'] = np.nan
                result[f'cs_entropy_context_{context_len}'] = np.nan
                result[f'mono_surprisal_context_{context_len}'] = np.nan
                result[f'mono_surprisal_per_token_context_{context_len}'] = np.nan
                result[f'mono_probability_context_{context_len}'] = np.nan
                result[f'mono_entropy_context_{context_len}'] = np.nan
                result[f'surprisal_difference_context_{context_len}'] = np.nan
                result[f'used_context_{context_len}'] = False
            result['cs_word'] = None
            result['cs_num_tokens'] = np.nan
            result['cs_num_valid_tokens'] = np.nan
            result['mono_word'] = None
            result['mono_num_tokens'] = np.nan
            result['mono_num_valid_tokens'] = np.nan
            result['calculation_success'] = False
            result['error_message'] = str(e)
            
            if show_progress:
                tqdm.write(f"Error processing row {idx}: {e}")
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out rows where calculation_success == False
    n_before_filter = len(results_df)
    results_df = results_df[results_df['calculation_success'] == True].copy()
    n_after_filter = len(results_df)
    n_filtered = n_before_filter - n_after_filter
    
    if show_progress and n_filtered > 0:
        tqdm.write(f"\nFiltered out {n_filtered} rows with failed calculations ({n_filtered/n_before_filter:.1%})")
        tqdm.write(f"Remaining rows: {n_after_filter}")
    
    return results_df


def compute_statistics(results_df: pd.DataFrame) -> Dict:
    """
    Compute statistical comparisons between CS and monolingual surprisal.
    
    Note: results_df should already be filtered to only include successful calculations
    (calculation_success == True). This function further filters to complete calculations.
    
    Args:
        results_df: DataFrame from calculate_surprisal_for_dataset() (already filtered for success)
        
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
    # Filter to only include successful calculations
    if 'calculation_success' not in results_df.columns:
        raise ValueError("results_df must have 'calculation_success' column")
    
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    n_filtered = len(results_df) - len(valid_df)
    
    # Further filter to only include complete calculations (all tokens valid)
    complete_df = valid_df[
        (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
        (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
    ].copy()
    
    stats_dict = {
        'n_total': len(results_df) + n_filtered,  # Original total before any filtering
        'n_valid': len(valid_df),
        'n_complete': len(complete_df),
        'n_filtered': n_filtered,
        'success_rate': len(valid_df) / (len(results_df) + n_filtered) if (len(results_df) + n_filtered) > 0 else 0,
        'complete_rate': len(complete_df) / (len(results_df) + n_filtered) if (len(results_df) + n_filtered) > 0 else 0
    }
    
    # Track context usage if column exists
    if 'used_context' in complete_df.columns:
        n_with_context = complete_df['used_context'].sum()
        stats_dict['n_with_context'] = int(n_with_context)
        stats_dict['n_without_context'] = len(complete_df) - int(n_with_context)
    
    if len(complete_df) == 0:
        return stats_dict
    
    # Use complete_df for all statistics (only fully valid token calculations)
    # Basic statistics
    stats_dict['cs_surprisal_mean'] = complete_df['cs_surprisal_total'].mean()
    stats_dict['cs_surprisal_std'] = complete_df['cs_surprisal_total'].std()
    stats_dict['cs_surprisal_median'] = complete_df['cs_surprisal_total'].median()
    
    stats_dict['mono_surprisal_mean'] = complete_df['mono_surprisal_total'].mean()
    stats_dict['mono_surprisal_std'] = complete_df['mono_surprisal_total'].std()
    stats_dict['mono_surprisal_median'] = complete_df['mono_surprisal_total'].median()
    
    stats_dict['difference_mean'] = complete_df['surprisal_difference'].mean()
    stats_dict['difference_std'] = complete_df['surprisal_difference'].std()
    stats_dict['difference_median'] = complete_df['surprisal_difference'].median()
    
    # Paired t-test (filter out any remaining infinite values)
    valid_pairs = complete_df[
        np.isfinite(complete_df['cs_surprisal_total']) & 
        np.isfinite(complete_df['mono_surprisal_total'])
    ]
    
    if len(valid_pairs) >= 2:
        t_stat, p_value = stats.ttest_rel(
            valid_pairs['cs_surprisal_total'],
            valid_pairs['mono_surprisal_total']
        )
    else:
        t_stat, p_value = np.nan, np.nan
        
    stats_dict['ttest_statistic'] = t_stat
    stats_dict['ttest_pvalue'] = p_value
    
    # Cohen's d effect size
    diff = complete_df['cs_surprisal_total'] - complete_df['mono_surprisal_total']
    cohens_d = diff.mean() / diff.std()
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
