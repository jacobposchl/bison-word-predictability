"""
Surprisal analysis for code-switching predictability experiment.

This module contains functions for calculating and comparing surprisal values
between code-switched translations and matched monolingual baseline sentences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from scipy import stats
from tqdm import tqdm
import pycantonese

from src.experiments.surprisal_calculator import MaskedLMSurprisalCalculator, AutoregressiveLMSurprisalCalculator
from src.analysis.pos_tagging import parse_pattern_segments


def calculate_surprisal_for_dataset(
    analysis_df: pd.DataFrame,
    surprisal_calc: Union[MaskedLMSurprisalCalculator, AutoregressiveLMSurprisalCalculator],
    show_progress: bool = True,
    use_context: bool = True
) -> pd.DataFrame:
    """
    Calculate surprisal values for both CS translations and matched monolingual sentences.
    
    Args:
        analysis_df: DataFrame from analysis_dataset.csv with columns:
            - cs_sentence: Original code-switched sentence
            - cs_translation: Cantonese translation
            - matched_mono: Matched monolingual sentence
            - switch_index: Position of switch in CS translation
            - matched_switch_index: Corresponding position in matched sentence
            - similarity: Similarity score
            - pattern: Code-switch pattern
            - cs_participant: Participant ID
            - cs_context, mono_context: Optional discourse context
        surprisal_calc: Initialized surprisal calculator (masked or autoregressive)
        show_progress: Whether to show progress bar
        use_context: Whether to use discourse context in calculations
        
    Returns:
        DataFrame with added columns:
            - cs_surprisal_total: Total surprisal at switch point in CS translation
            - cs_surprisal_per_token: Per-token average surprisal for CS
            - cs_word: Word at switch point in CS translation
            - cs_num_tokens: Number of subword tokens for CS word
            - mono_surprisal_total: Total surprisal at matched position in mono sentence
            - mono_surprisal_per_token: Per-token average surprisal for mono
            - mono_word: Word at matched position in mono sentence
            - mono_num_tokens: Number of subword tokens for mono word
            - surprisal_difference: cs_surprisal_total - mono_surprisal_total
            - calculation_success: Boolean indicating successful calculation
            - used_context: Whether context was actually used for this calculation
    """
    results = []
    
    iterator = tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Calculating surprisal") \
        if show_progress else analysis_df.iterrows()
    
    for idx, row in iterator:
        result = row.to_dict()
        
        # Determine whether to use context for this row
        cs_context = None
        mono_context = None
        used_context = False
        
        if use_context:
            # Check if context columns exist and are valid
            if 'cs_context' in row and row.get('cs_context_valid', False):
                cs_context = row['cs_context']
            if 'mono_context' in row and row.get('mono_context_valid', False):
                mono_context = row['mono_context']
            
            used_context = (cs_context is not None and mono_context is not None)
        
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
            
            cs_result = surprisal_calc.calculate_surprisal(
                word_index=switch_token_idx,  # Calculate surprisal at the switch token (first word that was English, now translated)
                words=cs_words,
                context=cs_context  # Pass context
            )
            
            result['cs_surprisal_total'] = cs_result['surprisal']
            result['cs_surprisal_per_token'] = cs_result['surprisal'] / cs_result['num_tokens'] if cs_result['num_tokens'] > 0 else np.nan
            result['cs_word'] = cs_result['word']
            result['cs_num_tokens'] = cs_result['num_tokens']
            result['cs_num_valid_tokens'] = cs_result['num_valid_tokens']
            
            # Calculate matched monolingual surprisal
            # Use PyCantonese segmentation because matched_switch_index is based on POS sequence,
            # which was created by re-segmenting with PyCantonese
            mono_words = pycantonese.segment(row['matched_mono'])
            # matched_switch_index is the direct mapping of switch_index from CS sentence to mono sentence
            # It points to the equivalent of the last Cantonese word before the switch
            # We want to measure surprisal at the equivalent of the switch word, so we add +1
            matched_switch_idx = int(row['matched_switch_index']) + 1
            
            # Check if the switch word position is available in the matched sentence
            # If not, we can't measure surprisal at the correct position, so skip this row
            if matched_switch_idx >= len(mono_words):
                raise ValueError(f"Matched switch index {matched_switch_idx} out of bounds for sentence with {len(mono_words)} words. Cannot measure surprisal at switch word position (matched_switch_index={row['matched_switch_index']}, sentence_length={len(mono_words)}).")
            
            mono_result = surprisal_calc.calculate_surprisal(
                word_index=matched_switch_idx,  # Calculate surprisal at equivalent of switch word
                words=mono_words,
                context=mono_context  # Pass context
            )
            
            result['mono_surprisal_total'] = mono_result['surprisal']
            result['mono_surprisal_per_token'] = mono_result['surprisal'] / mono_result['num_tokens'] if mono_result['num_tokens'] > 0 else np.nan
            result['mono_word'] = mono_result['word']
            result['mono_num_tokens'] = mono_result['num_tokens']
            result['mono_num_valid_tokens'] = mono_result['num_valid_tokens']
            
            # Calculate difference
            result['surprisal_difference'] = result['cs_surprisal_total'] - result['mono_surprisal_total']
            result['calculation_success'] = True
            result['used_context'] = used_context
            
        except Exception as e:
            # Handle errors gracefully
            result['cs_surprisal_total'] = np.nan
            result['cs_surprisal_per_token'] = np.nan
            result['cs_word'] = None
            result['cs_num_tokens'] = np.nan
            result['cs_num_valid_tokens'] = np.nan
            result['mono_surprisal_total'] = np.nan
            result['mono_surprisal_per_token'] = np.nan
            result['mono_word'] = None
            result['mono_num_tokens'] = np.nan
            result['mono_num_valid_tokens'] = np.nan
            result['surprisal_difference'] = np.nan
            result['calculation_success'] = False
            result['used_context'] = False
            result['error_message'] = str(e)
            
            if show_progress:
                tqdm.write(f"Error processing row {idx}: {e}")
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_statistics(results_df: pd.DataFrame) -> Dict:
    """
    Compute statistical comparisons between CS and monolingual surprisal.
    
    Args:
        results_df: DataFrame from calculate_surprisal_for_dataset()
        aggregated_df: Optional aggregated DataFrame from aggregate_mono_surprisals_by_sentence()
        
    Returns:
        Dictionary containing:
            - n_total: Total number of comparisons
            - n_valid: Number of successful calculations
            - cs_mean: Mean CS surprisal
            - cs_std: Std dev CS surprisal
            - mono_mean: Mean mono surprisal
            - mono_std: Std dev mono surprisal
            - difference_mean: Mean difference (CS - mono)
            - difference_std: Std dev of difference
            - paired_ttest: Results of paired t-test
            - cohens_d: Cohen's d effect size
            - aggregated_stats: Statistics from aggregated data (if provided)
    """
    # Filter valid calculations
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    
    # Further filter to only include complete calculations (all tokens valid)
    complete_df = valid_df[
        (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
        (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
    ].copy()
    
    stats_dict = {
        'n_total': len(results_df),
        'n_valid': len(valid_df),
        'n_complete': len(complete_df),
        'success_rate': len(valid_df) / len(results_df) if len(results_df) > 0 else 0,
        'complete_rate': len(complete_df) / len(results_df) if len(results_df) > 0 else 0
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
