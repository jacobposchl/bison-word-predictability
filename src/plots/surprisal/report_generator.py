"""
Report generation functions for surprisal analysis.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_surprisal_statistics_report(
    stats_dict: Dict,
    all_stats: Dict[int, Dict],
    context_lengths: List[int],
    model_type: str,
    mode_name: str,
    primary_context_length: Optional[int] = None,
    context_clipped_count: Optional[int] = None
) -> str:
    """
    Generate comprehensive statistics report for surprisal analysis.
    
    Args:
        stats_dict: Primary statistics dictionary from compute_statistics()
        all_stats: Dictionary mapping context length to statistics dict
        context_lengths: List of context lengths analyzed
        model_type: Type of model used ("masked" or "autoregressive")
        mode_name: Analysis mode ("with_context" or "without_context")
        primary_context_length: Primary context length to report (usually first)
        context_clipped_count: Number of sentences where context was clipped due to max_length
        
    Returns:
        Formatted text report
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("SURPRISAL COMPARISON STATISTICS")
    if primary_context_length:
        lines.append(f"Context Length: {primary_context_length} sentences")
    lines.append(f"Mode: {mode_name.replace('_', ' ').upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model type: {model_type}")
    lines.append("")
    
    # Dataset Statistics
    lines.append("Dataset Statistics:")
    lines.append(f"  Total comparisons: {stats_dict['n_total']}")
    if stats_dict.get('n_filtered', 0) > 0:
        filtered_pct = stats_dict['n_filtered'] / stats_dict['n_total'] * 100
        lines.append(f"  Filtered out (failed calculations): {stats_dict['n_filtered']} ({filtered_pct:.1f}%)")
    lines.append(f"  Valid calculations: {stats_dict['n_valid']}")
    lines.append(f"  Complete calculations: {stats_dict['n_complete']}")
    lines.append(f"  Success rate: {stats_dict['success_rate']:.1%}")
    lines.append(f"  Complete rate: {stats_dict['complete_rate']:.1%}")
    
    if context_clipped_count and context_clipped_count > 0:
        lines.append("")
        lines.append("Context Clipping:")
        lines.append(f"  Sentences with clipped context: {context_clipped_count}")
        lines.append(f"  (Context truncated from left to fit model max_length)")
    
    # Context Usage
    if 'n_with_context' in stats_dict:
        lines.append("")
        lines.append("Context Usage:")
        lines.append(f"  With context: {stats_dict['n_with_context']}")
        lines.append(f"  Without context: {stats_dict['n_without_context']}")
    
    # Code-Switched Translation Surprisal
    lines.append("")
    lines.append("Code-Switched Translation Surprisal:")
    lines.append(f"  Mean:   {stats_dict['cs_surprisal_mean']:.4f}")
    lines.append(f"  Median: {stats_dict['cs_surprisal_median']:.4f}")
    lines.append(f"  Std:    {stats_dict['cs_surprisal_std']:.4f}")
    
    # Monolingual Baseline Surprisal
    lines.append("")
    lines.append("Monolingual Baseline Surprisal:")
    lines.append(f"  Mean:   {stats_dict['mono_surprisal_mean']:.4f}")
    lines.append(f"  Median: {stats_dict['mono_surprisal_median']:.4f}")
    lines.append(f"  Std:    {stats_dict['mono_surprisal_std']:.4f}")
    
    # Difference
    lines.append("")
    lines.append("  Difference (CS - Monolingual):")
    lines.append(f"  Mean:   {stats_dict['difference_mean']:.4f}")
    lines.append(f"  Median: {stats_dict['difference_median']:.4f}")
    lines.append(f"  Std:    {stats_dict['difference_std']:.4f}")
    
    # Statistical Tests
    lines.append("")
    lines.append("Paired t-test:")
    lines.append(f"  t-statistic: {stats_dict['ttest_statistic']:.4f}")
    lines.append(f"  p-value:     {stats_dict['ttest_pvalue']:.6f}")
    
    # Effect Size
    lines.append("")
    lines.append("Effect Size:")
    lines.append(f"  Cohen's d: {stats_dict['cohens_d']:.4f}")
    
    # All Context Lengths Summary
    if len(context_lengths) > 1:
        lines.append("")
        lines.append("=" * 80)
        lines.append("STATISTICS FOR ALL CONTEXT LENGTHS")
        lines.append("=" * 80)
        lines.append("")
        for ctx_len in context_lengths:
            if ctx_len in all_stats:
                ctx_stats = all_stats[ctx_len]
                lines.append(f"Context Length {ctx_len}:")
                lines.append(f"  Complete calculations: {ctx_stats['n_complete']}")
                lines.append(f"  CS Mean: {ctx_stats['cs_surprisal_mean']:.4f}")
                lines.append(f"  Mono Mean: {ctx_stats['mono_surprisal_mean']:.4f}")
                lines.append(f"  Difference Mean: {ctx_stats['difference_mean']:.4f}")
                lines.append(f"  p-value: {ctx_stats['ttest_pvalue']:.6f}")
                lines.append("")
    
    return "\n".join(lines)

