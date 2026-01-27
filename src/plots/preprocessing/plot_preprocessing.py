"""
Visualization functions for code-switching preprocessing data.

This module provides functions for creating plots and printing analysis summaries
for code-switching data (WITHOUT fillers only).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from typing import List, Dict, Optional
import logging
import os
import re

logger = logging.getLogger(__name__)


def print_analysis_summary_simple(without_fillers: List[Dict]) -> None:
    """
    Print detailed text-based analysis summary for WITHOUT fillers data only.
    
    Args:
        without_fillers: List of sentences with fillers excluded
    """
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Basic dataset statistics
    print(f"\nDataset size:")
    print(f"  WITHOUT fillers: {len(without_fillers)} code-switching sentences")
    
    # Group distributions
    print(f"\n" + "-"*80)
    print("Sentences by Speaker Group:")
    print("-"*80)
    
    group_counts = Counter([s['group'] for s in without_fillers])
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} ({count/len(without_fillers)*100:.1f}%)")
    
    # Matrix language distributions
    print(f"\n" + "-"*80)
    print("Matrix Language Distribution:")
    print("-"*80)
    
    matrix_counts = Counter([s['matrix_language'] for s in without_fillers])
    print(f"  Cantonese: {matrix_counts['Cantonese']} ({matrix_counts['Cantonese']/len(without_fillers)*100:.1f}%)")
    print(f"  English: {matrix_counts['English']} ({matrix_counts['English']/len(without_fillers)*100:.1f}%)")
    if 'Equal' in matrix_counts:
        print(f"  Equal: {matrix_counts['Equal']} ({matrix_counts['Equal']/len(without_fillers)*100:.1f}%)")
    
    # Detailed breakdown by group AND matrix language
    print(f"\n" + "-"*80)
    print("Matrix Language by Participant Group:")
    print("-"*80)
    
    for group in sorted(groups):
        group_sentences = [s for s in without_fillers if s['group'] == group]
        if group_sentences:
            cant_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            print(
                f"  {group:18} (n={len(group_sentences):4}): "
                f"Cantonese {cant_matrix:4} ({cant_matrix/len(group_sentences)*100:5.1f}%)  |  "
                f"English {eng_matrix:4} ({eng_matrix/len(group_sentences)*100:5.1f}%)  |  "
                f"Equal {equal_matrix:4} ({equal_matrix/len(group_sentences)*100:5.1f}%)"
            )
    
    # Print detailed comparison table
    print("\n" + "="*80)
    print("MATRIX LANGUAGE DISTRIBUTION TABLE")
    print("="*80)
    print(f"{'Group':<12} {'Total':<8} {'Cantonese':<20} {'English':<20} {'Equal':<15}")
    print("-"*80)
    
    for group in groups:
        group_sentences = [s for s in without_fillers if s['group'] == group]
        if group_sentences:
            total = len(group_sentences)
            cant_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_count = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            
            print(
                f"{group:<12} {total:<8} "
                f"{cant_count:4} ({cant_count/total*100:5.1f}%){'':<8} "
                f"{eng_count:4} ({eng_count/total*100:5.1f}%){'':<8} "
                f"{equal_count:4} ({equal_count/total*100:5.1f}%)"
            )


def _parse_pattern_segments(pattern: str) -> List[tuple]:
    """Parse pattern string into segments (e.g., 'C5-E3-C2' -> [('C', 5), ('E', 3), ('C', 2)])."""
    if not pattern or pattern == 'FILLER_ONLY':
        return []
    segments = re.findall(r'([CE])(\d+)', pattern)
    return [(lang, int(count)) for lang, count in segments]


def _count_switches(pattern: str) -> int:
    """Count number of language switches in a pattern."""
    segments = _parse_pattern_segments(pattern)
    if len(segments) <= 1:
        return 0
    switches = 0
    for i in range(1, len(segments)):
        if segments[i][0] != segments[i-1][0]:
            switches += 1
    return switches


def _get_pattern_length(pattern: str) -> int:
    """Get total word count from pattern."""
    segments = _parse_pattern_segments(pattern)
    return sum(count for _, count in segments)


def _get_english_word_count(pattern: str) -> int:
    """Get total English words in pattern."""
    segments = _parse_pattern_segments(pattern)
    return sum(count for lang, count in segments if lang == 'E')


def _get_cantonese_word_count(pattern: str) -> int:
    """Get total Cantonese words in pattern."""
    segments = _parse_pattern_segments(pattern)
    return sum(count for lang, count in segments if lang == 'C')


def _get_switch_position(pattern: str, sentence_length: Optional[int] = None) -> Optional[float]:
    """Get normalized switch position (0-1) from pattern. Returns position of first switch."""
    segments = _parse_pattern_segments(pattern)
    if len(segments) < 2:
        return None
    
    # Find first switch (first E segment after C, or first C segment after E)
    first_lang = segments[0][0]
    first_count = segments[0][1]
    
    # If starts with C and switches to E, switch position is after first_count words
    if first_lang == 'C' and len(segments) > 1 and segments[1][0] == 'E':
        switch_index = first_count
        if sentence_length:
            return switch_index / sentence_length if sentence_length > 0 else None
        # If no sentence length, use pattern length
        pattern_len = _get_pattern_length(pattern)
        return switch_index / pattern_len if pattern_len > 0 else None
    
    # If starts with E, switch is at position 0
    if first_lang == 'E':
        return 0.0
    
    return None


def _get_raw_switch_position(pattern: str) -> Optional[int]:
    """Get raw switch position (word index) from pattern. Returns position of first switch."""
    segments = _parse_pattern_segments(pattern)
    if len(segments) < 2:
        return None
    
    # Find first switch (first E segment after C, or first C segment after E)
    first_lang = segments[0][0]
    first_count = segments[0][1]
    
    # If starts with C and switches to E, switch position is after first_count words
    if first_lang == 'C' and len(segments) > 1 and segments[1][0] == 'E':
        return first_count
    
    # If starts with E, switch is at position 0
    if first_lang == 'E':
        return 0
    
    return None


def _categorize_pattern(pattern: str) -> str:
    """Categorize pattern into common types (e.g., 'C-E', 'C-E-C', 'E-C', etc.)."""
    segments = _parse_pattern_segments(pattern)
    if not segments:
        return 'Other'
    
    # Get simplified pattern (just language sequence)
    lang_sequence = '-'.join([lang for lang, _ in segments])
    
    # Map to common categories
    if lang_sequence == 'C-E':
        return 'C-E'
    elif lang_sequence == 'E-C':
        return 'E-C'
    elif lang_sequence == 'C-E-C':
        return 'C-E-C'
    elif lang_sequence == 'E-C-E':
        return 'E-C-E'
    elif lang_sequence.startswith('C-') and lang_sequence.endswith('-C'):
        return 'C-E-C+'
    elif lang_sequence.startswith('E-') and lang_sequence.endswith('-E'):
        return 'E-C-E+'
    elif len(segments) == 2:
        return lang_sequence
    else:
        return 'Complex'


def plot_matrix_language_distribution(
    df: pd.DataFrame,
    figures_dir: str
) -> None:
    """
    Plot matrix language distribution for code-switching data.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'group' and 'matrix_language' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Count matrix languages by group
    group_matrix_counts = {}
    for group in groups:
        group_sentences = df[df['group'] == group]
        if len(group_sentences) > 0:
            cant_count = len(group_sentences[group_sentences['matrix_language'] == 'Cantonese'])
            eng_count = len(group_sentences[group_sentences['matrix_language'] == 'English'])
            equal_count = len(group_sentences[group_sentences['matrix_language'] == 'Equal'])
            group_matrix_counts[group] = {
                'Cantonese': cant_count,
                'English': eng_count,
                'Equal': equal_count,
                'Total': len(group_sentences)
            }
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Prepare data for stacked bar chart
    x = np.arange(len(groups))
    width = 0.65
    
    cantonese_counts = [group_matrix_counts.get(g, {}).get('Cantonese', 0) for g in groups]
    english_counts = [group_matrix_counts.get(g, {}).get('English', 0) for g in groups]
    equal_counts = [group_matrix_counts.get(g, {}).get('Equal', 0) for g in groups]
    
    # Professional color palette
    colors = {
        'Cantonese': '#3498db',  # Blue
        'English': '#e74c3c',     # Red
        'Equal': '#95a5a6'        # Gray
    }
    
    # Create stacked bars with better styling
    p1 = ax.bar(x, cantonese_counts, width, label='Cantonese', 
                color=colors['Cantonese'], edgecolor='white', linewidth=1.5, alpha=0.9)
    p2 = ax.bar(x, english_counts, width, bottom=cantonese_counts, label='English', 
                color=colors['English'], edgecolor='white', linewidth=1.5, alpha=0.9)
    p3 = ax.bar(x, equal_counts, width, 
                bottom=np.array(cantonese_counts) + np.array(english_counts), 
                label='Equal', color=colors['Equal'], edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Styling
    ax.set_xlabel('Speaker Group', fontsize=13, fontweight='medium')
    ax.set_ylabel('Number of Sentences', fontsize=13, fontweight='medium')
    ax.set_title('Matrix Language Distribution by Speaker Group', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    
    # Improve legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=11, framealpha=0.95)
    
    # Better grid styling
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    # Add value labels on bars with better positioning
    for i, group in enumerate(groups):
        total = group_matrix_counts.get(group, {}).get('Total', 0)
        if total > 0:
            # Calculate percentages for each segment
            cant_pct = (cantonese_counts[i] / total * 100) if total > 0 else 0
            eng_pct = (english_counts[i] / total * 100) if total > 0 else 0
            equal_pct = (equal_counts[i] / total * 100) if total > 0 else 0
            
            # Add total label at top
            ax.text(i, total + total*0.02, f'n={total}', ha='center', va='bottom', 
                   fontsize=10, fontweight='medium')
            
            # Add percentage labels on each segment (if segment is large enough)
            if cantonese_counts[i] > 0 and cant_pct > 5:
                ax.text(i, cantonese_counts[i] / 2, f'{cant_pct:.0f}%', 
                       ha='center', va='center', fontsize=9, fontweight='medium', color='white')
            if english_counts[i] > 0 and eng_pct > 5:
                ax.text(i, cantonese_counts[i] + english_counts[i] / 2, f'{eng_pct:.0f}%', 
                       ha='center', va='center', fontsize=9, fontweight='medium', color='white')
            if equal_counts[i] > 0 and equal_pct > 5:
                ax.text(i, cantonese_counts[i] + english_counts[i] + equal_counts[i] / 2, f'{equal_pct:.0f}%', 
                       ha='center', va='center', fontsize=9, fontweight='medium', color='white')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'matrix_language_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix language distribution plot to {output_path}")


def plot_pattern_complexity(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot distribution of code-switching pattern complexity (number of switches).
    
    Args:
        df: DataFrame with code-switching sentences (must have 'pattern' and 'group' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate number of switches for each sentence
    df = df.copy()
    df['num_switches'] = df['pattern'].apply(_count_switches)
    
    # Categorize switches (0, 1, 2, 3+)
    df['switch_category'] = df['num_switches'].apply(
        lambda x: '0' if x == 0 else ('1' if x == 1 else ('2' if x == 2 else '3+'))
    )
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    categories = ['0', '1', '2', '3+']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Stacked bar chart by group
    ax1 = axes[0]
    x = np.arange(len(groups))
    width = 0.6
    
    # Calculate counts for each category by group
    bottom = np.zeros(len(groups))
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, cat in enumerate(categories):
        counts = []
        for group in groups:
            group_df = df[df['group'] == group]
            count = len(group_df[group_df['switch_category'] == cat])
            counts.append(count)
        
        ax1.bar(x, counts, width, bottom=bottom, label=f'{cat} switch{"es" if cat != "1" else ""}', 
                color=colors[i % len(colors)])
        bottom += np.array(counts)
    
    ax1.set_xlabel('Speaker Group', fontsize=12)
    ax1.set_ylabel('Number of Sentences', fontsize=12)
    ax1.set_title('Code-Switch Complexity by Speaker Group', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Histogram of switch counts
    ax2 = axes[1]
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            ax2.hist(group_df['num_switches'], bins=range(0, max(df['num_switches'])+2), 
                    alpha=0.6, label=group, density=True)
    
    ax2.set_xlabel('Number of Switches', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Switch Counts', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'pattern_complexity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pattern complexity plot to {output_path}")


def plot_switch_position(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot distribution of code-switch positions in sentences (raw word positions).
    
    Args:
        df: DataFrame with code-switching sentences (must have 'switch_index' or 'pattern' column, and 'group' column)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    df = df.copy()
    
    # Check if switch_index column exists (from cantonese_translated_WITHOUT_fillers.csv)
    if 'switch_index' in df.columns:
        # Use the switch_index column directly
        df['switch_position'] = df['switch_index']
        # Filter out any NaN values
        df = df[df['switch_position'].notna()].copy()
    else:
        # Fall back to calculating from pattern (for all_sentences.csv)
        # Filter to only sentences with actual switches (pattern must have both C and E)
        df = df[
            df['pattern'].str.contains('C', na=False) & 
            df['pattern'].str.contains('E', na=False) &
            (df['pattern'] != 'FILLER_ONLY')
        ].copy()
        
        # Calculate raw switch positions (word index, not normalized)
        df['switch_position'] = df['pattern'].apply(_get_raw_switch_position)
        df = df[df['switch_position'].notna()].copy()
    
    if len(df) == 0:
        logger.warning("No valid switch positions found, skipping switch position plot")
        return
    
    # Create figure (single plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Plot single distribution for all switch positions (all groups combined)
    sns.histplot(data=df, x='switch_position', bins='auto', kde=True,
                color='#3498db', ax=ax, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Set x-axis limits from minimum switch index to 50
    min_pos = int(df['switch_position'].min())
    ax.set_xlim(left=min_pos, right=50)
    
    ax.set_xlabel('Switch Position (word index)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Count', fontsize=13, fontweight='medium')
    ax.set_title('Distribution of Code-Switch Positions', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'switch_position.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved switch position plot to {output_path}")


def plot_pattern_type_distribution(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot frequency distribution of common code-switching pattern types.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'pattern' and 'group' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Categorize patterns
    df = df.copy()
    df['pattern_type'] = df['pattern'].apply(_categorize_pattern)
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    pattern_types = sorted(df['pattern_type'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pattern_types))
    width = 0.25
    
    # Calculate counts for each pattern type by group
    for i, group in enumerate(groups):
        group_df = df[df['group'] == group]
        counts = [len(group_df[group_df['pattern_type'] == pt]) for pt in pattern_types]
        ax.bar(x + i*width, counts, width, label=group, alpha=0.8)
    
    ax.set_xlabel('Pattern Type', fontsize=12)
    ax.set_ylabel('Number of Sentences', fontsize=12)
    ax.set_title('Code-Switching Pattern Type Distribution by Speaker Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pattern_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'pattern_type_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pattern type distribution plot to {output_path}")


def plot_sentence_length_distribution(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot distribution of sentence lengths (word counts).
    
    Args:
        df: DataFrame with code-switching sentences (must have 'reconstructed_sentence', 'group', and 'matrix_language' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate sentence lengths
    df = df.copy()
    df['sentence_length'] = df['reconstructed_sentence'].str.split().str.len()
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    matrix_lang_colors = {'Cantonese': '#3498db', 'English': '#e74c3c', 'Equal': '#95a5a6'}
    group_colors = {'Homeland': '#e74c3c', 'Heritage': '#3498db', 'Immersed': '#2ecc71'}
    
    # Create figure (single plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Violin plot by group
    data_for_violin = [df[df['group'] == group]['sentence_length'].values for group in groups if len(df[df['group'] == group]) > 0]
    labels_for_violin = [group for group in groups if len(df[df['group'] == group]) > 0]
    
    parts = ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)), 
                          showmeans=True, showmedians=True)
    
    # Color each violin by group
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(group_colors.get(labels_for_violin[i], '#3498db'))
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')
        pc.set_linewidth(1.5)
    
    # Style the means and medians
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            parts[partname].set_edgecolor('#333333')
            parts[partname].set_linewidth(1.5)
    
    ax.set_xticks(range(len(labels_for_violin)))
    ax.set_xticklabels(labels_for_violin, fontsize=12)
    ax.set_ylabel('Sentence Length (words)', fontsize=13, fontweight='medium')
    ax.set_xlabel('Speaker Group', fontsize=13, fontweight='medium')
    ax.set_title('Sentence Length Distribution by Speaker Group', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'sentence_length_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sentence length distribution plot to {output_path}")


def plot_participant_variation(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot participant-level variation in code-switching behavior.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'participant_id' and 'group' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Count sentences per participant
    participant_counts = df.groupby(['participant_id', 'group']).size().reset_index(name='num_sentences')
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    group_colors = {'Homeland': '#e74c3c', 'Heritage': '#3498db', 'Immersed': '#2ecc71'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Box plot by group (without outliers)
    data_for_box = [participant_counts[participant_counts['group'] == group]['num_sentences'].values 
                    for group in groups if len(participant_counts[participant_counts['group'] == group]) > 0]
    labels_for_box = [group for group in groups if len(participant_counts[participant_counts['group'] == group]) > 0]
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True, 
                   showfliers=False,  # Remove outlier visuals
                   widths=0.6,
                   medianprops=dict(linewidth=2.5, color='#333333'),
                   boxprops=dict(linewidth=1.5),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
    
    # Color each box by group
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(group_colors.get(labels_for_box[i], '#9b59b6'))
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    
    # Style median, whiskers, and caps
    for element in ['medians', 'whiskers', 'caps']:
        if element in bp:
            for item in bp[element]:
                item.set_color('#333333')
    
    ax.set_ylabel('Number of Code-Switched Sentences', fontsize=13, fontweight='medium')
    ax.set_xlabel('Speaker Group', fontsize=13, fontweight='medium')
    ax.set_title('Participant-Level Variation in Code-Switching', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticklabels(labels_for_box, fontsize=12)
    
    # Professional styling
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'participant_variation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved participant variation plot to {output_path}")


def plot_matrix_language_proportions(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot overall matrix language proportions.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'matrix_language' column)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Count matrix languages
    matrix_counts = df['matrix_language'].value_counts()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Pie chart
    ax1 = axes[0]
    colors = {'Cantonese': '#2ecc71', 'English': '#e74c3c', 'Equal': '#95a5a6'}
    pie_colors = [colors.get(lang, '#3498db') for lang in matrix_counts.index]
    
    wedges, texts, autotexts = ax1.pie(matrix_counts.values, labels=matrix_counts.index, 
                                       autopct='%1.1f%%', colors=pie_colors, startangle=90)
    ax1.set_title('Overall Matrix Language Distribution', fontsize=14, fontweight='bold')
    
    # Subplot 2: Bar chart
    ax2 = axes[1]
    bars = ax2.bar(matrix_counts.index, matrix_counts.values, 
                   color=[colors.get(lang, '#3498db') for lang in matrix_counts.index], alpha=0.8)
    ax2.set_ylabel('Number of Sentences', fontsize=12)
    ax2.set_xlabel('Matrix Language', fontsize=12)
    ax2.set_title('Matrix Language Counts', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'matrix_language_proportions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix language proportions plot to {output_path}")


def plot_pattern_length_vs_switch_position(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot relationship between pattern length and switch position.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'pattern', 'reconstructed_sentence', and 'group' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate pattern length and switch position
    df = df.copy()
    df['pattern_length'] = df['pattern'].apply(_get_pattern_length)
    df['sentence_length'] = df['reconstructed_sentence'].str.split().str.len()
    df['switch_position'] = df.apply(
        lambda row: _get_switch_position(row['pattern'], row['sentence_length']), axis=1
    )
    df = df[df['switch_position'].notna()].copy()
    
    if len(df) == 0:
        logger.warning("No valid switch positions found, skipping pattern length vs switch position plot")
        return
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    colors = {'Homeland': '#e74c3c', 'Heritage': '#3498db', 'Immersed': '#2ecc71'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot by group
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            ax.scatter(group_df['pattern_length'], group_df['switch_position'], 
                      alpha=0.6, label=group, color=colors.get(group, '#95a5a6'), s=50)
    
    ax.set_xlabel('Pattern Length (total words)', fontsize=12)
    ax.set_ylabel('Normalized Switch Position', fontsize=12)
    ax.set_title('Pattern Length vs. Switch Position', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'pattern_length_vs_switch_position.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pattern length vs switch position plot to {output_path}")


def plot_code_switch_density(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot distribution of Cantonese to English word ratio per sentence by group.
    
    Args:
        df: DataFrame with code-switching sentences (must have 'pattern' and 'group' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calculate English and Cantonese word counts
    df = df.copy()
    df['english_word_count'] = df['pattern'].apply(_get_english_word_count)
    df['cantonese_word_count'] = df['pattern'].apply(_get_cantonese_word_count)
    
    # Calculate ratio: Cantonese / English
    # Filter out sentences with 0 English words to avoid division by zero
    df = df[df['english_word_count'] > 0].copy()
    df['cantonese_english_ratio'] = df['cantonese_word_count'] / df['english_word_count']
    
    if len(df) == 0:
        logger.warning("No valid sentences found for ratio calculation, skipping plot")
        return
    
    # Clip extreme outliers to 95th percentile for better visualization
    # This helps focus on the main distribution while still showing most data
    ratio_95th = df['cantonese_english_ratio'].quantile(0.95)
    df['ratio_clipped'] = df['cantonese_english_ratio'].clip(upper=ratio_95th)
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    colors = {'Homeland': '#e74c3c', 'Heritage': '#3498db', 'Immersed': '#2ecc71'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE (kernel density estimation) for smoother distributions
    # Use lower alpha for better visibility of overlapping distributions
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            # Use seaborn's kdeplot for smoother distribution with increased transparency
            sns.kdeplot(data=group_df, x='ratio_clipped', label=group, 
                       color=colors.get(group, '#95a5a6'), ax=ax, linewidth=2.5, 
                       alpha=0.3, fill=True, common_norm=False)
    
    ax.set_xlabel('Cantonese to English Word Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Cantonese to English Word Ratio per Sentence\nby Speaker Group', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(left=0)  # Start x-axis at 0 for better readability
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'language_ratio_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Cantonese to English ratio distribution plot to {output_path}")

