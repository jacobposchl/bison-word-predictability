"""
Visualization functions for code-switching preprocessing data.

This module provides functions for creating plots and printing analysis summaries for code-switching data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import os

from ...utils.data_helpers import (
    get_english_word_count,
    get_cantonese_word_count
)

logger = logging.getLogger(__name__)


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
    width = 0.45  # Narrower bars to make room for labels
    
    cantonese_counts = [group_matrix_counts.get(g, {}).get('Cantonese', 0) for g in groups]
    english_counts = [group_matrix_counts.get(g, {}).get('English', 0) for g in groups]
    equal_counts = [group_matrix_counts.get(g, {}).get('Equal', 0) for g in groups]
    
    # ColorBrewer YlOrBr palette (Yellow-Orange-Brown)
    cb_colors = sns.color_palette("YlOrBr", 3)
    colors = {
        'Cantonese': cb_colors[0],  # Light yellow
        'English': cb_colors[1],     # Orange
        'Equal': cb_colors[2]        # Brown
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
    ax.set_xlabel('Speaker Group', fontsize=14, fontweight='medium')
    ax.set_ylabel('Number of Sentences', fontsize=14, fontweight='medium')
    ax.set_title('Matrix Language Distribution by Speaker Group', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    
    # Improve legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=14, framealpha=0.95)
    
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
                   fontsize=12, fontweight='medium')
            
            # All percentage labels positioned to the right of bars with brackets
            # Cantonese segment
            if cantonese_counts[i] > 0:
                cant_y_pos = cantonese_counts[i] / 2
                # Draw bracket line from bar to text
                ax.plot([i + width/2, i + width/2 + 0.05, i + width/2 + 0.05], 
                       [cant_y_pos, cant_y_pos, cant_y_pos], 
                       color='#333333', linewidth=1.2, alpha=0.7)
                ax.text(i + width/2 + 0.08, cant_y_pos, f'{cant_pct:.0f}%', 
                       ha='left', va='center', fontsize=12, fontweight='medium')
            
            # English segment
            if english_counts[i] > 0:
                eng_y_pos = cantonese_counts[i] + english_counts[i] / 2
                # Draw bracket line from bar to text
                ax.plot([i + width/2, i + width/2 + 0.05, i + width/2 + 0.05], 
                       [eng_y_pos, eng_y_pos, eng_y_pos], 
                       color='#333333', linewidth=1.2, alpha=0.7)
                ax.text(i + width/2 + 0.08, eng_y_pos, f'{eng_pct:.0f}%', 
                       ha='left', va='center', fontsize=12, fontweight='medium')
            
            # Equal segment
            if equal_counts[i] > 0:
                equal_y_pos = cantonese_counts[i] + english_counts[i] + equal_counts[i] / 2
                # Draw bracket line from bar to text
                ax.plot([i + width/2, i + width/2 + 0.05, i + width/2 + 0.05], 
                       [equal_y_pos, equal_y_pos, equal_y_pos], 
                       color='#333333', linewidth=1.2, alpha=0.7)
                ax.text(i + width/2 + 0.08, equal_y_pos, f'{equal_pct:.0f}%', 
                       ha='left', va='center', fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'matrix_language_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix language distribution plot to {output_path}")


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
    
    # Create figure (single plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    
    # Set up integer bins (one bin per position) for accurate representation
    min_pos = int(df['switch_position'].min())
    max_pos = min(int(df['switch_position'].max()), 50)  # Cap at 50 for visualization
    bins = range(min_pos, max_pos + 2)  # +2 to include max_pos in a bin
    
    # Plot single distribution for all switch positions (all groups combined)
    sns.histplot(data=df, x='switch_position', bins=bins, kde=True,
                color=cb_colors[1], ax=ax, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Add vertical line at x=2 to show minimum cutoff
    ax.axvline(x=2, color='#333333', linestyle='--', linewidth=2, alpha=0.7,
               label='Minimum position (2 words)')
    
    # Set x-axis limits
    ax.set_xlim(left=min_pos - 0.5, right=50)
    
    # Set x-axis ticks every 5 positions
    ax.set_xticks(range(0, 51, 5))
    ax.tick_params(axis='both', labelsize=14)
    
    ax.set_xlabel('Switch Position (word index)', fontsize=14, fontweight='medium')
    ax.set_ylabel('Count', fontsize=14, fontweight='medium')
    ax.set_title('Distribution of Code-Switch Positions', 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=14, framealpha=0.95)
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
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    group_colors = {'Homeland': cb_colors[0], 'Heritage': cb_colors[1], 'Immersed': cb_colors[2]}
    
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
        patch.set_facecolor(group_colors.get(labels_for_box[i], cb_colors[0]))
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    
    # Style median, whiskers, and caps
    for element in ['medians', 'whiskers', 'caps']:
        if element in bp:
            for item in bp[element]:
                item.set_color('#333333')
    
    ax.set_ylabel('Number of Code-Switched Sentences', fontsize=14, fontweight='medium')
    ax.set_xlabel('Speaker Group', fontsize=14, fontweight='medium')
    ax.set_title('Participant-Level Variation in Code-Switching', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticklabels(labels_for_box, fontsize=14)
    
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
    df['english_word_count'] = df['pattern'].apply(get_english_word_count)
    df['cantonese_word_count'] = df['pattern'].apply(get_cantonese_word_count)
    
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
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    colors = {'Homeland': cb_colors[0], 'Heritage': cb_colors[1], 'Immersed': cb_colors[2]}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE (kernel density estimation) for smoother distributions
    # Use lower alpha for better visibility of overlapping distributions
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            # Use seaborn's kdeplot for smoother distribution with increased transparency
            sns.kdeplot(data=group_df, x='ratio_clipped', label=group, 
                       color=colors.get(group, cb_colors[2]), ax=ax, linewidth=2.5, 
                       alpha=0.3, fill=True, common_norm=False)
    
    ax.set_xlabel('Cantonese to English Word Ratio', fontsize=14, fontweight='medium')
    ax.set_ylabel('Density', fontsize=14, fontweight='medium')
    ax.set_title('Cantonese to English Word Ratio Per Code-Switched Sentence', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks every 5 units
    x_min = int(df['ratio_clipped'].min() // 5) * 5
    x_max = int(df['ratio_clipped'].max() // 5 + 1) * 5
    ax.set_xticks(range(x_min, x_max + 1, 5))
    ax.tick_params(axis='both', labelsize=14)
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=14, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'language_ratio_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Cantonese to English ratio distribution plot to {output_path}")

