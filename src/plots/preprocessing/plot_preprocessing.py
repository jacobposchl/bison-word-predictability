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
from collections import Counter
from typing import List

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
    
    # Create figure with single-column format
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # Prepare data for stacked bar chart
    x = np.arange(len(groups))
    width = 0.6
    
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
    ax.set_xlabel('Speaker Group', fontsize=9, fontweight='medium')
    ax.set_ylabel('Number of Sentences', fontsize=9, fontweight='medium')
    ax.set_title('Matrix Language Distribution', 
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    # Legend to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             frameon=True, fancybox=True, shadow=True, 
             fontsize=8, framealpha=0.95)
    
    # Grid styling
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = os.path.join(figures_dir, 'matrix_language_distribution.png')
    output_path_pdf = os.path.join(figures_dir, 'matrix_language_distribution.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix language distribution plot to {output_path_png} and {output_path_pdf}")


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
    
    # Create figure (single-column format)
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
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
    
    # Set x-axis ticks every 10 positions for cleaner look
    ax.set_xticks(range(0, 51, 10))
    ax.tick_params(axis='both', labelsize=8)
    
    ax.set_xlabel('Switch Position (word index)', fontsize=9, fontweight='medium')
    ax.set_ylabel('Count', fontsize=9, fontweight='medium')
    ax.set_title('Code-Switch Position Distribution', 
                fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=7, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = os.path.join(figures_dir, 'switch_position.png')
    output_path_pdf = os.path.join(figures_dir, 'switch_position.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved switch position plot to {output_path_png} and {output_path_pdf}")


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
    
    # Create figure (single-column format)
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
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
    
    ax.set_ylabel('Number of CS Sentences', fontsize=9, fontweight='medium')
    ax.set_xlabel('Speaker Group', fontsize=9, fontweight='medium')
    ax.set_title('Participant Variation', fontsize=10, fontweight='bold', pad=10)
    ax.set_xticklabels(labels_for_box, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    # Professional styling
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = os.path.join(figures_dir, 'participant_variation.png')
    output_path_pdf = os.path.join(figures_dir, 'participant_variation.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved participant variation plot to {output_path_png} and {output_path_pdf}")


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
    
    # Create figure (single-column format)
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # Plot KDE (kernel density estimation) for smoother distributions
    # Use lower alpha for better visibility of overlapping distributions
    for group in groups:
        group_df = df[df['group'] == group]
        if len(group_df) > 0:
            # Use seaborn's kdeplot for smoother distribution with increased transparency
            sns.kdeplot(data=group_df, x='ratio_clipped', label=group, 
                       color=colors.get(group, cb_colors[2]), ax=ax, linewidth=2.5, 
                       alpha=0.3, fill=True, common_norm=False)
    
    ax.set_xlabel('Cantonese:English Ratio', fontsize=9, fontweight='medium')
    ax.set_ylabel('Density', fontsize=9, fontweight='medium')
    ax.set_title('Language Ratio Distribution', 
                fontsize=10, fontweight='bold', pad=10)
    
    # Set x-axis ticks every 10 units
    x_min = int(df['ratio_clipped'].min() // 10) * 10
    x_max = int(df['ratio_clipped'].max() // 10 + 1) * 10
    ax.set_xticks(range(x_min, x_max + 1, 10))
    ax.tick_params(axis='both', labelsize=8)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             frameon=True, fancybox=True, shadow=True, 
             fontsize=8, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = os.path.join(figures_dir, 'language_ratio_distribution.png')
    output_path_pdf = os.path.join(figures_dir, 'language_ratio_distribution.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Cantonese to English ratio distribution plot to {output_path_png} and {output_path_pdf}")


def plot_participant_sentence_counts(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Plot sentence counts by participant and type (English, Monolingual Cantonese, Code-switched).

    Args:
        df: DataFrame with all sentences (must have 'participant_id' and 'pattern' columns)
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)

    def categorize_sentence(pattern):
        if pd.isna(pattern):
            return 'Other'
        pattern_str = str(pattern)
        has_c = 'C' in pattern_str
        has_e = 'E' in pattern_str
        if has_c and has_e:
            return 'Code-switched'
        elif has_c and not has_e:
            return 'Monolingual Cantonese'
        elif has_e and not has_c:
            return 'English'
        else:
            return 'Other'

    df = df.copy()
    df['sentence_type'] = df['pattern'].apply(categorize_sentence)

    counts = df.groupby(['participant_id', 'sentence_type']).size().unstack(fill_value=0)

    for cat in ['English', 'Monolingual Cantonese', 'Code-switched']:
        if cat not in counts.columns:
            counts[cat] = 0

    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(16, 4))

    x = np.arange(len(counts))
    width = 0.25

    cb_colors = sns.color_palette("YlOrBr", 3)

    ax.bar(x - width, counts['English'], width, label='English',
           color=cb_colors[0], alpha=0.9, edgecolor='white', linewidth=1.5)
    ax.bar(x, counts['Monolingual Cantonese'], width, label='Monolingual Cantonese',
           color=cb_colors[1], alpha=0.9, edgecolor='white', linewidth=1.5)
    ax.bar(x + width, counts['Code-switched'], width, label='Code-switched',
           color=cb_colors[2], alpha=0.9, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Participant ID', fontsize=9, fontweight='medium')
    ax.set_ylabel('Number of Sentences', fontsize=9, fontweight='medium')
    ax.set_title('Sentence Counts by Participant and Type', fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='both', labelsize=8)

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
              fontsize=8, framealpha=0.95)

    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')

    plt.tight_layout()

    output_path_png = os.path.join(figures_dir, 'participant_sentence_counts.png')
    output_path_pdf = os.path.join(figures_dir, 'participant_sentence_counts.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    logger.info(f"Saved participant sentence counts plot to {output_path_png} and {output_path_pdf}")


def plot_pos_distribution(
    mono_csv_path: str,
    cs_csv_path: str,
    figures_dir: str,
    top_n: int = 15
) -> None:
    """
    Plot POS tag distribution across groups (Heritage, Homeland, Immersed).
    
    Creates a faceted stacked bar chart with one subplot per group.
    Within each subplot: two stacked bars (Code-Switched vs Monolingual).
    Each bar shows the percentage composition of 3 POS categories: Verb, Noun, Other.
    
    Args:
        mono_csv_path: Path to cantonese_monolingual_WITHOUT_fillers.csv
        cs_csv_path: Path to cantonese_translated_WITHOUT_fillers.csv
        figures_dir: Directory to save figures
        top_n: Not used (kept for backward compatibility)
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    logger.info("Loading POS data...")
    mono_df = pd.read_csv(mono_csv_path)
    cs_df = pd.read_csv(cs_csv_path)
    
    # Helper function to parse POS tags
    def parse_pos_tags(pos_string) -> List[str]:
        if pd.isna(pos_string) or not isinstance(pos_string, str):
            return []
        return pos_string.split()
    
    # Helper function to categorize POS tags into 3 categories
    def categorize_pos_tag(pos_tag: str) -> str:
        """Map POS tags to Verb, Noun, or Other."""
        if pos_tag in ['VERB', 'AUX']:
            return 'Verb'
        elif pos_tag in ['NOUN', 'PROPN']:
            return 'Noun'
        else:
            return 'Other'
    
    # Calculate POS distributions for each group and sentence type
    results = []
    groups = ['Heritage', 'Homeland', 'Immersed']
    
    for group in groups:
        # Process monolingual sentences
        mono_group = mono_df[mono_df['group'] == group]
        mono_pos_tags = []
        for pos_string in mono_group['pos']:
            mono_pos_tags.extend(parse_pos_tags(pos_string))
        
        # Categorize and count
        mono_categorized = [categorize_pos_tag(tag) for tag in mono_pos_tags]
        mono_counter = Counter(mono_categorized)
        mono_total = sum(mono_counter.values())
        
        for pos_category, count in mono_counter.items():
            results.append({
                'group': group,
                'sentence_type': 'Monolingual',
                'pos_category': pos_category,
                'count': count,
                'percentage': (count / mono_total * 100) if mono_total > 0 else 0
            })
        
        # Process code-switched sentences
        cs_group = cs_df[cs_df['group'] == group]
        cs_pos_tags = []
        for pos_string in cs_group['translated_pos']:
            cs_pos_tags.extend(parse_pos_tags(pos_string))
        
        # Categorize and count
        cs_categorized = [categorize_pos_tag(tag) for tag in cs_pos_tags]
        cs_counter = Counter(cs_categorized)
        cs_total = sum(cs_counter.values())
        
        for pos_category, count in cs_counter.items():
            results.append({
                'group': group,
                'sentence_type': 'Code-Switched',
                'pos_category': pos_category,
                'count': count,
                'percentage': (count / cs_total * 100) if cs_total > 0 else 0
            })
    
    distribution_df = pd.DataFrame(results)
    
    logger.info(f"Creating POS distribution plot (3 categories: Verb, Noun, Other)...")
    
    # Create single plot with grouped bars (single-column format)
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # Use distinct, colorblind-friendly colors for the 3 categories
    colors = {
        'Verb': '#E74C3C',      # Red
        'Noun': '#3498DB',      # Blue
        'Other': '#95A5A6'      # Gray
    }
    categories = ['Verb', 'Noun', 'Other']
    groups = ['Heritage', 'Homeland', 'Immersed']
    
    # Set up bar positions
    bar_width = 0.12
    spacing = 0.05
    
    # Plot grouped stacked bars
    for group_idx, group in enumerate(groups):
        group_data = distribution_df[distribution_df['group'] == group]
        
        # Get data for Code-Switched and Monolingual
        for sent_idx, sent_type in enumerate(['Code-Switched', 'Monolingual']):
            sent_data = group_data[group_data['sentence_type'] == sent_type]
            
            # Get percentages for each category
            percentages = {}
            for cat in categories:
                cat_data = sent_data[sent_data['pos_category'] == cat]
                percentages[cat] = cat_data['percentage'].values[0] if len(cat_data) > 0 else 0
            
            # Calculate x position
            x_pos = group_idx * (2 * bar_width + spacing) + sent_idx * bar_width
            
            # Create stacked bars
            bottom = 0
            for cat in categories:
                ax.bar(x_pos, percentages[cat], bar_width, 
                      bottom=bottom, 
                      color=colors[cat],
                      edgecolor='white',
                      linewidth=1)
                bottom += percentages[cat]
    
    # Customize plot
    ax.set_ylabel('% of POS Categories', fontsize=9, fontweight='medium')
    ax.set_ylim(0, 100)
    
    # Set x-axis ticks at the center of each group's pair of bars
    group_centers = [i * (2 * bar_width + spacing) + bar_width/2 for i in range(len(groups))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(groups, fontsize=8, fontweight='medium')
    
    ax.tick_params(axis='both', labelsize=8, length=3)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Professional styling matching other preprocessing plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    # Add legend to the right
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[cat], edgecolor='white', linewidth=1, label=cat) 
               for cat in categories]
    
    ax.legend(handles=handles, loc='center left', 
             bbox_to_anchor=(1.02, 0.5),
             frameon=True, fontsize=8, 
             fancybox=True, shadow=True, framealpha=0.95)
    
    # Title
    ax.set_title('POS Distribution by Group', 
                fontsize=10, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = os.path.join(figures_dir, 'pos_distribution_by_group.png')
    output_path_pdf = os.path.join(figures_dir, 'pos_distribution_by_group.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved POS distribution plot to {output_path_png} and {output_path_pdf}")

