"""
Visualization functions for matching analysis results.

This module provides functions for creating plots from matching analysis datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


def plot_similarity_distributions_from_csv(
    window_datasets: list,
    output_dir: str
) -> str:
    """
    Create visualization showing similarity score distributions for each window size
    from CSV files.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity distribution plots from CSV files...")
    
    # Set professional style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    
    # Extract data for plotting with proper window size extraction and sorting
    plot_data = []
    window_size_map = {}  # Map window_size -> label for proper sorting
    
    import re
    for dataset_path in window_datasets:
        # Extract window size from filename (e.g., "analysis_dataset_window_2.csv" -> 2)
        match = re.search(r'window_(\d+)', dataset_path.name if isinstance(dataset_path, Path) else str(dataset_path))
        if not match:
            logger.warning(f"Could not extract window size from {dataset_path}, skipping...")
            continue
        
        window_size = int(match.group(1))
        window_label = f'n={window_size}'
        window_size_map[window_size] = window_label
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Extract similarity scores
        if 'similarity' not in df.columns:
            logger.warning(f"No 'similarity' column in {dataset_path}, skipping...")
            continue
        
        similarity_scores = df['similarity'].dropna().tolist()
        
        for score in similarity_scores:
            plot_data.append({
                'window_size': window_size,  # Keep numeric for sorting
                'Window Size': window_label,
                'Similarity Score': score
            })
    
    if not plot_data:
        logger.warning("No similarity scores to plot")
        return ""
    
    df_plot = pd.DataFrame(plot_data)
    
    # Sort by window size to ensure proper order (n=1, n=2, n=3, etc.)
    df_plot = df_plot.sort_values('window_size')
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Box plot - ensure proper order
    window_order = [f'n={w}' for w in sorted(window_size_map.keys())]
    sns.boxplot(
        data=df_plot,
        x='Window Size',
        y='Similarity Score',
        ax=axes[0],
        palette='Set2',
        order=window_order
    )
    axes[0].set_title('Distribution of Similarity Scores by Window Size', fontweight='bold')
    axes[0].set_ylabel('Levenshtein Similarity')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Histogram with KDE - use sorted order
    for window_size in sorted(window_size_map.keys()):
        window_label = window_size_map[window_size]
        scores = df_plot[df_plot['Window Size'] == window_label]['Similarity Score'].values
        if len(scores) > 0:
            axes[1].hist(
                scores,
                bins=20,
                alpha=0.5,
                label=window_label,
                density=True
            )
    
    axes[1].set_title('Similarity Score Distributions (Overlaid)', fontweight='bold')
    axes[1].set_xlabel('Levenshtein Similarity')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'window_matching_similarity_distributions.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved similarity distribution plot to: {output_path}")
    
    return str(output_path)


def _extract_window_size_from_path(dataset_path):
    """Extract window size from dataset path."""
    import re
    match = re.search(r'window_(\d+)', dataset_path.name if isinstance(dataset_path, Path) else str(dataset_path))
    if match:
        return int(match.group(1))
    return None


def _load_all_window_datasets(window_datasets):
    """Load all window datasets and return a dictionary mapping window_size to DataFrame."""
    datasets = {}
    for dataset_path in window_datasets:
        window_size = _extract_window_size_from_path(dataset_path)
        if window_size is None:
            continue
        try:
            df = pd.read_csv(dataset_path)
            datasets[window_size] = df
        except Exception as e:
            logger.warning(f"Error loading {dataset_path}: {e}")
    return datasets


def plot_match_success_rate(window_datasets: list, output_dir: str) -> str:
    """
    Plot match success rate (percentage of sentences with matches) by window size.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating match success rate plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Calculate match success rate for each window size
    window_sizes = sorted(datasets.keys())
    match_rates = []
    total_sentences = []
    matched_sentences = []
    
    for window_size in window_sizes:
        df = datasets[window_size]
        total = len(df)
        # Sentences with matches have non-null similarity
        matched = df['similarity'].notna().sum()
        match_rate = (matched / total * 100) if total > 0 else 0
        
        total_sentences.append(total)
        matched_sentences.append(matched)
        match_rates.append(match_rate)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(window_sizes))
    labels = [f'n={w}' for w in window_sizes]
    
    bars = ax.bar(x, match_rates, alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, rate, matched, total) in enumerate(zip(bars, match_rates, matched_sentences, total_sentences)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%\n({matched}/{total})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Match Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Match Success Rate by Window Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max(match_rates) * 1.15 if match_rates else 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'match_success_rate.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved match success rate plot to: {output_path}")
    return str(output_path)


def plot_matches_per_sentence_distribution(window_datasets: list, output_dir: str) -> str:
    """
    Plot distribution of number of matches per sentence.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating matches per sentence distribution plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    window_sizes = sorted(datasets.keys())
    
    # Create figure (single plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Box plot by window size (no outliers)
    data_for_box = []
    labels_for_box = []
    window_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}
    
    for window_size in window_sizes:
        df = datasets[window_size]
        if 'total_matches_above_threshold' in df.columns:
            matches = df['total_matches_above_threshold'].dropna().values
            if len(matches) > 0:
                data_for_box.append(matches)
                labels_for_box.append(f'n={window_size}')
    
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                       showfliers=False,  # Remove outliers
                       widths=0.6,
                       medianprops=dict(linewidth=2.5, color='#333333'),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color each box by window size
        for i, patch in enumerate(bp['boxes']):
            window_size = window_sizes[i]
            patch.set_facecolor(window_colors.get(window_size, '#9b59b6'))
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1.5)
        
        # Style median, whiskers, and caps
        for element in ['medians', 'whiskers', 'caps']:
            if element in bp:
                for item in bp[element]:
                    item.set_color('#333333')
    
    ax.set_ylabel('Number of Matches', fontsize=13, fontweight='medium')
    ax.set_xlabel('Window Size', fontsize=13, fontweight='medium')
    ax.set_title('Matches per Sentence by Window Size', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'matches_per_sentence_distribution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matches per sentence distribution plot to: {output_path}")
    return str(output_path)


def plot_match_quality_by_group_speaker(window_datasets: list, output_dir: str) -> str:
    """
    Plot match quality metrics showing proportion of matches from same group/speaker.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating match quality by group/speaker plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    window_sizes = sorted(datasets.keys())
    
    # Calculate proportions
    same_group_rates = []
    same_speaker_rates = []
    
    for window_size in window_sizes:
        df = datasets[window_size]
        if 'matches_same_group' not in df.columns or 'matches_same_speaker' not in df.columns:
            continue
        
        # Calculate rates (proportion of sentences with same group/speaker matches)
        total = len(df)
        same_group = (df['matches_same_group'] > 0).sum() if 'matches_same_group' in df.columns else 0
        same_speaker = (df['matches_same_speaker'] > 0).sum() if 'matches_same_speaker' in df.columns else 0
        
        same_group_rates.append((same_group / total * 100) if total > 0 else 0)
        same_speaker_rates.append((same_speaker / total * 100) if total > 0 else 0)
    
    if not same_group_rates:
        logger.warning("No group/speaker data available")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(window_sizes))
    width = 0.35
    labels = [f'n={w}' for w in window_sizes]
    
    bars1 = ax.bar(x - width/2, same_group_rates, width, label='Same Group', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, same_speaker_rates, width, label='Same Speaker', 
                   color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Sentences (%)', fontsize=12, fontweight='bold')
    ax.set_title('Match Quality: Same Group/Speaker Matches', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(max(same_group_rates), max(same_speaker_rates)) * 1.15 if same_group_rates else 100])
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'match_quality_group_speaker.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved match quality plot to: {output_path}")
    return str(output_path)


def plot_similarity_vs_characteristics(window_datasets: list, output_dir: str) -> str:
    """
    Plot similarity scores vs. various sentence characteristics.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity vs characteristics plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Combine all datasets for analysis
    all_data = []
    for window_size, df in datasets.items():
        df_copy = df.copy()
        df_copy['window_size'] = window_size
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['similarity'].notna()].copy()
    
    if len(combined_df) == 0:
        logger.warning("No similarity data available")
        return ""
    
    # Calculate sentence length
    combined_df['cs_sentence_length'] = combined_df['cs_translation'].str.split().str.len()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Similarity vs sentence length
    ax1 = axes[0, 0]
    ax1.scatter(combined_df['cs_sentence_length'], combined_df['similarity'], 
               alpha=0.5, s=20, color='#3498db')
    ax1.set_xlabel('CS Sentence Length (words)', fontsize=11)
    ax1.set_ylabel('Similarity Score', fontsize=11)
    ax1.set_title('Similarity vs. Sentence Length', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Similarity vs switch index
    ax2 = axes[0, 1]
    if 'switch_index' in combined_df.columns:
        ax2.scatter(combined_df['switch_index'], combined_df['similarity'], 
                   alpha=0.5, s=20, color='#e74c3c')
        ax2.set_xlabel('Switch Index', fontsize=11)
        ax2.set_ylabel('Similarity Score', fontsize=11)
        ax2.set_title('Similarity vs. Switch Position', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
    
    # Subplot 3: Similarity by group
    ax3 = axes[1, 0]
    if 'cs_group' in combined_df.columns:
        groups = combined_df['cs_group'].unique()
        data_for_box = [combined_df[combined_df['cs_group'] == g]['similarity'].values for g in groups if len(combined_df[combined_df['cs_group'] == g]) > 0]
        labels_for_box = [g for g in groups if len(combined_df[combined_df['cs_group'] == g]) > 0]
        
        if data_for_box:
            bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#2ecc71')
                patch.set_alpha(0.7)
        
        ax3.set_ylabel('Similarity Score', fontsize=11)
        ax3.set_xlabel('Speaker Group', fontsize=11)
        ax3.set_title('Similarity by Speaker Group', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Similarity by window size
    ax4 = axes[1, 1]
    window_sizes = sorted(combined_df['window_size'].unique())
    data_for_box = [combined_df[combined_df['window_size'] == w]['similarity'].values for w in window_sizes]
    labels_for_box = [f'n={w}' for w in window_sizes]
    
    if data_for_box:
        bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#9b59b6')
            patch.set_alpha(0.7)
    
    ax4.set_ylabel('Similarity Score', fontsize=11)
    ax4.set_xlabel('Window Size', fontsize=11)
    ax4.set_title('Similarity by Window Size', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'similarity_vs_characteristics.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity vs characteristics plot to: {output_path}")
    return str(output_path)


def plot_window_size_comparison(window_datasets: list, output_dir: str) -> str:
    """
    Plot multi-metric comparison across window sizes.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating window size comparison plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    window_sizes = sorted(datasets.keys())
    
    # Calculate metrics for each window size
    metrics = {
        'avg_similarity': [],
        'median_similarity': [],
        'match_rate': [],
        'avg_matches_per_sent': []
    }
    
    for window_size in window_sizes:
        df = datasets[window_size]
        total = len(df)
        matched = df['similarity'].notna().sum()
        
        # Average similarity
        avg_sim = df['similarity'].mean() if 'similarity' in df.columns else 0
        metrics['avg_similarity'].append(avg_sim)
        
        # Median similarity
        median_sim = df['similarity'].median() if 'similarity' in df.columns else 0
        metrics['median_similarity'].append(median_sim)
        
        # Match rate
        match_rate = (matched / total * 100) if total > 0 else 0
        metrics['match_rate'].append(match_rate)
        
        # Average matches per sentence
        if 'total_matches_above_threshold' in df.columns:
            avg_matches = df['total_matches_above_threshold'].mean()
        else:
            avg_matches = 0
        metrics['avg_matches_per_sent'].append(avg_matches)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    labels = [f'n={w}' for w in window_sizes]
    x = np.arange(len(window_sizes))
    
    # Subplot 1: Average similarity
    ax1 = axes[0, 0]
    bars = ax1.bar(x, metrics['avg_similarity'], alpha=0.8, color='#3498db', edgecolor='black')
    for bar, val in zip(bars, metrics['avg_similarity']):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('Average Similarity', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Window Size', fontsize=11, fontweight='bold')
    ax1.set_title('Average Similarity by Window Size', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(metrics['avg_similarity']) * 1.15 if metrics['avg_similarity'] else 1])
    
    # Subplot 2: Median similarity
    ax2 = axes[0, 1]
    bars = ax2.bar(x, metrics['median_similarity'], alpha=0.8, color='#2ecc71', edgecolor='black')
    for bar, val in zip(bars, metrics['median_similarity']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Median Similarity', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Window Size', fontsize=11, fontweight='bold')
    ax2.set_title('Median Similarity by Window Size', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(metrics['median_similarity']) * 1.15 if metrics['median_similarity'] else 1])
    
    # Subplot 3: Match rate
    ax3 = axes[1, 0]
    bars = ax3.bar(x, metrics['match_rate'], alpha=0.8, color='#e74c3c', edgecolor='black')
    for bar, val in zip(bars, metrics['match_rate']):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax3.set_ylabel('Match Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Window Size', fontsize=11, fontweight='bold')
    ax3.set_title('Match Success Rate by Window Size', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, max(metrics['match_rate']) * 1.15 if metrics['match_rate'] else 100])
    
    # Subplot 4: Average matches per sentence
    ax4 = axes[1, 1]
    bars = ax4.bar(x, metrics['avg_matches_per_sent'], alpha=0.8, color='#9b59b6', edgecolor='black')
    for bar, val in zip(bars, metrics['avg_matches_per_sent']):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax4.set_ylabel('Average Matches per Sentence', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Window Size', fontsize=11, fontweight='bold')
    ax4.set_title('Average Matches per Sentence by Window Size', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.grid(axis='y', alpha=0.3)
    if metrics['avg_matches_per_sent']:
        ax4.set_ylim([0, max(metrics['avg_matches_per_sent']) * 1.15])
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'window_size_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved window size comparison plot to: {output_path}")
    return str(output_path)


def plot_match_distribution_by_group(window_datasets: list, output_dir: str) -> str:
    """
    Plot distribution of matches across speaker groups.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating match distribution by group plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Combine all datasets
    all_data = []
    for df in datasets.values():
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if 'cs_group' not in combined_df.columns or 'matched_group' not in combined_df.columns:
        logger.warning("Group columns not found")
        return ""
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Stacked bar chart showing CS group vs matched group
    ax1 = axes[0]
    group_counts = {}
    for cs_group in groups:
        group_df = combined_df[combined_df['cs_group'] == cs_group]
        matched_counts = group_df['matched_group'].value_counts()
        group_counts[cs_group] = {g: matched_counts.get(g, 0) for g in groups}
    
    x = np.arange(len(groups))
    width = 0.6
    bottom = np.zeros(len(groups))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, matched_group in enumerate(groups):
        counts = [group_counts[cs_group].get(matched_group, 0) for cs_group in groups]
        ax1.bar(x, counts, width, bottom=bottom, label=matched_group, color=colors[i], alpha=0.8)
        bottom += np.array(counts)
    
    ax1.set_xlabel('Code-Switched Sentence Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
    ax1.set_title('Match Distribution: CS Group vs. Matched Group', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.legend(title='Matched Group')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Heatmap of group matching
    ax2 = axes[1]
    heatmap_data = []
    for cs_group in groups:
        row = []
        for matched_group in groups:
            count = len(combined_df[(combined_df['cs_group'] == cs_group) & 
                                   (combined_df['matched_group'] == matched_group)])
            row.append(count)
        heatmap_data.append(row)
    
    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(groups)))
    ax2.set_yticks(range(len(groups)))
    ax2.set_xticklabels(groups)
    ax2.set_yticklabels(groups)
    ax2.set_xlabel('Matched Group', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Code-Switched Group', fontsize=12, fontweight='bold')
    ax2.set_title('Group Matching Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(groups)):
        for j in range(len(groups)):
            text = ax2.text(j, i, heatmap_data[i][j], ha="center", va="center", 
                           color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'match_distribution_by_group.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved match distribution by group plot to: {output_path}")
    return str(output_path)


def plot_similarity_threshold_analysis(window_datasets: list, output_dir: str) -> str:
    """
    Plot similarity score threshold analysis showing distribution with threshold lines.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity threshold analysis plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Create figure (single plot - distribution with three window sizes overlapping)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    window_sizes = sorted(datasets.keys())
    window_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}
    
    # Similarity threshold (default is 0.4)
    similarity_threshold = 0.4
    
    # Plot KDE distributions for each window size (only scores >= threshold)
    for window_size in window_sizes:
        df = datasets[window_size]
        if 'similarity' in df.columns:
            scores = df['similarity'].dropna()
            # Filter to only scores above threshold
            scores_above_threshold = scores[scores >= similarity_threshold]
            if len(scores_above_threshold) > 0:
                sns.kdeplot(data=scores_above_threshold, label=f'n={window_size}',
                           color=window_colors.get(window_size, '#95a5a6'),
                           ax=ax, linewidth=2.5, alpha=0.6, fill=True, common_norm=False)
    
    # Add threshold line for reference
    ax.axvline(similarity_threshold, color='#333333', linestyle='--', linewidth=2, 
              alpha=0.7, label=f'Threshold ({similarity_threshold})')
    
    # Set x-axis to start at threshold (0.4)
    ax.set_xlim(left=similarity_threshold)
    
    ax.set_xlabel('Similarity Score', fontsize=13, fontweight='medium')
    ax.set_ylabel('Density', fontsize=13, fontweight='medium')
    ax.set_title('Similarity Score Distribution by Window Size', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'similarity_threshold_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity threshold analysis plot to: {output_path}")
    return str(output_path)


def plot_pos_window_alignment_quality(window_datasets: list, output_dir: str) -> str:
    """
    Plot POS window alignment quality comparison between CS and matched sentences.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating POS window alignment quality plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Combine all datasets
    all_data = []
    for df in datasets.values():
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if 'pos_window' not in combined_df.columns or 'matched_pos' not in combined_df.columns:
        logger.warning("POS columns not found")
        return ""
    
    # Calculate POS sequence lengths
    combined_df['cs_pos_length'] = combined_df['pos_window'].str.split().str.len()
    combined_df['matched_pos_length'] = combined_df['matched_pos'].str.split().str.len()
    
    # Create figure (single plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # POS length difference distribution
    combined_df['pos_length_diff'] = abs(combined_df['cs_pos_length'] - combined_df['matched_pos_length'])
    
    # Use histogram with KDE overlay for professional look
    sns.histplot(data=combined_df, x='pos_length_diff', bins='auto', kde=True,
                color='#2ecc71', ax=ax, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Absolute POS Length Difference', fontsize=13, fontweight='medium')
    ax.set_ylabel('Count', fontsize=13, fontweight='medium')
    ax.set_title('Distribution of POS Length Differences', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'pos_window_alignment_quality.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved POS window alignment quality plot to: {output_path}")
    return str(output_path)

