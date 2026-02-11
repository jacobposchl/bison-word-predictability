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
    
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    
    # Box plot by window size (no outliers)
    data_for_box = []
    labels_for_box = []
    window_colors = {1: cb_colors[0], 2: cb_colors[1], 3: cb_colors[2]}
    
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
            patch.set_facecolor(window_colors.get(window_size, cb_colors[2]))
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1.5)
        
        # Style median, whiskers, and caps
        for element in ['medians', 'whiskers', 'caps']:
            if element in bp:
                for item in bp[element]:
                    item.set_color('#333333')
    
    ax.set_ylabel('Number of Matches', fontsize=14, fontweight='medium')
    ax.set_xlabel('Window Size', fontsize=14, fontweight='medium')
    ax.set_title('Matches per Sentence by Window Size', fontsize=18, fontweight='bold', pad=20)
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


def plot_similarity_threshold_analysis(window_datasets: list, output_dir: str) -> str:
    """
    Plot similarity score histogram showing distribution of all matches.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity score histogram...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Create figure with subplots: one for each window size
    window_sizes = sorted(datasets.keys())
    n_windows = len(window_sizes)
    
    fig, axes = plt.subplots(1, n_windows, figsize=(11 * n_windows / 3, 7), sharey=True)
    if n_windows == 1:
        axes = [axes]
    
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    window_colors = {1: cb_colors[0], 2: cb_colors[1], 3: cb_colors[2]}
    
    # Similarity threshold (default is 0.4)
    similarity_threshold = 0.4
    
    for idx, window_size in enumerate(window_sizes):
        ax = axes[idx]
        df = datasets[window_size]
        
        # Parse all_similarity_scores column (stored as string representation of list)
        all_scores = []
        if 'all_similarity_scores' in df.columns:
            for scores_str in df['all_similarity_scores'].dropna():
                try:
                    # Parse the string representation of list
                    import ast
                    scores = ast.literal_eval(scores_str)
                    if isinstance(scores, list):
                        all_scores.extend(scores)
                except:
                    continue
        
        if not all_scores:
            logger.warning(f"No similarity scores found for window size {window_size}")
            continue
        
        # Create histogram
        bins = np.arange(0, 1.05, 0.05)  # Bins from 0 to 1 in 0.05 increments
        ax.hist(all_scores, bins=bins, color=window_colors.get(window_size, cb_colors[2]),
                alpha=0.7, edgecolor='white', linewidth=1.5)
        
        # Add threshold line
        ax.axvline(similarity_threshold, color='#333333', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Threshold ({similarity_threshold})')
        
        # Set x-axis limits and ticks
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='both', labelsize=14)
        
        # Labels
        ax.set_xlabel('Similarity Score', fontsize=14, fontweight='medium')
        if idx == 0:
            ax.set_ylabel('Number of Matches', fontsize=14, fontweight='medium')
        ax.set_title(f'Window Size n={window_size}', fontsize=16, fontweight='bold', pad=15)
        
        # Legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                 fontsize=12, framealpha=0.95)
        
        # Grid and styling
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#d0d0d0')
        ax.spines['bottom'].set_color('#d0d0d0')
        
        # Add count annotation
        total_matches = len(all_scores)
        above_threshold = sum(1 for s in all_scores if s >= similarity_threshold)
        ax.text(0.98, 0.98, f'Total: {total_matches}\nAbove threshold: {above_threshold}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    if n_windows > 1:
        fig.suptitle('Similarity Score Distribution by Window Size', 
                    fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'similarity_threshold_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity score histogram to: {output_path}")
    return str(output_path)


def plot_similarity_by_group(window_datasets: list, output_dir: str) -> str:
    """
    Plot similarity score histogram by speaker group.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity score histogram by group...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    window_sizes = sorted(datasets.keys())
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Create figure with subplots: rows = groups, cols = window sizes
    fig, axes = plt.subplots(len(groups), len(window_sizes), 
                             figsize=(11 * len(window_sizes) / 3, 6 * len(groups) / 3),
                             sharey='row')
    
    if len(groups) == 1 and len(window_sizes) == 1:
        axes = [[axes]]
    elif len(groups) == 1:
        axes = [axes]
    elif len(window_sizes) == 1:
        axes = [[ax] for ax in axes]
    
    # ColorBrewer YlOrBr palette
    cb_colors = sns.color_palette("YlOrBr", 3)
    group_colors = {'Homeland': cb_colors[0], 'Heritage': cb_colors[1], 'Immersed': cb_colors[2]}
    
    # Similarity threshold
    similarity_threshold = 0.4
    
    for row_idx, group in enumerate(groups):
        for col_idx, window_size in enumerate(window_sizes):
            ax = axes[row_idx][col_idx]
            df = datasets[window_size]
            
            # Filter by group
            group_df = df[df['cs_group'] == group]
            
            # Parse all_similarity_scores column
            all_scores = []
            if 'all_similarity_scores' in group_df.columns:
                for scores_str in group_df['all_similarity_scores'].dropna():
                    try:
                        import ast
                        scores = ast.literal_eval(scores_str)
                        if isinstance(scores, list):
                            all_scores.extend(scores)
                    except:
                        continue
            
            if not all_scores:
                # Empty plot with message
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                # Create histogram
                bins = np.arange(0, 1.05, 0.05)
                ax.hist(all_scores, bins=bins, color=group_colors.get(group, cb_colors[2]),
                       alpha=0.7, edgecolor='white', linewidth=1.5)
                
                # Add threshold line
                ax.axvline(similarity_threshold, color='#333333', linestyle='--', 
                          linewidth=2, alpha=0.7)
                
                # Add count annotation
                total_matches = len(all_scores)
                above_threshold = sum(1 for s in all_scores if s >= similarity_threshold)
                ax.text(0.98, 0.98, f'n={total_matches}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set x-axis
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.tick_params(axis='both', labelsize=12)
            
            # Labels
            if row_idx == len(groups) - 1:
                ax.set_xlabel('Similarity Score', fontsize=13, fontweight='medium')
            if col_idx == 0:
                ax.set_ylabel('Count', fontsize=13, fontweight='medium')
            
            # Title
            if row_idx == 0:
                ax.set_title(f'Window n={window_size}', fontsize=14, fontweight='bold', pad=10)
            if col_idx == len(window_sizes) - 1:
                ax.text(1.05, 0.5, group, transform=ax.transAxes, rotation=270,
                       va='center', ha='left', fontsize=14, fontweight='bold')
            
            # Grid and styling
            ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#d0d0d0')
            ax.spines['bottom'].set_color('#d0d0d0')
    
    # Overall title
    fig.suptitle('Similarity Score Distribution by Group and Window Size', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'similarity_by_group.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity by group histogram to: {output_path}")
    return str(output_path)


def plot_levenshtein_similarity_distribution(window_datasets: list, output_dir: str) -> list:
    """
    Plot Levenshtein similarity distribution histograms for each window size separately.
    
    Creates one figure per window size showing the distribution of similarity scores.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figures
        
    Returns:
        List of paths to saved figures
    """
    logger.info("Creating Levenshtein similarity distribution plots...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return []
    
    window_sizes = sorted(datasets.keys())
    cb_colors = sns.color_palette("YlOrBr", 3)
    window_colors = {1: cb_colors[0], 2: cb_colors[1], 3: cb_colors[2]}
    similarity_threshold = 0.4
    
    output_paths = []
    
    for window_size in window_sizes:
        df = datasets[window_size]
        
        # Parse all_similarity_scores column
        all_scores = []
        if 'all_similarity_scores' in df.columns:
            for scores_str in df['all_similarity_scores'].dropna():
                try:
                    import ast
                    scores = ast.literal_eval(scores_str)
                    if isinstance(scores, list):
                        all_scores.extend(scores)
                except:
                    continue
        
        if not all_scores:
            logger.warning(f"No similarity scores found for window size {window_size}")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Create histogram
        bins = np.arange(0, 1.05, 0.05)
        n, bins_edges, patches = ax.hist(all_scores, bins=bins, 
                                          color=window_colors.get(window_size, cb_colors[2]),
                                          alpha=0.7, edgecolor='white', linewidth=1.5)
        
        # Add threshold line
        ax.axvline(similarity_threshold, color='#333333', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Threshold ({similarity_threshold})')
        
        # Set x-axis
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='both', labelsize=14)
        
        # Labels
        ax.set_xlabel('Levenshtein Similarity Score', fontsize=14, fontweight='medium')
        ax.set_ylabel('Number of Matches', fontsize=14, fontweight='medium')
        ax.set_title(f'Levenshtein Similarity Distribution (Window Size n={window_size})', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                 fontsize=14, framealpha=0.95)
        
        # Grid and styling
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#d0d0d0')
        ax.spines['bottom'].set_color('#d0d0d0')
        
        # Statistics box
        total_matches = len(all_scores)
        above_threshold = sum(1 for s in all_scores if s >= similarity_threshold)
        mean_sim = np.mean(all_scores)
        median_sim = np.median(all_scores)
        
        stats_text = f'Total matches: {total_matches}\n'
        stats_text += f'Above threshold: {above_threshold} ({above_threshold/total_matches*100:.1f}%)\n'
        stats_text += f'Mean: {mean_sim:.3f}\n'
        stats_text += f'Median: {median_sim:.3f}'
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'levenshtein_similarity_distribution_window_{window_size}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved window {window_size} plot to: {output_path}")
        output_paths.append(str(output_path))
    
    return output_paths


def plot_similarity_violin_by_group(window_datasets: list, output_dir: str) -> list:
    """
    Plot violin plots of similarity scores by group for each window size.
    
    Creates one figure per window size showing violin plots for each speaker group.
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figures
        
    Returns:
        List of paths to saved figures
    """
    logger.info("Creating similarity violin plots by group...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return []
    
    window_sizes = sorted(datasets.keys())
    groups = ['Homeland', 'Heritage', 'Immersed']
    cb_colors = sns.color_palette("YlOrBr", 3)
    group_colors = {'Homeland': cb_colors[0], 'Heritage': cb_colors[1], 'Immersed': cb_colors[2]}
    
    output_paths = []
    
    for window_size in window_sizes:
        df = datasets[window_size]
        
        # Parse all_similarity_scores and create a long-form dataframe for plotting
        plot_data = []
        
        for _, row in df.iterrows():
            group = row.get('cs_group', 'Unknown')
            scores_str = row.get('all_similarity_scores', '[]')
            
            try:
                import ast
                scores = ast.literal_eval(scores_str)
                if isinstance(scores, list):
                    for score in scores:
                        plot_data.append({'Group': group, 'Similarity': score})
            except:
                continue
        
        if not plot_data:
            logger.warning(f"No data found for window size {window_size}")
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Create violin plot
        parts = ax.violinplot(
            [plot_df[plot_df['Group'] == g]['Similarity'].values for g in groups if g in plot_df['Group'].values],
            positions=range(len([g for g in groups if g in plot_df['Group'].values])),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )
        
        # Color the violins
        present_groups = [g for g in groups if g in plot_df['Group'].values]
        for i, pc in enumerate(parts['bodies']):
            group = present_groups[i]
            pc.set_facecolor(group_colors.get(group, cb_colors[0]))
            pc.set_alpha(0.7)
            pc.set_edgecolor('white')
            pc.set_linewidth(1.5)
        
        # Style the median and mean lines
        parts['cmedians'].set_color('#333333')
        parts['cmedians'].set_linewidth(2)
        parts['cmeans'].set_color('#d62728')
        parts['cmeans'].set_linewidth(2)
        parts['cbars'].set_color('#333333')
        parts['cmaxes'].set_color('#333333')
        parts['cmins'].set_color('#333333')
        
        # Set labels
        ax.set_xticks(range(len(present_groups)))
        ax.set_xticklabels(present_groups, fontsize=14)
        ax.set_ylabel('Similarity Score', fontsize=14, fontweight='medium')
        ax.set_xlabel('Speaker Group', fontsize=14, fontweight='medium')
        ax.set_title(f'Similarity Score Distribution by Group (Window Size n={window_size})', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='both', labelsize=14)
        
        # Grid and styling
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#d0d0d0')
        ax.spines['bottom'].set_color('#d0d0d0')
        
        # Add sample size annotations
        for i, group in enumerate(present_groups):
            n = len(plot_df[plot_df['Group'] == group])
            ax.text(i, 0.02, f'n={n}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'similarity_violin_by_group_window_{window_size}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved window {window_size} plot to: {output_path}")
        output_paths.append(str(output_path))
    
    return output_paths


def plot_match_ranking_distribution(window_datasets: list, output_dir: str) -> str:
    """
    Plot match ranking distribution showing percentage of matches by category.
    
    Creates a grouped bar chart showing the breakdown of matches by:
    - Same speaker
    - Same group (different speaker)
    - Different group
    
    Args:
        window_datasets: List of paths to analysis_dataset_window_*.csv files
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating match ranking distribution plot...")
    
    datasets = _load_all_window_datasets(window_datasets)
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    window_sizes = sorted(datasets.keys())
    cb_colors = sns.color_palette("YlOrBr", 3)
    
    # Collect statistics for each window size
    stats = {}
    
    for window_size in window_sizes:
        df = datasets[window_size]
        
        # Count matches by category
        # Each row represents a sentence, and we need to look at all its matches
        total_sentence_matches = 0
        same_speaker_matches = 0
        same_group_matches = 0
        different_group_matches = 0
        
        for _, row in df.iterrows():
            # Use the statistics columns that track ALL matches
            total = row.get('total_matches_above_threshold', 0)
            same_speaker = row.get('matches_same_speaker', 0)
            same_group = row.get('matches_same_group', 0)
            
            total_sentence_matches += total
            same_speaker_matches += same_speaker
            # Same group but different speaker
            same_group_matches += (same_group - same_speaker)
            # Different group
            different_group_matches += (total - same_group)
        
        if total_sentence_matches > 0:
            stats[window_size] = {
                'Same Speaker': (same_speaker_matches / total_sentence_matches) * 100,
                'Same Group': (same_group_matches / total_sentence_matches) * 100,
                'Different Group': (different_group_matches / total_sentence_matches) * 100,
                'total': total_sentence_matches
            }
    
    if not stats:
        logger.warning("No statistics to plot")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Prepare data for grouped bar chart
    categories = ['Same Speaker', 'Same Group', 'Different Group']
    x = np.arange(len(window_sizes))
    width = 0.25
    
    # Plot bars for each category
    for i, category in enumerate(categories):
        values = [stats[ws][category] for ws in window_sizes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=category, 
                     color=cb_colors[i], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Window Size', fontsize=14, fontweight='medium')
    ax.set_ylabel('Percentage of Matches (%)', fontsize=14, fontweight='medium')
    ax.set_title('Match Distribution by Speaker/Group Relationship', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={ws}' for ws in window_sizes], fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
             fontsize=14, framealpha=0.95)
    
    # Grid and styling
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    # Set y-axis limit
    ax.set_ylim(0, 100)
    
    # Add total matches annotation
    for i, ws in enumerate(window_sizes):
        total = stats[ws]['total']
        ax.text(i, -8, f'Total: {total}', ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'match_ranking_distribution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved match ranking distribution plot to: {output_path}")
    return str(output_path)

