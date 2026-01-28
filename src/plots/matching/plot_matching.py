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

