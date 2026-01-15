"""
Plotting functions for exploratory analysis.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def plot_similarity_distributions(
    window_results: Dict,
    output_dir: str
) -> str:
    """
    Create visualization showing similarity score distributions for each window size.
    
    Args:
        window_results: Results from analyze_window_matching()
        output_dir: Directory to save the figure
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating similarity distribution plots...")
    
    # Set professional style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    
    # Extract data for plotting
    plot_data = []
    for window_key, results in window_results.items():
        window_size = results['window_size']
        similarity_scores = results['similarity_scores']
        
        for score in similarity_scores:
            plot_data.append({
                'Window Size': f'n={window_size}',
                'Similarity Score': score
            })
    
    if not plot_data:
        logger.warning("No similarity scores to plot")
        return ""
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Box plot
    sns.boxplot(
        data=df_plot,
        x='Window Size',
        y='Similarity Score',
        ax=axes[0],
        palette='Set2'
    )
    axes[0].set_title('Distribution of Similarity Scores by Window Size', fontweight='bold')
    axes[0].set_ylabel('Levenshtein Similarity')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Histogram with KDE
    for window_key, results in window_results.items():
        window_size = results['window_size']
        similarity_scores = results['similarity_scores']
        
        if similarity_scores:
            axes[1].hist(
                similarity_scores,
                bins=20,
                alpha=0.5,
                label=f'n={window_size}',
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
