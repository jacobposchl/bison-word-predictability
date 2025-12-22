#!/usr/bin/env python3
"""
Generate multi-panel figure showing distributions of matching algorithm results.

This script visualizes:
1. Distribution of number of matches per sentence
2. Distribution of best similarity scores
3. Distribution of top-3 match similarities (box plot)
4. Match success rate breakdown
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

def setup_style():
    """Set professional matplotlib style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def load_data(csv_path: str) -> pd.DataFrame:
    """Load matching results from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['num_matches', 'best_similarity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def create_figure(df: pd.DataFrame, output_path: str, similarity_threshold: float = 0.4):
    """Create and save the matching distributions figure."""
    setup_style()
    
    # Create figure with 1x2 subplots (only top two)
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Distribution of number of matches
    ax1 = fig.add_subplot(gs[0, 0])
    
    num_matches = df['num_matches'].values
    max_matches = int(num_matches.max())
    
    # Count frequency of each unique value (for discrete data)
    unique_values, counts = np.unique(num_matches, return_counts=True)
    
    # Create bar chart for discrete values
    bars = ax1.bar(
        unique_values, counts, color='#4ECDC4', 
        edgecolor='black', linewidth=0.5, alpha=0.7, width=0.8
    )
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
    
    # Add mean line
    mean_matches = np.mean(num_matches)
    ax1.axvline(mean_matches, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_matches:.1f}')
    
    ax1.set_xlabel('Number of Matches', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Distribution of Number of Matches\n(n={len(df)} sentences)',
        fontsize=13, fontweight='bold'
    )
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add statistics
    stats_text = f'Mean: {mean_matches:.1f}\nMedian: {np.median(num_matches):.1f}\nMax: {max_matches}'
    ax1.text(
        0.98, 0.98, stats_text,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Subplot 2: Distribution of best similarity scores
    ax2 = fig.add_subplot(gs[0, 1])
    
    best_similarities = df[df['best_similarity'] > 0]['best_similarity'].values
    
    if len(best_similarities) > 0:
        n_bins = 30
        counts, bins_edges, patches = ax2.hist(
            best_similarities, bins=n_bins, color='#FF6B6B',
            edgecolor='black', linewidth=0.5, alpha=0.7
        )
        
        # Add threshold line
        ax2.axvline(similarity_threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold: {similarity_threshold}')
        
        # Add mean line
        mean_sim = np.mean(best_similarities)
        ax2.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_sim:.3f}')
        
        ax2.set_xlabel('Best Similarity Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(
            f'Distribution of Best Similarity Scores\n(n={len(best_similarities)} with matches)',
            fontsize=13, fontweight='bold'
        )
        ax2.set_xlim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics
        stats_text = f'Mean: {mean_sim:.3f}\nMedian: {np.median(best_similarities):.3f}'
        ax2.text(
            0.02, 0.98, stats_text,
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        ax2.text(0.5, 0.5, 'No matches found', ha='center', va='center', 
                transform=ax2.transAxes)
        ax2.set_title('Distribution of Best Similarity Scores', fontsize=13, fontweight='bold')
    
    # Add overall title
    fig.suptitle(
        'Matching Algorithm Results: Distribution Analysis',
        fontsize=15, fontweight='bold', y=0.98
    )
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate matching distributions figure'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for figure (default: figures/exploratory/matching_distributions.png)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to matching results CSV file (default: matching_results_sample.csv from config)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.4,
        help='Similarity threshold used in matching (default: 0.4)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(config_path=args.config)
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            config.get_exploratory_results_dir(),
            'matching_results_sample.csv'
        )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            config.get_exploratory_figures_dir(),
            'matching_distributions.png'
        )
    
    try:
        # Load data
        print(f"Loading data from: {csv_path}")
        df = load_data(csv_path)
        print(f"Loaded {len(df)} matching results")
        
        # Create figure
        print("Generating figure...")
        create_figure(df, output_path, similarity_threshold=args.threshold)
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

