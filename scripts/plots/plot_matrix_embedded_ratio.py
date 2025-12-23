#!/usr/bin/env python3
"""
Generate figure showing distribution of matrix/embedded word ratios
in code-switched sentences.

This script calculates the ratio of matrix language words to embedded language
words for each sentence and visualizes the distribution separately for
Cantonese matrix and English matrix sentences.
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.analysis.pos_tagging import parse_pattern_segments

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

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate matrix/embedded word ratios for each sentence."""
    df = df.copy()
    
    ratios = []
    matrix_words_list = []
    embedded_words_list = []
    
    for idx, row in df.iterrows():
        pattern = str(row['pattern'])
        matrix_lang = str(row['matrix_language'])
        
        # Skip if not code-switched or if matrix language is 'Equal'
        if matrix_lang not in ['Cantonese', 'English']:
            ratios.append(np.nan)
            matrix_words_list.append(0)
            embedded_words_list.append(0)
            continue
        
        # Parse pattern segments
        segments = parse_pattern_segments(pattern)
        
        # Determine matrix and embedded language codes
        matrix_code = 'C' if matrix_lang == 'Cantonese' else 'E'
        embedded_code = 'E' if matrix_code == 'C' else 'C'
        
        # Sum words in each language
        matrix_words = sum(count for lang, count in segments if lang == matrix_code)
        embedded_words = sum(count for lang, count in segments if lang == embedded_code)
        
        # Calculate ratio (avoid division by zero)
        if embedded_words > 0:
            ratio = matrix_words / embedded_words
        else:
            ratio = np.inf  # All words are matrix language
        
        ratios.append(ratio)
        matrix_words_list.append(matrix_words)
        embedded_words_list.append(embedded_words)
    
    df['matrix_words'] = matrix_words_list
    df['embedded_words'] = embedded_words_list
    df['ratio'] = ratios
    
    return df

def load_data(csv_path: str) -> pd.DataFrame:
    """Load code-switching data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['pattern', 'matrix_language']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def create_figure(df: pd.DataFrame, output_path: str):
    """Create and save the matrix/embedded ratio distribution figure."""
    setup_style()
    
    # Calculate ratios
    df_with_ratios = calculate_ratios(df)
    
    # Filter out invalid ratios
    df_valid = df_with_ratios[
        (df_with_ratios['ratio'].notna()) & 
        (df_with_ratios['ratio'] != np.inf) &
        (df_with_ratios['ratio'] > 0)
    ].copy()
    
    # Separate by matrix language
    cantonese_matrix = df_valid[df_valid['matrix_language'] == 'Cantonese']
    english_matrix = df_valid[df_valid['matrix_language'] == 'English']
    
    # Cap ratios at reasonable maximum for visualization (e.g., 10)
    max_ratio = 10
    cantonese_ratios = np.clip(cantonese_matrix['ratio'].values, 0, max_ratio)
    english_ratios = np.clip(english_matrix['ratio'].values, 0, max_ratio)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Cantonese matrix
    if len(cantonese_ratios) > 0:
        n_bins = min(30, len(cantonese_ratios) // 2) if len(cantonese_ratios) > 4 else 10
        n_bins = max(10, n_bins)  # At least 10 bins
        
        counts, bins, patches = ax1.hist(
            cantonese_ratios, bins=n_bins, color='#4ECDC4', 
            edgecolor='black', linewidth=0.5, alpha=0.7
        )
        
        # Add mean and median lines
        mean_ratio = np.mean(cantonese_ratios)
        median_ratio = np.median(cantonese_ratios)
        
        ax1.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
        ax1.axvline(median_ratio, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_ratio:.2f}')
        
        ax1.set_xlabel('Matrix/Embedded Word Ratio', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Cantonese Matrix Sentences\n(n={len(cantonese_ratios)})',
            fontsize=13, fontweight='bold'
        )
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics text
        stats_text = f'Mean: {mean_ratio:.2f}\nMedian: {median_ratio:.2f}\nStd: {np.std(cantonese_ratios):.2f}'
        ax1.text(
            0.98, 0.98, stats_text,
            transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Cantonese Matrix Sentences\n(n=0)', fontsize=13, fontweight='bold')
    
    # Right plot: English matrix
    if len(english_ratios) > 0:
        n_bins = min(30, len(english_ratios) // 2) if len(english_ratios) > 4 else 10
        n_bins = max(10, n_bins)  # At least 10 bins
        
        counts, bins, patches = ax2.hist(
            english_ratios, bins=n_bins, color='#FF6B6B',
            edgecolor='black', linewidth=0.5, alpha=0.7
        )
        
        # Add mean and median lines
        mean_ratio = np.mean(english_ratios)
        median_ratio = np.median(english_ratios)
        
        ax2.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
        ax2.axvline(median_ratio, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_ratio:.2f}')
        
        ax2.set_xlabel('Matrix/Embedded Word Ratio', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(
            f'English Matrix Sentences\n(n={len(english_ratios)})',
            fontsize=13, fontweight='bold'
        )
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics text
        stats_text = f'Mean: {mean_ratio:.2f}\nMedian: {median_ratio:.2f}\nStd: {np.std(english_ratios):.2f}'
        ax2.text(
            0.98, 0.98, stats_text,
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('English Matrix Sentences\n(n=0)', fontsize=13, fontweight='bold')
    
    # Add overall title
    fig.suptitle(
        'Distribution of Matrix/Embedded Word Ratios in Code-Switched Sentences',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate matrix/embedded word ratio distribution figure'
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
        help='Output path for figure (default: figures/exploratory/matrix_embedded_ratio_distribution.png)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file (default: code_switching_WITHOUT_fillers.csv from config)'
    )
    parser.add_argument(
        '--max-ratio',
        type=float,
        default=10.0,
        help='Maximum ratio to display (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(config_path=args.config)
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = config.get_csv_without_fillers_path()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            config.get_exploratory_figures_dir(),
            'matrix_embedded_ratio_distribution.png'
        )
    
    try:
        # Load data
        print(f"Loading data from: {csv_path}")
        df = load_data(csv_path)
        print(f"Loaded {len(df)} sentences")
        
        # Create figure
        print("Generating figure...")
        create_figure(df, output_path)
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

