#!/usr/bin/env python3
"""
Generate figure showing distribution of matrix language (Cantonese vs English)
in code-switched sentences.

This script counts sentences by matrix language and creates a professional
visualization showing the distribution across speaker groups.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config

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
    """Load code-switching data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['matrix_language', 'group']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def create_figure(df: pd.DataFrame, output_path: str):
    """Create and save the matrix language distribution figure."""
    setup_style()
    
    # Filter out 'Equal' matrix language cases for cleaner visualization
    df_filtered = df[df['matrix_language'].isin(['Cantonese', 'English'])].copy()
    
    # Get speaker groups
    groups = sorted(df_filtered['group'].unique())
    
    # Prepare data for plotting
    cantonese_counts = []
    english_counts = []
    
    for group in groups:
        group_data = df_filtered[df_filtered['group'] == group]
        cant_count = len(group_data[group_data['matrix_language'] == 'Cantonese'])
        eng_count = len(group_data[group_data['matrix_language'] == 'English'])
        cantonese_counts.append(cant_count)
        english_counts.append(eng_count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create grouped bar chart
    x = range(len(groups))
    width = 0.35
    
    bars1 = ax.bar(
        [i - width/2 for i in x], cantonese_counts, width,
        label='Cantonese Matrix', color='#4ECDC4', edgecolor='black', linewidth=0.5
    )
    bars2 = ax.bar(
        [i + width/2 for i in x], english_counts, width,
        label='English Matrix', color='#FF6B6B', edgecolor='black', linewidth=0.5
    )
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold'
                )
    
    # Add percentage labels
    for i, group in enumerate(groups):
        total = cantonese_counts[i] + english_counts[i]
        if total > 0:
            cant_pct = (cantonese_counts[i] / total) * 100
            eng_pct = (english_counts[i] / total) * 100
            
            # Add percentage text above bars
            if cantonese_counts[i] > 20:  # Only show if bar is large enough
                ax.text(
                    i - width/2, cantonese_counts[i] + total * 0.02,
                    f'{cant_pct:.1f}%',
                    ha='center', va='bottom', fontsize=8, style='italic'
                )
            if english_counts[i] > 20:
                ax.text(
                    i + width/2, english_counts[i] + total * 0.02,
                    f'{eng_pct:.1f}%',
                    ha='center', va='bottom', fontsize=8, style='italic'
                )
    
    # Formatting
    ax.set_xlabel('Speaker Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sentences', fontsize=12, fontweight='bold')
    ax.set_title(
        'Distribution of Matrix Language in Code-Switched Sentences',
        fontsize=13, fontweight='bold', pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total count annotation
    total_sentences = len(df_filtered)
    ax.text(
        0.02, 0.98, f'Total: {total_sentences} sentences',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
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
        description='Generate matrix language distribution figure'
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
        help='Output path for figure (default: figures/exploratory/matrix_embedded_distribution.png)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file (default: from config)'
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
            'matrix_embedded_distribution.png'
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

