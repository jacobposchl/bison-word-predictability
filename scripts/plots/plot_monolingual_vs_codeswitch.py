#!/usr/bin/env python3
"""
Generate figure comparing number of monolingual vs code-switched sentences.

This script classifies sentences as monolingual Cantonese, monolingual English,
or code-switched, and creates a professional visualization.
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
from src.analysis.feasibility import is_monolingual

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

def classify_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Classify sentences as monolingual or code-switched."""
    df = df.copy()
    
    # Classify each sentence
    classifications = []
    for pattern in df['pattern']:
        lang_type = is_monolingual(str(pattern))
        if lang_type == 'Cantonese':
            classifications.append('Monolingual Cantonese')
        elif lang_type == 'English':
            classifications.append('Monolingual English')
        else:
            classifications.append('Code-switched')
    
    df['classification'] = classifications
    return df

def load_data(csv_path: str) -> pd.DataFrame:
    """Load all sentences data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['pattern']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def create_figure(df: pd.DataFrame, output_path: str):
    """Create and save the monolingual vs code-switched comparison figure."""
    setup_style()
    
    # Classify sentences
    df_classified = classify_sentences(df)
    
    # Get speaker groups if available
    has_groups = 'group' in df_classified.columns
    if has_groups:
        groups = sorted(df_classified['group'].unique())
    else:
        groups = ['All']
        df_classified['group'] = 'All'
    
    # Prepare data for plotting
    categories = ['Monolingual Cantonese', 'Monolingual English', 'Code-switched']
    data_by_group = {}
    
    for group in groups:
        group_data = df_classified[df_classified['group'] == group]
        counts = {
            'Monolingual Cantonese': len(group_data[group_data['classification'] == 'Monolingual Cantonese']),
            'Monolingual English': len(group_data[group_data['classification'] == 'Monolingual English']),
            'Code-switched': len(group_data[group_data['classification'] == 'Code-switched'])
        }
        data_by_group[group] = counts
    
    # Create figure with subplots
    if has_groups and len(groups) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot: Overall distribution
        ax1 = axes[0]
        overall_counts = [
            data_by_group[group][cat] for cat in categories
            for group in groups
        ]
        overall_cant = sum(data_by_group[g]['Monolingual Cantonese'] for g in groups)
        overall_eng = sum(data_by_group[g]['Monolingual English'] for g in groups)
        overall_cs = sum(data_by_group[g]['Code-switched'] for g in groups)
        
        overall_counts = [overall_cant, overall_eng, overall_cs]
        colors = ['#4ECDC4', '#FF6B6B', '#F7B801']
        
        bars = ax1.bar(categories, overall_counts, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, count in zip(bars, overall_counts):
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )
        
        # Add percentage labels
        total = sum(overall_counts)
        for i, (bar, count) in enumerate(zip(bars, overall_counts)):
            pct = (count / total) * 100 if total > 0 else 0
            ax1.text(
                bar.get_x() + bar.get_width()/2., count + total * 0.01,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9, style='italic'
            )
        
        ax1.set_ylabel('Number of Sentences', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Distribution\n(All Groups Combined)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_xticklabels(categories, rotation=15, ha='right')
        
        # Right plot: By group
        ax2 = axes[1]
        x = range(len(groups))
        width = 0.25
        
        cant_counts = [data_by_group[g]['Monolingual Cantonese'] for g in groups]
        eng_counts = [data_by_group[g]['Monolingual English'] for g in groups]
        cs_counts = [data_by_group[g]['Code-switched'] for g in groups]
        
        bars1 = ax2.bar(
            [i - width for i in x], cant_counts, width,
            label='Monolingual Cantonese', color='#4ECDC4', edgecolor='black', linewidth=0.5
        )
        bars2 = ax2.bar(
            x, eng_counts, width,
            label='Monolingual English', color='#FF6B6B', edgecolor='black', linewidth=0.5
        )
        bars3 = ax2.bar(
            [i + width for i in x], cs_counts, width,
            label='Code-switched', color='#F7B801', edgecolor='black', linewidth=0.5
        )
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold'
                    )
        
        ax2.set_xlabel('Speaker Group', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Sentences', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution by Speaker Group', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
    else:
        # Single plot for overall or single group
        fig, ax = plt.subplots(figsize=(10, 7))
        
        overall_cant = data_by_group[groups[0]]['Monolingual Cantonese']
        overall_eng = data_by_group[groups[0]]['Monolingual English']
        overall_cs = data_by_group[groups[0]]['Code-switched']
        
        overall_counts = [overall_cant, overall_eng, overall_cs]
        colors = ['#4ECDC4', '#FF6B6B', '#F7B801']
        
        bars = ax.bar(categories, overall_counts, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, count in zip(bars, overall_counts):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold'
                )
        
        # Add percentage labels
        total = sum(overall_counts)
        for i, (bar, count) in enumerate(zip(bars, overall_counts)):
            pct = (count / total) * 100 if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width()/2., count + total * 0.02,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, style='italic'
            )
        
        ax.set_ylabel('Number of Sentences', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentence Type', fontsize=12, fontweight='bold')
        ax.set_title(
            'Distribution of Monolingual vs Code-Switched Sentences',
            fontsize=13, fontweight='bold', pad=20
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(categories, rotation=15, ha='right')
    
    # Add total count annotation
    total_sentences = len(df_classified)
    fig.text(
        0.02, 0.02, f'Total: {total_sentences} sentences',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
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
        description='Generate monolingual vs code-switched comparison figure'
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
        help='Output path for figure (default: figures/exploratory/monolingual_vs_codeswitch.png)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file (default: all_sentences.csv from config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(config_path=args.config)
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = config.get_csv_all_sentences_path()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            config.get_exploratory_figures_dir(),
            'monolingual_vs_codeswitch.png'
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

