"""
Simplified visualization functions for code-switching data (WITHOUT fillers only).

This module provides simplified functions that work with only WITHOUT fillers data,
since WITH fillers datasets are no longer produced.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)


def print_analysis_summary_simple(without_fillers: List[Dict]) -> None:
    """
    Print detailed text-based analysis summary for WITHOUT fillers data only.
    
    Args:
        without_fillers: List of sentences with fillers excluded
    """
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Basic dataset statistics
    print(f"\nDataset size:")
    print(f"  WITHOUT fillers: {len(without_fillers)} code-switching sentences")
    
    # Group distributions
    print(f"\n" + "-"*80)
    print("Sentences by Speaker Group:")
    print("-"*80)
    
    group_counts = Counter([s['group'] for s in without_fillers])
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} ({count/len(without_fillers)*100:.1f}%)")
    
    # Matrix language distributions
    print(f"\n" + "-"*80)
    print("Matrix Language Distribution:")
    print("-"*80)
    
    matrix_counts = Counter([s['matrix_language'] for s in without_fillers])
    print(f"  Cantonese: {matrix_counts['Cantonese']} ({matrix_counts['Cantonese']/len(without_fillers)*100:.1f}%)")
    print(f"  English: {matrix_counts['English']} ({matrix_counts['English']/len(without_fillers)*100:.1f}%)")
    if 'Equal' in matrix_counts:
        print(f"  Equal: {matrix_counts['Equal']} ({matrix_counts['Equal']/len(without_fillers)*100:.1f}%)")
    
    # Detailed breakdown by group AND matrix language
    print(f"\n" + "-"*80)
    print("Matrix Language by Participant Group:")
    print("-"*80)
    
    for group in sorted(groups):
        group_sentences = [s for s in without_fillers if s['group'] == group]
        if group_sentences:
            cant_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            print(
                f"  {group:18} (n={len(group_sentences):4}): "
                f"Cantonese {cant_matrix:4} ({cant_matrix/len(group_sentences)*100:5.1f}%)  |  "
                f"English {eng_matrix:4} ({eng_matrix/len(group_sentences)*100:5.1f}%)  |  "
                f"Equal {equal_matrix:4} ({equal_matrix/len(group_sentences)*100:5.1f}%)"
            )
    
    # Print detailed comparison table
    print("\n" + "="*80)
    print("MATRIX LANGUAGE DISTRIBUTION TABLE")
    print("="*80)
    print(f"{'Group':<12} {'Total':<8} {'Cantonese':<20} {'English':<20} {'Equal':<15}")
    print("-"*80)
    
    for group in groups:
        group_sentences = [s for s in without_fillers if s['group'] == group]
        if group_sentences:
            total = len(group_sentences)
            cant_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_count = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            
            print(
                f"{group:<12} {total:<8} "
                f"{cant_count:4} ({cant_count/total*100:5.1f}%){'':<8} "
                f"{eng_count:4} ({eng_count/total*100:5.1f}%){'':<8} "
                f"{equal_count:4} ({equal_count/total*100:5.1f}%)"
            )


def plot_matrix_language_distribution_simple(
    without_fillers: List[Dict],
    figures_dir: str
) -> None:
    """
    Plot matrix language distribution for WITHOUT fillers data.
    
    Args:
        without_fillers: List of sentences with fillers excluded
        figures_dir: Directory to save figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Count matrix languages by group
    group_matrix_counts = {}
    for group in groups:
        group_sentences = [s for s in without_fillers if s['group'] == group]
        if group_sentences:
            cant_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_count = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            group_matrix_counts[group] = {
                'Cantonese': cant_count,
                'English': eng_count,
                'Equal': equal_count,
                'Total': len(group_sentences)
            }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for stacked bar chart
    x = np.arange(len(groups))
    width = 0.6
    
    cantonese_counts = [group_matrix_counts.get(g, {}).get('Cantonese', 0) for g in groups]
    english_counts = [group_matrix_counts.get(g, {}).get('English', 0) for g in groups]
    equal_counts = [group_matrix_counts.get(g, {}).get('Equal', 0) for g in groups]
    
    # Create stacked bars
    p1 = ax.bar(x, cantonese_counts, width, label='Cantonese', color='#2ecc71')
    p2 = ax.bar(x, english_counts, width, bottom=cantonese_counts, label='English', color='#e74c3c')
    p3 = ax.bar(x, equal_counts, width, 
                bottom=np.array(cantonese_counts) + np.array(english_counts), 
                label='Equal', color='#95a5a6')
    
    ax.set_xlabel('Speaker Group', fontsize=12)
    ax.set_ylabel('Number of Sentences', fontsize=12)
    ax.set_title('Matrix Language Distribution by Speaker Group\n(WITHOUT Fillers)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, group in enumerate(groups):
        total = group_matrix_counts.get(group, {}).get('Total', 0)
        if total > 0:
            ax.text(i, total + total*0.01, str(total), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(figures_dir, 'matrix_language_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix language distribution plot to {output_path}")

