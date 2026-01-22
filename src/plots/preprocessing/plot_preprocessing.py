"""
Visualization and analysis summaries for code-switching data.

This module provides functions for creating plots and printing analysis summaries.

NOTE: This module contains functions that compare WITH and WITHOUT fillers datasets.
For simplified functions that only work with WITHOUT fillers data, see plot_preprocessing_simple.py
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)


def print_analysis_summary(
    with_fillers: List[Dict],
    without_fillers: List[Dict]
) -> None:
    """
    Print detailed text-based analysis summaries.
    
    Args:
        with_fillers: List of sentences with fillers included
        without_fillers: List of sentences with fillers excluded
    """
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    # Basic dataset statistics
    print(f"\nDataset sizes:")
    print(f"  WITH fillers: {len(with_fillers)} code-switching sentences")
    print(f"  WITHOUT fillers: {len(without_fillers)} code-switching sentences")
    print(
        f"  Difference: {len(with_fillers) - len(without_fillers)} sentences "
        f"({(len(with_fillers) - len(without_fillers))/len(with_fillers)*100:.1f}% reduction)"
    )
    
    # Compare group distributions
    print(f"\n" + "-"*80)
    print("Sentences by Speaker Group:")
    print("-"*80)
    
    for dataset_name, dataset in [("WITH fillers", with_fillers), ("WITHOUT fillers", without_fillers)]:
        group_counts = Counter([s['group'] for s in dataset])
        print(f"\n{dataset_name}:")
        for group, count in sorted(group_counts.items()):
            print(f"  {group}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Compare matrix language distributions
    print(f"\n" + "-"*80)
    print("Matrix Language Distribution:")
    print("-"*80)
    
    for dataset_name, dataset in [("WITH fillers", with_fillers), ("WITHOUT fillers", without_fillers)]:
        matrix_counts = Counter([s['matrix_language'] for s in dataset])
        print(f"\n{dataset_name}:")
        print(f"  Cantonese: {matrix_counts['Cantonese']} ({matrix_counts['Cantonese']/len(dataset)*100:.1f}%)")
        print(f"  English: {matrix_counts['English']} ({matrix_counts['English']/len(dataset)*100:.1f}%)")
        if 'Equal' in matrix_counts:
            print(f"  Equal: {matrix_counts['Equal']} ({matrix_counts['Equal']/len(dataset)*100:.1f}%)")
    
    # Detailed breakdown by group AND matrix language
    print(f"\n" + "-"*80)
    print("Matrix Language by Participant Group:")
    print("-"*80)
    
    for group in sorted(groups):
        print(f"\n{group}:")
        for dataset_name, dataset in [("WITH fillers", with_fillers), ("WITHOUT fillers", without_fillers)]:
            group_sentences = [s for s in dataset if s['group'] == group]
            if group_sentences:
                cant_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
                eng_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
                equal_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
                print(
                    f"  {dataset_name:18} (n={len(group_sentences):4}): "
                    f"Cantonese {cant_matrix:4} ({cant_matrix/len(group_sentences)*100:5.1f}%)  |  "
                    f"English {eng_matrix:4} ({eng_matrix/len(group_sentences)*100:5.1f}%)  |  "
                    f"Equal {equal_matrix:4} ({equal_matrix/len(group_sentences)*100:5.1f}%)"
                )
    
    # Print detailed comparison table
    print("\n" + "="*80)
    print("MATRIX LANGUAGE DISTRIBUTION COMPARISON TABLE")
    print("="*80)
    
    for dataset_name, dataset in [("WITH Fillers", with_fillers), ("WITHOUT Fillers", without_fillers)]:
        print(f"\n{dataset_name}:")
        print(f"{'Group':<12} {'Total':<8} {'Cantonese':<20} {'English':<20} {'Equal':<15}")
        print("-"*80)
        
        for group in groups:
            group_sentences = [s for s in dataset if s['group'] == group]
            if group_sentences:
                total = len(group_sentences)
                cant_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
                eng_count = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
                equal_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
                
                print(
                    f"{group:<12} {total:<8} {cant_count:<5} ({cant_count/total*100:>5.1f}%)  "
                    f"{eng_count:<5} ({eng_count/total*100:>5.1f}%)  "
                    f"{equal_count:<5} ({equal_count/total*100:>5.1f}%)"
                )
    
    # Impact of filler removal
    print("\n" + "="*80)
    print("IMPACT OF FILLER REMOVAL")
    print("="*80)
    
    for group in groups:
        # WITH fillers
        with_group = [s for s in with_fillers if s['group'] == group]
        with_cant_pct = (
            sum(1 for s in with_group if s['matrix_language'] == 'Cantonese') / len(with_group) * 100
            if with_group else 0
        )
        
        # WITHOUT fillers
        without_group = [s for s in without_fillers if s['group'] == group]
        without_cant_pct = (
            sum(1 for s in without_group if s['matrix_language'] == 'Cantonese') / len(without_group) * 100
            if without_group else 0
        )
        
        change = without_cant_pct - with_cant_pct
        print(f"{group}:")
        print(f"  Cantonese matrix WITH fillers: {with_cant_pct:.1f}%")
        print(f"  Cantonese matrix WITHOUT fillers: {without_cant_pct:.1f}%")
        print(f"  Change: {change:+.1f} percentage points\n")


def plot_matrix_language_distribution(
    with_fillers: List[Dict],
    without_fillers: List[Dict],
    output_dir: str = "figures/preprocessing"
) -> None:
    """
    Create stacked bar charts showing matrix language distribution.
    
    Args:
        with_fillers: List of sentences with fillers included
        without_fillers: List of sentences with fillers excluded
        output_dir: Directory to save figures
    """
    # Create organized subfolder for these plots
    plot_dir = os.path.join(output_dir, "matrix_language_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    groups = ['Homeland', 'Heritage', 'Immersed']
    datasets = [
        ("WITH Fillers", with_fillers),
        ("WITHOUT Fillers", without_fillers)
    ]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (dataset_name, dataset) in zip([ax1, ax2], datasets):
        # Prepare counts for this dataset
        cantonese_counts = []
        english_counts = []
        
        for group in groups:
            group_sentences = [s for s in dataset if s['group'] == group]
            cant_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_matrix = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            cantonese_counts.append(cant_matrix)
            english_counts.append(eng_matrix)
        
        # Create stacked bar chart
        x = np.arange(len(groups))
        width = 0.6
        
        p1 = ax.bar(x, english_counts, width, label='English', color='#FF6B6B')
        p2 = ax.bar(x, cantonese_counts, width, bottom=english_counts, label='Cantonese', color='#4ECDC4')
        
        # Labels and formatting
        ax.set_ylabel('Count', fontsize=11)
        ax.set_xlabel('Speaker Group', fontsize=11)
        ax.set_title(
            f'Matrix Language Distribution\n{dataset_name}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, group in enumerate(groups):
            total = cantonese_counts[i] + english_counts[i]
            
            if total > 0:  # Avoid division by zero
                # English percentage (bottom section)
                eng_pct = english_counts[i] / total * 100
                if english_counts[i] > 30:  # Only show label if bar is big enough
                    ax.text(
                        i, english_counts[i]/2, f'{english_counts[i]}\n{eng_pct:.1f}%',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10
                    )
                
                # Cantonese percentage (top section)
                cant_pct = cantonese_counts[i] / total * 100
                if cantonese_counts[i] > 30:  # Only show label if bar is big enough
                    ax.text(
                        i, english_counts[i] + cantonese_counts[i]/2,
                        f'{cantonese_counts[i]}\n{cant_pct:.1f}%',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10
                    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(plot_dir, 'matrix_language_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()


def plot_equal_matrix_cases(
    with_fillers: List[Dict],
    without_fillers: List[Dict],
    output_dir: str = "figures/preprocessing"
) -> None:
    """
    Create visualization comparing equal matrix language cases across groups.
    
    Args:
        with_fillers: List of sentences with fillers included
        without_fillers: List of sentences with fillers excluded
        output_dir: Directory to save figures
    """
    # Create organized subfolder for these plots
    plot_dir = os.path.join(output_dir, "matrix_language_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    groups = ['Homeland', 'Heritage', 'Immersed']
    datasets = [
        ("WITH Fillers", with_fillers),
        ("WITHOUT Fillers", without_fillers)
    ]
    
    # Create visualization with grouped bars showing all three categories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax, (dataset_name, dataset) in zip([ax1, ax2], datasets):
        # Prepare counts for all three matrix language types
        cantonese_counts = []
        english_counts = []
        equal_counts = []
        
        for group in groups:
            group_sentences = [s for s in dataset if s['group'] == group]
            cant_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Cantonese')
            eng_count = sum(1 for s in group_sentences if s['matrix_language'] == 'English')
            equal_count = sum(1 for s in group_sentences if s['matrix_language'] == 'Equal')
            
            cantonese_counts.append(cant_count)
            english_counts.append(eng_count)
            equal_counts.append(equal_count)
        
        # Create grouped bar chart with three bars per group
        x = np.arange(len(groups))
        width = 0.25  # Width of each bar
        
        # Position bars side by side
        bars1 = ax.bar(
            x - width, cantonese_counts, width, label='Cantonese',
            color='#4ECDC4', edgecolor='black', linewidth=0.5
        )
        bars2 = ax.bar(
            x, english_counts, width, label='English',
            color='#FF6B6B', edgecolor='black', linewidth=0.5
        )
        bars3 = ax.bar(
            x + width, equal_counts, width, label='Equal',
            color='#95E1D3', edgecolor='black', linewidth=0.5
        )
        
        # Labels and formatting
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Speaker Group', fontsize=12)
        ax.set_title(
            f'Complete Matrix Language Distribution\n{dataset_name}',
            fontsize=13, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add count labels on top of each bar
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show label if there's a count
                    ax.text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold'
                    )
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'equal_matrix_cases.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()
    
    # Create a focused visualization comparing Equal cases across groups
    fig, ax = plt.subplots(figsize=(12, 6))
    
    equal_percentages_with = []
    equal_percentages_without = []
    
    for group in groups:
        # WITH fillers
        with_group = [s for s in with_fillers if s['group'] == group]
        with_equal_count = sum(1 for s in with_group if s['matrix_language'] == 'Equal')
        with_equal_pct = (with_equal_count / len(with_group) * 100) if with_group else 0
        equal_percentages_with.append(with_equal_pct)
        
        # WITHOUT fillers
        without_group = [s for s in without_fillers if s['group'] == group]
        without_equal_count = sum(1 for s in without_group if s['matrix_language'] == 'Equal')
        without_equal_pct = (without_equal_count / len(without_group) * 100) if without_group else 0
        equal_percentages_without.append(without_equal_pct)
    
    # Create grouped bar chart
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(
        x - width/2, equal_percentages_with, width, label='WITH Fillers',
        color='#95E1D3', edgecolor='black', linewidth=1, alpha=0.8
    )
    bars2 = ax.bar(
        x + width/2, equal_percentages_without, width, label='WITHOUT Fillers',
        color='#5AB9AC', edgecolor='black', linewidth=1, alpha=0.8
    )
    
    # Labels and formatting
    ax.set_ylabel('Percentage of Equal Matrix Cases (%)', fontsize=12)
    ax.set_xlabel('Speaker Group', fontsize=12)
    ax.set_title(
        'Prevalence of Equal Matrix Language Cases\nAcross Speaker Groups and Datasets',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'equal_matrix_prevalence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()


def plot_filler_impact(
    with_fillers: List[Dict],
    without_fillers: List[Dict],
    output_dir: str = "figures/preprocessing"
) -> None:
    """
    Create visualization showing the impact of filler removal on matrix language.
    
    Args:
        with_fillers: List of sentences with fillers included
        without_fillers: List of sentences with fillers excluded
        output_dir: Directory to save figures
    """
    groups = ['Homeland', 'Heritage', 'Immersed']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the change in Cantonese matrix percentage for each group
    changes = []
    for group in groups:
        # WITH fillers
        with_group = [s for s in with_fillers if s['group'] == group]
        with_cant_pct = (
            sum(1 for s in with_group if s['matrix_language'] == 'Cantonese') / len(with_group) * 100
            if with_group else 0
        )
        
        # WITHOUT fillers
        without_group = [s for s in without_fillers if s['group'] == group]
        without_cant_pct = (
            sum(1 for s in without_group if s['matrix_language'] == 'Cantonese') / len(without_group) * 100
            if without_group else 0
        )
        
        change = without_cant_pct - with_cant_pct
        changes.append(change)
    
    # Plot the changes
    x = np.arange(len(groups))
    colors = ['#FF6B6B' if c < 0 else '#4ECDC4' for c in changes]
    bars = ax.bar(x, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Change in Cantonese Matrix % (percentage points)', fontsize=11)
    ax.set_xlabel('Speaker Group', fontsize=11)
    ax.set_title(
        'Impact of Filler Removal on Cantonese Matrix Language Percentage',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, change) in enumerate(zip(bars, changes)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{change:+.1f}pp',
            ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=10
        )
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'filler_impact.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    plt.close()

