"""
Visualization functions for surprisal analysis results.

This module provides functions for creating plots from surprisal analysis datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def _load_all_surprisal_data(results_base: Path) -> dict:
    """
    Load all surprisal results from all window sizes and context lengths.
    
    Args:
        results_base: Base directory containing window_* subdirectories
        
    Returns:
        Dictionary with structure: {window_size: {context_length: DataFrame}}
    """
    datasets = {}
    
    # Find all window size directories
    window_dirs = sorted([d for d in results_base.iterdir() 
                         if d.is_dir() and d.name.startswith('window_')])
    
    for window_dir in window_dirs:
        window_size = int(window_dir.name.replace('window_', ''))
        datasets[window_size] = {}
        
        results_csv = window_dir / "surprisal_results.csv"
        
        if not results_csv.exists():
            logger.warning(f"Results CSV not found: {results_csv}")
            continue
        
        try:
            df = pd.read_csv(results_csv)
            
            # Extract all context lengths from column names
            cs_cols = [col for col in df.columns if col.startswith('cs_surprisal_context_')]
            for col in cs_cols:
                match = re.search(r'context_(\d+)', col)
                if match:
                    context_length = int(match.group(1))
                    # Store data for this context length
                    datasets[window_size][context_length] = df
                    
        except Exception as e:
            logger.warning(f"Error loading {results_csv}: {e}")
    
    return datasets


def plot_surprisal_distributions(
    results_base: Path,
    output_path: Path,
    model_type: str = "autoregressive"
) -> str:
    """
    Plot distribution of surprisal for monolingual vs code-switched sentences.
    Combines all window sizes and context lengths into single distributions.
    
    Args:
        results_base: Base directory containing surprisal results
        output_path: Path to save the figure
        model_type: Model type (autoregressive or masked)
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating surprisal distribution plot...")
    
    # Load all data
    datasets = _load_all_surprisal_data(results_base)
    
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Collect all surprisal values
    all_cs_surprisals = []
    all_mono_surprisals = []
    
    for window_size in sorted(datasets.keys()):
        for context_length in sorted(datasets[window_size].keys()):
            df = datasets[window_size][context_length]
            
            # Extract CS surprisal for this context
            cs_col = f'cs_surprisal_context_{context_length}'
            mono_col = f'mono_surprisal_context_{context_length}'
            
            if cs_col in df.columns:
                cs_values = df[cs_col].dropna()
                all_cs_surprisals.extend(cs_values.tolist())
            
            if mono_col in df.columns:
                mono_values = df[mono_col].dropna()
                all_mono_surprisals.extend(mono_values.tolist())
    
    if not all_cs_surprisals or not all_mono_surprisals:
        logger.warning("No surprisal values found")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create DataFrames for plotting
    cs_df = pd.DataFrame({'surprisal': all_cs_surprisals, 'type': 'Code-Switched'})
    mono_df = pd.DataFrame({'surprisal': all_mono_surprisals, 'type': 'Monolingual'})
    combined_df = pd.concat([cs_df, mono_df], ignore_index=True)
    
    # Plot KDE distributions separately to ensure proper legend
    sns.kdeplot(data=cs_df, x='surprisal', label='Code-Switched',
               color='#e74c3c', ax=ax, linewidth=2.5, alpha=0.6, fill=True, common_norm=False)
    sns.kdeplot(data=mono_df, x='surprisal', label='Monolingual',
               color='#3498db', ax=ax, linewidth=2.5, alpha=0.6, fill=True, common_norm=False)
    
    # Add mean lines (dotted)
    cs_mean = np.mean(all_cs_surprisals)
    mono_mean = np.mean(all_mono_surprisals)
    ax.axvline(cs_mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
               label=f'CS Mean: {cs_mean:.2f}')
    ax.axvline(mono_mean, color='#3498db', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Mono Mean: {mono_mean:.2f}')
    
    ax.set_xlabel('Surprisal', fontsize=13, fontweight='medium')
    ax.set_ylabel('Density', fontsize=13, fontweight='medium')
    ax.set_title(f'Surprisal Distribution: Code-Switched vs Monolingual\n({model_type.capitalize()} Model, All Window Sizes & Context Lengths)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved surprisal distribution plot to: {output_path}")
    return str(output_path)


def plot_surprisal_distributions_by_context(
    results_base: Path,
    output_path: Path,
    context_length: int,
    model_type: str = "autoregressive"
) -> str:
    """
    Plot distribution of surprisal for monolingual vs code-switched sentences
    for a specific context length, averaging across all window sizes.
    
    Args:
        results_base: Base directory containing surprisal results
        output_path: Path to save the figure
        context_length: Context length (1, 2, or 3)
        model_type: Model type (autoregressive or masked)
        
    Returns:
        Path to saved figure
    """
    logger.info(f"Creating surprisal distribution plot for context length {context_length}...")
    
    # Load all data
    datasets = _load_all_surprisal_data(results_base)
    
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Collect all surprisal values for this context length (across all window sizes)
    all_cs_surprisals = []
    all_mono_surprisals = []
    
    for window_size in sorted(datasets.keys()):
        if context_length in datasets[window_size]:
            df = datasets[window_size][context_length]
            
            # Extract CS surprisal for this context
            cs_col = f'cs_surprisal_context_{context_length}'
            mono_col = f'mono_surprisal_context_{context_length}'
            
            if cs_col in df.columns:
                cs_values = df[cs_col].dropna()
                all_cs_surprisals.extend(cs_values.tolist())
            
            if mono_col in df.columns:
                mono_values = df[mono_col].dropna()
                all_mono_surprisals.extend(mono_values.tolist())
    
    if not all_cs_surprisals or not all_mono_surprisals:
        logger.warning(f"No surprisal values found for context length {context_length}")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create DataFrames for plotting
    cs_df = pd.DataFrame({'surprisal': all_cs_surprisals, 'type': 'Code-Switched'})
    mono_df = pd.DataFrame({'surprisal': all_mono_surprisals, 'type': 'Monolingual'})
    
    # Plot KDE distributions separately to ensure proper legend
    sns.kdeplot(data=cs_df, x='surprisal', label='Code-Switched',
               color='#e74c3c', ax=ax, linewidth=2.5, alpha=0.6, fill=True, common_norm=False)
    sns.kdeplot(data=mono_df, x='surprisal', label='Monolingual',
               color='#3498db', ax=ax, linewidth=2.5, alpha=0.6, fill=True, common_norm=False)
    
    # Add mean lines (dotted)
    cs_mean = np.mean(all_cs_surprisals)
    mono_mean = np.mean(all_mono_surprisals)
    ax.axvline(cs_mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8,
               label=f'CS Mean: {cs_mean:.2f}')
    ax.axvline(mono_mean, color='#3498db', linestyle='--', linewidth=2, alpha=0.8,
               label=f'Mono Mean: {mono_mean:.2f}')
    
    ax.set_xlabel('Surprisal', fontsize=13, fontweight='medium')
    ax.set_ylabel('Density', fontsize=13, fontweight='medium')
    ax.set_title(f'Surprisal Distribution: Code-Switched vs Monolingual\n({model_type.capitalize()} Model, Context Length {context_length}, Averaged Across Window Sizes)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved surprisal distribution plot to: {output_path}")
    return str(output_path)


def plot_surprisal_differences_by_context(
    results_base: Path,
    output_path: Path,
    context_length: int,
    model_type: str = "autoregressive"
) -> str:
    """
    Plot distribution of surprisal differences for a specific context length,
    averaging across all window sizes.
    
    Args:
        results_base: Base directory containing surprisal results
        output_path: Path to save the figure
        context_length: Context length (1, 2, or 3)
        model_type: Model type (autoregressive or masked)
        
    Returns:
        Path to saved figure
    """
    logger.info(f"Creating surprisal difference plot for context length {context_length}...")
    
    # Load all data
    datasets = _load_all_surprisal_data(results_base)
    
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Collect all difference values for this context length (across all window sizes)
    all_differences = []
    
    for window_size in sorted(datasets.keys()):
        if context_length in datasets[window_size]:
            df = datasets[window_size][context_length]
            
            # Filter for complete calculations
            valid_df = df[df['calculation_success'] == True].copy()
            complete_df = valid_df[
                (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
                (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
            ].copy()
            
            # Extract difference for this context
            diff_col = f'surprisal_difference_context_{context_length}'
            if diff_col in complete_df.columns:
                diff_values = complete_df[diff_col].dropna()
                all_differences.extend(diff_values.tolist())
    
    if not all_differences:
        logger.warning(f"No difference values found for context length {context_length}")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create DataFrame for plotting
    diff_df = pd.DataFrame({'difference': all_differences})
    
    # Plot histogram with KDE overlay
    sns.histplot(data=diff_df, x='difference', bins=30, kde=True,
                color='#8DA0CB', ax=ax, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Add mean line
    mean_diff = np.mean(all_differences)
    ax.axvline(mean_diff, color='#333333', linestyle='-', linewidth=2.5,
              label=f'Mean: {mean_diff:.2f}')
    
    ax.set_xlabel('Surprisal Difference (CS - Mono, bits)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Count', fontsize=13, fontweight='medium')
    ax.set_title(f'Distribution of Surprisal Differences\n({model_type.capitalize()} Model, Context Length {context_length}, Averaged Across Window Sizes, n={len(all_differences)})',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved surprisal difference plot to: {output_path}")
    return str(output_path)


def plot_surprisal_distributions_matrix(
    results_base: Path,
    output_path: Path,
    model_type: str = "autoregressive"
) -> str:
    """
    Plot a 3x3 matrix of surprisal distributions for all context length and window size combinations.
    
    Args:
        results_base: Base directory containing surprisal results
        output_path: Path to save the figure
        model_type: Model type (autoregressive or masked)
        
    Returns:
        Path to saved figure
    """
    logger.info("Creating surprisal distribution matrix plot...")
    
    # Load all data
    datasets = _load_all_surprisal_data(results_base)
    
    if not datasets:
        logger.warning("No valid datasets found")
        return ""
    
    # Create 3x3 subplot grid (3 context lengths × 3 window sizes)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'Surprisal Distributions: All Context × Window Combinations\n({model_type.capitalize()} Model)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    context_lengths = [1, 2, 3]
    window_sizes = sorted(datasets.keys())
    
    for i, context_length in enumerate(context_lengths):
        for j, window_size in enumerate(window_sizes):
            ax = axes[i, j]
            
            # Get data for this context/window combination
            if window_size in datasets and context_length in datasets[window_size]:
                df = datasets[window_size][context_length]
                
                # Extract surprisal values
                cs_col = f'cs_surprisal_context_{context_length}'
                mono_col = f'mono_surprisal_context_{context_length}'
                
                cs_values = df[cs_col].dropna() if cs_col in df.columns else []
                mono_values = df[mono_col].dropna() if mono_col in df.columns else []
                
                if len(cs_values) > 0 and len(mono_values) > 0:
                    # Create DataFrames for plotting
                    cs_df = pd.DataFrame({'surprisal': cs_values.tolist()})
                    mono_df = pd.DataFrame({'surprisal': mono_values.tolist()})
                    
                    # Plot KDE distributions
                    sns.kdeplot(data=cs_df, x='surprisal', label='Code-Switched',
                               color='#e74c3c', ax=ax, linewidth=2, alpha=0.6, fill=True, common_norm=False)
                    sns.kdeplot(data=mono_df, x='surprisal', label='Monolingual',
                               color='#3498db', ax=ax, linewidth=2, alpha=0.6, fill=True, common_norm=False)
                    
                    # Add mean lines (dotted)
                    cs_mean = np.mean(cs_values)
                    mono_mean = np.mean(mono_values)
                    ax.axvline(cs_mean, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.axvline(mono_mean, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    # Set labels and title
                    if i == 2:  # Bottom row
                        ax.set_xlabel('Surprisal', fontsize=10, fontweight='medium')
                    if j == 0:  # Left column
                        ax.set_ylabel('Density', fontsize=10, fontweight='medium')
                    
                    ax.set_title(f'Context {context_length}, Window {window_size}\n(n={len(cs_values)})',
                                fontsize=11, fontweight='bold')
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
                    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
                    ax.set_axisbelow(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#d0d0d0')
                    ax.spines['bottom'].set_color('#d0d0d0')
                else:
                    # No data available
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'Context {context_length}, Window {window_size}',
                                fontsize=11, fontweight='bold')
                    ax.axis('off')
            else:
                # No data available
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Context {context_length}, Window {window_size}',
                            fontsize=11, fontweight='bold')
                ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved surprisal distribution matrix plot to: {output_path}")
    return str(output_path)

