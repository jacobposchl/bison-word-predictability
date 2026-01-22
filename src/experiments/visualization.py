"""
Visualization functions for surprisal analysis results.

This module provides plotting functions for comparing surprisal values
between code-switched translations and matched monolingual baselines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


def setup_plot_style():
    """Set up consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'


def plot_surprisal_distributions(
    results_df: pd.DataFrame,
    output_path: Path,
    context_length: Optional[int] = None
):
    """
    Plot distribution comparison between CS and monolingual surprisal.
    
    Creates violin plots and box plots showing the distributions of
    surprisal values for both conditions.
    
    Args:
        results_df: DataFrame with cs_surprisal_total and mono_surprisal_total (or context-specific columns)
        output_path: Path to save the figure
        context_length: Context length to use. If None, tries to find any context length column or uses old column names.
    """
    setup_plot_style()
    
    # Filter for complete calculations only (all tokens valid)
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    complete_df = valid_df[
        (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
        (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
    ].copy()
    
    # Determine which columns to use
    if context_length is not None:
        cs_col = f'cs_surprisal_context_{context_length}'
        mono_col = f'mono_surprisal_context_{context_length}'
    else:
        # Try to find context length columns, or fall back to old column names
        context_cols = [col for col in complete_df.columns if 'cs_surprisal_context_' in col]
        if context_cols:
            import re
            match = re.search(r'context_(\d+)', context_cols[0])
            if match:
                context_length = int(match.group(1))
                cs_col = f'cs_surprisal_context_{context_length}'
                mono_col = f'mono_surprisal_context_{context_length}'
            else:
                cs_col = 'cs_surprisal_total'
                mono_col = 'mono_surprisal_total'
        else:
            cs_col = 'cs_surprisal_total'
            mono_col = 'mono_surprisal_total'
    
    # Filter to rows with valid surprisal values
    complete_df = complete_df[
        pd.notna(complete_df[cs_col]) &
        pd.notna(complete_df[mono_col])
    ].copy()
    
    # Prepare data for plotting
    cs_data = complete_df[cs_col].values
    mono_data = complete_df[mono_col].values
    
    # Create melted dataframe for seaborn
    plot_data = pd.DataFrame({
        'Surprisal': np.concatenate([cs_data, mono_data]),
        'Condition': ['Code-Switched\nTranslation'] * len(cs_data) + ['Monolingual\nBaseline'] * len(mono_data)
    })
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Violin + box plot
    sns.violinplot(data=plot_data, x='Condition', y='Surprisal', ax=ax, palette='Set2')
    sns.boxplot(data=plot_data, x='Condition', y='Surprisal', ax=ax, 
                width=0.3, palette='Set2', showfliers=False, 
                boxprops=dict(alpha=0.7))
    
    ax.set_ylabel('Surprisal (bits)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title(f'Surprisal Distribution Comparison\n(n={len(valid_df)} comparisons)', 
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add mean lines
    cs_mean = cs_data.mean()
    mono_mean = mono_data.mean()
    ax.axhline(cs_mean, color='#8DA0CB', linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'CS mean: {cs_mean:.2f}')
    ax.axhline(mono_mean, color='#FC8D62', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Mono mean: {mono_mean:.2f}')
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_path}")


def plot_scatter_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
    context_length: Optional[int] = None
):
    """
    Plot scatter comparison of CS vs monolingual surprisal.
    
    Creates scatter plots with identity line to visualize
    the relationship between CS and monolingual surprisal values.
    
    Args:
        results_df: DataFrame with cs_surprisal_total and mono_surprisal_total (or context-specific columns)
        output_path: Path to save the figure
        context_length: Context length to use. If None, tries to find any context length column or uses old column names.
    """
    setup_plot_style()
    
    # Filter valid data
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    
    # Determine which columns to use
    if context_length is not None:
        cs_col = f'cs_surprisal_context_{context_length}'
        mono_col = f'mono_surprisal_context_{context_length}'
    else:
        # Try to find context length columns, or fall back to old column names
        context_cols = [col for col in valid_df.columns if 'cs_surprisal_context_' in col]
        if context_cols:
            import re
            match = re.search(r'context_(\d+)', context_cols[0])
            if match:
                context_length = int(match.group(1))
                cs_col = f'cs_surprisal_context_{context_length}'
                mono_col = f'mono_surprisal_context_{context_length}'
            else:
                cs_col = 'cs_surprisal_total'
                mono_col = 'mono_surprisal_total'
        else:
            cs_col = 'cs_surprisal_total'
            mono_col = 'mono_surprisal_total'
    
    # Filter to rows with valid surprisal values
    valid_df = valid_df[
        pd.notna(valid_df[cs_col]) &
        pd.notna(valid_df[mono_col])
    ].copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(valid_df[mono_col], valid_df[cs_col],
               alpha=0.5, s=30, color='#66C2A5', edgecolors='black', linewidth=0.5)
    
    # Identity line
    max_val = max(valid_df[cs_col].max(), valid_df[mono_col].max())
    min_val = min(valid_df[cs_col].min(), valid_df[mono_col].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, 
            label='Identity (CS = Mono)')
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        valid_df[mono_col], valid_df[cs_col]
    )
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', alpha=0.7, linewidth=2,
            label=f'Regression (r²={r_value**2:.3f})')
    
    ax.set_xlabel('Monolingual Baseline Surprisal (bits)', fontweight='bold')
    ax.set_ylabel('Code-Switched Translation Surprisal (bits)', fontweight='bold')
    ax.set_title(f'CS vs. Monolingual Surprisal\n(n={len(valid_df)} complete comparisons)',
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def plot_difference_histogram(
    results_df: pd.DataFrame,
    output_path: Path,
    context_length: Optional[int] = None
):
    """
    Plot histogram of surprisal differences (CS - Monolingual).
    
    Shows the distribution of differences to visualize whether
    CS translations have systematically higher or lower surprisal.
    
    Args:
        results_df: DataFrame with surprisal_difference column (or context-specific)
        output_path: Path to save the figure
        context_length: Context length to use. If None, tries to find any context length column or uses old column names.
    """
    setup_plot_style()
    
    # Filter for complete calculations only (all tokens valid)
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    complete_df = valid_df[
        (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
        (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
    ].copy()
    
    # Determine which column to use
    if context_length is not None:
        diff_col = f'surprisal_difference_context_{context_length}'
    else:
        # Try to find context length columns, or fall back to old column names
        context_cols = [col for col in complete_df.columns if 'surprisal_difference_context_' in col]
        if context_cols:
            import re
            match = re.search(r'context_(\d+)', context_cols[0])
            if match:
                context_length = int(match.group(1))
                diff_col = f'surprisal_difference_context_{context_length}'
            else:
                diff_col = 'surprisal_difference'
        else:
            diff_col = 'surprisal_difference'
    
    # Filter to rows with valid difference values
    complete_df = complete_df[pd.notna(complete_df[diff_col])].copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram
    differences = complete_df[diff_col].values
    
    ax.hist(differences, bins=30, alpha=0.7, color='#8DA0CB', edgecolor='black', linewidth=1.2)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax.axvline(differences.mean(), color='darkblue', linestyle='-', linewidth=2,
              label=f'Mean: {differences.mean():.2f}')
    
    ax.set_xlabel('Surprisal Difference (CS - Mono, bits)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'Distribution of Surprisal Differences\n(n={len(complete_df)} complete comparisons)',
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'Mean: {differences.mean():.3f}\nMedian: {np.median(differences):.3f}\nStd: {differences.std():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved difference histogram to {output_path}")


def plot_summary_statistics(
    results_df: pd.DataFrame,
    output_path: Path,
    stats_dict: dict = None,
    context_length: Optional[int] = None
):
    """
    Create a comprehensive multi-panel summary figure.
    
    Combines multiple visualization types into a single summary figure:
    - Violin/box plots of distributions
    - Scatter plot with regression
    - Histogram of differences
    - Statistical summary text
    
    Args:
        results_df: DataFrame with surprisal results
        output_path: Path to save the figure
        stats_dict: Statistics dictionary from compute_statistics() (optional)
        context_length: Context length to use. If None, tries to find any context length column or uses old column names.
    """
    setup_plot_style()
    
    # Filter for complete calculations only (all tokens valid)
    valid_df = results_df[results_df['calculation_success'] == True].copy()
    complete_df = valid_df[
        (valid_df['cs_num_valid_tokens'] == valid_df['cs_num_tokens']) &
        (valid_df['mono_num_valid_tokens'] == valid_df['mono_num_tokens'])
    ].copy()
    
    # Determine which columns to use
    if context_length is not None:
        cs_col = f'cs_surprisal_context_{context_length}'
        mono_col = f'mono_surprisal_context_{context_length}'
        diff_col = f'surprisal_difference_context_{context_length}'
    else:
        # Try to find context length columns, or fall back to old column names
        context_cols = [col for col in complete_df.columns if 'cs_surprisal_context_' in col]
        if context_cols:
            import re
            match = re.search(r'context_(\d+)', context_cols[0])
            if match:
                context_length = int(match.group(1))
                cs_col = f'cs_surprisal_context_{context_length}'
                mono_col = f'mono_surprisal_context_{context_length}'
                diff_col = f'surprisal_difference_context_{context_length}'
            else:
                cs_col = 'cs_surprisal_total'
                mono_col = 'mono_surprisal_total'
                diff_col = 'surprisal_difference'
        else:
            cs_col = 'cs_surprisal_total'
            mono_col = 'mono_surprisal_total'
            diff_col = 'surprisal_difference'
    
    # Filter to rows with valid surprisal values
    complete_df = complete_df[
        pd.notna(complete_df[cs_col]) &
        pd.notna(complete_df[mono_col])
    ].copy()
    
    # Create 2x2 subplot figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Distribution comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data = pd.DataFrame({
        'Surprisal': np.concatenate([
            complete_df[cs_col].values,
            complete_df[mono_col].values
        ]),
        'Condition': ['CS Translation'] * len(complete_df) + ['Mono Baseline'] * len(complete_df)
    })
    sns.violinplot(data=plot_data, x='Condition', y='Surprisal', ax=ax1, palette='Set2')
    sns.boxplot(data=plot_data, x='Condition', y='Surprisal', ax=ax1,
               width=0.3, palette='Set2', showfliers=False, boxprops=dict(alpha=0.7))
    ax1.set_ylabel('Surprisal (bits)', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_title('Distribution Comparison', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Scatter plot (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(complete_df[mono_col], complete_df[cs_col],
               alpha=0.5, s=25, color='#66C2A5', edgecolors='black', linewidth=0.5)
    max_val = max(complete_df[cs_col].max(), complete_df[mono_col].max())
    min_val = min(complete_df[cs_col].min(), complete_df[mono_col].min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Monolingual Surprisal (bits)', fontweight='bold')
    ax2.set_ylabel('CS Translation Surprisal (bits)', fontweight='bold')
    ax2.set_title('CS vs. Monolingual Correlation', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Difference histogram (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    differences = complete_df[diff_col].values
    ax3.hist(differences, bins=30, alpha=0.7, color='#8DA0CB', edgecolor='black', linewidth=1.2)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(differences.mean(), color='darkblue', linestyle='-', linewidth=2)
    ax3.set_xlabel('Surprisal Difference (CS - Mono, bits)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Differences', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Statistical summary (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""
STATISTICAL SUMMARY
{'='*40}

Sample Size:
  Total comparisons: {stats_dict['n_total']}
  Complete comparisons: {stats_dict['n_complete']}
  Complete rate: {stats_dict['complete_rate']:.1%}

Code-Switched Translation:
  Mean: {stats_dict['cs_surprisal_mean']:.4f} bits
  Median: {stats_dict['cs_surprisal_median']:.4f} bits
  Std: {stats_dict['cs_surprisal_std']:.4f} bits

Monolingual Baseline:
  Mean: {stats_dict['mono_surprisal_mean']:.4f} bits
  Median: {stats_dict['mono_surprisal_median']:.4f} bits
  Std: {stats_dict['mono_surprisal_std']:.4f} bits

Difference (CS - Mono):
  Mean: {stats_dict['difference_mean']:.4f} bits
  Median: {stats_dict['difference_median']:.4f} bits
  Std: {stats_dict['difference_std']:.4f} bits

Paired t-test:
  t-statistic: {stats_dict['ttest_statistic']:.4f}
  p-value: {stats_dict['ttest_pvalue']:.6f}
  Significance: {'***' if stats_dict['ttest_pvalue'] < 0.001 else '**' if stats_dict['ttest_pvalue'] < 0.01 else '*' if stats_dict['ttest_pvalue'] < 0.05 else 'ns'}

Effect Size:
  Cohen's d: {stats_dict['cohens_d']:.4f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle('Surprisal Comparison: Code-Switched Translation vs. Monolingual Baseline',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary figure to {output_path}")
