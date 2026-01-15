"""
Comprehensive visualization script for code_switching_WITHOUT_fillers.csv
Analyzes code-switching patterns between English and Cantonese.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = Path("results/preprocessing/code_switching_WITHOUT_fillers.csv")
OUTPUT_DIR = Path("figures/preprocessing/code_switching")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate derived metrics
df['duration'] = (df['end_time'] - df['start_time']) / 1000  # Convert to seconds
df['sentence_length'] = df['reconstructed_sentence'].str.split().str.len()

# Parse pattern to get language segments
def parse_pattern(pattern):
    """Extract language segments from pattern string."""
    segments = re.findall(r'([EC])(\d+)', str(pattern))
    return segments

def count_switches(pattern):
    """Count number of language switches in a pattern."""
    segments = parse_pattern(pattern)
    if len(segments) <= 1:
        return 0
    switches = 0
    for i in range(1, len(segments)):
        if segments[i][0] != segments[i-1][0]:
            switches += 1
    return switches

def get_english_words(pattern):
    """Get total English words in pattern."""
    segments = parse_pattern(pattern)
    return sum(int(length) for lang, length in segments if lang == 'E')

def get_cantonese_words(pattern):
    """Get total Cantonese words in pattern."""
    segments = parse_pattern(pattern)
    return sum(int(length) for lang, length in segments if lang == 'C')

df['num_switches'] = df['pattern'].apply(count_switches)
df['num_segments'] = df['pattern'].apply(lambda x: len(parse_pattern(x)))
df['english_words'] = df['pattern'].apply(get_english_words)
df['cantonese_words'] = df['pattern'].apply(get_cantonese_words)
df['english_ratio'] = df['english_words'] / (df['english_words'] + df['cantonese_words'])

print(f"Loaded {len(df)} code-switching sentences")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Groups: {df['group'].unique()}")

# ============================================================================
# 1. Distribution of Code-Switching Points
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['num_switches'], bins=range(0, df['num_switches'].max() + 2),
        edgecolor='black', alpha=0.7, color='#8e44ad')
ax.set_xlabel('Number of Language Switches', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Code-Switching: Distribution of Language Switch Points', 
             fontsize=14, fontweight='bold')
ax.axvline(df['num_switches'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["num_switches"].mean():.2f}')
ax.axvline(df['num_switches'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {df["num_switches"].median():.0f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'switch_points_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved switch_points_distribution.png")
plt.close()

# ============================================================================
# 2. Number of Language Segments
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['num_segments'], bins=range(0, df['num_segments'].max() + 2),
        edgecolor='black', alpha=0.7, color='#27ae60')
ax.set_xlabel('Number of Language Segments', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Code-Switching: Number of Language Segments per Sentence', 
             fontsize=14, fontweight='bold')
ax.axvline(df['num_segments'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["num_segments"].mean():.2f}')
ax.axvline(df['num_segments'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {df["num_segments"].median():.0f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'language_segments_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved language_segments_distribution.png")
plt.close()

# ============================================================================
# 3. English-Cantonese Balance Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['english_ratio'], bins=30, edgecolor='black', alpha=0.7, color='#e67e22')
ax.set_xlabel('English Word Ratio', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Code-Switching: English vs Cantonese Balance\n(0 = All Cantonese, 1 = All English)', 
             fontsize=14, fontweight='bold')
ax.axvline(0.5, color='green', linestyle='-', linewidth=2, alpha=0.7,
           label='Balanced (50/50)')
ax.axvline(df['english_ratio'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["english_ratio"].mean():.2f}')
ax.axvline(df['english_ratio'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {df["english_ratio"].median():.2f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'english_cantonese_balance.png', dpi=300, bbox_inches='tight')
print("✓ Saved english_cantonese_balance.png")
plt.close()

# ============================================================================
# 4. Matrix Language Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
matrix_counts = df['matrix_language'].value_counts()
colors = ['#3498db', '#e74c3c', '#95a5a6']
bars = ax.bar(range(len(matrix_counts)), matrix_counts.values,
              color=colors[:len(matrix_counts)], edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(matrix_counts)))
ax.set_xticklabels(matrix_counts.index, fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Code-Switching: Matrix Language Distribution', 
             fontsize=14, fontweight='bold')

for i, (idx, val) in enumerate(matrix_counts.items()):
    pct = (val / len(df)) * 100
    ax.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'matrix_language_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved matrix_language_distribution.png")
plt.close()

# ============================================================================
# 5. Switches vs Sentence Duration
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(df['num_switches'], df['duration'], 
                     c=df['english_ratio'], cmap='RdYlGn_r', 
                     alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Language Switches', fontsize=12)
ax.set_ylabel('Duration (seconds)', fontsize=12)
ax.set_title('Code-Switching: Number of Switches vs Duration', 
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('English Ratio', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'switches_vs_duration_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved switches_vs_duration_scatter.png")
plt.close()

# ============================================================================
# 6. Participant Contributions
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
participant_counts = df['participant_id'].value_counts().sort_values(ascending=False)
colors = plt.cm.plasma(np.linspace(0, 1, len(participant_counts)))
ax.bar(range(len(participant_counts)), participant_counts.values,
       edgecolor='black', alpha=0.8, color=colors)
ax.set_xlabel('Participant (sorted by contribution)', fontsize=12)
ax.set_ylabel('Number of Code-Switching Sentences', fontsize=12)
ax.set_title('Code-Switching: Participant Contributions', 
             fontsize=14, fontweight='bold')
ax.axhline(participant_counts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {participant_counts.mean():.1f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'participant_contributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved participant_contributions.png")
plt.close()

# ============================================================================
# 7. Group Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
group_counts = df['group'].value_counts()
colors_group = ['#1abc9c', '#f39c12', '#9b59b6']
bars = ax.bar(range(len(group_counts)), group_counts.values,
              color=colors_group[:len(group_counts)], edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(group_counts)))
ax.set_xticklabels(group_counts.index, fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Code-Switching: Distribution by Group', 
             fontsize=14, fontweight='bold')

for i, (idx, val) in enumerate(group_counts.items()):
    pct = (val / len(df)) * 100
    ax.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'group_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved group_distribution.png")
plt.close()

# ============================================================================
# 8. Matrix Language by Number of Switches
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
switch_by_matrix = df.groupby(['num_switches', 'matrix_language']).size().unstack(fill_value=0)
switch_by_matrix.plot(kind='bar', stacked=True, ax=ax, 
                       color=['#3498db', '#e74c3c', '#95a5a6'],
                       edgecolor='black', alpha=0.8)
ax.set_xlabel('Number of Switches', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Code-Switching: Matrix Language by Number of Switches', 
             fontsize=14, fontweight='bold')
ax.legend(title='Matrix Language', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'matrix_by_switches.png', dpi=300, bbox_inches='tight')
print("✓ Saved matrix_by_switches.png")
plt.close()

# ============================================================================
# 9. English vs Cantonese Word Counts Scatter
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(df['cantonese_words'], df['english_words'], 
                     c=df['num_switches'], cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Cantonese Words', fontsize=12)
ax.set_ylabel('English Words', fontsize=12)
ax.set_title('Code-Switching: English vs Cantonese Word Counts', 
             fontsize=14, fontweight='bold')

# Add diagonal line (50/50 balance)
max_val = max(df['cantonese_words'].max(), df['english_words'].max())
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, linewidth=2, 
        label='Balanced (50/50)')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Switches', fontsize=12)

ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'english_vs_cantonese_words.png', dpi=300, bbox_inches='tight')
print("✓ Saved english_vs_cantonese_words.png")
plt.close()

# ============================================================================
# 10. Top 10 Most Common Patterns
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
top_patterns = df['pattern'].value_counts().head(10)
ax.barh(range(len(top_patterns)), top_patterns.values, 
        color='steelblue', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(top_patterns)))
ax.set_yticklabels(top_patterns.index, fontsize=10)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Code-Switching: Top 10 Most Common Patterns', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, val in enumerate(top_patterns.values):
    ax.text(val, i, f' {val}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved top_patterns.png")
plt.close()

# ============================================================================
# 11. Summary Statistics
# ============================================================================
summary_stats = pd.DataFrame({
    'Total CS Sentences': [len(df)],
    'Unique Participants': [df['participant_id'].nunique()],
    'Mean Switches': [df['num_switches'].mean()],
    'Median Switches': [df['num_switches'].median()],
    'Mean Segments': [df['num_segments'].mean()],
    'Mean English Ratio': [df['english_ratio'].mean()],
    'Median English Ratio': [df['english_ratio'].median()],
    'Mean Duration (s)': [df['duration'].mean()],
    'Cantonese Matrix Count': [(df['matrix_language'] == 'Cantonese').sum()],
    'English Matrix Count': [(df['matrix_language'] == 'English').sum()],
    'Unique Patterns': [df['pattern'].nunique()]
})

print("\n" + "="*70)
print("CODE-SWITCHING SUMMARY STATISTICS")
print("="*70)
for col in summary_stats.columns:
    val = summary_stats[col].values[0]
    if isinstance(val, float):
        print(f"{col:.<45} {val:>20.2f}")
    else:
        print(f"{col:.<45} {val:>20}")
print("="*70)

# Save summary to CSV
summary_stats.T.to_csv(OUTPUT_DIR / 'summary_statistics.csv', header=['Value'])
print("\n✓ Saved summary_statistics.csv")

print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
