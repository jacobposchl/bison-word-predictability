"""
Comprehensive visualization script for all_sentences.csv
Generates informative plots about the complete sentence dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = Path("results/preprocessing/all_sentences.csv")
OUTPUT_DIR = Path("figures/preprocessing/all_sentences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate derived metrics
df['duration'] = (df['end_time'] - df['start_time']) / 1000  # Convert to seconds
df['sentence_length'] = df['reconstructed_sentence'].str.split().str.len()
df['contains_english'] = df['pattern'].str.contains('E', na=False)
df['contains_cantonese'] = df['pattern'].str.contains('C', na=False)
df['is_code_switching'] = df['contains_english'] & df['contains_cantonese']

print(f"Loaded {len(df)} sentences")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Groups: {df['group'].unique()}")

# ============================================================================
# 1. Distribution of Sentence Durations
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['duration'], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Duration (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Sentence Durations', fontsize=14, fontweight='bold')
ax.axvline(df['duration'].median(), color='red', linestyle='--', 
           label=f'Median: {df["duration"].median():.2f}s')
ax.axvline(df['duration'].mean(), color='green', linestyle='--', 
           label=f'Mean: {df["duration"].mean():.2f}s')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved duration_distribution.png")
plt.close()

# ============================================================================
# 2. Sentence Length Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['sentence_length'], bins=range(0, df['sentence_length'].max() + 2), 
        edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Number of Words', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Sentence Lengths (Word Count)', fontsize=14, fontweight='bold')
ax.axvline(df['sentence_length'].median(), color='red', linestyle='--', 
           label=f'Median: {df["sentence_length"].median():.0f} words')
ax.axvline(df['sentence_length'].mean(), color='green', linestyle='--', 
           label=f'Mean: {df["sentence_length"].mean():.1f} words')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sentence_length_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved sentence_length_distribution.png")
plt.close()

# ============================================================================
# 3. Matrix Language Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
matrix_counts = df['matrix_language'].value_counts()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(range(len(matrix_counts)), matrix_counts.values, 
              color=colors[:len(matrix_counts)], edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(matrix_counts)))
ax.set_xticklabels(matrix_counts.index, fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution by Matrix Language', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (idx, val) in enumerate(matrix_counts.items()):
    pct = (val / len(df)) * 100
    ax.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'matrix_language_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved matrix_language_distribution.png")
plt.close()

# ============================================================================
# 4. Code-Switching vs Monolingual
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cs_counts = df['is_code_switching'].value_counts()
labels = ['Monolingual', 'Code-Switching']
colors = ['#66c2a5', '#fc8d62']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax.pie(cs_counts.values, labels=labels, autopct='%1.1f%%',
                                     colors=colors, explode=explode, startangle=90,
                                     textprops={'fontsize': 12})
ax.set_title('Code-Switching vs Monolingual Sentences', fontsize=14, fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'code_switching_vs_monolingual.png', dpi=300, bbox_inches='tight')
print("✓ Saved code_switching_vs_monolingual.png")
plt.close()

# ============================================================================
# 5. Participant Contribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
participant_counts = df['participant_id'].value_counts().sort_values(ascending=False)
ax.bar(range(len(participant_counts)), participant_counts.values, 
       edgecolor='black', alpha=0.7, color='mediumpurple')
ax.set_xlabel('Participant (sorted by contribution)', fontsize=12)
ax.set_ylabel('Number of Sentences', fontsize=12)
ax.set_title('Sentence Contributions per Participant', fontsize=14, fontweight='bold')
ax.axhline(participant_counts.mean(), color='red', linestyle='--', 
           label=f'Mean: {participant_counts.mean():.1f}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'participant_contributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved participant_contributions.png")
plt.close()

# ============================================================================
# 6. Group Comparison
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
group_counts = df['group'].value_counts()
bars = ax.bar(range(len(group_counts)), group_counts.values, 
              color='coral', edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(group_counts)))
ax.set_xticklabels(group_counts.index, fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Sentence Distribution by Group', fontsize=14, fontweight='bold')

for i, (idx, val) in enumerate(group_counts.items()):
    pct = (val / len(df)) * 100
    ax.text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'group_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved group_distribution.png")
plt.close()

# ============================================================================
# 7. Pattern Complexity (number of switches)
# ============================================================================
df['pattern_segments'] = df['pattern'].str.findall(r'[EC]\d+').str.len()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['pattern_segments'], bins=range(0, df['pattern_segments'].max() + 2),
        edgecolor='black', alpha=0.7, color='teal')
ax.set_xlabel('Number of Language Segments in Pattern', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Pattern Complexity: Number of Language Segments', fontsize=14, fontweight='bold')
ax.axvline(df['pattern_segments'].median(), color='red', linestyle='--',
           label=f'Median: {df["pattern_segments"].median():.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pattern_complexity.png', dpi=300, bbox_inches='tight')
print("✓ Saved pattern_complexity.png")
plt.close()

# ============================================================================
# 8. Duration vs Sentence Length Scatter
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(df['sentence_length'], df['duration'], 
                     c=df['is_code_switching'].astype(int), 
                     cmap='viridis', alpha=0.5, s=20)
ax.set_xlabel('Sentence Length (words)', fontsize=12)
ax.set_ylabel('Duration (seconds)', fontsize=12)
ax.set_title('Sentence Length vs Duration', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_label('Sentence Type', fontsize=12)
cbar.ax.set_yticklabels(['Monolingual', 'Code-Switching'])

# Add trend line
z = np.polyfit(df['sentence_length'], df['duration'], 1)
p = np.poly1d(z)
ax.plot(df['sentence_length'].sort_values(), 
        p(df['sentence_length'].sort_values()), 
        "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'length_vs_duration_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved length_vs_duration_scatter.png")
plt.close()

# ============================================================================
# 9. Matrix Language by Group
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
matrix_by_group = pd.crosstab(df['group'], df['matrix_language'], normalize='index') * 100

matrix_by_group.plot(kind='bar', stacked=True, ax=ax, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                      edgecolor='black', alpha=0.8)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Matrix Language Distribution by Group', fontsize=14, fontweight='bold')
ax.legend(title='Matrix Language', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'matrix_language_by_group.png', dpi=300, bbox_inches='tight')
print("✓ Saved matrix_language_by_group.png")
plt.close()

# ============================================================================
# 10. Summary Statistics Table
# ============================================================================
summary_stats = pd.DataFrame({
    'Total Sentences': [len(df)],
    'Unique Participants': [df['participant_id'].nunique()],
    'Code-Switching Sentences': [df['is_code_switching'].sum()],
    'Monolingual Sentences': [(~df['is_code_switching']).sum()],
    'Mean Duration (s)': [df['duration'].mean()],
    'Mean Sentence Length': [df['sentence_length'].mean()],
    'English Matrix': [(df['matrix_language'] == 'English').sum()],
    'Cantonese Matrix': [(df['matrix_language'] == 'Cantonese').sum()]
})

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
for col in summary_stats.columns:
    print(f"{col:.<40} {summary_stats[col].values[0]:>20.2f}" if isinstance(summary_stats[col].values[0], float) 
          else f"{col:.<40} {summary_stats[col].values[0]:>20}")
print("="*70)

# Save summary to CSV
summary_stats.T.to_csv(OUTPUT_DIR / 'summary_statistics.csv', header=['Value'])
print("\n✓ Saved summary_statistics.csv")

print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
