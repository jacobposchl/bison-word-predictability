"""
Comprehensive visualization script for cantonese_monolingual_WITHOUT_fillers.csv
Focuses on pure Cantonese utterances without code-switching.
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
DATA_PATH = Path("results/preprocessing/cantonese_monolingual_WITHOUT_fillers.csv")
OUTPUT_DIR = Path("figures/preprocessing/cantonese_monolingual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate derived metrics
df['duration'] = (df['end_time'] - df['start_time']) / 1000  # Convert to seconds
df['sentence_length'] = df['reconstructed_sentence'].str.split().str.len()
df['pattern_length'] = df['pattern'].str.extract(r'C(\d+)')[0].astype(int)

print(f"Loaded {len(df)} Cantonese monolingual sentences")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Groups: {df['group'].unique()}")

# ============================================================================
# 1. Distribution of Sentence Durations
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['duration'], bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
ax.set_xlabel('Duration (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Cantonese Monolingual: Sentence Duration Distribution', 
             fontsize=14, fontweight='bold')
ax.axvline(df['duration'].median(), color='darkred', linestyle='--', linewidth=2,
           label=f'Median: {df["duration"].median():.2f}s')
ax.axvline(df['duration'].mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Mean: {df["duration"].mean():.2f}s')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved duration_distribution.png")
plt.close()

# ============================================================================
# 2. Sentence Length Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['sentence_length'], bins=range(0, min(df['sentence_length'].max() + 2, 51)),
        edgecolor='black', alpha=0.7, color='#3498db')
ax.set_xlabel('Number of Words', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Cantonese Monolingual: Sentence Length Distribution', 
             fontsize=14, fontweight='bold')
ax.axvline(df['sentence_length'].median(), color='navy', linestyle='--', linewidth=2,
           label=f'Median: {df["sentence_length"].median():.0f} words')
ax.axvline(df['sentence_length'].mean(), color='cyan', linestyle='--', linewidth=2,
           label=f'Mean: {df["sentence_length"].mean():.1f} words')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sentence_length_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved sentence_length_distribution.png")
plt.close()

# ============================================================================
# 3. Participant Contribution Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
participant_counts = df['participant_id'].value_counts().sort_values(ascending=False)

colors = plt.cm.viridis(np.linspace(0, 1, len(participant_counts)))
ax.bar(range(len(participant_counts)), participant_counts.values,
       edgecolor='black', alpha=0.8, color=colors)
ax.set_xlabel('Participant (sorted by contribution)', fontsize=12)
ax.set_ylabel('Number of Cantonese Sentences', fontsize=12)
ax.set_title('Cantonese Monolingual: Participant Contributions', 
             fontsize=14, fontweight='bold')
ax.axhline(participant_counts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {participant_counts.mean():.1f}')
ax.axhline(participant_counts.median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {participant_counts.median():.1f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'participant_contributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved participant_contributions.png")
plt.close()

# ============================================================================
# 4. Group Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
group_counts = df['group'].value_counts()
colors_group = ['#2ecc71', '#e67e22', '#9b59b6']
bars = ax.bar(range(len(group_counts)), group_counts.values,
              color=colors_group[:len(group_counts)], edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(group_counts)))
ax.set_xticklabels(group_counts.index, fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Cantonese Monolingual: Distribution by Group', 
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
# 5. Pattern Length Distribution
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['pattern_length'], bins=range(0, df['pattern_length'].max() + 2),
        edgecolor='black', alpha=0.7, color='#16a085')
ax.set_xlabel('Pattern Length (Number of Cantonese Words)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Cantonese Monolingual: Pattern Length Distribution', 
             fontsize=14, fontweight='bold')
ax.axvline(df['pattern_length'].median(), color='darkgreen', linestyle='--', linewidth=2,
           label=f'Median: {df["pattern_length"].median():.0f}')
ax.axvline(df['pattern_length'].mean(), color='lime', linestyle='--', linewidth=2,
           label=f'Mean: {df["pattern_length"].mean():.1f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pattern_length_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved pattern_length_distribution.png")
plt.close()

# ============================================================================
# 6. Duration vs Sentence Length Relationship
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
scatter = ax.scatter(df['sentence_length'], df['duration'], 
                     alpha=0.6, s=30, c=df['pattern_length'], cmap='plasma')
ax.set_xlabel('Sentence Length (words)', fontsize=12)
ax.set_ylabel('Duration (seconds)', fontsize=12)
ax.set_title('Cantonese Monolingual: Sentence Length vs Duration', 
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Pattern Length', fontsize=12)

# Add regression line
z = np.polyfit(df['sentence_length'], df['duration'], 1)
p = np.poly1d(z)
sorted_lengths = np.sort(df['sentence_length'].unique())
ax.plot(sorted_lengths, p(sorted_lengths), "r--", alpha=0.8, linewidth=2,
        label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')

# Calculate correlation
corr = df[['sentence_length', 'duration']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'length_vs_duration_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved length_vs_duration_scatter.png")
plt.close()

# ============================================================================
# 7. Speaking Rate Distribution
# ============================================================================
df['speaking_rate'] = df['sentence_length'] / df['duration']  # words per second

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.hist(df['speaking_rate'], bins=50, edgecolor='black', alpha=0.7, color='#c0392b')
ax.set_xlabel('Speaking Rate (words/second)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Cantonese Monolingual: Speaking Rate Distribution', 
             fontsize=14, fontweight='bold')
ax.axvline(df['speaking_rate'].median(), color='darkred', linestyle='--', linewidth=2,
           label=f'Median: {df["speaking_rate"].median():.2f} w/s')
ax.axvline(df['speaking_rate'].mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Mean: {df["speaking_rate"].mean():.2f} w/s')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'speaking_rate_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved speaking_rate_distribution.png")
plt.close()

# ============================================================================
# 8. Box Plot: Duration by Group
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
df.boxplot(column='duration', by='group', ax=ax, patch_artist=True)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Duration (seconds)', fontsize=12)
ax.set_title('Cantonese Monolingual: Duration Distribution by Group', 
             fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'duration_by_group_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved duration_by_group_boxplot.png")
plt.close()

# ============================================================================
# 9. Box Plot: Sentence Length by Group
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
df.boxplot(column='sentence_length', by='group', ax=ax, patch_artist=True)
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Sentence Length (words)', fontsize=12)
ax.set_title('Cantonese Monolingual: Sentence Length by Group', 
             fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sentence_length_by_group_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved sentence_length_by_group_boxplot.png")
plt.close()

# ============================================================================
# 10. Summary Statistics
# ============================================================================
summary_stats = pd.DataFrame({
    'Total Sentences': [len(df)],
    'Unique Participants': [df['participant_id'].nunique()],
    'Mean Duration (s)': [df['duration'].mean()],
    'Median Duration (s)': [df['duration'].median()],
    'Std Duration (s)': [df['duration'].std()],
    'Mean Sentence Length': [df['sentence_length'].mean()],
    'Median Sentence Length': [df['sentence_length'].median()],
    'Mean Pattern Length': [df['pattern_length'].mean()],
    'Mean Speaking Rate (w/s)': [df['speaking_rate'].mean()],
    'Median Speaking Rate (w/s)': [df['speaking_rate'].median()]
})

print("\n" + "="*70)
print("CANTONESE MONOLINGUAL SUMMARY STATISTICS")
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
