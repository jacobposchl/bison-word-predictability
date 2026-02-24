import pandas as pd

# Path to your CSV file
csv_path = r"results/surprisal/masked/window_1/surprisal_results.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Find rows where 'word' appears in 'context'
matches = df[df.apply(lambda row: pd.notna(row['context']) and pd.notna(row['word']) and str(row['word']) in str(row['context']), axis=1)]

# Prepare output DataFrame with required columns
output_df = pd.DataFrame({
    'switch_word': matches['word'],
    'cs_sentence': matches['sentence'],
    'context': matches['context'],
    'is_switch': matches['is_switch'] if 'is_switch' in matches.columns else [1]*len(matches)
})

# Add 'full_dialogue' column: context + cs_sentence
output_df['full_dialogue'] = output_df['context'].astype(str) + ' ||| ' + output_df['cs_sentence'].astype(str)

# Save to CSV
output_df.to_csv("results/surprisal/masked/window_1/surprisal_results_word_in_context.csv", index=False)

print(f"Number of is_switch = True: {output_df['is_switch'].sum()}, Number of is_switch = False: {len(output_df) - output_df['is_switch'].sum()}")