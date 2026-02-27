"""
Demo script to show matching algorithm for a specific code-switched sentence.

This script illustrates:
1. Matching failures (similarity < 0.4)
2. Successful matches (similarity >= 0.4)
3. Same-speaker vs different-speaker matches

Modify the INPUT VARIABLES section to choose which CS dataset row(s) to demo.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.analysis.matching_algorithm import (
    build_monolingual_pos_cache,
    levenshtein_similarity,
    filter_by_full_sentence_similarity,
    find_window_matches
)

# ============================================================================
# INPUT VARIABLES - MODIFY THESE TO TEST DIFFERENT SENTENCES
# ============================================================================

# Row index/indices into the code-switched sentences dataset (0-based).
# Set to a single int to run one sentence, or a list of ints for multiple.
CS_ROW_INDICES = 252, 294, 360, 378

WINDOW_SIZE = 1           # Window size for matching
SIMILARITY_THRESHOLD = 0.4  # Levenshtein similarity threshold

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def run_demo_for_row(row, monolingual_sentences, mono_pos_cache, window_size, similarity_threshold):
    """Run the matching demo for a single CS dataset row (dict)."""

    CS_SENTENCE   = row['code_switch_original']
    CS_TRANSLATION = row['cantonese_translation']
    CS_TRANSLATED_POS = row['translated_pos']
    SWITCH_INDEX  = int(row['switch_index'])
    PARTICIPANT_ID = row['participant_id']
    GROUP         = row['group']
    START_TIME    = float(row['start_time'])

    # Display input sentence
    print("\n" + "=" * 80)
    print("INPUT CODE-SWITCHED SENTENCE")
    print("=" * 80)
    print(f"Original:       {CS_SENTENCE}")
    print(f"Translation:    {CS_TRANSLATION}")
    print(f"POS:            {CS_TRANSLATED_POS}")
    print(f"Switch Index:   {SWITCH_INDEX}")
    print(f"Window Size:    ±{window_size}")
    print(f"Speaker:        {PARTICIPANT_ID} ({GROUP})")
    print(f"Threshold:      {similarity_threshold}")

    # Extract POS window
    pos_sequence = CS_TRANSLATED_POS.split()
    window_start = max(0, SWITCH_INDEX - window_size)
    window_end = min(len(pos_sequence), SWITCH_INDEX + window_size + 1)
    pos_window = pos_sequence[window_start:window_end]

    print(f"\nPOS Window:     {' '.join(pos_window)}")
    print(f"Window range:   [{window_start}:{window_end}]")

    # STAGE 1: Filter by full sentence similarity
    print("\n" + "=" * 80)
    print("STAGE 1: FULL SENTENCE SIMILARITY FILTERING")
    print("=" * 80)

    candidates, all_similarity_scores = filter_by_full_sentence_similarity(
        cs_pos_sequence=pos_sequence,
        monolingual_sentences=monolingual_sentences,
        mono_pos_cache=mono_pos_cache,
        threshold=similarity_threshold
    )

    print(f"\nTotal sentences evaluated:     {len(monolingual_sentences)}")
    print(f"Sentences passing threshold:   {len(candidates)} ({len(candidates)/len(monolingual_sentences)*100:.1f}%)")
    print(f"Sentences below threshold:     {len(monolingual_sentences) - len(candidates)}")

    # Find matching failures (similarity < threshold)
    print("\n" + "-" * 80)
    print(f"MATCHING FAILURES (Similarity < {similarity_threshold})")
    print("-" * 80)

    failures = []
    for idx, mono_sent in enumerate(monolingual_sentences):
        mono_pos_seq = mono_pos_cache.get(idx, [])
        if not mono_pos_seq:
            continue

        similarity = levenshtein_similarity(pos_sequence, mono_pos_seq)

        if similarity < similarity_threshold:
            failures.append({
                'sentence': mono_sent.get('reconstructed_sentence', ''),
                'pos': ' '.join(mono_pos_seq),
                'similarity': similarity,
                'speaker': mono_sent.get('participant_id', ''),
                'group': mono_sent.get('group', '')
            })

    # Show 3 examples of failures
    failures_sorted = sorted(failures, key=lambda x: x['similarity'], reverse=True)
    print(f"\nFound {len(failures)} sentences below threshold. Showing top 3 closest failures:")

    for i, failure in enumerate(failures_sorted[:3], 1):
        print(f"\n{i}. Similarity: {failure['similarity']:.3f}")
        print(f"   Speaker: {failure['speaker']} ({failure['group']})")
        print(f"   Sentence: {failure['sentence']}")
        print(f"   POS: {failure['pos']}")

    # STAGE 2: Exact window matching
    print("\n" + "=" * 80)
    print("STAGE 2: EXACT WINDOW MATCHING")
    print("=" * 80)

    cs_sent_dict = {
        'code_switch_original': CS_SENTENCE,
        'cantonese_translation': CS_TRANSLATION,
        'translated_pos': CS_TRANSLATED_POS,
        'switch_index': SWITCH_INDEX,
        'participant_id': PARTICIPANT_ID,
        'group': GROUP,
        'start_time': START_TIME,
        'pattern': row.get('pattern', 'C-E')
    }

    matches, stage1_count, stage2_count, is_cutoff, _ = find_window_matches(
        cs_sent_dict,
        monolingual_sentences,
        window_size=window_size,
        similarity_threshold=similarity_threshold,
        mono_pos_cache=mono_pos_cache
    )

    print(f"\nStage 1 passed: {stage1_count} sentences")
    print(f"Stage 2 passed: {stage2_count} exact window matches")
    print(f"Window cutoff:  {'Yes' if is_cutoff else 'No'}")

    if not matches:
        print("\n⚠️  No exact window matches found!")
        return

    same_speaker_matches = [m for m in matches
                            if m['match_sentence'].get('participant_id') == PARTICIPANT_ID]
    diff_speaker_diff_group_matches = [m for m in matches
                                       if m['match_sentence'].get('participant_id') != PARTICIPANT_ID
                                       and m['match_sentence'].get('group') != GROUP]

    print(f"\nSame speaker matches:           {len(same_speaker_matches)}")
    print(f"Different speaker & group:      {len(diff_speaker_diff_group_matches)}")

    # Display successful matches
    print("\n" + "-" * 80)
    print(f"SUCCESSFUL MATCHES (Similarity >= {similarity_threshold})")
    print("-" * 80)
    print(f"\nCS Sentence Start Time: {START_TIME:.1f}ms")

    print(f"\n{'SAME SPEAKER MATCHES':^80}")
    print("-" * 80)

    if not same_speaker_matches:
        print("\n⚠️  No same-speaker matches found!")
    else:
        for i, match in enumerate(same_speaker_matches[:2], 1):
            mono_sent = match['match_sentence']
            mono_start = mono_sent.get('start_time', 0)
            mono_pos_full = mono_sent.get('pos', '').split()
            mono_words = mono_sent.get('reconstructed_sentence', '').split()

            cs_words = CS_TRANSLATION.split()
            cs_window_start = max(0, SWITCH_INDEX - window_size)
            cs_window_end = min(len(cs_words), SWITCH_INDEX + window_size + 1)
            cs_window_words = cs_words[cs_window_start:cs_window_end]

            mono_window_start = match['matched_window_start']
            matched_pos_list = match['matched_pos'] if isinstance(match['matched_pos'], list) else match['matched_pos'].split()
            mono_window_end = mono_window_start + len(matched_pos_list)
            matched_words = mono_words[mono_window_start:mono_window_end]

            print(f"\nMatch {i}:")
            print(f"  Full Similarity:    {match['similarity']:.3f}")
            print(f"  Speaker:            {mono_sent.get('participant_id')} ({mono_sent.get('group')})")
            print(f"  CS Start Time:      {START_TIME:.1f}ms")
            print(f"  Match Start Time:   {mono_start:.1f}ms")
            print(f"  Time Distance:      {abs(START_TIME - mono_start):.1f}ms")
            print(f"  Matched Sentence:   {mono_sent.get('reconstructed_sentence', '')}")
            print(f"  Full Mono POS:      {' '.join(mono_pos_full)}")
            print(f"  CS POS Window:      {' '.join(pos_window)}")
            print(f"  CS Window Words:    {' '.join(cs_window_words)}")
            print(f"  Matched POS Window: {' '.join(matched_pos_list)} (index {mono_window_start}:{mono_window_end})")
            print(f"  Matched Window Words: {' '.join(matched_words)} [DEBUG: len(mono_words)={len(mono_words)}, indices={mono_window_start}:{mono_window_end}]")
            print(f"  CS Switch POS:      {match['cs_switch_pos']}")
            print(f"  Mono Switch POS:    {match['mono_switch_pos']}")

    print(f"\n{'DIFFERENT SPEAKER & GROUP MATCHES':^80}")
    print("-" * 80)

    if not diff_speaker_diff_group_matches:
        print("\n⚠️  No different-speaker, different-group matches found!")
    else:
        for i, match in enumerate(diff_speaker_diff_group_matches[:2], 1):
            mono_sent = match['match_sentence']
            mono_start = mono_sent.get('start_time', 0)
            mono_pos_full = mono_sent.get('pos', '').split()
            mono_words = mono_sent.get('reconstructed_sentence', '').split()

            cs_words = CS_TRANSLATION.split()
            cs_window_start = max(0, SWITCH_INDEX - window_size)
            cs_window_end = min(len(cs_words), SWITCH_INDEX + window_size + 1)
            cs_window_words = cs_words[cs_window_start:cs_window_end]

            mono_window_start = match['matched_window_start']
            matched_pos_list = match['matched_pos'] if isinstance(match['matched_pos'], list) else match['matched_pos'].split()
            mono_window_end = mono_window_start + len(matched_pos_list)
            matched_words = mono_words[mono_window_start:mono_window_end]

            print(f"\nMatch {i}:")
            print(f"  Full Similarity:    {match['similarity']:.3f}")
            print(f"  Speaker:            {mono_sent.get('participant_id')} ({mono_sent.get('group')})")
            print(f"  CS Start Time:      {START_TIME:.1f}ms")
            print(f"  Match Start Time:   {mono_start:.1f}ms")
            print(f"  Matched Sentence:   {mono_sent.get('reconstructed_sentence', '')}")
            print(f"  Full Mono POS:      {' '.join(mono_pos_full)}")
            print(f"  CS POS Window:      {' '.join(pos_window)}")
            print(f"  CS Window Words:    {' '.join(cs_window_words)}")
            print(f"  Matched POS Window: {' '.join(matched_pos_list)} (index {mono_window_start}:{mono_window_end})")
            print(f"  Matched Window Words: {' '.join(matched_words)} [DEBUG: len(mono_words)={len(mono_words)}, indices={mono_window_start}:{mono_window_end}]")
            print(f"  CS Switch POS:      {match['cs_switch_pos']}")
            print(f"  Mono Switch POS:    {match['mono_switch_pos']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total monolingual sentences:    {len(monolingual_sentences)}")
    print(f"Stage 1 pass rate:              {stage1_count/len(monolingual_sentences)*100:.1f}%")
    if stage1_count > 0:
        print(f"Stage 2 pass rate:              {stage2_count/stage1_count*100:.1f}% (of Stage 1)")
    else:
        print("Stage 2 pass rate:              N/A (no Stage 1 passes)")
    print(f"Overall match rate:             {stage2_count/len(monolingual_sentences)*100:.1f}%")
    print(f"Matching failures shown:        3")

    same_speaker_shown = min(2, len(same_speaker_matches))
    diff_speaker_shown = min(2, len(diff_speaker_diff_group_matches))
    print(f"Same speaker matches shown:     {same_speaker_shown}")
    print(f"Different speaker/group shown:  {diff_speaker_shown}")
    print(f"Total successful matches shown: {same_speaker_shown + diff_speaker_shown}")
    print("=" * 80)


def main():
    """Run matching demo for the specified CS dataset row(s)."""

    print("=" * 80)
    print("MATCHING ALGORITHM DEMONSTRATION")
    print("=" * 80)

    # Load configuration
    config = Config()
    preprocessing_dir = Path(config.get_preprocessing_results_dir())

    # Load code-switched sentences
    cs_csv = preprocessing_dir / config.get('output.csv_cantonese_translated')
    if not cs_csv.exists():
        print(f"ERROR: CS sentences CSV not found: {cs_csv}")
        print("Run preprocessing first: python scripts/preprocess/preprocess.py")
        return

    print(f"\nLoading code-switched sentences from: {cs_csv.name}")
    cs_df = pd.read_csv(cs_csv)
    print(f"Dataset has {len(cs_df)} rows (indices 0 – {len(cs_df) - 1})")

    # Resolve which row indices to run
    indices = [CS_ROW_INDICES] if isinstance(CS_ROW_INDICES, int) else list(CS_ROW_INDICES)

    # Load monolingual sentences (shared across all rows)
    monolingual_csv = preprocessing_dir / config.get('output.csv_cantonese_mono_without_fillers')
    if not monolingual_csv.exists():
        print(f"ERROR: Monolingual CSV not found: {monolingual_csv}")
        print("Run preprocessing first: python scripts/preprocess/preprocess.py")
        return

    print(f"\nLoading monolingual sentences from: {monolingual_csv.name}")
    monolingual_df = pd.read_csv(monolingual_csv)
    monolingual_sentences = monolingual_df.to_dict('records')
    print(f"Loaded {len(monolingual_sentences)} monolingual sentences")

    # Build POS cache (shared across all rows)
    print("\nBuilding POS cache...")
    mono_pos_cache = build_monolingual_pos_cache(monolingual_sentences)

    # Run demo for each requested row
    for idx in indices:
        if idx < 0 or idx >= len(cs_df):
            print(f"\nERROR: Row index {idx} is out of range (0 – {len(cs_df) - 1}). Skipping.")
            continue
        print(f"\n{'#' * 80}")
        print(f"  CS DATASET ROW {idx}")
        print(f"{'#' * 80}")
        row = cs_df.iloc[idx].to_dict()
        run_demo_for_row(row, monolingual_sentences, mono_pos_cache, WINDOW_SIZE, SIMILARITY_THRESHOLD)


if __name__ == '__main__':
    main()