"""
Demo script to show matching algorithm for a specific code-switched sentence.

This script illustrates:
1. Matching failures (similarity < 0.4)
2. Successful matches (similarity >= 0.4)
3. Same-speaker vs different-speaker matches

Modify the INPUT VARIABLES section to test different sentences.
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

# Example: A code-switched sentence
CS_SENTENCE = "即係 可以 買 即係 唔係 去 supermarket 買嘢 噉樣"
CS_SENTENCE_POS = "CCONJ AUX VERB CCONJ VERB VERB PROPN VERB PRON"
CS_TRANSLATION = "即係 同 啲 朋友 就 好 少 會 成個"
CS_TRANSLATED_POS = "CCONJ AUX VERB CCONJ VERB VERB PROPN VERB PRON"
SWITCH_INDEX = 3  # Index where switch occurs (0-based)
WINDOW_SIZE = 1  # Window size for matching
PARTICIPANT_ID = "ACH2023"  # Speaker ID
GROUP = "Homeland"  # Speaker group
START_TIME = 1605785  # Timestamp (for same-speaker matching)

SIMILARITY_THRESHOLD = 0.4  # Levenshtein similarity threshold

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Run matching demo for the specified code-switched sentence."""
    
    print("=" * 80)
    print("MATCHING ALGORITHM DEMONSTRATION")
    print("=" * 80)
    
    # Load configuration
    config = Config()
    preprocessing_dir = Path(config.get_preprocessing_results_dir())
    
    # Load monolingual sentences
    monolingual_csv = preprocessing_dir / config.get('output.csv_cantonese_mono_without_fillers')
    if not monolingual_csv.exists():
        print(f"ERROR: Monolingual CSV not found: {monolingual_csv}")
        print("Run preprocessing first: python scripts/preprocess/preprocess.py")
        return
    
    print(f"\nLoading monolingual sentences from: {monolingual_csv.name}")
    monolingual_df = pd.read_csv(monolingual_csv)
    monolingual_sentences = monolingual_df.to_dict('records')
    print(f"Loaded {len(monolingual_sentences)} monolingual sentences")
    
    # Build POS cache
    print("\nBuilding POS cache...")
    mono_pos_cache = build_monolingual_pos_cache(monolingual_sentences)
    
    # Display input sentence
    print("\n" + "=" * 80)
    print("INPUT CODE-SWITCHED SENTENCE")
    print("=" * 80)
    print(f"Original:       {CS_SENTENCE}")
    print(f"Translation:    {CS_TRANSLATION}")
    print(f"POS:            {CS_TRANSLATED_POS}")
    print(f"Switch Index:   {SWITCH_INDEX}")
    print(f"Window Size:    ±{WINDOW_SIZE}")
    print(f"Speaker:        {PARTICIPANT_ID} ({GROUP})")
    print(f"Threshold:      {SIMILARITY_THRESHOLD}")
    
    # Extract POS window
    pos_sequence = CS_TRANSLATED_POS.split()
    window_start = max(0, SWITCH_INDEX - WINDOW_SIZE)
    window_end = min(len(pos_sequence), SWITCH_INDEX + WINDOW_SIZE + 1)
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
        threshold=SIMILARITY_THRESHOLD
    )
    
    print(f"\nTotal sentences evaluated:     {len(monolingual_sentences)}")
    print(f"Sentences passing threshold:   {len(candidates)} ({len(candidates)/len(monolingual_sentences)*100:.1f}%)")
    print(f"Sentences below threshold:     {len(monolingual_sentences) - len(candidates)}")
    
    # Find matching failures (similarity < threshold)
    print("\n" + "-" * 80)
    print(f"MATCHING FAILURES (Similarity < {SIMILARITY_THRESHOLD})")
    print("-" * 80)
    
    failures = []
    for idx, mono_sent in enumerate(monolingual_sentences):
        mono_pos_seq = mono_pos_cache.get(idx, [])
        if not mono_pos_seq:
            continue
        
        similarity = levenshtein_similarity(pos_sequence, mono_pos_seq)
        
        if similarity < SIMILARITY_THRESHOLD:
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
    
    # Create a mock code-switched sentence dict for find_window_matches
    cs_sent_dict = {
        'code_switch_original': CS_SENTENCE,
        'cantonese_translation': CS_TRANSLATION,
        'translated_pos': CS_TRANSLATED_POS,
        'switch_index': SWITCH_INDEX,
        'participant_id': PARTICIPANT_ID,
        'group': GROUP,
        'start_time': START_TIME,
        'pattern': 'C-E'  # Mock pattern
    }
    
    # Find all matches
    matches, stage1_count, stage2_count, is_cutoff, _ = find_window_matches(
        cs_sent_dict,
        monolingual_sentences,
        window_size=WINDOW_SIZE,
        similarity_threshold=SIMILARITY_THRESHOLD,
        mono_pos_cache=mono_pos_cache
    )
    
    print(f"\nStage 1 passed: {stage1_count} sentences")
    print(f"Stage 2 passed: {stage2_count} exact window matches")
    print(f"Window cutoff:  {'Yes' if is_cutoff else 'No'}")
    
    if not matches:
        print("\n⚠️  No exact window matches found!")
        return
    
    # Separate matches by speaker and group
    same_speaker_matches = [m for m in matches 
                           if m['match_sentence'].get('participant_id') == PARTICIPANT_ID]
    
    diff_speaker_diff_group_matches = [m for m in matches 
                                       if m['match_sentence'].get('participant_id') != PARTICIPANT_ID
                                       and m['match_sentence'].get('group') != GROUP]
    
    print(f"\nSame speaker matches:           {len(same_speaker_matches)}")
    print(f"Different speaker & group:      {len(diff_speaker_diff_group_matches)}")
    
    # Display successful matches
    print("\n" + "-" * 80)
    print(f"SUCCESSFUL MATCHES (Similarity >= {SIMILARITY_THRESHOLD})")
    print("-" * 80)
    print(f"\nCS Sentence Start Time: {START_TIME:.1f}ms")
    
    # Show 2 same-speaker matches
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
            
            # Extract CS window words using the same indices as the POS window
            cs_words = CS_TRANSLATION.split()
            cs_window_start = max(0, SWITCH_INDEX - WINDOW_SIZE)
            cs_window_end = min(len(cs_words), SWITCH_INDEX + WINDOW_SIZE + 1)
            cs_window_words = cs_words[cs_window_start:cs_window_end]
            
            # Extract matched monolingual window words
            mono_window_start = match['matched_window_start']
            # matched_pos is a list of POS tags
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
    
    # Show 2 different-speaker, different-group matches
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
            
            # Extract CS window words using the same indices as the POS window
            cs_words = CS_TRANSLATION.split()
            cs_window_start = max(0, SWITCH_INDEX - WINDOW_SIZE)
            cs_window_end = min(len(cs_words), SWITCH_INDEX + WINDOW_SIZE + 1)
            cs_window_words = cs_words[cs_window_start:cs_window_end]
            
            # Extract matched monolingual window words
            mono_window_start = match['matched_window_start']
            # matched_pos is a list of POS tags
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
    print(f"Stage 2 pass rate:              {stage2_count/stage1_count*100:.1f}% (of Stage 1)" if stage1_count > 0 else "Stage 2 pass rate:              N/A (no Stage 1 passes)")
    print(f"Overall match rate:             {stage2_count/len(monolingual_sentences)*100:.1f}%")
    print(f"Matching failures shown:        3")
    
    same_speaker_shown = min(2, len(same_speaker_matches))
    diff_speaker_shown = min(2, len(diff_speaker_diff_group_matches))
    print(f"Same speaker matches shown:     {same_speaker_shown}")
    print(f"Different speaker/group shown:  {diff_speaker_shown}")
    print(f"Total successful matches shown: {same_speaker_shown + diff_speaker_shown}")
    print("=" * 80)


if __name__ == '__main__':
    main()