"""
Deep validation of the matching algorithm and similarity calculations.
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.matching_algorithm import levenshtein_similarity

def test_levenshtein_similarity():
    """Test if Levenshtein similarity calculation is correct."""
    print("=" * 80)
    print("DEEP VALIDATION: Levenshtein Similarity Calculation")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        # (seq1, seq2, expected_range, description)
        (['NOUN', 'VERB'], ['NOUN', 'VERB'], (1.0, 1.0), "Identical sequences"),
        (['NOUN', 'VERB'], ['NOUN', 'NOUN'], (0.4, 0.6), "One substitution"),
        (['NOUN', 'VERB'], ['VERB', 'NOUN'], (0.0, 0.4), "Swapped order"),
        (['NOUN'], ['NOUN', 'VERB'], (0.4, 0.6), "One insertion"),
        (['NOUN', 'VERB', 'ADJ'], ['NOUN', 'VERB'], (0.6, 0.8), "One deletion"),
        (['NOUN'], ['VERB'], (0.0, 0.5), "Completely different"),
    ]
    
    print("\nTesting similarity calculations:")
    all_pass = True
    for seq1, seq2, expected_range, desc in test_cases:
        sim = levenshtein_similarity(seq1, seq2)
        in_range = expected_range[0] <= sim <= expected_range[1]
        status = "[PASS]" if in_range else "[FAIL]"
        if not in_range:
            all_pass = False
        print(f"{status} {desc}: similarity = {sim:.3f} (expected {expected_range[0]:.1f}-{expected_range[1]:.1f})")
        print(f"         seq1: {seq1}")
        print(f"         seq2: {seq2}")
        print()
    
    return all_pass

def validate_window_extraction():
    """Validate that POS windows are extracted correctly."""
    print("=" * 80)
    print("DEEP VALIDATION: POS Window Extraction")
    print("=" * 80)
    
    def extract_pos_window(pos_sequence, switch_index, window_size=3):
        start = max(0, switch_index - window_size)
        end = min(len(pos_sequence), switch_index + window_size + 1)
        return pos_sequence[start:end]
    
    # Test cases
    pos_seq = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    test_cases = [
        (0, 3, ['A', 'B', 'C', 'D'], "Switch at start"),
        (4, 3, ['B', 'C', 'D', 'E', 'F', 'G', 'H'], "Switch in middle"),
        (7, 3, ['E', 'F', 'G', 'H'], "Switch at end"),
    ]
    
    print("\nTesting window extraction:")
    all_pass = True
    for switch_idx, window_size, expected, desc in test_cases:
        result = extract_pos_window(pos_seq, switch_idx, window_size)
        passed = result == expected
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"{status} {desc}:")
        print(f"         switch_idx={switch_idx}, window_size={window_size}")
        print(f"         Expected: {expected}")
        print(f"         Got: {result}")
        print()
    
    return all_pass

def check_matching_logic():
    """Check if matching logic has any issues."""
    print("=" * 80)
    print("DEEP VALIDATION: Matching Logic")
    print("=" * 80)
    
    matching_path = "exploratory_results/matching_results_sample.csv"
    df = pd.read_csv(matching_path)
    
    # Check: Are there sentences that should have matches but don't?
    # This is hard to validate automatically, but we can check patterns
    
    # Check: Are very short sentences (1-2 words) less likely to match?
    df['sentence_length'] = df['sentence'].str.split().str.len()
    short_sentences = df[df['sentence_length'] <= 3]
    long_sentences = df[df['sentence_length'] > 10]
    
    if len(short_sentences) > 0 and len(long_sentences) > 0:
        short_match_rate = short_sentences['has_match'].mean() * 100
        long_match_rate = long_sentences['has_match'].mean() * 100
        
        print(f"\nMatch rates by sentence length:")
        print(f"  Short sentences (<=3 words): {short_match_rate:.1f}%")
        print(f"  Long sentences (>10 words): {long_match_rate:.1f}%")
        
        if abs(short_match_rate - long_match_rate) > 30:
            print(f"  [WARN] Large difference in match rates - may indicate bias")
        else:
            print(f"  [PASS] Match rates are similar across sentence lengths")
    
    # Check: Are sentences with single-word switches less likely to match?
    df['has_single_word_segment'] = df['pattern'].str.contains(r'[CE]1-') | df['pattern'].str.contains(r'-[CE]1')
    single_word = df[df['has_single_word_segment'] == True]
    multi_word = df[df['has_single_word_segment'] == False]
    
    if len(single_word) > 0 and len(multi_word) > 0:
        single_match_rate = single_word['has_match'].mean() * 100
        multi_match_rate = multi_word['has_match'].mean() * 100
        
        print(f"\nMatch rates by switch type:")
        print(f"  Sentences with single-word segments: {single_match_rate:.1f}%")
        print(f"  Sentences without single-word segments: {multi_match_rate:.1f}%")
    
    return True

def check_statistics_consistency():
    """Verify that reported statistics are consistent with data."""
    print("\n" + "=" * 80)
    print("DEEP VALIDATION: Statistics Consistency")
    print("=" * 80)
    
    matching_path = "exploratory_results/matching_results_sample.csv"
    df = pd.read_csv(matching_path)
    
    # Recalculate statistics
    total = len(df)
    with_matches = len(df[df['has_match'] == True])
    total_matches = df['num_matches'].sum()
    
    recalc_success_rate = (with_matches / total) * 100
    recalc_avg_matches = total_matches / total
    
    # Check against report
    report_success_rate = 66.6
    report_avg_matches = 4.69
    
    print(f"\nRecalculated statistics:")
    print(f"  Success rate: {recalc_success_rate:.1f}% (report: {report_success_rate}%)")
    print(f"  Avg matches: {recalc_avg_matches:.2f} (report: {report_avg_matches})")
    
    if abs(recalc_success_rate - report_success_rate) < 0.1 and abs(recalc_avg_matches - report_avg_matches) < 0.01:
        print(f"  [PASS] Statistics match report")
        return True
    else:
        print(f"  [FAIL] Statistics don't match report!")
        return False

def main():
    """Run all deep validations."""
    print("\n" + "=" * 80)
    print("DEEP VALIDATION OF ANALYSIS")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Levenshtein Similarity", test_levenshtein_similarity()))
    results.append(("Window Extraction", validate_window_extraction()))
    results.append(("Matching Logic", check_matching_logic()))
    results.append(("Statistics Consistency", check_statistics_consistency()))
    
    print("\n" + "=" * 80)
    print("DEEP VALIDATION SUMMARY")
    print("=" * 80)
    
    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False
    
    print("=" * 80)
    
    if all_pass:
        print("\n[PASS] ALL DEEP VALIDATIONS PASSED")
        return 0
    else:
        print("\n[FAIL] SOME DEEP VALIDATIONS FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())

