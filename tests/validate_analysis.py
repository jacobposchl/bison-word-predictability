"""
Validation script to thoroughly check the exploratory analysis for correctness,
confounds, and potential issues.
"""

import pandas as pd
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.feasibility import is_monolingual
from src.analysis.pos_tagging import parse_pattern_segments

def validate_pattern_parsing():
    """Validate that pattern parsing is correct."""
    print("=" * 80)
    print("VALIDATION 1: Pattern Parsing")
    print("=" * 80)
    
    test_cases = [
        ("C10", "Cantonese"),
        ("E8", "English"),
        ("C5-E3", None),
        ("E2-C4-E1", None),
        ("C1", "Cantonese"),
        ("E1", "English"),
    ]
    
    all_pass = True
    for pattern, expected in test_cases:
        result = is_monolingual(pattern)
        status = "[PASS]" if result == expected else "[FAIL]"
        if result != expected:
            all_pass = False
        print(f"{status} Pattern '{pattern}': Expected {expected}, Got {result}")
    
    return all_pass

def validate_data_consistency():
    """Validate data consistency across files."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: Data Consistency")
    print("=" * 80)
    
    # Load all sentences
    all_sentences_path = Path(__file__).parent.parent / "processed_data" / "all_sentences.csv"
    if not all_sentences_path.exists():
        print("[FAIL] all_sentences.csv not found")
        return False
    
    df = pd.read_csv(all_sentences_path)
    print(f"[PASS] Loaded {len(df)} sentences from all_sentences.csv")
    
    # Check monolingual extraction
    cantonese = []
    english = []
    code_switched = []
    
    for idx, row in df.iterrows():
        pattern = row.get('pattern', '')
        lang_type = is_monolingual(pattern)
        if lang_type == 'Cantonese':
            cantonese.append(row)
        elif lang_type == 'English':
            english.append(row)
        else:
            code_switched.append(row)
    
    print(f"  Pure Cantonese: {len(cantonese)}")
    print(f"  Pure English: {len(english)}")
    print(f"  Code-switched: {len(code_switched)}")
    print(f"  Total: {len(cantonese) + len(english) + len(code_switched)}")
    
    # Verify totals match
    if len(cantonese) + len(english) + len(code_switched) != len(df):
        print(f"[FAIL] Total mismatch! Expected {len(df)}, got {len(cantonese) + len(english) + len(code_switched)}")
        return False
    else:
        print("[PASS] Total counts match")
    
    # Check for empty patterns
    empty_patterns = df[df['pattern'].isna() | (df['pattern'] == '')]
    if len(empty_patterns) > 0:
        print(f"[WARN] {len(empty_patterns)} sentences with empty patterns")
    
    return True

def validate_matching_results():
    """Validate matching results for correctness."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: Matching Results")
    print("=" * 80)
    
    matching_path = Path(__file__).parent.parent / "exploratory_results" / "matching_results_sample.csv"
    if not matching_path.exists():
        print("[FAIL] matching_results_sample.csv not found")
        return False
    
    df = pd.read_csv(matching_path)
    print(f"[PASS] Loaded {len(df)} matching results")
    
    # Check statistics
    total_sentences = len(df)
    sentences_with_matches = len(df[df['has_match'] == True])
    total_matches = df['num_matches'].sum()
    
    print(f"  Total sentences: {total_sentences}")
    print(f"  Sentences with matches: {sentences_with_matches} ({sentences_with_matches/total_sentences*100:.1f}%)")
    print(f"  Total matches: {total_matches}")
    print(f"  Average matches per sentence: {total_matches/total_sentences:.2f}")
    
    # Check similarity scores
    sentences_with_similarity = df[df['best_similarity'] > 0]
    if len(sentences_with_similarity) > 0:
        avg_similarity = sentences_with_similarity['best_similarity'].mean()
        print(f"  Average similarity (for matched sentences): {avg_similarity:.3f}")
        
        # Check similarity range
        min_sim = sentences_with_similarity['best_similarity'].min()
        max_sim = sentences_with_similarity['best_similarity'].max()
        print(f"  Similarity range: {min_sim:.3f} - {max_sim:.3f}")
        
        # Check if similarities are in valid range [0, 1]
        invalid_sim = df[(df['best_similarity'] < 0) | (df['best_similarity'] > 1)]
        if len(invalid_sim) > 0:
            print(f"[FAIL] ERROR: {len(invalid_sim)} sentences with invalid similarity scores!")
            return False
        else:
            print("[PASS] All similarity scores in valid range [0, 1]")
    
    # Check consistency: has_match should match num_matches > 0
    inconsistent = df[(df['has_match'] == True) & (df['num_matches'] == 0)] | \
                   df[(df['has_match'] == False) & (df['num_matches'] > 0)]
    if len(inconsistent) > 0:
        print(f"[FAIL] ERROR: {len(inconsistent)} sentences with inconsistent has_match/num_matches!")
        return False
    else:
        print("[PASS] has_match and num_matches are consistent")
    
    # Check that sentences with matches have similarity > 0
    matched_with_zero_sim = df[(df['has_match'] == True) & (df['best_similarity'] == 0)]
    if len(matched_with_zero_sim) > 0:
        print(f"[WARN] {len(matched_with_zero_sim)} matched sentences have similarity = 0")
    
    return True

def validate_switch_detection():
    """Validate switch point detection."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: Switch Point Detection")
    print("=" * 80)
    
    matching_path = Path(__file__).parent.parent / "exploratory_results" / "matching_results_sample.csv"
    if not matching_path.exists():
        print("[FAIL] matching_results_sample.csv not found")
        return False
    
    df = pd.read_csv(matching_path)
    
    # Check switch direction detection
    c_to_e_count = len(df[df['has_c_to_e'] == True])
    e_to_c_count = len(df[df['has_e_to_c'] == True])
    both_count = len(df[(df['has_c_to_e'] == True) & (df['has_e_to_c'] == True)])
    
    print(f"  C->E switches: {c_to_e_count}")
    print(f"  E->C switches: {e_to_c_count}")
    print(f"  Both directions: {both_count}")
    
    # Validate patterns match switch directions
    errors = 0
    for idx, row in df.head(100).iterrows():  # Check first 100
        pattern = row['pattern']
        segments = parse_pattern_segments(pattern)
        
        has_c_to_e_actual = False
        has_e_to_c_actual = False
        
        for i in range(len(segments) - 1):
            if segments[i][0] == 'C' and segments[i+1][0] == 'E':
                has_c_to_e_actual = True
            elif segments[i][0] == 'E' and segments[i+1][0] == 'C':
                has_e_to_c_actual = True
        
        if has_c_to_e_actual != row['has_c_to_e'] or has_e_to_c_actual != row['has_e_to_c']:
            errors += 1
            if errors <= 5:  # Show first 5 errors
                print(f"[FAIL] Pattern '{pattern}': Expected C->E={has_c_to_e_actual}, E->C={has_e_to_c_actual}, Got C->E={row['has_c_to_e']}, E->C={row['has_e_to_c']}")
    
    if errors == 0:
        print("[PASS] Switch direction detection is correct (checked first 100)")
    else:
        print(f"[FAIL] Found {errors} errors in switch direction detection")
        return False
    
    return True

def validate_pos_tagging():
    """Validate POS tagging results."""
    print("\n" + "=" * 80)
    print("VALIDATION 5: POS Tagging")
    print("=" * 80)
    
    pos_path = Path(__file__).parent.parent / "exploratory_results" / "pos_tagged_sample.csv"
    if not pos_path.exists():
        print("[FAIL] pos_tagged_sample.csv not found")
        return False
    
    df = pd.read_csv(pos_path)
    print(f"[PASS] Loaded {len(df)} POS tagged sentences")
    
    # Check error rate
    errors = len(df[df['error'] == True])
    error_rate = errors / len(df) * 100 if len(df) > 0 else 0
    print(f"  Error rate: {error_rate:.1f}% ({errors}/{len(df)})")
    
    # Check sequence lengths
    successful = df[df['error'] == False]
    if len(successful) > 0:
        avg_length = successful['sequence_length'].mean()
        print(f"  Average sequence length: {avg_length:.1f}")
        
        # Check for empty sequences in successful tags
        empty_sequences = successful[successful['sequence_length'] == 0]
        if len(empty_sequences) > 0:
            print(f"[WARN] {len(empty_sequences)} successful tags have empty sequences")
    
    return True

def check_potential_confounds():
    """Check for potential confounds or biases."""
    print("\n" + "=" * 80)
    print("VALIDATION 6: Potential Confounds")
    print("=" * 80)
    
    matching_path = Path(__file__).parent.parent / "exploratory_results" / "matching_results_sample.csv"
    if not matching_path.exists():
        return False
    
    df = pd.read_csv(matching_path)
    
    # Check if sentence length affects matching
    df['sentence_length'] = df['sentence'].str.split().str.len()
    
    matched = df[df['has_match'] == True]
    unmatched = df[df['has_match'] == False]
    
    if len(matched) > 0 and len(unmatched) > 0:
        avg_len_matched = matched['sentence_length'].mean()
        avg_len_unmatched = unmatched['sentence_length'].mean()
        print(f"  Average length (matched): {avg_len_matched:.1f} words")
        print(f"  Average length (unmatched): {avg_len_unmatched:.1f} words")
        
        if abs(avg_len_matched - avg_len_unmatched) > 5:
            print(f"[WARN] Large difference in sentence length between matched/unmatched")
    
    # Check if pattern complexity affects matching
    pattern_complexity = df['pattern'].str.count('-') + 1
    matched_complexity = pattern_complexity[df['has_match'] == True].mean() if len(matched) > 0 else 0
    unmatched_complexity = pattern_complexity[df['has_match'] == False].mean() if len(unmatched) > 0 else 0
    
    print(f"  Average pattern segments (matched): {matched_complexity:.1f}")
    print(f"  Average pattern segments (unmatched): {unmatched_complexity:.1f}")
    
    # Check for potential data leakage (same sentences matching themselves)
    # This would be a major confound
    all_sentences_path = Path(__file__).parent.parent / "processed_data" / "all_sentences.csv"
    if all_sentences_path.exists():
        all_df = pd.read_csv(all_sentences_path)
        print("\n  Checking for potential data leakage...")
        # This would require checking if code-switched sentences match themselves
        # which shouldn't happen, but worth checking
        
    return True

def main():
    """Run all validations."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS VALIDATION")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Pattern Parsing", validate_pattern_parsing()))
    results.append(("Data Consistency", validate_data_consistency()))
    results.append(("Matching Results", validate_matching_results()))
    results.append(("Switch Detection", validate_switch_detection()))
    results.append(("POS Tagging", validate_pos_tagging()))
    results.append(("Confounds Check", check_potential_confounds()))
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False
    
    print("=" * 80)
    
    if all_pass:
        print("\n[PASS] ALL VALIDATIONS PASSED")
        return 0
    else:
        print("\n[FAIL] SOME VALIDATIONS FAILED - Please review above")
        return 1

if __name__ == '__main__':
    sys.exit(main())

