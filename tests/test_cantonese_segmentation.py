"""
Comprehensive test for Cantonese segmentation and Levenshtein similarity.

This test verifies that:
1. Cantonese text is properly segmented using pycantonese
2. POS tagging works correctly with proper segmentation
3. Levenshtein similarity calculation is correct
4. The end-to-end pipeline handles Cantonese correctly
"""

import sys
from pathlib import Path
# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pos_tagging import pos_tag_cantonese, pos_tag_mixed_sentence, extract_pos_sequence
from src.analysis.matching_algorithm import levenshtein_similarity, _sequence_edit_distance

def explain_levenshtein_similarity():
    """
    Explain how Levenshtein similarity works in this codebase.
    
    Levenshtein Distance (Edit Distance):
    - Measures the minimum number of single-character edits (insertions, deletions, substitutions)
      needed to transform one string into another
    - Example: "cat" -> "bat" = 1 substitution, "cat" -> "cats" = 1 insertion
    
    In this codebase, we use SEQUENCE-LEVEL edit distance:
    - We compare sequences of POS tags (e.g., ['NOUN', 'VERB', 'ADJ'])
    - Each POS tag is treated as a single unit
    - Example: ['NOUN', 'VERB'] vs ['NOUN', 'NOUN'] = 1 substitution (VERB -> NOUN)
    
    Levenshtein Similarity:
    - Normalized version: similarity = 1 - (edit_distance / max_length)
    - Range: 0.0 (completely different) to 1.0 (identical)
    - Example: ['NOUN', 'VERB'] vs ['NOUN', 'NOUN'] with max_length=2:
      edit_distance = 1, similarity = 1 - (1/2) = 0.5
    
    Why this matters for code-switching:
    - We extract POS windows around switch points (e.g., 3 words before/after)
    - Compare these windows to similar windows in monolingual sentences
    - Higher similarity = more similar grammatical structure
    """
    print("=" * 80)
    print("LEVENSHTEIN SIMILARITY EXPLANATION")
    print("=" * 80)
    print()
    print("Levenshtein Distance (Edit Distance):")
    print("  - Measures minimum edits (insert/delete/substitute) to transform one sequence to another")
    print("  - We use SEQUENCE-LEVEL distance (comparing POS tag sequences, not characters)")
    print()
    print("Example calculations:")
    print()
    
    # Example 1: Identical
    seq1 = ['NOUN', 'VERB', 'ADJ']
    seq2 = ['NOUN', 'VERB', 'ADJ']
    dist1 = _sequence_edit_distance(seq1, seq2)
    sim1 = levenshtein_similarity(seq1, seq2)
    print(f"  Example 1 - Identical sequences:")
    print(f"    seq1: {seq1}")
    print(f"    seq2: {seq2}")
    print(f"    Edit distance: {dist1} (0 = identical)")
    print(f"    Similarity: {sim1:.3f} (1.0 = perfect match)")
    print()
    
    # Example 2: One substitution
    seq3 = ['NOUN', 'VERB', 'ADJ']
    seq4 = ['NOUN', 'NOUN', 'ADJ']  # VERB -> NOUN
    dist2 = _sequence_edit_distance(seq3, seq4)
    sim2 = levenshtein_similarity(seq3, seq4)
    print(f"  Example 2 - One substitution:")
    print(f"    seq1: {seq3}")
    print(f"    seq2: {seq4}")
    print(f"    Edit distance: {dist2} (1 substitution: VERB -> NOUN)")
    print(f"    Similarity: {sim2:.3f} (1 - 1/3 = 0.667)")
    print()
    
    # Example 3: Different lengths
    seq5 = ['NOUN', 'VERB']
    seq6 = ['NOUN', 'VERB', 'ADJ', 'NOUN']
    dist3 = _sequence_edit_distance(seq5, seq6)
    sim3 = levenshtein_similarity(seq5, seq6)
    print(f"  Example 3 - Different lengths:")
    print(f"    seq1: {seq5} (length 2)")
    print(f"    seq2: {seq6} (length 4)")
    print(f"    Edit distance: {dist3} (2 insertions needed)")
    print(f"    Similarity: {sim3:.3f} (1 - 2/4 = 0.5)")
    print()
    
    print("In code-switching analysis:")
    print("  - Extract POS window around switch point (e.g., 3 words before/after)")
    print("  - Compare to similar windows in monolingual sentences")
    print("  - Higher similarity = more similar grammatical structure")
    print("  - Threshold: typically 0.4 (40% similarity) to consider a match")
    print()


def test_cantonese_segmentation():
    """Test that Cantonese segmentation works correctly."""
    print("=" * 80)
    print("TEST: Cantonese Segmentation")
    print("=" * 80)
    print()
    
    test_cases = [
        # (input, description, expected_word_count_range)
        ("我係香港人", "Unsegmented Cantonese", (3, 4)),
        ("我 係 香港人", "Space-separated (may be incorrect)", (3, 4)),
        ("我 係 香 港 人", "Incorrectly segmented (should re-segment)", (3, 4)),
        ("你好", "Simple greeting", (1, 2)),
        ("我 想 去 香港", "Space-separated sentence", (3, 5)),
    ]
    
    all_pass = True
    for text, description, expected_range in test_cases:
        try:
            tagged = pos_tag_cantonese(text)
            word_count = len(tagged)
            in_range = expected_range[0] <= word_count <= expected_range[1]
            
            status = "[PASS]" if in_range else "[FAIL]"
            if not in_range:
                all_pass = False
            
            print(f"{status} {description}:")
            print(f"  Input: '{text}'")
            print(f"  Segmented words: {[word for word, _ in tagged]}")
            print(f"  Word count: {word_count} (expected {expected_range[0]}-{expected_range[1]})")
            print(f"  POS tags: {[pos for _, pos in tagged]}")
            print()
        except Exception as e:
            print(f"[FAIL] {description}: Error - {e}")
            print(f"  Input: '{text}'")
            print()
            all_pass = False
    
    return all_pass


def test_cantonese_pos_consistency():
    """Test that same Cantonese text produces consistent POS tags regardless of spacing."""
    print("=" * 80)
    print("TEST: Cantonese POS Tagging Consistency")
    print("=" * 80)
    print()
    
    # Same meaning, different spacing
    test_pairs = [
        ("我係香港人", "我 係 香港人"),
        ("我想去香港", "我 想 去 香港"),
    ]
    
    all_pass = True
    for unsegmented, segmented in test_pairs:
        try:
            tagged1 = pos_tag_cantonese(unsegmented)
            tagged2 = pos_tag_cantonese(segmented)
            
            # Extract just POS sequences
            pos1 = extract_pos_sequence(tagged1)
            pos2 = extract_pos_sequence(tagged2)
            
            # They should produce the same POS sequence
            same_pos = pos1 == pos2
            
            status = "[PASS]" if same_pos else "[WARN]"
            if not same_pos:
                # This might be okay if segmentation differs slightly
                # but POS tags should be similar
                all_pass = False
            
            print(f"{status} Consistency check:")
            print(f"  Unsegmented: '{unsegmented}' -> POS: {pos1}")
            print(f"  Segmented:   '{segmented}' -> POS: {pos2}")
            if not same_pos:
                print(f"  Note: POS sequences differ, but this may be acceptable if segmentation differs")
            print()
        except Exception as e:
            print(f"[FAIL] Error comparing '{unsegmented}' and '{segmented}': {e}")
            print()
            all_pass = False
    
    return all_pass


def test_mixed_sentence_segmentation():
    """Test that mixed sentences properly segment Cantonese portions."""
    print("=" * 80)
    print("TEST: Mixed Sentence Segmentation")
    print("=" * 80)
    print()
    
    test_cases = [
        # (sentence, pattern, description)
        ("我 係 local 人", "C2-E1-C1", "Cantonese-English-Cantonese"),
        ("I am 香港人", "E2-C1", "English-Cantonese"),
        ("你好 hello 再見", "C1-E1-C1", "Cantonese-English-Cantonese"),
    ]
    
    all_pass = True
    for sentence, pattern, description in test_cases:
        try:
            tagged = pos_tag_mixed_sentence(sentence, pattern)
            pos_seq = extract_pos_sequence(tagged)
            
            # Check that we got the right number of tags
            # Parse pattern to get expected word count
            words = sentence.split()
            expected_min = len(words) - 2  # Allow some flexibility
            expected_max = len(words) + 2
            
            word_count = len(tagged)
            in_range = expected_min <= word_count <= expected_max
            
            status = "[PASS]" if in_range else "[WARN]"
            if not in_range:
                all_pass = False
            
            print(f"{status} {description}:")
            print(f"  Sentence: '{sentence}'")
            print(f"  Pattern: {pattern}")
            print(f"  Tagged words: {len(tagged)} (expected ~{len(words)})")
            print(f"  Words: {[(w, pos, lang) for w, pos, lang in tagged[:5]]}...")
            print(f"  POS sequence: {pos_seq[:10]}...")
            print()
        except Exception as e:
            print(f"[FAIL] {description}: Error - {e}")
            print(f"  Sentence: '{sentence}', Pattern: {pattern}")
            print()
            all_pass = False
    
    return all_pass


def test_levenshtein_with_cantonese():
    """Test Levenshtein similarity with actual Cantonese POS tags."""
    print("=" * 80)
    print("TEST: Levenshtein Similarity with Cantonese POS Tags")
    print("=" * 80)
    print()
    
    # Get real Cantonese POS sequences
    test_sentences = [
        "我係香港人",
        "我想去香港",
        "你好嗎",
    ]
    
    all_pass = True
    pos_sequences = []
    
    for sent in test_sentences:
        try:
            tagged = pos_tag_cantonese(sent)
            pos_seq = extract_pos_sequence(tagged)
            pos_sequences.append((sent, pos_seq))
        except Exception as e:
            print(f"[FAIL] Error tagging '{sent}': {e}")
            all_pass = False
            continue
    
    if len(pos_sequences) >= 2:
        # Test similarity between different sentences
        sent1, pos1 = pos_sequences[0]
        sent2, pos2 = pos_sequences[1]
        
        sim = levenshtein_similarity(pos1, pos2)
        
        print(f"Comparing Cantonese sentences:")
        print(f"  Sentence 1: '{sent1}' -> POS: {pos1}")
        print(f"  Sentence 2: '{sent2}' -> POS: {pos2}")
        print(f"  Similarity: {sim:.3f}")
        print()
        
        # Test similarity with identical (should be 1.0)
        sim_identical = levenshtein_similarity(pos1, pos1)
        if abs(sim_identical - 1.0) < 0.001:
            print(f"[PASS] Identical sequences have similarity = 1.0")
        else:
            print(f"[FAIL] Identical sequences should have similarity = 1.0, got {sim_identical}")
            all_pass = False
        print()
    
    return all_pass


def test_end_to_end_pipeline():
    """Test the complete pipeline from Cantonese text to similarity calculation."""
    print("=" * 80)
    print("TEST: End-to-End Pipeline")
    print("=" * 80)
    print()
    
    # Simulate what happens in the matching algorithm
    cantonese_sent1 = "我係香港人"
    cantonese_sent2 = "我想去香港"
    
    try:
        # Step 1: POS tag both sentences
        tagged1 = pos_tag_cantonese(cantonese_sent1)
        tagged2 = pos_tag_cantonese(cantonese_sent2)
        
        pos_seq1 = extract_pos_sequence(tagged1)
        pos_seq2 = extract_pos_sequence(tagged2)
        
        # Step 2: Calculate similarity
        similarity = levenshtein_similarity(pos_seq1, pos_seq2)
        
        print(f"End-to-end test:")
        print(f"  Sentence 1: '{cantonese_sent1}'")
        print(f"    -> Words: {[w for w, _ in tagged1]}")
        print(f"    -> POS: {pos_seq1}")
        print()
        print(f"  Sentence 2: '{cantonese_sent2}'")
        print(f"    -> Words: {[w for w, _ in tagged2]}")
        print(f"    -> POS: {pos_seq2}")
        print()
        print(f"  Similarity: {similarity:.3f}")
        print()
        
        # Verify segmentation worked
        if len(tagged1) > 0 and len(tagged2) > 0:
            print(f"[PASS] End-to-end pipeline works correctly")
            return True
        else:
            print(f"[FAIL] Segmentation failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] End-to-end pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CANTONESE SEGMENTATION & LEVENSHTEIN SIMILARITY TEST")
    print("=" * 80)
    print()
    
    # Explain Levenshtein similarity
    explain_levenshtein_similarity()
    
    # Run tests
    results = []
    
    results.append(("Cantonese Segmentation", test_cantonese_segmentation()))
    results.append(("POS Tagging Consistency", test_cantonese_pos_consistency()))
    results.append(("Mixed Sentence Segmentation", test_mixed_sentence_segmentation()))
    results.append(("Levenshtein with Cantonese", test_levenshtein_with_cantonese()))
    results.append(("End-to-End Pipeline", test_end_to_end_pipeline()))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL/WARN]"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False
    
    print("=" * 80)
    
    if all_pass:
        print("\n[PASS] ALL TESTS PASSED - Cantonese segmentation is working correctly!")
        return 0
    else:
        print("\n[WARN] Some tests had warnings or failures - review above")
        return 1


if __name__ == '__main__':
    sys.exit(main())

