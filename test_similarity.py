"""Test Levenshtein similarity calculation to verify correctness."""

from Levenshtein import distance as levenshtein_distance

def test_string_vs_sequence():
    """Test if string-based Levenshtein gives correct results for sequences."""
    
    # Test case 1: Identical sequences
    seq1 = ['NOUN', 'VERB', 'ADJ']
    seq2 = ['NOUN', 'VERB', 'ADJ']
    
    seq1_str = '|'.join(seq1)
    seq2_str = '|'.join(seq2)
    
    dist_str = levenshtein_distance(seq1_str, seq2_str)
    print(f"Test 1 - Identical sequences:")
    print(f"  String distance: {dist_str} (should be 0)")
    print(f"  seq1_str: '{seq1_str}'")
    print(f"  seq2_str: '{seq2_str}'")
    
    # Test case 2: One substitution
    seq3 = ['NOUN', 'VERB', 'ADJ']
    seq4 = ['NOUN', 'NOUN', 'ADJ']  # VERB -> NOUN
    
    seq3_str = '|'.join(seq3)
    seq4_str = '|'.join(seq4)
    
    dist_str2 = levenshtein_distance(seq3_str, seq4_str)
    print(f"\nTest 2 - One substitution:")
    print(f"  String distance: {dist_str2}")
    print(f"  seq3_str: '{seq3_str}'")
    print(f"  seq4_str: '{seq4_str}'")
    
    # Test case 3: Problem case - tags that contain the separator
    # If a POS tag somehow contained '|', this would break
    # But POS tags shouldn't contain '|', so this should be fine
    
    # Test case 4: Check if we can use lists directly
    try:
        # python-Levenshtein distance can work on sequences
        # But we need to check the actual behavior
        print(f"\nTest 3 - Checking if distance works on lists:")
        # The distance function from python-Levenshtein works on strings
        # For sequences, we need to join them
        print(f"  Using string join method (current approach)")
        
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == '__main__':
    test_string_vs_sequence()

