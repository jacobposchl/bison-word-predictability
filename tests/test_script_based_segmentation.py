"""
Test script for visualizing script-based segmentation.

This script tests the script-based segmentation approach used to identify
Cantonese and English words in mixed text with inconsistent spacing.
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # If reconfigure doesn't work, wrap stdout
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.pattern_analysis import (
    _is_cjk_character,
    _is_ascii_alphabetic,
    segment_by_script,
    tokenize_main_tier_sentence,
    process_sentence_from_main_tier
)


def print_separator():
    """Print a separator line."""
    print("=" * 80)


def test_character_detection():
    """Test character detection functions."""
    print("\n" + "=" * 80)
    print("CHARACTER DETECTION TESTS")
    print("=" * 80)
    
    # Note: These functions only work on single characters
    test_chars = [
        ("我", True, False, "CJK character"),
        ("人", True, False, "CJK character"),
        ("h", False, True, "ASCII letter"),
        ("a", False, True, "ASCII letter"),
        ("5", False, False, "Digit"),
        ("，", True, False, "Chinese punctuation (CJK)"),
        (".", False, False, "Punctuation"),
        (" ", False, False, "Whitespace"),
    ]
    
    print(f"\n{'Character':<15} {'Is CJK':<10} {'Is ASCII':<12} {'Expected':<20} {'Result':<10}")
    print("-" * 80)
    
    for char, expected_cjk, expected_ascii, description in test_chars:
        # These functions expect single characters
        if len(char) != 1:
            print(f"{char:<15} {'SKIP':<10} {'SKIP':<12} {description:<20} {'SKIP':<10}")
            continue
        is_cjk = _is_cjk_character(char)
        is_ascii = _is_ascii_alphabetic(char)
        result = "✓" if (is_cjk == expected_cjk and is_ascii == expected_ascii) else "✗"
        try:
            print(f"{char:<15} {str(is_cjk):<10} {str(is_ascii):<12} {description:<20} {result:<10}")
        except UnicodeEncodeError:
            # Fallback for characters that can't be printed
            char_repr = repr(char) if len(char) == 1 else char
            print(f"{char_repr:<15} {str(is_cjk):<10} {str(is_ascii):<12} {description:<20} {result:<10}")


def test_script_segmentation():
    """Test script-based segmentation with various examples."""
    print("\n" + "=" * 80)
    print("SCRIPT SEGMENTATION TESTS")
    print("=" * 80)
    
    test_cases = [
        # (input, expected_segments, description)
        ("我local人", [("我", "C"), ("local", "E"), ("人", "C")], 
         "Mixed text without spaces"),
        ("我 local 人", [("我", "C"), ("local", "E"), ("人", "C")], 
         "Mixed text with spaces"),
        ("hello你好", [("hello", "E"), ("你好", "C")], 
         "English then Cantonese, no space"),
        ("你好hello", [("你好", "C"), ("hello", "E")], 
         "Cantonese then English, no space"),
        ("我係香港人", [("我係香港人", "C")], 
         "Pure Cantonese"),
        ("hello world", [("hello", "E"), ("world", "E")], 
         "Pure English"),
        ("我 係 local 人", [("我", "C"), ("係", "C"), ("local", "E"), ("人", "C")], 
         "Mixed with spaces between all words"),
        ("我係local人", [("我係", "C"), ("local", "E"), ("人", "C")], 
         "Cantonese compound before English"),
        ("我local係人", [("我", "C"), ("local", "E"), ("係人", "C")], 
         "English in middle of Cantonese"),
        ("我local", [("我", "C"), ("local", "E")], 
         "Simple mixed case"),
        ("local人", [("local", "E"), ("人", "C")], 
         "English then Cantonese, no space"),
    ]
    
    print(f"\n{'Input':<25} {'Segments':<35} {'Status':<10}")
    print("-" * 80)
    
    for input_text, expected, description in test_cases:
        result = segment_by_script(input_text)
        status = "✓" if result == expected else "✗"
        
        # Format segments for display
        segments_str = ", ".join([f"{seg}({lang})" for seg, lang in result])
        expected_str = ", ".join([f"{seg}({lang})" for seg, lang in expected])
        
        print(f"{input_text:<25} {segments_str:<35} {status:<10}")
        if status == "✗":
            print(f"{'Expected:':>26} {expected_str}")
            print(f"{'Description:':>26} {description}")
        print()


def test_tokenization():
    """Test full tokenization pipeline."""
    print("\n" + "=" * 80)
    print("TOKENIZATION TESTS")
    print("=" * 80)
    
    test_cases = [
        ("我local人", "Mixed without spaces"),
        ("我 local 人", "Mixed with spaces"),
        ("我係香港人", "Pure Cantonese"),
        ("hello world", "Pure English"),
        ("我 係 local 人", "Mixed spaced"),
        ("我係local係人", "English embedded in Cantonese"),
        ("local係人hello", "English on both sides"),
    ]
    
    for input_text, description in test_cases:
        print(f"\nInput: '{input_text}' ({description})")
        print("-" * 80)
        
        # Tokenize (using dummy timestamps)
        tokens = tokenize_main_tier_sentence(1000, 2000, input_text)
        
        if not tokens:
            print("  No tokens produced")
            continue
        
        # Display tokens with language labels
        token_strs = []
        for timestamp, word, lang in tokens:
            lang_label = "CANT" if lang == "C" else "ENG"
            token_strs.append(f"{word}({lang_label})")
        
        print(f"  Tokens ({len(tokens)}): {' | '.join(token_strs)}")
        
        # Show pattern
        pattern_parts = []
        current_lang = tokens[0][2]
        current_count = 0
        for _, _, lang in tokens:
            if lang == current_lang:
                current_count += 1
            else:
                pattern_parts.append(f"{current_lang}{current_count}")
                current_lang = lang
                current_count = 1
        pattern_parts.append(f"{current_lang}{current_count}")
        pattern = '-'.join(pattern_parts)
        
        print(f"  Pattern: {pattern}")


def test_full_processing():
    """Test the full sentence processing pipeline."""
    print("\n" + "=" * 80)
    print("FULL PROCESSING PIPELINE TESTS")
    print("=" * 80)
    
    test_cases = [
        (1000, 2000, "我local人", "Mixed without spaces"),
        (2000, 3000, "我 local 人", "Mixed with spaces"),
        (3000, 4000, "我係香港人", "Pure Cantonese"),
        (4000, 5000, "hello world", "Pure English"),
        (5000, 6000, "我 係 local 人", "Mixed spaced"),
    ]
    
    for start, end, text, description in test_cases:
        print(f"\nInput: '{text}' ({description})")
        print(f"Time: {start}ms - {end}ms")
        print("-" * 80)
        
        # Process sentence
        main_annotation = (start, end, text)
        sentence_data = process_sentence_from_main_tier(main_annotation)
        
        if not sentence_data:
            print("  No sentence data produced")
            continue
        
        print(f"  Total words: {sentence_data['total_words']}")
        print(f"  Cantonese words: {sentence_data['cant_words']}")
        print(f"  English words: {sentence_data['eng_words']}")
        print(f"  Pattern: {sentence_data['pattern']}")
        print(f"  Matrix language: {sentence_data['matrix_language']}")
        print(f"  Reconstructed: {sentence_data['reconstructed_text']}")
        
        # Show tokens
        token_strs = []
        for timestamp, word, lang in sentence_data['items']:
            lang_label = "CANT" if lang == "C" else "ENG"
            token_strs.append(f"{word}({lang_label})")
        print(f"  Tokens: {' | '.join(token_strs)}")


def test_edge_cases():
    """Test edge cases and potential issues."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    
    edge_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("我", "Single CJK character"),
        ("a", "Single ASCII character"),
        ("123", "Numbers only"),
        ("我，。", "CJK with punctuation"),
        ("hello, world", "English with punctuation"),
        ("我local，人", "Mixed with Chinese punctuation"),
        ("hello,我", "English then Cantonese with comma"),
        ("我local...人", "Mixed with ellipses"),
    ]
    
    for input_text, description in edge_cases:
        print(f"\n'{input_text}' - {description}")
        print("-" * 80)
        
        segments = segment_by_script(input_text)
        if segments:
            segments_str = ", ".join([f"{seg}({lang})" for seg, lang in segments])
            print(f"  Segments: {segments_str}")
        else:
            print("  No segments")
        
        tokens = tokenize_main_tier_sentence(1000, 2000, input_text)
        if tokens:
            token_strs = []
            for _, word, lang in tokens:
                lang_label = "CANT" if lang == "C" else "ENG"
                token_strs.append(f"{word}({lang_label})")
            print(f"  Tokens: {' | '.join(token_strs)}")
        else:
            print("  No tokens")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SCRIPT-BASED SEGMENTATION VISUALIZATION TEST")
    print("=" * 80)
    print("\nThis script tests the script-based segmentation approach used")
    print("to identify Cantonese and English words in mixed text without")
    print("relying on subtiers or consistent spacing.")
    
    try:
        test_character_detection()
        test_script_segmentation()
        test_tokenization()
        test_full_processing()
        test_edge_cases()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
