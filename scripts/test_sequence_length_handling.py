"""
Test script to verify sequence length handling in surprisal calculators.

This script tests:
1. Cases that fit within max_length (512 tokens for BERT)
2. Cases that require post-switch trimming
3. Cases where required content exceeds max_length (edge case)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.surprisal_calculator import MaskedLMSurprisalCalculator
import logging

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()


def test_case(calc, test_name, context, words, word_index):
    """
    Test a single case and show detailed information.
    
    Args:
        calc: Surprisal calculator instance
        test_name: Name of the test case
        context: Context sentences (or None)
        words: List of words in target sentence
        word_index: Index of target word
    """
    print_separator(f"TEST: {test_name}")
    
    # Show input
    print("INPUT:")
    if context:
        print(f"  Context: {context[:100]}{'...' if len(context) > 100 else ''}")
        print(f"  Context length: {len(context)} chars, {len(context.split())} words")
    else:
        print("  Context: None")
    
    print(f"  Sentence: {''.join(words)}")
    print(f"  Sentence length: {len(words)} words")
    print(f"  Target word index: {word_index} (word: '{words[word_index]}')")
    print(f"  Pre-switch words: {word_index}")
    print(f"  Post-switch words: {len(words) - word_index - 1}")
    
    # Calculate what should be kept
    if context:
        context_clean = context.replace(' ||| ', ' ')
        context_words = context_clean.strip().split()
    else:
        context_words = []
    
    required_words = context_words + words[:word_index + 1]
    required_text = "".join(required_words)
    
    # Tokenize to see actual token counts
    required_encoding = calc.tokenizer(required_text, add_special_tokens=True)
    required_tokens = len(required_encoding['input_ids'])
    
    full_text = "".join(context_words + words)
    full_encoding = calc.tokenizer(full_text, add_special_tokens=True)
    full_tokens = len(full_encoding['input_ids'])
    
    postswitch_words = words[word_index + 1:]
    available_for_postswitch = calc.max_length - required_tokens
    
    print(f"\nTOKEN ANALYSIS:")
    print(f"  Model max_length: {calc.max_length}")
    print(f"  Required content (context + pre-switch + target): {required_tokens} tokens")
    print(f"  Available for post-switch: {available_for_postswitch} tokens")
    print(f"  Full sentence (if all kept): {full_tokens} tokens")
    
    if required_tokens > calc.max_length:
        print(f"  ⚠️  WARNING: Required content exceeds max_length by {required_tokens - calc.max_length} tokens!")
        print(f"      This will cause errors or truncation of pre-switch context")
    elif full_tokens <= calc.max_length:
        print(f"  ✓ All content fits! ({full_tokens}/{calc.max_length} tokens)")
    else:
        print(f"  ⚠️  Need to trim {full_tokens - calc.max_length} tokens from post-switch")
        print(f"      ({len(postswitch_words)} post-switch words available)")
    
    # Run the calculation
    print(f"\nRUNNING CALCULATION...")
    try:
        result = calc.calculate_surprisal(
            word_index=word_index,
            words=words,
            context=context
        )
        
        print(f"\nRESULT:")
        print(f"  Surprisal: {result['surprisal']:.3f}" if result['surprisal'] == result['surprisal'] else "  Surprisal: NaN")
        print(f"  Probability: {result['probability']:.6f}")
        print(f"  Target word: {result['word']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Num tokens: {result['num_tokens']}")
        print(f"  Valid tokens: {result['num_valid_tokens']}")
        print(f"  ✓ SUCCESS")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run test cases."""
    
    print_separator("SEQUENCE LENGTH HANDLING TEST")
    print("This test verifies that the surprisal calculator correctly handles:")
    print("  1. Sequences that fit within max_length")
    print("  2. Sequences requiring post-switch trimming")
    print("  3. Edge cases where required content is very long")
    print()
    print("Using model: bert-base-chinese (max_length: 512)")
    
    # Initialize calculator
    print("\nLoading model...")
    calc = MaskedLMSurprisalCalculator(
        model_name="bert-base-chinese",
        device="cpu"  # Use CPU for testing
    )
    print(f"Model loaded. Max length: {calc.max_length}")
    
    # Test Case 1: Normal case - everything fits
    test_case(
        calc=calc,
        test_name="Case 1: Normal - Everything Fits",
        context="我今天去了商店 ||| 买了很多东西 ||| 然后回家了",
        words=["我", "想", "買", "apple", "和", "banana"],
        word_index=3  # "apple" is the switch point
    )
    
    # Test Case 2: Long post-switch content (should be trimmed)
    long_postswitch = ["post_word_" + str(i) for i in range(100)]  # 100 post-switch words
    test_case(
        calc=calc,
        test_name="Case 2: Long Post-Switch (Should Trim)",
        context="我今天去了商店 ||| 买了很多东西",
        words=["我", "想", "買", "apple"] + long_postswitch,
        word_index=3  # "apple"
    )
    
    # Test Case 3: No context
    test_case(
        calc=calc,
        test_name="Case 3: No Context",
        context=None,
        words=["我", "想", "買", "apple", "和", "banana"],
        word_index=3
    )
    
    # Test Case 4: Very long context (many sentences)
    long_context_sentences = []
    for i in range(10):  # 10 context sentences
        sentence = "".join([f"字{j}" for j in range(20)])  # 20 chars per sentence
        long_context_sentences.append(sentence)
    long_context = " ||| ".join(long_context_sentences)
    
    test_case(
        calc=calc,
        test_name="Case 4: Very Long Context",
        context=long_context,
        words=["我", "想", "買", "apple", "和", "banana", "還有", "orange"],
        word_index=3
    )
    
    # Test Case 5: Minimal case - target at beginning
    test_case(
        calc=calc,
        test_name="Case 5: Target at Beginning",
        context="之前的一些内容",
        words=["apple", "is", "good"],
        word_index=0  # First word
    )
    
    # Test Case 6: Target at end (no post-switch words)
    test_case(
        calc=calc,
        test_name="Case 6: Target at End (No Post-Switch)",
        context="我今天去了",
        words=["商店", "買", "了", "apple"],
        word_index=3  # Last word
    )
    
    # Test Case 7: Very long required content (edge case - should warn)
    very_long_preswitch = ["pre_" + str(i) for i in range(200)]  # 200 pre-switch words
    test_case(
        calc=calc,
        test_name="Case 7: EXTREME - Very Long Required Content",
        context=" ||| ".join(["很长的上下文句子" + str(i) for i in range(20)]),
        words=very_long_preswitch + ["TARGET"] + ["post"],
        word_index=len(very_long_preswitch)
    )
    
    print_separator("ALL TESTS COMPLETE")
    print("Review the output above to verify:")
    print("  ✓ Token counts are calculated correctly")
    print("  ✓ Required content is always kept")
    print("  ✓ Post-switch trimming happens when needed")
    print("  ✓ Warnings appear for edge cases")


if __name__ == "__main__":
    main()
