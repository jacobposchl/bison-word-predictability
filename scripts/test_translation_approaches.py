"""
Test script to compare old vs new translation segmentation approaches.

Compares:
1. Old approach: Append translated phrase as-is (unsegmented)
2. New approach: Segment translated phrase using pycantonese

Outputs results to CSV for comparison.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import pycantonese

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.nllb_translator import NLLBTranslator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Test sentences of varying complexity
TEST_SENTENCES = [
    # Simple phrases
    "I don't know",
    "hello world",
    "good morning",
    "thank you",
    "nice to meet you",
    "see you later",
    "have a great day",
    "take care",
    
    # Medium phrases
    "How are you doing today",
    "I really enjoyed the movie",
    "Can you help me with this problem",
    "What is your name",
    "Where do you live",
    "What time is it now",
    "I love to read books",
    "Do you want to go shopping",
    
    # Longer phrases
    "I would like to know what you think about this situation",
    "The weather today is very nice and sunny outside",
    "I have been working on this project for quite some time now",
    "Could you please explain the main idea behind this concept",
    "I think we should try a different approach to solve this problem",
    "The main reason why I came here is to discuss the upcoming plans",
    "I was wondering if you could provide me with some helpful advice",
    "It has been a long time since we last spoke to each other",
]


def test_old_approach(translator: NLLBTranslator, english_text: str) -> str:
    """
    Old approach: Translate and append as single phrase (unsegmented).
    
    Args:
        translator: NLLBTranslator instance
        english_text: English text to translate
        
    Returns:
        Unsegmented translated phrase
    """
    translation = translator.translate_english_to_cantonese(english_text)
    return translation


def test_new_approach(translator: NLLBTranslator, english_text: str) -> str:
    """
    New approach: Translate and segment using pycantonese.
    
    Args:
        translator: NLLBTranslator instance
        english_text: English text to translate
        
    Returns:
        Space-separated segmented words
    """
    translation = translator.translate_english_to_cantonese(english_text)
    segmented_words = list(pycantonese.segment(translation))
    return ' '.join(segmented_words)


def test_translation_approaches() -> None:
    """
    Test both translation approaches and compare results.
    """
    print("Initializing NLLB translator...")
    translator = NLLBTranslator(
        model_name="facebook/nllb-200-distilled-600M",
        device="auto",
        show_progress=False
    )
    
    print(f"Testing {len(TEST_SENTENCES)} sentences...\n")
    
    results = []
    
    for i, english_text in enumerate(TEST_SENTENCES, 1):
        print(f"[{i}/{len(TEST_SENTENCES)}] Testing: {english_text}")
        
        try:
            # Test old approach
            old_result = test_old_approach(translator, english_text)
            
            # Test new approach
            new_result = test_new_approach(translator, english_text)
            
            # Count words in each result
            old_word_count = len(old_result.split())
            new_word_count = len(new_result.split())
            
            result_row = {
                'english_phrase': english_text,
                'english_word_count': len(english_text.split()),
                'old_approach_output': old_result,
                'old_word_count': old_word_count,
                'new_approach_output': new_result,
                'new_word_count': new_word_count,
                'word_count_difference': new_word_count - old_word_count,
                'is_same': old_result == new_result
            }
            
            results.append(result_row)
            
            if old_result != new_result:
                print(f"  ✓ DIFFERENCE FOUND")
                print(f"    Old: {old_result} (words: {old_word_count})")
                print(f"    New: {new_result} (words: {new_word_count})")
            else:
                print(f"  ✓ Same output")
            print()
            
        except Exception as e:
            logger.error(f"Error testing sentence: {english_text}")
            logger.error(f"Error: {str(e)}")
            result_row = {
                'english_phrase': english_text,
                'english_word_count': len(english_text.split()),
                'old_approach_output': f"ERROR: {str(e)}",
                'old_word_count': None,
                'new_approach_output': f"ERROR: {str(e)}",
                'new_word_count': None,
                'word_count_difference': None,
                'is_same': None
            }
            results.append(result_row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Export to CSV
    output_path = Path(project_root) / 'results' / 'translation_approach_comparison.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n" + "="*80)
    print(f"Results exported to: {output_path}")
    print("="*80)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 80)
    
    same_count = df['is_same'].sum()
    different_count = len(df) - same_count - df['is_same'].isna().sum()
    
    print(f"Total sentences tested: {len(df)}")
    print(f"Same output (old vs new): {same_count}")
    print(f"Different output: {different_count}")
    
    if different_count > 0:
        print("\nSentences with differences:")
        different_df = df[df['is_same'] == False]
        for idx, row in different_df.iterrows():
            print(f"\n  {idx+1}. {row['english_phrase']}")
            print(f"     Old word count: {row['old_word_count']}")
            print(f"     New word count: {row['new_word_count']}")
            print(f"     Difference: {row['word_count_difference']} words")


if __name__ == '__main__':
    setup_logging()
    test_translation_approaches()
