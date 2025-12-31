"""
Script to translate code-switched sentences to full Cantonese and verify translations.

Processes rows in cantonese_translated_WITHOUT_fillers.csv one at a time,
translates them, verifies they are fully Cantonese, and updates the CSV after each.
Stops early when a translation fails verification.
"""

import argparse
import pandas as pd
import re
from typing import List, Tuple, Optional
from tqdm import tqdm
from src.experiments.nllb_translator import NLLBTranslator
from src.core.config import Config

# Load config
config = Config()


def is_english_word(word: str) -> bool:
    """
    Check if a word is likely English (contains only ASCII letters).
    
    Args:
        word: Word to check
        
    Returns:
        True if word appears to be English, False otherwise
    """
    # Remove punctuation and whitespace
    cleaned = re.sub(r'[^\w]', '', word)
    
    # If empty after cleaning, it's not an English word
    if not cleaned:
        return False
    
    # Check if all characters are ASCII letters (a-z, A-Z)
    # This is a simple heuristic: English words use ASCII, Cantonese uses Unicode
    return all(ord(c) < 128 and c.isalpha() for c in cleaned)


def contains_english_words(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains any English words.
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (has_english, english_words_found)
    """
    # Split by whitespace and punctuation to get potential words
    # This is a simple approach - could be improved with better tokenization
    words = re.findall(r'\b\w+\b', text)
    
    english_words = []
    for word in words:
        if is_english_word(word):
            # Filter out very short words that might be false positives
            # (e.g., single letters, common abbreviations)
            if len(word) > 2 or word.lower() in ['ok', 'okay', 'uh', 'um', 'ah', 'oh']:
                english_words.append(word)
    
    return len(english_words) > 0, english_words


def verify_cantonese_only(translation: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a translation is fully Cantonese (no English words).
    
    Args:
        translation: Translated text to verify
        
    Returns:
        Tuple of (is_valid, error_message)
        is_valid is True if translation is fully Cantonese
    """
    if not translation or not translation.strip():
        return False, "Translation is empty"
    
    has_english, english_words = contains_english_words(translation)
    
    if has_english:
        return False, f"Contains English words: {', '.join(english_words[:5])}"
    
    return True, None


def translate_and_verify_row(
    row: pd.Series,
    translator: NLLBTranslator,
    idx: int,
    skip_existing: bool = True
) -> Tuple[str, bool, Optional[str]]:
    """
    Translate a single row and verify the translation.
    
    Args:
        row: DataFrame row with 'code_switch_original' and 'pattern'
        translator: NLLBTranslator instance
        idx: Row index for error reporting
        skip_existing: If True, skip rows that already have valid translations
        
    Returns:
        Tuple of (translation, is_valid, error_message)
    """
    sentence = row['code_switch_original']
    pattern = row['pattern']
    
    # Check if already translated and valid (skip retranslation if valid)
    if skip_existing:
        existing_translation = row.get('cantonese_translation', '')
        if pd.notna(existing_translation) and existing_translation.strip():
            is_valid, error = verify_cantonese_only(str(existing_translation))
            if is_valid:
                return str(existing_translation), True, None
            # If invalid, we'll retranslate below
    
    # This function is now mainly for backward compatibility
    # The main loop does the translation directly to get segment details
    if pd.isna(sentence) or pd.isna(pattern) or not sentence or not pattern:
        return '', False, "Missing sentence or pattern"
    
    try:
        words = sentence.split()
        result = translator.translate_code_switched_sentence(
            sentence=sentence,
            pattern=str(pattern),
            words=words
        )
        translation = result['translated_sentence']
        is_valid, error = verify_cantonese_only(translation)
        return translation, is_valid, error
    except Exception as e:
        return '', False, f"Translation error: {str(e)}"


def main():
    """Main function to process CSV and translate sentences incrementally."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Translate code-switched sentences to full Cantonese with verification"
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (default: process until first invalid translation)'
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRANSLATING CODE-SWITCHED SENTENCES TO FULL CANTONESE")
    print("=" * 80)
    
    # Load CSV
    csv_path = config.get_csv_cantonese_translated_path()
    print(f"\n1. Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure cantonese_translation column is string type
    if 'cantonese_translation' in df.columns:
        df['cantonese_translation'] = df['cantonese_translation'].fillna('').astype(str)
    print(f"   Loaded {len(df)} rows")
    
    if args.max_rows:
        print(f"   Processing up to {args.max_rows} rows")
    
    # Initialize translator
    print("\n2. Initializing NLLB translator...")
    translator = NLLBTranslator(
        model_name=config.get_translation_model(),
        device=config.get_translation_device(),
        show_progress=False
    )
    print("   ✓ Translator ready")
    
    # Test simple translation first
    print("\n3. Testing simple translation:")
    test_english = "I was born in Canada"
    test_translation = translator.translate_english_to_cantonese(test_english)
    print(f"   Input:  '{test_english}'")
    print(f"   Output: '{test_translation}'")
    is_valid, error = verify_cantonese_only(test_translation)
    print(f"   Verification: {'✓ Valid' if is_valid else f'✗ {error}'}")
    
    # Process rows one at a time
    print("\n4. Processing rows (saving after each translation)...")
    print("=" * 80)
    
    processed = 0
    translated_count = 0
    skipped_count = 0
    stopped_early = False
    
    for idx, row in df.iterrows():
        # Check if we've reached the max rows limit
        if args.max_rows and processed >= args.max_rows:
            print(f"\n   Reached max-rows limit ({args.max_rows})")
            break
        
        # Check if already has valid translation
        existing_translation = row.get('cantonese_translation', '')
        if pd.notna(existing_translation) and existing_translation.strip():
            is_valid, _ = verify_cantonese_only(str(existing_translation))
            if is_valid:
                skipped_count += 1
                processed += 1
                continue
        
        # Translate and verify
        print(f"\n   Row {idx}:")
        print(f"   Original: {row['code_switch_original'][:80]}...")
        print(f"   Pattern: {row['pattern']}")
        
        # Debug: Show word segmentation
        sentence = row['code_switch_original']
        pattern = row['pattern']
        words = sentence.split()
        print(f"   Words ({len(words)}): {words}")
        
        # Parse pattern and show segments
        segments = []
        for segment in pattern.split('-'):
            lang = segment[0]
            count = int(segment[1:])
            segments.append((lang, count))
        
        total_from_pattern = sum(count for _, count in segments)
        print(f"   Pattern expects {total_from_pattern} words")
        if len(words) != total_from_pattern:
            print(f"   ⚠ WARNING: Word count mismatch! Pattern expects {total_from_pattern} but sentence has {len(words)}")
        
        # Show how words are segmented
        word_idx = 0
        for lang, count in segments:
            segment_words = words[word_idx:word_idx + count] if word_idx < len(words) else []
            print(f"     {lang}{count}: {segment_words}")
            word_idx += count
        
        # Translate and get detailed result
        translation_result = None
        translation = ''
        is_valid = False
        error_msg = None
        
        if not pd.isna(sentence) and not pd.isna(pattern) and sentence and pattern:
            try:
                translation_result = translator.translate_code_switched_sentence(
                    sentence, pattern, words
                )
                translation = translation_result['translated_sentence']
                is_valid, error_msg = verify_cantonese_only(translation)
                
                # Show translation segments
                if translation_result.get('segments'):
                    print(f"   Translation segments:")
                    for i, seg in enumerate(translation_result['segments']):
                        lang_name = seg.get('language', 'Unknown')
                        orig = seg.get('original', '')
                        trans = seg.get('translated', '')
                        print(f"     {lang_name}: '{orig}' → '{trans}'")
            except Exception as e:
                translation = ''
                is_valid = False
                error_msg = f"Translation error: {str(e)}"
        else:
            translation = ''
            is_valid = False
            error_msg = "Missing sentence or pattern"
        
        # Update the row in the DataFrame (ensure string type)
        df.at[idx, 'cantonese_translation'] = str(translation) if translation else ''
        
        # Save CSV after each translation
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        if not translation:
            print(f"   ✗ Error: {error_msg}")
            skipped_count += 1
        elif is_valid:
            print(f"   ✓ Translation: {translation[:80]}...")
            print(f"   ✓ Verification: Valid (fully Cantonese)")
            translated_count += 1
        else:
            print(f"   ✗ Translation: {translation[:80]}...")
            print(f"   ✗ Verification FAILED: {error_msg}")
            print(f"\n   ⚠ STOPPING: Found invalid translation at row {idx}")
            stopped_early = True
            break
        
        processed += 1
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("TRANSLATION SUMMARY")
    print("=" * 80)
    print(f"Rows processed: {processed}")
    print(f"  - Newly translated: {translated_count}")
    print(f"  - Skipped (already valid): {skipped_count}")
    if stopped_early:
        print(f"  - Stopped early due to invalid translation")
    print(f"\nCSV saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
