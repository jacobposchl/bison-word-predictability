"""Debug script to test why NLLB translation isn't working properly."""

import pandas as pd
from src.experiments.nllb_translator import NLLBTranslator
from src.core.config import Config

# Load config
config = Config()

# Read one problematic sentence from CSV
csv_path = config.get_csv_cantonese_translated_path()
df = pd.read_csv(csv_path, nrows=5)

print("=" * 80)
print("DEBUGGING TRANSLATION ISSUES")
print("=" * 80)

# Initialize translator
print("\n1. Initializing NLLB translator...")
translator = NLLBTranslator(
    model_name=config.get_translation_model(),
    device=config.get_translation_device(),
    show_progress=False
)

# Test a simple English phrase first
print("\n2. Testing simple English-to-Cantonese translation:")
simple_english = "I was born in Canada"
simple_translation = translator.translate_english_to_cantonese(simple_english)
print(f"   Input:  '{simple_english}'")
print(f"   Output: '{simple_translation}'")
print(f"   Is it actually translated? {simple_english != simple_translation}")

# Now test the first problematic sentence
print("\n3. Testing first sentence from CSV:")
row = df.iloc[0]
print(f"   Reconstructed: {row['reconstructed_sentence'][:100]}...")
print(f"   Pattern: {row['pattern']}")
print(f"   Translation: {row['cantonese_translation'][:100]}...")

# Manually parse and translate
sentence = row['reconstructed_sentence']
pattern = row['pattern']
words = sentence.split()

print(f"\n4. Manual step-through:")
print(f"   Number of words: {len(words)}")
print(f"   Pattern: {pattern}")

# Parse pattern
segments = []
for segment in pattern.split('-'):
    lang = segment[0]
    count = int(segment[1:])
    segments.append((lang, count))
    print(f"   Segment: {lang}{count}")

print(f"\n5. Word-to-segment mapping:")
word_idx = 0
for lang, count in segments:
    segment_words = words[word_idx:word_idx + count]
    print(f"   {lang}{count}: {' '.join(segment_words)}")
    
    if lang == 'E':
        english_text = ' '.join(segment_words)
        cantonese = translator.translate_english_to_cantonese(english_text)
        print(f"      -> Translation: '{cantonese}'")
        print(f"      -> Changed? {english_text != cantonese}")
    
    word_idx += count

print("\n6. Full translation result:")
result = translator.translate_code_switched_sentence(sentence, pattern, words)
print(f"   Original:   {result['original_sentence'][:100]}...")
print(f"   Translated: {result['translated_sentence'][:100]}...")

print("\n" + "=" * 80)
