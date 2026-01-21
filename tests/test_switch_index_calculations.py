"""
Test script demonstrating switch index calculations and surprisal input preparation.

This script shows:
1. How switch_index is calculated from patterns
2. How matched_switch_index is calculated via POS window matching
3. What words are sent to BERT-style (masked) surprisal calculator
4. What words are sent to GPT-style (autoregressive) surprisal calculator
"""

import re
from typing import List, Tuple


def parse_pattern_segments(pattern: str) -> List[Tuple[str, int]]:
    """
    Parse pattern string into language segments.
    
    Args:
        pattern: Pattern like "C5-E3-C2"
        
    Returns:
        List of (language, count) tuples
    """
    if not pattern:
        return []
    
    matches = re.findall(r'([CE])(\d+)', pattern)
    return [(lang, int(count)) for lang, count in matches]


def get_switch_index(pattern: str) -> int:
    """Extract the index of the first English word (the switch word).
    
    For pattern C18-E1, returns 18 (the first English word, 0-based indexing).
    This is the actual switch word position.
    """
    try:
        segments = parse_pattern_segments(pattern)
        if len(segments) < 2:
            return -1
        first_lang, first_count = segments[0]
        if first_lang == 'C' and first_count > 0:
            return first_count  # First English word (0-based, so count is the index)
        return -1  # English starts at beginning, no Cantonese word before switch
    except Exception:
        return -1


def extract_pos_window(pos_sequence, switch_index, window_size=3):
    """Extract POS window around switch point."""
    window_start = max(0, switch_index - window_size)
    window_end = min(len(pos_sequence), switch_index + window_size + 1)
    pos_window = pos_sequence[window_start:window_end]
    switch_index_in_window = switch_index - window_start
    return pos_window, switch_index_in_window, window_start


def calculate_matched_switch_index(best_start_idx, switch_index_in_window, mono_pos_length):
    """Calculate matched switch index in monolingual sentence."""
    matched_switch_index = best_start_idx + switch_index_in_window
    # Ensure within bounds
    matched_switch_index = min(matched_switch_index, mono_pos_length - 1)
    return matched_switch_index


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_example(example_num, title):
    """Print example header."""
    print(f"\n{'-'*80}")
    print(f"EXAMPLE {example_num}: {title}")
    print(f"{'-'*80}")


def demonstrate_switch_index_calculation():
    """Demonstrate how switch_index is calculated from patterns."""
    print_section("PART 1: Switch Index Calculation from Patterns")
    
    examples = [
        ("C18-E1", "18 Cantonese words, then 1 English word"),
        ("C5-E3-C2", "5 Cantonese, 3 English, 2 Cantonese"),
        ("C10-E2", "10 Cantonese, 2 English"),
        ("C3-E1-C5", "3 Cantonese, 1 English, 5 Cantonese"),
    ]
    
    for pattern, description in examples:
        print(f"\nPattern: {pattern} ({description})")
        segments = parse_pattern_segments(pattern)
        print(f"  Segments: {segments}")
        
        switch_index = get_switch_index(pattern)
        print(f"  switch_index = {switch_index}")
        
        if switch_index >= 0:
            print(f"  => Switch word is at position {switch_index} (0-based)")
            print(f"  => Words 0 to {switch_index-1} are Cantonese")
            print(f"  => Word {switch_index} is the first English word (switch word)")
        else:
            print(f"  => No valid switch index (monolingual or English-first)")


def demonstrate_window_extraction():
    """Demonstrate POS window extraction around switch point."""
    print_section("PART 2: POS Window Extraction Around Switch Point")
    
    # Example: Pattern C18-E1, switch_index = 18
    switch_index = 18
    window_size = 3
    
    # Simulated POS sequence (25 POS tags total)
    pos_sequence = [f"POS{i}" for i in range(25)]
    
    print(f"\nCode-Switched Sentence POS Sequence:")
    print(f"  Length: {len(pos_sequence)}")
    print(f"  switch_index: {switch_index}")
    print(f"  window_size: {window_size}")
    
    pos_window, switch_index_in_window, window_start = extract_pos_window(
        pos_sequence, switch_index, window_size
    )
    
    print(f"\nExtracted Window:")
    print(f"  window_start: {window_start}")
    print(f"  window_end: {window_start + len(pos_window)}")
    print(f"  Window POS tags: {pos_window}")
    print(f"  switch_index_in_window: {switch_index_in_window}")
    print(f"    => Switch is at position {switch_index_in_window} within the window")
    
    # Visual representation
    print(f"\nVisual Representation:")
    print(f"  Full POS sequence indices: {' '.join([f'{i:2d}' for i in range(len(pos_sequence))])}")
    print(f"  Window region:            {' '*window_start*3}{'-'*len(pos_window)*3}")
    print(f"  Switch position:          {' '*switch_index*3}^")


def demonstrate_matched_switch_index():
    """Demonstrate matched switch index calculation."""
    print_section("PART 3: Matched Switch Index Calculation")
    
    # Code-switched sentence
    cs_switch_index = 18
    cs_window_size = 3
    cs_pos_sequence = [f"CS_POS{i}" for i in range(25)]
    
    cs_window, cs_switch_in_window, cs_window_start = extract_pos_window(
        cs_pos_sequence, cs_switch_index, cs_window_size
    )
    
    print(f"\nCode-Switched Sentence:")
    print(f"  switch_index: {cs_switch_index}")
    print(f"  Window: {cs_window}")
    print(f"  switch_index_in_window: {cs_switch_in_window}")
    
    # Monolingual sentence (different length, similar POS structure)
    mono_pos_sequence = [f"MONO_POS{i}" for i in range(20)]
    
    # Simulate finding a match starting at position 10
    best_start_idx = 10
    
    print(f"\nMonolingual Sentence:")
    print(f"  Length: {len(mono_pos_sequence)}")
    print(f"  Best match window starts at: {best_start_idx}")
    print(f"  Matched window: {mono_pos_sequence[best_start_idx:best_start_idx+len(cs_window)]}")
    
    # Calculate matched switch index
    matched_switch_index = calculate_matched_switch_index(
        best_start_idx, cs_switch_in_window, len(mono_pos_sequence)
    )
    
    print(f"\nMatched Switch Index Calculation:")
    print(f"  matched_switch_index = best_start_idx + switch_index_in_window")
    print(f"  matched_switch_index = {best_start_idx} + {cs_switch_in_window}")
    print(f"  matched_switch_index = {matched_switch_index}")
    
    print(f"\nVisual Mapping:")
    print(f"  CS sentence: switch at position {cs_switch_index} (position {cs_switch_in_window} in window)")
    print(f"  Mono sentence: switch at position {matched_switch_index} (position {cs_switch_in_window} in matched window)")


def demonstrate_bert_input():
    """Demonstrate what words are sent to BERT (masked LM) surprisal calculator."""
    print_section("PART 4: BERT-Style (Masked LM) Surprisal Input")
    
    # Example sentence
    words = [f"word{i}" for i in range(25)]
    switch_index = 18
    
    print(f"\nFull Sentence:")
    print(f"  Words: {words}")
    print(f"  switch_index: {switch_index}")
    print(f"  Target word: {words[switch_index]}")
    
    print(f"\nBERT Input Preparation:")
    print(f"  => Input: ALL words (full sentence)")
    print(f"  => Target word at index {switch_index} is MASKED")
    print(f"  => Model predicts masked word using bidirectional context")
    
    # Show what gets masked
    masked_sentence = words.copy()
    masked_sentence[switch_index] = "[MASK]"
    
    print(f"\nInput to BERT:")
    print(f"  Words: {masked_sentence}")
    print(f"  => Word {switch_index} is replaced with [MASK]")
    print(f"  => Model sees: {words[:switch_index]} [MASK] {words[switch_index+1:]}")
    print(f"  => Model predicts word {switch_index} using context from both sides")


def demonstrate_gpt_input():
    """Demonstrate what words are sent to GPT (autoregressive) surprisal calculator."""
    print_section("PART 5: GPT-Style (Autoregressive) Surprisal Input")
    
    # Example sentence
    words = [f"word{i}" for i in range(25)]
    switch_index = 18
    
    print(f"\nFull Sentence:")
    print(f"  Words: {words}")
    print(f"  switch_index: {switch_index}")
    print(f"  Target word: {words[switch_index]}")
    
    print(f"\nGPT Input Preparation:")
    print(f"  => Input: Only words UP TO (but NOT including) target word")
    print(f"  => Target word is NOT included in initial input")
    print(f"  => Model predicts target word given preceding context only")
    
    # Show context only
    context_words = words[:switch_index]
    
    print(f"\nInitial Input to GPT:")
    print(f"  Context words: {context_words}")
    print(f"  => Only words 0 to {switch_index-1} (total: {len(context_words)} words)")
    print(f"  => Target word {words[switch_index]} is NOT included")
    
    print(f"\nPrediction Process:")
    print(f"  1. Input context => Model predicts first token of word {switch_index}")
    print(f"  2. Compare prediction to actual first token of '{words[switch_index]}'")
    print(f"  3. If multi-token word:")
    print(f"     - Append actual first token to input")
    print(f"     - Model predicts second token")
    print(f"     - Continue for all tokens")
    
    # Show iterative process for multi-token word
    print(f"\nExample: If '{words[switch_index]}' has 2 tokens:")
    print(f"  Iteration 1: Input {context_words} => Predict token1")
    print(f"  Iteration 2: Input {context_words} + [token1] => Predict token2")
    print(f"  => Surprisal = sum of surprisals for token1 and token2")


def demonstrate_complete_example():
    """Demonstrate a complete example from pattern to surprisal calculation."""
    print_section("PART 6: Complete Example - Pattern to Surprisal")
    
    # Pattern
    pattern = "C18-E1"
    print(f"\nPattern: {pattern}")
    
    # Calculate switch index
    switch_index = get_switch_index(pattern)
    print(f"switch_index: {switch_index}")
    
    # Create example words
    cs_words = [f"CS_word{i}" for i in range(25)]
    print(f"\nCode-Switched Sentence ({len(cs_words)} words):")
    print(f"  Words: {cs_words}")
    print(f"  Switch word: {cs_words[switch_index]}")
    
    # BERT input
    print(f"\nBERT Input:")
    print(f"  Full sentence: {cs_words}")
    print(f"  Masked word: {cs_words[switch_index]} => [MASK]")
    print(f"  Context: {cs_words[:switch_index]} [MASK] {cs_words[switch_index+1:]}")
    
    # GPT input
    print(f"\nGPT Input:")
    print(f"  Context only: {cs_words[:switch_index]}")
    print(f"  Predict: {cs_words[switch_index]}")
    
    # Matched monolingual
    mono_words = [f"MONO_word{i}" for i in range(20)]
    matched_switch_index = 13
    
    print(f"\nMatched Monolingual Sentence ({len(mono_words)} words):")
    print(f"  Words: {mono_words}")
    print(f"  matched_switch_index: {matched_switch_index}")
    print(f"  Switch word: {mono_words[matched_switch_index]}")
    
    # BERT input for mono
    print(f"\nBERT Input (Monolingual):")
    print(f"  Full sentence: {mono_words}")
    print(f"  Masked word: {mono_words[matched_switch_index]} => [MASK]")
    print(f"  Context: {mono_words[:matched_switch_index]} [MASK] {mono_words[matched_switch_index+1:]}")
    
    # GPT input for mono
    print(f"\nGPT Input (Monolingual):")
    print(f"  Context only: {mono_words[:matched_switch_index]}")
    print(f"  Predict: {mono_words[matched_switch_index]}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("  SWITCH INDEX CALCULATIONS & SURPRISAL INPUT DEMONSTRATION")
    print("="*80)
    
    demonstrate_switch_index_calculation()
    demonstrate_window_extraction()
    demonstrate_matched_switch_index()
    demonstrate_bert_input()
    demonstrate_gpt_input()
    demonstrate_complete_example()
    
    print("\n" + "="*80)
    print("  DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

