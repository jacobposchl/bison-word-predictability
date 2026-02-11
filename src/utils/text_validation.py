"""
Text validation utilities for code-switching analysis.

This module provides functions for validating text content, detecting English words,
and verifying translation quality.
"""

import re
from typing import Tuple, List, Optional


def is_english_word(word: str) -> bool:
    """
    Check if a word is likely English (contains only ASCII letters).
    
    Args:
        word: Word to check
        
    Returns:
        True if word appears to be English, False otherwise
    """
    if not word:
        return False
    alpha_chars = [c for c in word if c.isalpha()]
    if not alpha_chars:
        return False
    return all(ord(c) < 128 for c in alpha_chars)


def contains_english_words(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains any English words.
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (has_english, english_words_found)
    """
    # Split by whitespace and punctuation to get potential words
    words = re.findall(r'\b\w+\b', text)
    
    english_words = []
    for word in words:
        if is_english_word(word):
            # Filter out very short words that might be false positives
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


def _is_cjk_character(char: str) -> bool:
    """
    Check if a character is a CJK character.
    
    Args:
        char: Single character
        
    Returns:
        True if character is CJK, False otherwise
    """
    code_point = ord(char)
    return (
        (0x4E00 <= code_point <= 0x9FFF) or      # CJK Unified Ideographs
        (0x3400 <= code_point <= 0x4DBF) or      # CJK Extension A
        (0x20000 <= code_point <= 0x2A6DF) or    # CJK Extension B
        (0x2A700 <= code_point <= 0x2B73F) or    # CJK Extension C
        (0x2B740 <= code_point <= 0x2B81F) or    # CJK Extension D
        (0xF900 <= code_point <= 0xFAFF) or      # CJK Compatibility Ideographs
        (0x2F800 <= code_point <= 0x2FA1F)       # CJK Compatibility Ideographs Supplement
    )


def _is_ascii_alphabetic(char: str) -> bool:
    """
    Check if a character is an ASCII alphabetic character.
    
    Args:
        char: Single character
        
    Returns:
        True if character is ASCII letter (a-z, A-Z), False otherwise
    """
    return ord(char) < 128 and char.isalpha()


def segment_by_script(text: str) -> List[Tuple[str, str]]:
    """
    Segment text into runs of same script (CJK vs ASCII).
    
    This handles cases like "我local人" → [("我", "C"), ("local", "E"), ("人", "C")]
    and "我 local 人" → [("我", "C"), ("local", "E"), ("人", "C")]
    
    Args:
        text: Input text that may contain mixed scripts
        
    Returns:
        List of (segment_text, language_code) tuples
        language_code is 'C' for Cantonese or 'E' for English
    """
    if not text:
        return []
    
    segments = []
    current_segment = []
    current_script = None
    
    for char in text:
        # Determine script type for this character
        if _is_cjk_character(char):
            script = 'C'
        elif _is_ascii_alphabetic(char):
            script = 'E'
        else:
            # Punctuation, whitespace, numbers, etc.
            # Attach to previous segment if exists, otherwise skip
            if current_segment:
                current_segment.append(char)
            continue
        
        # Check if script changed
        if current_script is None:
            # First character
            current_script = script
            current_segment = [char]
        elif current_script == script:
            # Same script, continue segment
            current_segment.append(char)
        else:
            # Script changed, finalize current segment and start new one
            if current_segment:
                segments.append((''.join(current_segment), current_script))
            current_script = script
            current_segment = [char]
    
    # Don't forget the last segment
    if current_segment:
        segments.append((''.join(current_segment), current_script))
    
    return segments
