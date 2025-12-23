"""
Text cleaning utilities for processing annotation data.

This module provides functions to clean and validate text from EAF annotations,
including handling of punctuation, annotation markers, and filler words.
"""

import re
import string
from typing import Optional

# Unicode dash characters that should be normalized
UNICODE_DASHES = {
    '\u2010',  # hyphen
    '\u2011',  # non-breaking hyphen
    '\u2012',  # figure dash
    '\u2013',  # en dash
    '\u2014',  # em dash
    '\u2015',  # horizontal bar
    '\u2212',  # minus sign
}

# Regex pattern for matching dashes
DASH_CLASS = '[' + ''.join(UNICODE_DASHES) + r'\-]'  # ASCII hyphen + unicode dashes
DASH_RE = re.compile(DASH_CLASS)

# Common filler words in English
ENGLISH_FILLERS = {
    'uh', 'um', 'er', 'err', 'ah', 'eh', 'mm', 'hmm', 'mhm',
    'uh-huh', 'mm-hmm', 'uh-uh', 'mm-mm', 'em', 'emm', 'ehh'
}

# Common filler words in Cantonese
CANTONESE_FILLERS = {
    '呃', '嗯', '啊', '哦', '唔',  # Common Cantonese hesitation sounds
}


def normalize_dashes(text: str) -> str:
    """
    Map all Unicode dash-like characters to ASCII '-' for consistent handling.
    
    Args:
        text: Input text string
        
    Returns:
        Text with all Unicode dashes normalized to ASCII '-'
    """
    if not isinstance(text, str):
        return text
    return ''.join('-' if ch in UNICODE_DASHES else ch for ch in text)


def split_on_internal_dashes(token: str) -> list[str]:
    """
    Split tokens on internal hyphens/dashes after normalization.
    
    Returns a list of dash-free parts. Pure-dash tokens return empty list.
    
    Examples:
        'love–hate' -> ['love', 'hate']
        '我-哋' -> ['我', '哋']
        '--' -> []
    
    Args:
        token: Token string to split
        
    Returns:
        List of dash-free parts
    """
    t = normalize_dashes(token)
    # If token is only dashes, drop it
    if t and all(ch == '-' for ch in t):
        return []
    # Split anywhere a '-' appears
    parts = [p for p in t.split('-') if p != '']
    return parts


def is_annotation_marker(text: str) -> bool:
    """
    Check if text is an annotation marker like X, XX, XXX, XXXX, etc.
    
    Also checks for patterns like (X), (XX), etc. and lowercase x.
    These represent sensitive information that should be filtered out.
    
    Args:
        text: Text to check
        
    Returns:
        True if the text consists ONLY of annotation markers
    """
    # Remove parentheses and whitespace for checking
    cleaned = text.strip().replace('(', '').replace(')', '').strip()
    
    # Check if it's only X characters (any number of them)
    # The pattern ^X+$ means: start of string, one or more X's, end of string
    if re.match(r'^x+$', cleaned, re.IGNORECASE):
        return True
    
    return False


def is_chinese_punctuation_only(text: str) -> bool:
    """
    Check if text contains ONLY Chinese punctuation marks.
    
    Chinese punctuation includes: 。，、？！：；「」『』（）【】《》〈〉…—·
    
    This is important because Chinese punctuation gets time-stamped in the
    Cantonese tier but should not count as Cantonese words.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains only Chinese punctuation
    """
    chinese_punctuation = '。，、？！：；「」『』（）【】《》〈〉…—·'
    
    # Remove all Chinese punctuation and whitespace
    cleaned = text
    for char in chinese_punctuation:
        cleaned = cleaned.replace(char, '')
    cleaned = cleaned.strip()
    
    # If nothing remains after removing punctuation, it was punctuation-only
    return len(cleaned) == 0


def has_content(text: str) -> bool:
    """
    Content checker that filters out:
    1. Western punctuation (like . , ? ! ; :)
    2. Chinese punctuation (like 。，？！)
    3. Annotation markers (X, XX, XXX, etc.)
    4. Pure whitespace
    
    Returns True only if there's actual linguistic content.
    
    This fixes two major issues:
    - XX being counted as English words (it's an annotation marker)
    - Chinese punctuation being counted as Cantonese words
    
    Args:
        text: Text to check
        
    Returns:
        True if text has real linguistic content
    """
    # First check if it's an annotation marker
    if is_annotation_marker(text):
        return False
    
    # Check if it's only Chinese punctuation
    if is_chinese_punctuation_only(text):
        return False
    
    # Remove all Western and Chinese punctuation and whitespace
    chinese_punctuation = '。，、？！：；「」『』（）【】《》〈〉…—·'
    cleaned = ''.join(char for char in text
                     if char not in string.punctuation
                     and not char.isspace()
                     and char not in chinese_punctuation)
    
    return len(cleaned) > 0


def clean_word(word: str) -> Optional[str]:
    """
    Clean individual words by removing attached punctuation and annotation markup.
    
    This handles both edge punctuation (e.g., "word.") and internal annotation
    markers (e.g., "lo::cal" which represents elongated pronunciation).
    
    Args:
        word: Word string to clean
        
    Returns:
        Cleaned word without punctuation, or None if the word should be filtered out
    """
    # First check if this is an annotation marker like X, XX, XXX
    if is_annotation_marker(word):
        return None
    
    # Remove Western and Chinese punctuation from the word
    chinese_punctuation = '。，、？！：；「」『』（）《》〈〉…—·'
    cleaned = word.strip()
    
    # Remove punctuation from start and end
    # This handles cases like "word." or ",word" or "word，"
    while cleaned and (cleaned[0] in string.punctuation or cleaned[0] in chinese_punctuation):
        cleaned = cleaned[1:]
    while cleaned and (cleaned[-1] in string.punctuation or cleaned[-1] in chinese_punctuation):
        cleaned = cleaned[:-1]
    
    cleaned = cleaned.strip()
    
    # Removes colons from anywhere within the word
    # We replace them with empty string, so "lo::cal" becomes "local"
    cleaned = cleaned.replace(':', '')
    
    if not cleaned or is_annotation_marker(cleaned):
        return None
    
    return cleaned


def is_filler(word: str, lang: str) -> bool:
    """
    Determine if a word is a filler/hesitation marker rather than meaningful content.
    
    The logic here is conservative: we only mark something as a filler if we're
    confident it's a hesitation marker. When in doubt, we treat it as real content.
    
    Args:
        word: The cleaned word string
        lang: The language code ('C' for Cantonese or 'E' for English)
        
    Returns:
        True if the word is a filler, False otherwise
    """
    if not word:
        return False
    
    # Normalize to lowercase for comparison (English is case-insensitive for this)
    word_lower = word.lower()
    
    if lang == 'E':
        # For English, check against our filler list
        return word_lower in ENGLISH_FILLERS
    elif lang == 'C':
        # For Cantonese, check against Cantonese filler list
        # Note: some Cantonese particles can be fillers
        return word in CANTONESE_FILLERS
    
    return False

