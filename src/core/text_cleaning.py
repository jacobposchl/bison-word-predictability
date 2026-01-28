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
DASH_CLASS = '[' + ''.join(UNICODE_DASHES) + r'\-]'
DASH_RE = re.compile(DASH_CLASS)

# Common filler words in English
ENGLISH_FILLERS = {
    'uh', 'um', 'uhm', 'er', 'err', 'ah', 'eh', 'mm', 'hm', 'hmm', 'mhm', 'ehm',
    'uh-huh', 'mm-hmm', 'uh-uh', 'mm-mm', 'em', 'emm', 'ehh',
    'umm', 'ummm', 'uhh', 'mmm', 'huh', 'oh', 'ohh', 'ohhh', 'ummmm', 'uhmm'
}

# Common filler words in Cantonese
CANTONESE_FILLERS = {
    '呃', '嗯', '啊', '哦', '唔', '吖', '哎', '咦'
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
    
    # Remove ellipses (three dots or Unicode ellipsis) from anywhere
    cleaned = cleaned.replace('...', '')
    cleaned = cleaned.replace('…', '')  # Unicode ellipsis
    
    cleaned = cleaned.strip()
    
    if not cleaned or is_annotation_marker(cleaned):
        return None
    
    return cleaned


def is_filler(word: str, lang: str) -> bool:
    """
    Determine if a word is a filler rather than meaningful content.
    
    Args:
        word: The cleaned word string
        lang: The language code ('C' for Cantonese or 'E' for English)
        
    Returns:
        True if the word is a filler, False otherwise
    """

    if not word:
        return False
    
    # Normalize to lowercase for comparison
    word_lower = word.lower()
    
    if lang == 'E':
        # check against filler list
        return word_lower in ENGLISH_FILLERS
    elif lang == 'C':
        # check against filler list
        return word in CANTONESE_FILLERS
    
    return False


def remove_fillers_from_text(text: str, lang: str = None) -> str:
    """
    Remove filler words from a sentence.
    
    Args:
        text: The sentence text to clean
        lang: Language code ('C' for Cantonese, 'E' for English).
              If None, removes both English and Cantonese fillers.
    
    Returns:
        Text with filler words removed
    """
    
    if not text:
        return text
    
    # Determine which fillers to remove
    if lang == 'C':
        fillers_to_remove = CANTONESE_FILLERS
    elif lang == 'E':
        fillers_to_remove = ENGLISH_FILLERS
    else:
        # Remove both if language not specified
        fillers_to_remove = ENGLISH_FILLERS | CANTONESE_FILLERS
    
    # Split, filter, and rejoin
    words = text.split()
    filtered_words = [
        w for w in words 
        if w.lower() not in fillers_to_remove and w not in fillers_to_remove
    ]
    
    return ' '.join(filtered_words)

