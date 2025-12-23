"""
Tokenization utilities for processing annotations.

This module handles tokenization of annotations with language-specific splitting
and per-word timestamp calculation.
"""

from typing import List, Tuple
import pycantonese
from .text_cleaning import split_on_internal_dashes, clean_word, has_content


def per_word_times(start: float, end: float, n: int) -> List[float]:
    """
    Calculate per-word timestamps by evenly distributing time across words.
    
    Args:
        start: Start timestamp
        end: End timestamp
        n: Number of words
        
    Returns:
        List of timestamps, one per word
    """
    s, e = float(start), float(end)
    if n <= 1 or e <= s:
        return [s] * max(1, n)
    step = (e - s) / n
    return [s + i * step for i in range(n)]


def tokenize_annotation(start: float, end: float, text: str, lang: str) -> List[Tuple[float, str, str]]:
    """
    Tokenize an annotation into per-word items with per-word timestamps.
    
    Splits based on language:
    - Cantonese: Uses PyCantonese for word segmentation
    - English: Uses whitespace splitting
    
    Also handles dash normalization and splitting, and filters out
    non-content tokens.
    
    Args:
        start: Start timestamp of the annotation
        end: End timestamp of the annotation
        text: Text content of the annotation
        lang: Language code ('C' for Cantonese, 'E' for English)
        
    Returns:
        List of tuples: (timestamp, word, language_code)
    """
    # Split based on language - PyCantonese for Cantonese, whitespace for English
    if lang == 'C':  # Cantonese
        raw_tokens = pycantonese.segment(text)
    else:  # English
        raw_tokens = text.split()
    
    toks = []
    
    for w in raw_tokens:
        # Normalize and split on internal dashes/en-dashes/em-dashes
        subparts = split_on_internal_dashes(w)
        if not subparts:  # either pure dash or empty after split
            continue
        for piece in subparts:
            cw = clean_word(piece)
            if cw and has_content(cw):  # keep only real content
                toks.append(cw)
    
    if not toks:
        return []
    
    times = per_word_times(start, end, len(toks))
    return [(t, tok, lang) for t, tok in zip(times, toks)]


def segment_cantonese_sentence(sentence: str) -> List[str]:
    """
    Segment Cantonese sentence into words using PyCantonese.
    
    Args:
        sentence: Cantonese sentence string
        
    Returns:
        List of words
    """
    # Parse with PyCantonese
    parsed = pycantonese.parse_text(sentence)
    
    # Extract words
    words = []
    for token in parsed:
        if hasattr(token, 'word'):
            words.append(token.word)
        else:
            # Fallback for simple tokens
            words.append(str(token))
    
    return words

