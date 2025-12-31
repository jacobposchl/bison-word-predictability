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
        # First, split on ellipses (three dots or Unicode ellipsis)
        # This handles cases like "Um...uh" which should be split into "Um" and "uh"
        ellipsis_parts = []
        if '...' in w:
            ellipsis_parts = w.split('...')
        elif '…' in w:  # Unicode ellipsis
            ellipsis_parts = w.split('…')
        else:
            ellipsis_parts = [w]
        
        # Process each part (after ellipsis split)
        for part in ellipsis_parts:
            if not part.strip():
                continue
            
            # Split on commas in the middle of tokens
            # This handles cases like "eh,eh" which should be split into "eh" and "eh"
            comma_parts = []
            if ',' in part:
                comma_parts = part.split(',')
            else:
                comma_parts = [part]
            
            for comma_part in comma_parts:
                if not comma_part.strip():
                    continue
                # Normalize and split on internal dashes/en-dashes/em-dashes
                subparts = split_on_internal_dashes(comma_part)
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
        sentence: Cantonese sentence string (may be space-separated)
        
    Returns:
        List of words
    """
    # Use PyCantonese's segment function (same as used in tokenize_annotation)
    # This properly segments Cantonese text regardless of input format
    words = pycantonese.segment(sentence)
    return words if words else []

