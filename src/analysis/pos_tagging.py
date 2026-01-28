"""
Part-of-speech tagging utilities for Cantonese and English.

"""

import logging
import re
from typing import List, Tuple, Optional, Dict
import pycantonese

logger = logging.getLogger(__name__)

# Global spaCy model
_spacy_model = None


def _load_spacy_model():
    """Load spaCy English model on first use."""
    global _spacy_model
    if _spacy_model is None:
        try:
            import spacy
            _spacy_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model (en_core_web_sm)")
        except OSError:
            logger.error(
                "spaCy English model not found. Please install it with: "
                "python -m spacy download en_core_web_sm"
            )
            raise
    return _spacy_model


def pos_tag_cantonese(sentence: str) -> List[Tuple[str, str]]:
    """
    Tag Cantonese text using PyCantonese.
    
    Uses PyCantonese's word segmentation to properly segment the text before POS tagging
    
    Args:
        sentence: Cantonese text string (may be space-separated or unsegmented)
        
    Returns:
        List of (word, pos_tag) tuples
        
    Example:
        >>> pos_tag_cantonese("我係香港人")
        [('我', 'PRON'), ('係', 'VERB'), ('香港人', 'NOUN')]
    """

    if not sentence or not sentence.strip():
        return []
    
    try:
        # Use PyCantonese's segmentation to properly segment Cantonese text
        words = pycantonese.segment(sentence)
        if not words:
            return []
        tagged = pycantonese.pos_tag(words)
        return [(word, pos) for word, pos in tagged if word.strip()]
    except Exception as e:
        logger.warning(f"Error tagging Cantonese sentence '{sentence[:50]}...': {e}")
        return []


def pos_tag_english(sentence: str) -> List[Tuple[str, str]]:
    """
    Tag English text using spaCy.
    
    Args:
        sentence: English text string
        
    Returns:
        List of (word, pos_tag) tuples
        
    Example:
        >>> pos_tag_english("I am from Hong Kong")
        [('I', 'PRON'), ('am', 'AUX'), ('from', 'ADP'), ('Hong', 'PROPN'), ('Kong', 'PROPN')]
    """
    if not sentence or not sentence.strip():
        return []
    
    try:
        nlp = _load_spacy_model()
        doc = nlp(sentence)
        
        return [(token.text, token.pos_) for token in doc if not token.is_space and not token.is_punct]
    except Exception as e:
        logger.warning(f"Error tagging English sentence '{sentence[:50]}...': {e}")
        return []


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


def is_monolingual(pattern: str) -> Optional[str]:
    """
    Determine if a pattern represents a monolingual sentence.
    
    Args:
        pattern: Pattern string like "C10", "E8", or "C5-E3"
        
    Returns:
        'Cantonese' if pure Cantonese, 'English' if pure English,
        None if code-switched
    """

    segments = parse_pattern_segments(pattern)
    languages = {lang for lang, _ in segments}
    
    if len(languages) == 0:
        return None
    elif len(languages) == 1:
        lang = languages.pop()
        return 'Cantonese' if lang == 'C' else 'English'
    else:
        return None


def split_sentence_by_pattern(sentence: str, pattern: str) -> List[Tuple[str, str]]:
    """
    Split a mixed-language sentence according to its pattern.
    
    Args:
        sentence: Space-separated sentence text
        pattern: Pattern like "C5-E3-C2"
        
    Returns:
        List of (text_segment, language) tuples
    """

    words = sentence.split()
    segments = parse_pattern_segments(pattern)
    
    if not segments:
        return []
    
    result = []
    word_idx = 0
    
    for lang, count in segments:
        if word_idx >= len(words):
            break
        
        # Extract the segment
        segment_words = words[word_idx:word_idx + count]
        segment_text = ' '.join(segment_words)
        result.append((segment_text, lang))
        word_idx += count
    
    return result


def pos_tag_mixed_sentence( sentence: str, pattern: str ) -> List[Tuple[str, str, str]]:
    """
    Tag a mixed-language sentence by segmenting and tagging each language portion.
    
    Args:
        sentence: Space-separated sentence text
        pattern: Pattern like "C5-E3-C2"
        
    Returns:
        List of (word, pos_tag, language) tuples
        
    Example:
        >>> pos_tag_mixed_sentence("我 係 local 人", "C2-E1-C1")
        [('我', 'PRON', 'C'), ('係', 'VERB', 'C'), ('local', 'ADJ', 'E'), ('人', 'NOUN', 'C')]
    """
    
    segments = split_sentence_by_pattern(sentence, pattern)
    result = []
    
    for segment_text, lang in segments:
        if lang == 'C':
            tagged = pos_tag_cantonese(segment_text)
            result.extend([(word, pos, 'C') for word, pos in tagged])
        elif lang == 'E':
            tagged = pos_tag_english(segment_text)
            result.extend([(word, pos, 'E') for word, pos in tagged])
    
    return result


def extract_pos_sequence(tagged: List[Tuple]) -> List[str]:
    """
    Extract just the POS tags from tagged output.
    
    Args:
        tagged: List of tuples (word, pos) or (word, pos, lang)
        
    Returns:
        List of POS tag strings
    """
    if not tagged:
        return []
    
    # Handle both (word, pos) and (word, pos, lang) formats
    if len(tagged[0]) == 3:
        return [pos for _, pos, _ in tagged]
    else:
        return [pos for _, pos in tagged]

