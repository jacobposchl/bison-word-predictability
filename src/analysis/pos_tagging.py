"""
Part-of-speech tagging utilities for Cantonese and English.

"""

import logging
import re
from typing import List, Tuple, Optional, Dict
import pycantonese

from ..utils.data_helpers import parse_pattern_segments, split_sentence_by_pattern

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


def extract_pos_sequence(tagged: List[Tuple]) -> List[str]:
    """
    Extract just the POS tags from tagged output.
    
    Args:
        tagged: List of tuples (word, pos) or (word, pos, lang)
        
    Returns:
        List of POS tag strings
    """
    
    assert len(tagged[0]) == 2, "Pycantonese POS tagging error, should return tuple of length 2..."
    
    return [pos for _, pos in tagged]

