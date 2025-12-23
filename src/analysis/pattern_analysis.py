"""
Code-switching pattern analysis.

This module handles processing sentences, identifying matrix languages,
and building code-switching patterns with and without fillers.
"""

from typing import List, Dict, Tuple, Optional
import os
import logging
from ..data.eaf_processor import load_eaf_file, get_main_tier, extract_tiers, parse_participant_info, get_all_eaf_files
from ..core.tokenization import tokenize_annotation
from ..core.text_cleaning import is_filler

logger = logging.getLogger(__name__)


def identify_matrix_language(items: List[Tuple[float, str, str]]) -> str:
    """
    Identify the matrix language based on the MLF (Matrix Language Framework) model.
    
    The matrix language is determined by:
    1. The language with more words (higher count)
    2. If equal, return "Equal"
    
    Args:
        items: List of (timestamp, word, language_code) tuples
        
    Returns:
        'Cantonese', 'English', or 'Equal'
    """
    # Count words in each language
    cant_count = sum(1 for _, _, lang in items if lang == 'C')
    eng_count = sum(1 for _, _, lang in items if lang == 'E')
    
    # Rule 1: Language with more words
    if cant_count > eng_count:
        return 'Cantonese'
    elif eng_count > cant_count:
        return 'English'
    
    # Rule 2: If equal, return "Equal"
    return "Equal"


def process_sentence_with_reconstruction(
    ach_sentence: Tuple[int, int, str],
    cant_annotations: List[Tuple[int, int, str]],
    eng_annotations: List[Tuple[int, int, str]],
    buffer: float = 0.050
) -> Optional[Dict]:
    """
    Process a sentence and create both the original text and reconstructed text.
    
    How it works:
    1. Use the main tier sentence's start/end times as boundaries
    2. Collect all annotation words that fall within those boundaries
    3. Sort them by timestamp to preserve order
    4. Concatenate to build the reconstructed sentence
    
    Args:
        ach_sentence: Tuple of (start_time, end_time, original_text)
        cant_annotations: List of (start, end, text) tuples for Cantonese tier
        eng_annotations: List of (start, end, text) tuples for English tier
        buffer: Time buffer in seconds for overlap detection
        
    Returns:
        Dictionary with sentence data, or None if no words found
    """
    start, end, original_text = ach_sentence
    
    def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
        """Check if two time intervals overlap."""
        return (a1 > b0) and (a0 < b1)
    
    s0, s1 = start - buffer, end + buffer
    sentence_words = []
    
    # Collect per-word tokens from overlapping annotations
    for tier, lang in ((cant_annotations, 'C'), (eng_annotations, 'E')):
        for a_start, a_end, a_text in tier:
            if overlaps(a_start, a_end, s0, s1):
                sentence_words.extend(tokenize_annotation(a_start, a_end, a_text, lang))
    
    # Sort by time to preserve spoken order
    sentence_words.sort(key=lambda x: x[0])
    
    if not sentence_words:
        return None
    
    # Reconstructed text from the same tokens used for the pattern
    reconstructed_text = ' '.join(tok for _, tok, _ in sentence_words)
    
    # Build pattern from runs of language labels
    pattern_parts = []
    current_lang = sentence_words[0][2]
    current_count = 0
    for _, _, lang in sentence_words:
        if lang == current_lang:
            current_count += 1
        else:
            pattern_parts.append(f"{current_lang}{current_count}")
            current_lang = lang
            current_count = 1
    pattern_parts.append(f"{current_lang}{current_count}")
    pattern = '-'.join(pattern_parts)
    
    # Matrix language
    matrix_language = identify_matrix_language(sentence_words)
    
    # Counts
    cant_count = sum(1 for _, _, lang in sentence_words if lang == 'C')
    eng_count = sum(1 for _, _, lang in sentence_words if lang == 'E')
    
    return {
        'start_time': start,
        'end_time': end,
        'text': original_text,  # original ACH text for reference
        'reconstructed_text': reconstructed_text,
        'pattern': pattern,
        'matrix_language': matrix_language,
        'items': sentence_words,
        'total_words': len(sentence_words),
        'cant_words': cant_count,
        'eng_words': eng_count
    }


def build_patterns_with_fillers(sentence_data: Dict) -> Dict:
    """
    Build code-switching patterns both with and without fillers.
    
    Args:
        sentence_data: Dictionary containing sentence data with 'items' key
        
    Returns:
        Updated sentence_data dictionary with pattern fields added
    """
    items = sentence_data['items']
    
    # First, identify which items are fillers and which are content
    filler_positions = []
    content_items = []
    
    for idx, (timestamp, word, lang) in enumerate(items):
        if is_filler(word, lang):
            # This word is a hesitation marker, not a real code-switch
            filler_positions.append({
                'position': idx,
                'word': word,
                'lang': lang
            })
        else:
            # This is real linguistic content, keep it
            content_items.append((timestamp, word, lang))
    
    # Build the ORIGINAL pattern (includes fillers as if they were real words)
    seq_with_fillers = []
    if items:
        current_lang = items[0][2]
        count = 0
        
        for timestamp, word, lang in items:
            if lang == current_lang:
                count += 1
            else:
                seq_with_fillers.append(f"{current_lang}{count}")
                current_lang = lang
                count = 1
        seq_with_fillers.append(f"{current_lang}{count}")
        
        sentence_data['pattern_with_fillers'] = '-'.join(seq_with_fillers)
    else:
        sentence_data['pattern_with_fillers'] = ''
    
    # Build the CONTENT-ONLY pattern (excludes fillers)
    if content_items:  # Make sure we have content after removing fillers
        seq_content = []
        current_lang = content_items[0][2]
        count = 0
        
        for timestamp, word, lang in content_items:
            if lang == current_lang:
                count += 1
            else:
                seq_content.append(f"{current_lang}{count}")
                current_lang = lang
                count = 1
        seq_content.append(f"{current_lang}{count}")
        
        sentence_data['pattern_content_only'] = '-'.join(seq_content)
        
        # Determine matrix language based on CONTENT words only (excluding fillers)
        sentence_data['matrix_language'] = identify_matrix_language(content_items)
    else:
        # Edge case: the entire sentence was just fillers with no real content
        sentence_data['pattern_content_only'] = 'FILLER_ONLY'
        sentence_data['matrix_language'] = 'UNKNOWN'
    
    # Store metadata about fillers for later analysis
    sentence_data['filler_count'] = len(filler_positions)
    sentence_data['filler_info'] = filler_positions
    sentence_data['has_fillers'] = len(filler_positions) > 0
    
    # For main analysis, use the content-only pattern
    sentence_data['pattern'] = sentence_data['pattern_content_only']
    
    return sentence_data


def process_all_files(
    data_path: str,
    buffer_ms: float = 0.050,
    min_sentence_words: int = 2
) -> List[Dict]:
    """
    Process all EAF files in the data directory.
    
    Args:
        data_path: Path to directory containing EAF files
        buffer_ms: Time buffer in seconds for sentence overlap detection
        min_sentence_words: Minimum number of words to keep a sentence
        
    Returns:
        List of sentence data dictionaries
    """
    eaf_files = get_all_eaf_files(data_path)
    all_sentence_patterns = []
    
    logger.info(f"Found {len(eaf_files)} EAF files to process")
    
    for file_idx, eaf_file in enumerate(eaf_files, 1):
        file_path = os.path.join(data_path, eaf_file)
        
        try:
            eaf = load_eaf_file(file_path)
            
            # Get participant tier
            main_tier = get_main_tier(eaf)
            if not main_tier:
                logger.warning(f"Skipping {eaf_file} - no participant tier found")
                continue
            
            # Parse participant info
            participant_info = parse_participant_info(main_tier)
            participant_id = main_tier
            group_code = participant_info['group_code']
            group = participant_info['group']
            
            # Get annotations
            ach_sentences = eaf.get_annotation_data_for_tier(main_tier)
            cant_c, eng_c = extract_tiers(eaf, main_tier)
            
            # Process each sentence
            file_sentence_count = 0
            for ach_sentence in ach_sentences:
                sentence_data = process_sentence_with_reconstruction(
                    ach_sentence, cant_c, eng_c, buffer=buffer_ms
                )
                
                # Only keep sentences with multiple words
                if sentence_data and len(sentence_data['items']) >= min_sentence_words:
                    # Add participant info to the sentence data
                    sentence_data['participant_id'] = participant_id
                    sentence_data['group_code'] = group_code
                    sentence_data['group'] = group
                    sentence_data['file_name'] = eaf_file
                    
                    # Build patterns with filler detection
                    sentence_data = build_patterns_with_fillers(sentence_data)
                    
                    all_sentence_patterns.append(sentence_data)
                    file_sentence_count += 1
            
            logger.info(
                f"Processed {file_idx}/{len(eaf_files)}: {eaf_file} - "
                f"{file_sentence_count} sentences"
            )
            
        except Exception as e:
            logger.error(f"Error processing {eaf_file}: {e}")
            continue
    
    logger.info(f"Total sentences collected: {len(all_sentence_patterns)}")
    return all_sentence_patterns


def find_switch_positions(pattern: str) -> List[int]:
    """
    Find word indices where code-switches occur.
    
    Args:
        pattern: Pattern like "C5-E2-C3"
        
    Returns:
        List of switch positions (word indices where switch happens)
    """
    segments = pattern.split('-')
    positions = []
    current_pos = 0
    
    for i in range(len(segments) - 1):
        lang, count = segments[i][0], int(segments[i][1:])
        current_pos += count
        positions.append(current_pos)
    
    return positions


