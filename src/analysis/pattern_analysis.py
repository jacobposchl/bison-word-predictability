"""
Code-switching pattern analysis.

This module handles processing sentences, identifying matrix languages,
and building code-switching patterns with and without fillers.
"""

from typing import List, Dict, Tuple, Optional
import os
import logging
import pycantonese
from ..data.eaf_processor import load_eaf_file, get_main_tier, parse_participant_info, get_all_eaf_files
from ..core.tokenization import per_word_times
from ..core.text_cleaning import is_filler, clean_word, has_content, split_on_internal_dashes

logger = logging.getLogger(__name__)


def _is_cjk_character(char: str) -> bool:
    """
    Check if a character is a CJK (Chinese/Japanese/Korean) character.
    
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
        language_code is 'C' for Cantonese (CJK) or 'E' for English (ASCII)
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


def tokenize_main_tier_sentence(start: int, end: int, text: str) -> List[Tuple[float, str, str]]:
    """
    Tokenize a main tier sentence and assign language labels based on script.
    
    This handles mixed text with inconsistent spacing by using script-based segmentation.
    
    Args:
        start: Start time (ms)
        end: End time (ms)
        text: Sentence text from main tier (may have mixed scripts, inconsistent spacing)
        
    Returns:
        List of (timestamp, word, language) tuples
    """
    # First, segment by script boundaries
    script_segments = segment_by_script(text)
    
    if not script_segments:
        return []
    
    all_tokens = []
    
    for segment_text, lang in script_segments:
        if not segment_text.strip():
            continue
        
        # Tokenize based on language
        if lang == 'C':
            # Cantonese: use PyCantonese segmentation
            raw_tokens = pycantonese.segment(segment_text)
        else:
            # English: split by whitespace
            raw_tokens = segment_text.split()
        
        # Clean and process tokens (similar to tokenize_annotation)
        for w in raw_tokens:
            if not w.strip():
                continue
            
            # Split on ellipses
            ellipsis_parts = []
            if '...' in w:
                ellipsis_parts = w.split('...')
            elif '…' in w:
                ellipsis_parts = w.split('…')
            else:
                ellipsis_parts = [w]
            
            for part in ellipsis_parts:
                if not part.strip():
                    continue
                
                # Split on commas
                comma_parts = part.split(',') if ',' in part else [part]
                
                for comma_part in comma_parts:
                    if not comma_part.strip():
                        continue
                    
                    # Split on internal dashes
                    subparts = split_on_internal_dashes(comma_part)
                    if not subparts:
                        continue
                    
                    for piece in subparts:
                        cw = clean_word(piece)
                        if cw and has_content(cw):
                            all_tokens.append((cw, lang))
    
    if not all_tokens:
        return []
    
    # Assign timestamps
    times = per_word_times(start, end, len(all_tokens))
    return [(t, tok, lang) for t, (tok, lang) in zip(times, all_tokens)]




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


def process_sentence_from_main_tier(
    main_annotation: Tuple[int, int, str]
) -> Optional[Dict]:
    """
    Process a main tier sentence annotation and create sentence data.
    
    This function processes sentences from main tier annotations using script-based
    segmentation to identify Cantonese and English words without relying on subtiers.
    
    Args:
        main_annotation: (start_time, end_time, text) tuple from main tier
        
    Returns:
        Dictionary with sentence data, or None if no words found
    """
    start, end, original_text = main_annotation
    
    # Tokenize the sentence using script-based segmentation
    sentence_words = tokenize_main_tier_sentence(start, end, original_text)
    
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
        # Check if word is a filler in EITHER language
        # This handles cases where English fillers appear in Cantonese segments
        # or vice versa (e.g., "eh" in a Cantonese annotation)
        is_english_filler = is_filler(word, 'E')
        is_cantonese_filler = is_filler(word, 'C')
        
        if is_english_filler or is_cantonese_filler:
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
        
        # Build reconstructed text WITHOUT fillers
        sentence_data['reconstructed_text_without_fillers'] = ' '.join(word for _, word, _ in content_items)
    else:
        # Edge case: the entire sentence was just fillers with no real content
        sentence_data['pattern_content_only'] = 'FILLER_ONLY'
        sentence_data['matrix_language'] = 'UNKNOWN'
        sentence_data['reconstructed_text_without_fillers'] = ''
    
    # Store metadata about fillers for later analysis
    sentence_data['filler_count'] = len(filler_positions)
    sentence_data['filler_info'] = filler_positions
    sentence_data['has_fillers'] = len(filler_positions) > 0
    
    # For main analysis, use the content-only pattern
    sentence_data['pattern'] = sentence_data['pattern_content_only']
    
    return sentence_data


def process_all_files(
    data_path: str
) -> List[Dict]:
    """
    Process all EAF files in the data directory using main tier sentence boundaries.
    
    This function uses main tier annotations directly as sentence boundaries,
    then extracts the relevant Cantonese and English subtier annotations for each sentence.
    
    Note: min_sentence_words filtering is applied later in export functions,
    after filler removal, to ensure accurate word counts.
    
    Args:
        data_path: Path to directory containing EAF files
        
    Returns:
        List of sentence data dictionaries
    """
    eaf_files = get_all_eaf_files(data_path)
    all_sentence_patterns = []
    
    logger.info(f"Found {len(eaf_files)} EAF files to process")
    logger.info("Using main tier annotations for sentence boundaries")
    
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
            
            # Get main tier annotations (sentences)
            try:
                main_annotations = eaf.get_annotation_data_for_tier(main_tier)
            except KeyError as e:
                logger.warning(f"Skipping {eaf_file} - main tier not found: {e}")
                continue
            
            # Process each main tier annotation as a sentence
            file_sentence_count = 0
            for main_annotation in main_annotations:
                sentence_data = process_sentence_from_main_tier(main_annotation)
                
                # Only keep sentences with at least some words (basic check)
                # Full min_sentence_words filtering happens AFTER filler removal in export functions
                if sentence_data and len(sentence_data['items']) > 0:
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


