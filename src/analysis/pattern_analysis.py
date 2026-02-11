"""
Code-switching pattern analysis.

This module handles processing sentences, identifying matrix languages,
and building code-switching patterns with and without fillers.
"""

from typing import List, Dict, Tuple, Optional
import os
import logging
import pycantonese
from tqdm import tqdm

from ..core.text_cleaning import is_filler, clean_word, has_content, split_on_internal_dashes
from ..utils.text_validation import _is_cjk_character, _is_ascii_alphabetic, segment_by_script
from ..utils.data_helpers import find_switch_points

logger = logging.getLogger(__name__)


def tokenize_main_tier_sentence(text: str) -> List[Tuple[str, str]]:
    """
    Tokenize a sentence and assign language labels based on script.
    
    
    Args:
        text: Sentence text from main tier
        
    Returns:
        List of (word, language) tuples
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
        
        # Clean and process tokens
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
    
    return all_tokens




def identify_matrix_language(items: List[Tuple[str, str]]) -> str:
    """
    Identify the matrix language based on the MLF (Matrix Language Framework) model.
    
    The matrix language is determined by:
    1. The language with more words (higher count)
    2. If equal, return "Equal"
    
    Args:
        items: List of (word, language_code) tuples
        
    Returns:
        'Cantonese', 'English', or 'Equal'
    """

    # Count words in each language
    cant_count = sum(1 for _, lang in items if lang == 'C')
    eng_count = sum(1 for _, lang in items if lang == 'E')
    
    if cant_count > eng_count:
        return 'Cantonese'
    elif eng_count > cant_count:
        return 'English'
    
    return "Equal"


def process_sentence_from_main_tier(main_annotation: Tuple[int, int, str]) -> Optional[Dict]:
    """
    Process a main tier sentence annotation and create sentence data.
    
    Args:
        main_annotation: (start_time, end_time, text) tuple from main tier
        
    Returns:
        Dictionary with sentence data, or None if no words found
    """

    start, end, original_text = main_annotation
    
    # Tokenize the sentence using script-based segmentation
    sentence_words = tokenize_main_tier_sentence(original_text)
    
    if not sentence_words:
        return None
    
    # Reconstructed text from the same tokens
    reconstructed_text = ' '.join(tok for tok, _ in sentence_words)
    
    # Counts (before filler removal)
    cant_count = sum(1 for _, lang in sentence_words if lang == 'C')
    eng_count = sum(1 for _, lang in sentence_words if lang == 'E')
    
    return {
        'start_time': start,
        'end_time': end,
        'text': original_text,
        'reconstructed_text': reconstructed_text,
        'items': sentence_words,
        'total_words': len(sentence_words),
        'cant_words': cant_count,
        'eng_words': eng_count
        # Pattern and matrix_language will be set in build_patterns_with_fillers after filler removal
    }


def build_patterns_with_fillers(sentence_data: Dict) -> Dict:
    """
    Build code-switching pattern after removing fillers.
    
    This function:
    1. Identifies and removes fillers from the sentence
    2. Creates the pattern from content words only (excluding fillers)
    3. Determines matrix language from content words only
    
    Args:
        sentence_data: Dictionary containing sentence data with 'items' key
        
    Returns:
        Updated sentence_data dictionary with pattern and matrix_language fields
    """
    
    items = sentence_data['items']
    
    # Identify which items are fillers and which are content
    filler_positions = []
    content_items = []
    
    for idx, (word, lang) in enumerate(items):
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
            content_items.append((word, lang))
    
    # Build pattern from content words only (excluding fillers)
    if content_items:
        seq_content = []
        current_lang = content_items[0][1]
        count = 0
        
        for _, lang in content_items:
            if lang == current_lang:
                count += 1
            else:
                seq_content.append(f"{current_lang}{count}")
                current_lang = lang
                count = 1
        seq_content.append(f"{current_lang}{count}")
        
        sentence_data['pattern'] = '-'.join(seq_content)
        
        # Determine matrix language based on CONTENT words only (excluding fillers)
        sentence_data['matrix_language'] = identify_matrix_language(content_items)
        
        # Build reconstructed text WITHOUT fillers
        sentence_data['reconstructed_text_without_fillers'] = ' '.join(word for word, _ in content_items)
    else:
        # Edge case: the entire sentence was just fillers with no real content
        sentence_data['pattern'] = 'FILLER_ONLY'
        sentence_data['matrix_language'] = 'UNKNOWN'
        sentence_data['reconstructed_text_without_fillers'] = ''
    
    # Store metadata about fillers for later analysis
    sentence_data['filler_count'] = len(filler_positions)
    sentence_data['filler_info'] = filler_positions
    sentence_data['has_fillers'] = len(filler_positions) > 0
    
    return sentence_data


def process_all_files( data_path: str ) -> List[Dict]:
    """
    Process all EAF files in the data directory.
    
    Args:
        data_path: Path to directory containing EAF files
        
    Returns:
        List of sentence data dictionaries
    """

    eaf_files = get_all_eaf_files(data_path)
    all_sentence_patterns = []
    
    logger.info(f"Found {len(eaf_files)} EAF files to process")
    
    for eaf_file in tqdm(eaf_files, desc="Processing EAF files"):
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
            for main_annotation in main_annotations:
                sentence_data = process_sentence_from_main_tier(main_annotation)
                
                # Only keep sentences with at least some words (basic check)
                # Full min_sentence_words filtering happens after filler removal in export functions
                if sentence_data and len(sentence_data['items']) > 0:
                    # Add participant info to the sentence data
                    sentence_data['participant_id'] = participant_id
                    sentence_data['group_code'] = group_code
                    sentence_data['group'] = group
                    sentence_data['file_name'] = eaf_file
                    
                    # Build patterns with filler detection
                    sentence_data = build_patterns_with_fillers(sentence_data)
                    
                    all_sentence_patterns.append(sentence_data)
            
        except Exception as e:
            logger.error(f"Error processing {eaf_file}: {e}")
            continue
    
    logger.info(f"Total sentences collected: {len(all_sentence_patterns)}")
    
    return all_sentence_patterns


