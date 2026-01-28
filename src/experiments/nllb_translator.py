"""
NLLB (No Language Left Behind) translator for code-switching.

"""

import logging
import torch
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NLLBTranslator:
    """
    Translator using Meta's NLLB model for English to Cantonese translation.
    
    Language codes:
    - English: eng_Latn
    - Cantonese (Traditional): yue_Hant
    """
    
    # NLLB language codes
    ENGLISH_CODE = "eng_Latn"
    CANTONESE_CODE = "yue_Hant"
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        show_progress: bool = True
    ):
        """
        Initialize NLLB translator.
        
        Args:
            model_name: NLLB model variant
                - facebook/nllb-200-distilled-600M (faster, ~2.4GB)
                - facebook/nllb-200-1.3B (better quality, ~5GB)
                - facebook/nllb-200-3.3B (best quality, ~13GB)
            device: Device to run on ("auto", "cpu", "cuda")
            show_progress: Whether to show progress bars
        """

        self.model_name = model_name
        self.show_progress = show_progress
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing NLLB translator: {model_name}")
        logger.info(f"Device: {self.device}")
        
        import transformers
        transformers.logging.set_verbosity_error()
        
        # Load model and tokenizer
        if show_progress:
            print(f"Loading NLLB model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.ENGLISH_CODE,
            tgt_lang=self.CANTONESE_CODE
        )
        

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=False  # Avoid meta tensor issues
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Restore normal logging
        transformers.logging.set_verbosity_warning()
        
        logger.info("NLLB translator ready!")
        if show_progress:
            print(f" NLLB model loaded successfully on {self.device}")
    
    def translate_english_to_cantonese( self, english_text: str, max_length: int = 512 ) -> str:
        """
        Translate English text to Cantonese.
        
        Args:
            english_text: English text to translate
            max_length: Maximum length of translation
            
        Returns:
            Cantonese translation
        """

        # Validate input
        if not english_text or not english_text.strip():
            logger.warning("Empty or whitespace-only English text provided for translation")
            return ""
        
        # Set source language
        self.tokenizer.src_lang = self.ENGLISH_CODE
        
        # Tokenize input
        inputs = self.tokenizer(
            english_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            # Get the token ID for Cantonese language code
            cantonese_token_id = self.tokenizer.convert_tokens_to_ids(self.CANTONESE_CODE)
            
            # Validate token ID
            if cantonese_token_id is None:
                raise ValueError(
                    f"Could not find token ID for language code '{self.CANTONESE_CODE}'. "
                    f"Please check that the tokenizer supports this language code."
                )
            
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=cantonese_token_id,
                max_length=max_length,
                num_beams=5,  # Beam search for better quality
                early_stopping=True
            )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(
            translated,
            skip_special_tokens=True
        )[0]
        
        return translation.strip()
    
    def translate_code_switched_sentence( self, sentence: str, pattern: str, words: List[str] ) -> Dict:
        """
        Translate a code-switched sentence to fully Cantonese.
        
        Args:
            sentence: Original code-switched sentence
            pattern: Pattern string like "C5-E2-C3"
            words: List of words (should match pattern)
            
        Returns:
            Dictionary with translation details
        """
        # Parse pattern
        segments = self._parse_pattern(pattern)
        
        # Validate pattern matches word count
        total_pattern_words = sum(count for _, count in segments)
        if total_pattern_words != len(words):
            logger.warning(
                f"Pattern word count ({total_pattern_words}) doesn't match sentence word count ({len(words)}) "
                f"for pattern '{pattern}' and sentence '{sentence[:50]}...'"
            )
        
        # Extract segments from words
        word_segments = []
        word_idx = 0
        for lang, count in segments:
            segment_words = words[word_idx:word_idx + count]
            word_segments.append((lang, segment_words, word_idx, word_idx + count))
            word_idx += count
        
        # Translate English segments
        translated_words = []
        segment_translations = []
        
        for lang, segment_words, start_idx, end_idx in word_segments:
            if lang == 'C':
                # Keep Cantonese as-is (preserve spaces)
                cantonese_segment = ' '.join(segment_words)
                translated_words.append(cantonese_segment)
                segment_translations.append({
                    'language': 'Cantonese',
                    'original': cantonese_segment,
                    'translated': cantonese_segment,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            else:  # lang == 'E'
                # Check for empty segments before translation
                if not segment_words:
                    logger.warning(f"Empty English segment in pattern {pattern}, skipping")
                    continue
                
                english_text = ' '.join(segment_words)
                
                if not english_text.strip():
                    logger.warning(f"Empty or whitespace-only English text in segment, skipping")
                    continue
                
                cantonese_translation = self.translate_english_to_cantonese(english_text)
                
                # Add translation (preserve as space-separated)
                translated_words.append(cantonese_translation)
                
                segment_translations.append({
                    'language': 'English',
                    'original': english_text,
                    'translated': cantonese_translation,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        # Combine into full sentence with spaces
        translated_sentence = ' '.join(translated_words)
        
        return {
            'translated_sentence': translated_sentence,
            'original_sentence': sentence,
            'pattern': pattern,
            'segments': segment_translations
        }
    
    def translate_batch( self, sentences: List[str], patterns: List[str], words_list: List[List[str]] ) -> List[Dict]:
        """
        Translate multiple code-switched sentences.
        
        Args:
            sentences: List of code-switched sentences
            patterns: List of pattern strings
            words_list: List of word lists
            
        Returns:
            List of translation dictionaries
        """
        
        if not (len(sentences) == len(patterns) == len(words_list)):
            raise ValueError("sentences, patterns, and words_list must have same length")
        
        results = []
        
        # Create progress bar if enabled
        iterator = zip(sentences, patterns, words_list)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=len(sentences),
                desc="Translating (NLLB)",
                unit="sentence",
                ncols=80
            )
        
        for sentence, pattern, words in iterator:
            result = self.translate_code_switched_sentence(sentence, pattern, words)
            results.append(result)
        
        return results
    
    def _parse_pattern(self, pattern: str) -> List[Tuple[str, int]]:
        """
        Parse pattern string into segments.
        
        Args:
            pattern: Pattern like "C5-E2-C3"
            
        Returns:
            List of (language, count) tuples
        """
        segments = []
        for segment in pattern.split('-'):
            lang = segment[0]
            count = int(segment[1:])
            segments.append((lang, count))
        return segments
