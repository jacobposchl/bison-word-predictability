'''
Main goal is to translate from code-switch (Matrix: Cantonese + Embedded: English) to pure Cantonese.

Workflow:
1. Take code-switched sentence from CSV (reconstructed_sentence field)
2. Parse the pattern field (e.g., "C5-E2-C3") to identify language segments
3. Use GPT to translate English segments to Cantonese
4. Return full Cantonese sentence

Example:
  Input: reconstructed_sentence="我哋 go 咗 公園", pattern="C2-E1-C2"
  Output: translated_sentence="我哋去咗公園" (English "go" translated to Cantonese)
'''

import os
import json
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TranslationCache:
    '''
    File-based cache for translations to avoid redundant API calls
    '''

    def __init__(self, cache_dir: str):
        '''
        Initialize translation cache.
        
        Args:
            cache_dir: Directory to store cache files
        '''
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _make_key(self, text: str, context: str = "") -> str:
        """Create cache key from text and context."""
        combined = f"{text}||{context}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get(self, text: str, context: str = "") -> Optional[str]:
        """Get cached translation."""
        key = self._make_key(text, context)
        return self.cache.get(key)
    
    def set(self, text: str, translation: str, context: str = ""):
        """Store translation in cache."""
        key = self._make_key(text, context)
        self.cache[key] = translation
        self._save_cache()

class CodeSwitchTranslator:
    '''
    Translator for code-switched sentences.

    Translates English segments in Cantonese-English code-switched sentences to produce monolingual Cantonese versions.
    Works with CSV data structure containing 'reconstructed_sentence' and 'pattern' fields.
    '''

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 use_cache: bool = True,
                 cache_dir: str = "cache/translations",
                 temperature: float = 0.3,
                 max_tokens: int = 200,
                 show_progress: bool = True):
        '''
        Initialize translator.
        
        Args:
            api_key: OpenAI API key (REQUIRED)
            model: OpenAI model to use (default: "gpt-4")
            use_cache: Whether to cache translations (default: True)
            cache_dir: Directory for translation cache (default: "cache/translations")
            temperature: Temperature for API calls (default: 0.3)
            max_tokens: Max tokens for responses (default: 200)
            show_progress: Whether to show progress bar (default: True)
        '''
        # Validate API key
        if not api_key:
            raise ValueError("API key is required. Pass it as an argument for security.")
        
        # Suppress HTTP request logging from httpx (used by openai)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Initialize OpenAI client with API key
        self.client = OpenAI(api_key=api_key)
        logger.info("Initialized with provided API key")
        
        # Set translation parameters
        self.model = model
        self.use_cache = use_cache
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_progress = show_progress
        
        # Initialize cache
        self.cache = TranslationCache(cache_dir) if self.use_cache else None

        logger.info(f"Translator ready: model={self.model}, cache={self.use_cache}")

    def _parse_pattern(self, pattern: str) -> List[Tuple[str, int]]:
        """
        Parse pattern string into segments.
        
        Args:
            pattern: Pattern like "C5-E2-C3"
            
        Returns:
            List of (language, count) tuples: [('C', 5), ('E', 2), ('C', 3)]
        """

        segments = []
        for segment in pattern.split('-'):
            lang = segment[0]
            count = int(segment[1:])
            segments.append((lang, count))
        return segments
    
    def _extract_segments_from_sentence(
        self, 
        words: List[str], 
        pattern: str
    ) -> List[Tuple[str, List[str], int, int]]:
        """
        Extract language segments from word list based on pattern.
        
        Args:
            words: List of words in sentence
            pattern: Pattern string like "C5-E2-C3"
            
        Returns:
            List of (language, word_list, start_idx, end_idx) tuples
        """
        segments = self._parse_pattern(pattern)
        word_segments = []
        word_idx = 0
        
        for lang, count in segments:
            segment_words = words[word_idx:word_idx + count]
            word_segments.append((lang, segment_words, word_idx, word_idx + count))
            word_idx += count
        
        return word_segments
    
    def _translate_english_to_cantonese(
        self, 
        english_text: str,
        cantonese_context: str = ""
    ) -> str:
        """
        Translate English text to Cantonese using GPT-4.
        
        Args:
            english_text: English text to translate
            cantonese_context: Surrounding Cantonese context for better translation
            
        Returns:
            Cantonese translation
        """
        # Check cache first
        if self.use_cache:
            cached = self.cache.get(english_text, cantonese_context)
            if cached:
                logger.debug(f"Cache hit for: {english_text}")
                return cached
        
        # Build prompt
        if cantonese_context:
            prompt = f"""Translate the following English text to Cantonese. 

Context (Cantonese): {cantonese_context}
English text to translate: {english_text}

Provide ONLY the Cantonese translation, no explanations or additional text.
Use natural, colloquial Cantonese that fits the context."""
        else:
            prompt = f"""Translate the following English text to natural, colloquial Cantonese.

English: {english_text}

Provide ONLY the Cantonese translation, no explanations or additional text."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in English to Cantonese translation. Provide natural, colloquial translations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            translation = response.choices[0].message.content.strip()
            
            # Cache the result
            if self.use_cache:
                self.cache.set(english_text, translation, cantonese_context)
            
            logger.debug(f"Translated: {english_text} -> {translation}")
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise
    
    def translate_code_switched_sentence(
        self,
        sentence: str,
        pattern: str,
        words: List[str]
    ) -> Dict:
        """
        Translate a code-switched sentence to fully Cantonese.
        
        Args:
            sentence: Original code-switched sentence
            pattern: Pattern string like "C5-E2-C3"
            words: List of words (should match pattern)
            
        Returns:
            Dictionary with:
                - translated_sentence: Full Cantonese translation
                - original_sentence: Original code-switched sentence
                - pattern: Original pattern
                - segments: List of segment translations
        """
        # Extract segments
        segments = self._extract_segments_from_sentence(words, pattern)
        
        # Translate English segments
        translated_words = []
        segment_translations = []
        
        for lang, segment_words, start_idx, end_idx in segments:
            if lang == 'C':
                # Keep Cantonese as-is
                translated_words.extend(segment_words)
                segment_translations.append({
                    'language': 'Cantonese',
                    'original': ''.join(segment_words),
                    'translated': ''.join(segment_words),
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            else:  # lang == 'E'
                # Translate English to Cantonese
                english_text = ' '.join(segment_words)
                
                # Get surrounding Cantonese context
                context_words = []
                if start_idx > 0:
                    # Get previous Cantonese segment
                    prev_segment = segments[segments.index((lang, segment_words, start_idx, end_idx)) - 1]
                    if prev_segment[0] == 'C':
                        context_words.extend(prev_segment[1][-3:])  # Last 3 words
                
                context = ''.join(context_words) if context_words else ""
                
                # Translate
                cantonese_translation = self._translate_english_to_cantonese(english_text, context)
                
                # For word-level analysis, we need to segment the translation
                # For now, treat translation as single unit (can be refined with PyCantonese)
                translated_words.append(cantonese_translation)
                
                segment_translations.append({
                    'language': 'English',
                    'original': english_text,
                    'translated': cantonese_translation,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
        
        # Combine into full sentence
        translated_sentence = ''.join(translated_words)
        
        return {
            'translated_sentence': translated_sentence,
            'original_sentence': sentence,
            'pattern': pattern,
            'segments': segment_translations
        }
    
    def translate_batch(
        self,
        sentences: List[str],
        patterns: List[str],
        words_list: List[List[str]]
    ) -> List[Dict]:
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
                desc="Translating",
                unit="sentence",
                ncols=80
            )
        
        for sentence, pattern, words in iterator:
            result = self.translate_code_switched_sentence(sentence, pattern, words)
            results.append(result)
        
        return results