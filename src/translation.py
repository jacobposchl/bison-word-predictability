'''
Main goal is to translate from code-switch (Matrix: Cantonese + Embedded: English) to pure Cantonese.

1. Take full code switch sentence as input
2. Use GPT to translate english parts to Cantonese
3. Return full Cantonese sentence as output
'''

# ***IMPORTANT*** | NEEDS TO BE REFACTORED TO USE CONFIG FILE NOT HARDCODED
# ***IMPORTANT*** | NEEDS TO ALLOW FOR OPENAI API KEY TO INPUT AS AN ARGUMENT
# ***IMPORTANT*** | NEEDS TO ACTUALLY WORK IN CONGRUENCY WITH MY CSV FILES FROM PREPROCESSING

import os
import json
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import openai

logger = logging.getLogger(__name__)

class TranslationCache:
    '''
    File-based cache for translations to avoid reundant API calls
    '''

    def __init__(self,
                cache_dir: str = "cache/translations"):
        '''
        Docstring for __init__
        
        :param self: self
        :param cache_dir: dir to store cache files
        :type cache_dir: str
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

    Translates English segments into Cantonese-English code-switched sentences to produce a monolingual Cantonese version

    '''

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 use_cache: bool = True,
                 cache_dir: str = "cache/translations"
                ):
        '''
        Docstring for __init__
        
        :param api_key: OpenAI API key
        :type api_key: Optional[str]
        :param model: which model for translation
        :type model: str
        '''
        if api_key:
            openai.api_key = api_key
        else:
            logger.warning("NO API KEY FOUND!")


        self.model = model
        self.use_cache = use_cache
        self.cache = TranslationCache(cache_dir) if use_cache else None

        logger.info(f"Initialized translator with model: {model}")

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
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in English to Cantonese translation. Provide natural, colloquial translations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=200
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
        for sentence, pattern, words in zip(sentences, patterns, words_list):
            result = self.translate_code_switched_sentence(sentence, pattern, words)
            results.append(result)
        
        return results