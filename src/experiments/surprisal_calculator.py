'''
Core surprisal calculation using a language model (in this case, hon9kon9ize/bert-large-cantonese)

1. Load the pre-trained model
2. Given a cantonese sentence + word position, calculate surprisal for that word
3. Return surprisal value

'''

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class CantoneseSurprisalCalculator:
    '''
    Class for word-level surprisal calculation using Cant BERT
    '''

    def __init__(self,
                model_name: str = "hon9kon9ize/bert-large-cantonese",
                device: str = None):
        '''
        Docstring for __init__
        
        :param model_name: HuggingFace model identifier
        :type model_name: str
        :param device: device to use 
        :type device: str
        '''

        logger.info(f"Loading model: {model_name}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Using {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval() 

        logger.info("Model Loaded!")

    def _align_word_to_tokens(self,
                              words: List[str],
                              word_index: int,
                              ) -> Tuple[List[int], List[str]]:
        '''

        Align a word to its corresponding BERT subword tokens.
        
        :param words: list of words in the given sentence
        :type words: List[str]
        :param word_index: position of the target word for surprisal calculation
        :type word_index: int
        '''

        sentence = "".join(words)
        target_word = words[word_index]

        char_start = sum(len(w) for w in words[:word_index])
        char_end = char_start + len(target_word)

        encoding = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        token_ids = encoding['input_ids'][0].tolist()

        token_indices = []
        current_char_pos = 0

        for i, token_id in enumerate(token_ids[1: -1], start = 1): #Skip [CLS] and [SEP]
            token_str = self.tokenizer.decode([token_id])
            token_len = len(token_str)
            token_start = current_char_pos
            token_end = current_char_pos + token_len

            if token_start < char_end and token_end > char_start:
                token_indices.append(i)
            
            current_char_pos += token_len

        token_strings = [self.tokenizer.decode([token_ids[i]]) for i in token_indices]

        return token_indices, token_strings

    def calculate_surprisal(self,
                            word_index: int,
                            words: List[str] = None,
                            ) -> Dict:
        '''
        Calculate the surprisal for a word at a given position in the sentence
        
        :param self: self
        :param word_index: index of the target word to measure surprisal (0-based)
        :type word_index: int
        :param words: pre-segmented words list
        :type words: List[str]
        '''

        if words is None:
            raise ValueError("The 'words' list must contain words")

        if word_index < 0 or word_index >= len(words):
            raise ValueError("Your word index is out of bounds")
        
        target_word = words[word_index]

        token_indices, token_strings = self._align_word_to_tokens(words, word_index)

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'word': target_word,
                'tokens': [],
                'token_surprisals': []
            }

        token_surprisals = []
        token_probs = []

        sentence = "".join(words)

        for token_idx in token_indices:
            tokens = self.tokenizer.tokenize(sentence, add_special_tokens = True)
            input_ids = self.tokernizer.convert_tokens_to_ids(tokens)

            original_token_id = input_ids[token_idx]
            input_ids[token_idx] = self.tokenizer.mask_token_id

            input_ids_tensor = torch.tensor([input_ids]).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids_tensor)
                predictions = outputs.logits
            
            masked_token_logits = predictions[0, token_idx, :]
            probs = torch.softmax(masked_token_logits, dim = 0)

            token_prob = probs[original_token_id].item()

            if token_prob > 0:
                token_surprisal = -np.log2(token_prob)
            else:
                logger.warning(f"Warning: infinite surprisal, token_prob <= 0 | token_prob: {token_prob}")
                token_surprisal = float('inf')

            token_surprisals.append(token_surprisal)
            token_probs.append(token_prob)

        total_surprisal = sum(token_surprisals)
        total_prob = np.prod(token_probs)

        return {
            'surprisal': total_surprisal,
            'probability': total_prob,
            'word': target_word,
            'tokens': token_strings,
            'token_surprisals': token_surprisals,
            'num_tokens': len(token_indices)
        }
