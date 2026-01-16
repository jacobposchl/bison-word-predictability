'''
Core surprisal calculation using language models.

Supports two types of models:
1. Masked Language Models (BERT-style)
2. Autoregressive Models (GPT-style)

'''

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from functools import lru_cache
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MaskedLMSurprisalCalculator:
    '''
    Class for word-level surprisal calculation using Masked Language Models (BERT-style).
    Uses bidirectional context by masking the target word.
    '''

    def __init__(self,
                model_name: str,
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
                              sentence: str,
                              words: List[str],
                              word_index: int,
                              ) -> Tuple[List[int], List[str]]:
        '''

        Align a word to its corresponding BERT subword tokens.
        
        :param sentence: the full sentence string to tokenize
        :type sentence: str
        :param words: list of words in the given sentence
        :type words: List[str]
        :param word_index: position of the target word for surprisal calculation
        :type word_index: int
        '''

        target_word = words[word_index]

        # Calculate character position based on how sentence was constructed
        char_start = sum(len(w) for w in words[:word_index])
        char_end = char_start + len(target_word)

        encoding = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
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
                            context: str = None,
                            ) -> Dict:
        '''
        Calculate the surprisal for a word at a given position in the sentence.
        
        :param self: self
        :param word_index: index of the target word to measure surprisal (0-based)
        :type word_index: int
        :param words: pre-segmented words list
        :type words: List[str]
        :param context: optional discourse context (previous sentences)
        :type context: str
        '''

        if words is None:
            raise ValueError("The 'words' list must contain words")

        if word_index < 0 or word_index >= len(words):
            raise ValueError("Your word index is out of bounds")
        
        # Build full input with context if provided
        # Use consistent spacing: no spaces between Cantonese characters
        if context and context.strip():
            # For masked LM, prepend context to sentence
            # Join context and words without spaces (Cantonese doesn't need them)
            context_words = context.strip().split()
            full_sentence = "".join(context_words) + "".join(words)
            # Adjust word index to account for context words at the beginning
            adjusted_word_index = len(context_words) + word_index
            full_words = context_words + words
        else:
            full_sentence = "".join(words)
            adjusted_word_index = word_index
            full_words = words
        
        target_word = words[word_index]

        token_indices, token_strings = self._align_word_to_tokens(full_sentence, full_words, adjusted_word_index)

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_tokens': 0,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_probs = []

        # Tokenize once before the loop for efficiency
        tokens = self.tokenizer.tokenize(full_sentence, add_special_tokens = True)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        for token_idx in token_indices:
            # Create a copy of input_ids for this iteration
            masked_input_ids = input_ids.copy()
            original_token_id = masked_input_ids[token_idx]
            masked_input_ids[token_idx] = self.tokenizer.mask_token_id

            input_ids_tensor = torch.tensor([masked_input_ids]).to(self.device)

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

        # Filter out invalid values for validation
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s) and not np.isinf(s)]
        
        total_surprisal = sum(token_surprisals)
        total_prob = np.prod(token_probs)

        return {
            'surprisal': total_surprisal,
            'probability': total_prob,
            'word': target_word,
            'tokens': token_strings,
            'token_surprisals': token_surprisals,
            'num_tokens': len(token_indices),
            'num_valid_tokens': len(valid_surprisals)
        }


class AutoregressiveLMSurprisalCalculator:
    '''
    Class for word-level surprisal calculation using Autoregressive Language Models.
    '''

    def __init__(self,
                model_name: str,
                device: str = None,
                use_4bit: bool = False):
        '''
        Initialize autoregressive LM for surprisal calculation.
        
        :param model_name: HuggingFace model identifier for causal LM
        :type model_name: str
        :param device: device to use (cuda/cpu)
        :type device: str
        :param use_4bit: whether to use 4-bit quantization (saves memory)
        :type use_4bit: bool
        '''

        logger.info(f"Loading autoregressive model: {model_name}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Using {device}")
        if use_4bit:
            logger.info("Using 4-bit quantization to save memory")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional 4-bit quantization
        if use_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
        
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Autoregressive model loaded!")

    def _align_word_to_tokens(self,
                              sentence: str,
                              words: List[str],
                              word_index: int,
                              ) -> Tuple[List[int], List[str]]:
        '''
        Align a word to its corresponding subword tokens.
        
        :param sentence: the full sentence string to tokenize
        :type sentence: str
        :param words: list of words in the given sentence
        :type words: List[str]
        :param word_index: position of the target word for surprisal calculation
        :type word_index: int
        '''

        target_word = words[word_index]

        # Calculate character position based on how sentence was constructed
        char_start = sum(len(w) for w in words[:word_index])
        char_end = char_start + len(target_word)

        encoding = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        token_ids = encoding['input_ids'][0].tolist()

        # For autoregressive models, we don't skip special tokens at the start
        token_indices = []
        current_char_pos = 0

        for i, token_id in enumerate(token_ids):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            token_len = len(token_str)
            token_start = current_char_pos
            token_end = current_char_pos + token_len

            if token_start < char_end and token_end > char_start:
                token_indices.append(i)
            
            current_char_pos += token_len

        token_strings = [self.tokenizer.decode([token_ids[i]], skip_special_tokens=True) for i in token_indices]

        return token_indices, token_strings

    def calculate_surprisal(self,
                            word_index: int,
                            words: List[str] = None,
                            context: str = None,
                            ) -> Dict:
        '''
        Calculate the surprisal for a word at a given position using autoregressive LM.
        
        :param word_index: index of the target word to measure surprisal (0-based)
        :type word_index: int
        :param words: pre-segmented words list
        :type words: List[str]
        :param context: optional discourse context (previous sentences)
        :type context: str
        '''

        if words is None:
            raise ValueError("The 'words' list must contain words")

        if word_index < 0 or word_index >= len(words):
            raise ValueError("Your word index is out of bounds")
        
        # Build full input with context if provided
        # Use consistent spacing: no spaces between Cantonese characters
        # For autoregressive models, only include words UP TO and INCLUDING the target word
        words_up_to_target = words[:word_index + 1]
        
        if context and context.strip():
            # For autoregressive LM, prepend context to sentence
            # Join context and words without spaces (Cantonese doesn't need them)
            context_words = context.strip().split()
            full_sentence = "".join(context_words) + "".join(words_up_to_target)
            # Adjust word index to account for context words at the beginning
            adjusted_word_index = len(context_words) + word_index
            full_words = context_words + words_up_to_target
        else:
            full_sentence = "".join(words_up_to_target)
            adjusted_word_index = word_index
            full_words = words_up_to_target
        
        target_word = words[word_index]

        token_indices, token_strings = self._align_word_to_tokens(full_sentence, full_words, adjusted_word_index)

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_tokens': 0,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_probs = []

        # Tokenize the full sentence once
        # Use the model's maximum length from its config
        max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        # If the tokenizer has an unreasonably large default, cap it
        if max_length > 1000000:
            max_length = 2048
        
        encoding = self.tokenizer(full_sentence, return_tensors="pt", add_special_tokens=True, 
                                  truncation=True, max_length=max_length)
        input_ids = encoding['input_ids'].to(self.device)
        
        # Check if target tokens are within the truncated sequence
        if token_indices and max(token_indices) >= input_ids.shape[1]:
            logger.warning(f"Target word tokens beyond truncation limit")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'word': target_word,
                'tokens': token_strings,
                'token_surprisals': [],
                'num_tokens': len(token_indices),
                'num_valid_tokens': 0
            }

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # For each target token, calculate surprisal based on preceding context
        for token_idx in token_indices:
            if token_idx == 0:
                # First token has no context, use uniform prior or skip
                logger.warning(f"Target word at first position, surprisal may be unreliable")
                token_surprisal = float('nan')
                token_prob = 0.0
            else:
                # Get logits from previous position to predict current token
                token_logits = logits[0, token_idx - 1, :]
                probs = torch.softmax(token_logits, dim=0)
                
                actual_token_id = input_ids[0, token_idx].item()
                token_prob = probs[actual_token_id].item()

                if token_prob > 1e-10:  # Use small threshold to avoid log(0)
                    token_surprisal = -np.log2(token_prob)
                else:
                    # Cap at maximum reasonable surprisal (~ 33 bits for 1e-10 probability)
                    token_surprisal = 33.0
                    logger.debug(f"Very low probability {token_prob:.2e}, capping surprisal at 33.0")

            token_surprisals.append(token_surprisal)
            token_probs.append(token_prob)

        # Filter out NaN values for aggregation
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s) and not np.isinf(s)]
        valid_probs = [p for p in token_probs if p > 0]

        if valid_surprisals:
            total_surprisal = sum(valid_surprisals)
        else:
            total_surprisal = float('nan')

        if valid_probs:
            total_prob = np.prod(valid_probs)
        else:
            total_prob = 0.0

        return {
            'surprisal': total_surprisal,
            'probability': total_prob,
            'word': target_word,
            'tokens': token_strings,
            'token_surprisals': token_surprisals,
            'num_tokens': len(token_indices),
            'num_valid_tokens': len(valid_surprisals)
        }


def create_surprisal_calculator(model_type: str, config, device: str = None):
    '''
    Factory function to create the appropriate surprisal calculator.
    
    :param model_type: Type of model - "masked" for BERT-style or "autoregressive" for GPT-style
    :type model_type: str
    :param config: Configuration object
    :type config: Config
    :param device: Device to use (cuda/cpu/auto)
    :type device: str
    :return: Surprisal calculator instance
    '''
    if model_type.lower() == "masked":
        model_name = config.get('experiment.masked_model', 'hon9kon9ize/bert-large-cantonese')
        return MaskedLMSurprisalCalculator(
            model_name=model_name,
            device=device
        )
    elif model_type.lower() == "autoregressive":
        model_name = config.get('experiment.autoregressive_model', 'uer/gpt2-chinese-cluecorpussmall')
        use_4bit = config.get('experiment.use_4bit_quantization', False)
        return AutoregressiveLMSurprisalCalculator(
            model_name=model_name,
            device=device,
            use_4bit=use_4bit
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'masked' or 'autoregressive'")
