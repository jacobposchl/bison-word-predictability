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

# Delimiter used to separate context sentences (must match analysis_dataset.py)
CONTEXT_SENTENCE_DELIMITER = ' ||| '

class MaskedLMSurprisalCalculator:
    '''
    Class for word-level surprisal calculation using Masked Language Models (BERT-style).
    Uses bidirectional context by masking the target word.
    '''

    def __init__(self, model_name: str, device: str = None):
        '''
        
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
        
        # Get model's maximum sequence length
        self.max_length = self.tokenizer.model_max_length
        if self.max_length > 1000000:  # Some models return very large values
            self.max_length = 512  # Use BERT's default
        logger.info(f"Model max sequence length: {self.max_length}")

        logger.info("Model Loaded!")

    def _align_word_to_tokens(self, sentence: str, words: List[str], word_index: int, ) -> Tuple[List[int], List[str]]:
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

    def calculate_surprisal(self, word_index: int, words: List[str] = None, context: str = None, ) -> Dict:
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
        
        
        
        # Build the required prefix: context + words up to and including target
        if context and context.strip():
            context_clean = context.replace(CONTEXT_SENTENCE_DELIMITER, ' ')
            context_words = context_clean.strip().split()
        else:
            context_words = []
        
        # Words that MUST be included: context + all words up to and including target
        required_words = context_words + words[:word_index + 1]
        required_text = "".join(required_words)
        
        # Encode required text to check token count
        required_encoding = self.tokenizer(required_text, add_special_tokens=True)
        required_token_count = len(required_encoding['input_ids'])
        
        # Calculate how many tokens we can use for post-switch words
        available_for_postswitch = self.max_length - required_token_count
        
        if required_token_count > self.max_length:
            logger.error(f"Required content (context + pre-switch + target) exceeds max_length!")
            logger.error(f"  Required tokens: {required_token_count}, Max: {self.max_length}")
            logger.error(f"  Context words: {len(context_words)}, Pre-switch words: {word_index}, Target: 1")
            logger.error(f"  This will cause truncation of pre-switch context, which may affect results")

            full_sentence = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words = required_words
        elif available_for_postswitch <= 0:
            # No room for post-switch words
            logger.debug(f"No room for post-switch words (required content uses {required_token_count}/{self.max_length} tokens)")
            full_sentence = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words = required_words
        else:
            # We have room for some/all post-switch words
            postswitch_words = words[word_index + 1:]
            
            if not postswitch_words:
                # No post-switch words anyway
                full_sentence = required_text
                adjusted_word_index = len(context_words) + word_index
                full_words = required_words
            else:
                # Add as many post-switch words as fit
                postswitch_to_use = []
                current_tokens = 0
                
                for word in postswitch_words:
                    word_tokens = len(self.tokenizer(word, add_special_tokens=False)['input_ids'])
                    if current_tokens + word_tokens <= available_for_postswitch:
                        postswitch_to_use.append(word)
                        current_tokens += word_tokens
                    else:
                        break
                
                if len(postswitch_to_use) < len(postswitch_words):
                    trimmed_count = len(postswitch_words) - len(postswitch_to_use)
                    logger.debug(f"Trimmed {trimmed_count}/{len(postswitch_words)} post-switch words to fit within {self.max_length} token limit")
                
                full_words = required_words + postswitch_to_use
                full_sentence = "".join(full_words)
                adjusted_word_index = len(context_words) + word_index
        
        target_word = words[word_index]

        token_indices, token_strings = self._align_word_to_tokens(full_sentence, full_words, adjusted_word_index)

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'entropy': float('nan'),
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_tokens': 0,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_probs = []
        token_entropies = []

        encoding = self.tokenizer(full_sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding['input_ids'][0].tolist()

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
            
            epsilon = 1e-20
            probs_clamped = torch.clamp(probs, min=epsilon, max=1.0)
            
            # Normalize to ensure sum is 1
            probs_normalized = probs_clamped / probs_clamped.sum()
            
            log_probs = torch.log2(probs_normalized)
            entropy_terms = probs_normalized * log_probs
            # Filter out any NaN or Inf terms
            valid_terms = entropy_terms[torch.isfinite(entropy_terms)]
            
            if len(valid_terms) > 0:
                token_entropy = -torch.sum(valid_terms).item()
            else:
                token_entropy = float('nan')
            
            if not (np.isfinite(token_entropy) and token_entropy >= 0):
                logger.debug(f"Invalid entropy: {token_entropy}, probs shape: {probs.shape}, probs sum: {probs.sum().item()}")
                token_entropy = float('nan')

            if token_prob > 0:
                token_surprisal = -np.log2(token_prob)
            else:
                logger.warning(f"Warning: infinite surprisal, token_prob <= 0 | token_prob: {token_prob}")
                token_surprisal = float('inf')

            token_surprisals.append(token_surprisal)
            token_probs.append(token_prob)
            token_entropies.append(token_entropy)

        # Filter out invalid values for aggregation
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s) and not np.isinf(s)]
        valid_probs = [p for p in token_probs if p > 0]
        valid_entropies = [e for e in token_entropies if not np.isnan(e)]
        
        # Sum only valid surprisals
        if valid_surprisals:
            total_surprisal = sum(valid_surprisals)
        else:
            total_surprisal = float('nan')
        
        # Multiply only valid probabilities
        if valid_probs:
            total_prob = np.prod(valid_probs)
        else:
            total_prob = 0.0
        
        # Average only valid entropies
        avg_entropy = np.mean(valid_entropies) if valid_entropies else float('nan')

        return {
            'surprisal': total_surprisal,
            'probability': total_prob,
            'entropy': avg_entropy,
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

    def __init__(self, model_name: str, device: str = None, use_4bit: bool = False):
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
        
        # Get model's maximum sequence length
        self.max_length = self.tokenizer.model_max_length
        if self.max_length > 1000000:  # Some models return very large values
            self.max_length = 2048  # Use a reasonable default for autoregressive models
        logger.info(f"Model max sequence length: {self.max_length}")

        logger.info("Autoregressive model loaded!")

    def _align_word_to_tokens(self, sentence: str, words: List[str], word_index: int, ) -> Tuple[List[int], List[str]]:
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

        # Skip BOS token at start (consistent with masked model skipping [CLS])
        # Most autoregressive tokenizers add BOS at position 0
        start_idx = 1 if (len(token_ids) > 0 and 
                         self.tokenizer.bos_token_id is not None and 
                         token_ids[0] == self.tokenizer.bos_token_id) else 0
        
        token_indices = []
        current_char_pos = 0

        for i, token_id in enumerate(token_ids[start_idx:], start=start_idx):
            token_str = self.tokenizer.decode([token_id])
            token_len = len(token_str)
            token_start = current_char_pos
            token_end = current_char_pos + token_len

            if token_start < char_end and token_end > char_start:
                token_indices.append(i)
            
            current_char_pos += token_len

        token_strings = [self.tokenizer.decode([token_ids[i]]) for i in token_indices]

        return token_indices, token_strings

    def calculate_surprisal(self, word_index: int, words: List[str] = None, context: str = None) -> Dict:
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
        

        words_context = words[:word_index]
        target_word = words[word_index]
        
        
        # Build the required prefix
        if context and context.strip():
            context_clean = context.replace(CONTEXT_SENTENCE_DELIMITER, ' ')
            context_words = context_clean.strip().split()
        else:
            context_words = []
        
        # Words that MUST be included: context + all words up to and including target
        required_words = context_words + words[:word_index + 1]
        required_text = "".join(required_words)
        
        # Encode required text to check token count
        required_encoding = self.tokenizer(required_text, add_special_tokens=True)
        required_token_count = len(required_encoding['input_ids'])
        
        # Calculate how many tokens we can use for post-switch words
        available_for_postswitch = self.max_length - required_token_count
        
        if required_token_count > self.max_length:
            # Required content exceeds max_length
            logger.error(f"Required content (context + pre-switch + target) exceeds max_length!")
            logger.error(f"  Required tokens: {required_token_count}, Max: {self.max_length}")
            logger.error(f"  This will cause truncation of pre-switch context")
            full_sentence_for_alignment = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words_for_alignment = required_words
        elif available_for_postswitch <= 0:
            # No room for post-switch words
            logger.debug(f"No room for post-switch words (required: {required_token_count}/{self.max_length} tokens)")
            full_sentence_for_alignment = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words_for_alignment = required_words
        else:
            # Add as many post-switch words as fit
            postswitch_words = words[word_index + 1:]
            
            if not postswitch_words:
                full_sentence_for_alignment = required_text
                adjusted_word_index = len(context_words) + word_index
                full_words_for_alignment = required_words
            else:
                postswitch_to_use = []
                current_tokens = 0
                
                for word in postswitch_words:
                    word_tokens = len(self.tokenizer(word, add_special_tokens=False)['input_ids'])
                    if current_tokens + word_tokens <= available_for_postswitch:
                        postswitch_to_use.append(word)
                        current_tokens += word_tokens
                    else:
                        break
                
                if len(postswitch_to_use) < len(postswitch_words):
                    trimmed_count = len(postswitch_words) - len(postswitch_to_use)
                    logger.debug(f"Trimmed {trimmed_count}/{len(postswitch_words)} post-switch words to fit within {self.max_length} token limit")
                
                full_sentence_for_alignment = "".join(required_words + postswitch_to_use)
                adjusted_word_index = len(context_words) + word_index
                full_words_for_alignment = required_words + postswitch_to_use
        
        # Align target word to tokens (using full sentence to get correct tokenization)
        token_indices, token_strings = self._align_word_to_tokens(
            full_sentence_for_alignment, 
            full_words_for_alignment, 
            adjusted_word_index
        )
        \
        input_sentence = "".join(required_words[:-1])

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': 0.0,
                'entropy': float('nan'),
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_tokens': 0,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_probs = []
        
        encoding = self.tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding['input_ids'].to(self.device)

        # Tokenize the full sentence (with target) to get actual token IDs
        full_encoding = self.tokenizer(full_sentence_for_alignment, return_tensors="pt", 
                                       add_special_tokens=True)
        full_input_ids = full_encoding['input_ids'][0].tolist()
        
        # Get the position where predictions start (end of context in input)
        context_token_length = input_ids.shape[1]
        
        current_input_ids = input_ids.clone()
        token_entropies = []
        
        for i, token_idx_in_full in enumerate(token_indices):
            # Get the actual token ID from the full sequence
            actual_token_id = full_input_ids[token_idx_in_full]
            
            if current_input_ids.shape[1] == 0:
                logger.warning(f"Target word at first position, surprisal may be unreliable")
                token_surprisal = float('nan')
                token_prob = 0.0
                token_entropy = float('nan')
            else:
                with torch.no_grad():
                    outputs = self.model(current_input_ids)
                    logits = outputs.logits
                
                # Get logits from the last position (which predicts the next token)
                last_position = current_input_ids.shape[1] - 1
                token_logits = logits[0, last_position, :]
                probs = torch.softmax(token_logits, dim=0)
                
                # Get probability of the actual token
                token_prob = probs[actual_token_id].item()
                

                epsilon = 1e-20
                probs_clamped = torch.clamp(probs, min=epsilon, max=1.0)
                
                # Normalize to ensure sum is 1 (handles quantization errors)
                probs_normalized = probs_clamped / probs_clamped.sum()
                
                # Calculate entropy: H = -sum(p * log2(p))
                log_probs = torch.log2(probs_normalized)
                entropy_terms = probs_normalized * log_probs

                # Filter out any NaN or Inf terms
                valid_terms = entropy_terms[torch.isfinite(entropy_terms)]
                
                if len(valid_terms) > 0:
                    token_entropy = -torch.sum(valid_terms).item()
                else:
                    token_entropy = float('nan')
                
                # Final check for invalid values
                if not (np.isfinite(token_entropy) and token_entropy >= 0):
                    logger.debug(f"Invalid entropy: {token_entropy}, probs shape: {probs.shape}, probs sum: {probs.sum().item()}")
                    token_entropy = float('nan')

                if token_prob > 1e-10:  # Use small threshold to avoid log(0)
                    token_surprisal = -np.log2(token_prob)
                else:
                    # Cap at maximum reasonable surprisal (~ 33 bits for 1e-10 probability)
                    token_surprisal = 33.0
                    logger.debug(f"Very low probability {token_prob:.2e}, capping surprisal at 33.0")
                
                # Append the actual token to input for next iteration (if not last token)
                if i < len(token_indices) - 1:
                    # Add the actual token to the sequence for predicting next token
                    new_token = torch.tensor([[actual_token_id]], device=self.device)
                    current_input_ids = torch.cat([current_input_ids, new_token], dim=1)
                    
                    # Check for truncation
                    if current_input_ids.shape[1] >= self.max_length:
                        logger.warning(f"Sequence length exceeded during iterative prediction")
                        break

            token_surprisals.append(token_surprisal)
            token_probs.append(token_prob)
            token_entropies.append(token_entropy)

        # Filter out NaN values for aggregation
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s) and not np.isinf(s)]
        valid_probs = [p for p in token_probs if p > 0]
        valid_entropies = [e for e in token_entropies if not np.isnan(e)]

        if valid_surprisals:
            total_surprisal = sum(valid_surprisals)
        else:
            total_surprisal = float('nan')

        if valid_probs:
            total_prob = np.prod(valid_probs)
        else:
            total_prob = 0.0
        
        # Average entropy across tokens
        avg_entropy = np.mean(valid_entropies) if valid_entropies else float('nan')

        return {
            'surprisal': total_surprisal,
            'probability': total_prob,
            'entropy': avg_entropy,
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
        model_name = config.get('experiment.masked_model')
        return MaskedLMSurprisalCalculator(
            model_name=model_name,
            device=device
        )
        
    elif model_type.lower() == "autoregressive":
        model_name = config.get('experiment.autoregressive_model')
        return AutoregressiveLMSurprisalCalculator(
            model_name=model_name,
            device=device,
            use_4bit=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'masked' or 'autoregressive'")
