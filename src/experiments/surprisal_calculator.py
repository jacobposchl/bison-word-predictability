'''
Core surprisal calculation using language models.

Supports two types of models:
1. Masked Language Models (BERT-style)
2. Autoregressive Models (GPT-style)

'''

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
import numpy as np
import math
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from functools import lru_cache
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Principled cap: -log2(minimum positive float32 value ≈ 1.18e-38)
MAX_SURPRISAL_BITS: float = 126.9
# Threshold above which a token's surprisal is logged as a low-probability event (~1e-9)
LOW_PROB_THRESHOLD_BITS: float = 30.0


def _compute_token_surprisal_and_entropy(
    logits: torch.Tensor,
    target_token_id: int,
) -> Tuple[float, float, bool]:
    """
    Compute surprisal and Shannon entropy (both in bits) for a target token.

    Uses log_softmax for numerical stability (log-sum-exp internally).
    Never performs a probability->log conversion, avoiding float32 underflow.

    Args:
        logits: Raw logit tensor of shape (vocab_size,)
        target_token_id: Index of the token whose surprisal to compute

    Returns:
        Tuple of (surprisal_bits, entropy_bits, was_capped):
            surprisal_bits: -log2(p(token)) in bits
            entropy_bits: Shannon entropy H of the full distribution in bits
            was_capped: True if log_prob was -inf and surprisal was set to MAX_SURPRISAL_BITS
    """
    LN2 = math.log(2)

    log_probs_nat = torch.log_softmax(logits.float(), dim=0)  # natural log, numerically stable

    # --- Surprisal ---
    log_prob_nat = log_probs_nat[target_token_id].item()
    if math.isfinite(log_prob_nat):
        surprisal = -log_prob_nat / LN2
        was_capped = False
    else:
        surprisal = MAX_SURPRISAL_BITS
        was_capped = True

    # --- Entropy: H = -(1/ln2) * sum(exp(log_p) * log_p) ---
    probs = torch.exp(log_probs_nat)
    entropy_terms = probs * log_probs_nat
    valid = entropy_terms[torch.isfinite(entropy_terms)]
    if len(valid) > 0:
        entropy = -valid.sum().item() / LN2
        if not (math.isfinite(entropy) and entropy >= 0):
            entropy = math.nan
    else:
        entropy = math.nan

    return surprisal, entropy, was_capped


# Delimiter used to separate context sentences
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

        self.context_clipped_count = 0
        self.low_prob_events: List[Dict] = []

        logger.info("Model Loaded!")

    def export_low_prob_events(self, path: str) -> int:
        """Write low-probability token events to CSV and reset the accumulator.

        Args:
            path: File path for the output CSV

        Returns:
            Number of events exported
        """
        n = len(self.low_prob_events)
        if n > 0:
            pd.DataFrame(self.low_prob_events).to_csv(path, index=False)
            logger.info(f"Exported {n} low-prob events to {path}")
        self.low_prob_events = []
        return n

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

        # DEBUG: Log token counts
        logger.info(f"=== TOKEN COUNT DEBUG ===")
        logger.info(f"Context words: {len(context_words)}")
        logger.info(f"Target sentence words: {len(words)}")
        logger.info(f"Required token count: {required_token_count}/{self.max_length}")
        logger.info(f"Will clip: {required_token_count > self.max_length}")
        if required_token_count > self.max_length:
            logger.warning(f"CLIPPING! Required: {required_token_count}, Max: {self.max_length}")
        logger.info(f"========================")

        # Calculate how many tokens we can use for post-switch words
        available_for_postswitch = self.max_length - required_token_count

        _left_clipped = False
        _right_trimmed = False

        if required_token_count > self.max_length:
            # Clip context from the left to fit
            self.context_clipped_count += 1
            _left_clipped = True
            _right_trimmed = len(words) > word_index + 1  # post-switch words exist but can't be added

            preswitch_words = words[:word_index + 1]
            preswitch_text = "".join(preswitch_words)
            preswitch_encoding = self.tokenizer(preswitch_text, add_special_tokens=True)
            preswitch_token_count = len(preswitch_encoding['input_ids'])

            tokens_for_context = self.max_length - preswitch_token_count

            # Clip context words from the left
            clipped_context_words = []
            current_tokens = 0
            for word in reversed(context_words):
                word_tokens = len(self.tokenizer(word, add_special_tokens=False)['input_ids'])
                if current_tokens + word_tokens <= tokens_for_context:
                    clipped_context_words.insert(0, word)
                    current_tokens += word_tokens
                else:
                    break

            required_words = clipped_context_words + preswitch_words
            required_text = "".join(required_words)
            full_sentence = required_text
            adjusted_word_index = len(clipped_context_words) + word_index
            full_words = required_words
            available_for_postswitch = 0
        elif available_for_postswitch <= 0:
            # No room for post-switch words
            logger.debug(f"No room for post-switch words (required content uses {required_token_count}/{self.max_length} tokens)")
            full_sentence = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words = required_words
            _right_trimmed = len(words) > word_index + 1
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

                _right_trimmed = len(postswitch_to_use) < len(postswitch_words)
                if _right_trimmed:
                    trimmed_count = len(postswitch_words) - len(postswitch_to_use)
                    logger.debug(f"Trimmed {trimmed_count}/{len(postswitch_words)} post-switch words to fit within {self.max_length} token limit")

                full_words = required_words + postswitch_to_use
                full_sentence = "".join(full_words)
                adjusted_word_index = len(context_words) + word_index

        target_word = words[word_index]

        if _left_clipped and _right_trimmed:
            _truncation = 'both'
        elif _left_clipped:
            _truncation = 'left_clipped'
        elif _right_trimmed:
            _truncation = 'right_trimmed'
        else:
            _truncation = 'clean'

        token_indices, token_strings = self._align_word_to_tokens(full_sentence, full_words, adjusted_word_index)

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': float('nan'),
                'entropy': float('nan'),
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_chars': len(target_word) if target_word else 0,
                'truncation': _truncation,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_entropies = []

        encoding = self.tokenizer(full_sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding['input_ids'][0].tolist()

        for i, token_idx in enumerate(token_indices):
            # Create a copy of input_ids for this iteration
            masked_input_ids = input_ids.copy()
            original_token_id = masked_input_ids[token_idx]
            masked_input_ids[token_idx] = self.tokenizer.mask_token_id

            input_ids_tensor = torch.tensor([masked_input_ids]).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids_tensor)
                predictions = outputs.logits

            masked_token_logits = predictions[0, token_idx, :]
            token_surprisal, token_entropy, was_capped = _compute_token_surprisal_and_entropy(
                masked_token_logits, original_token_id
            )

            if was_capped or token_surprisal >= LOW_PROB_THRESHOLD_BITS:
                token_str = token_strings[i] if i < len(token_strings) else ''
                self.low_prob_events.append({
                    'sentence':            ' '.join(words),
                    'word':                target_word,
                    'word_index':          word_index,
                    'token':               token_str,
                    'token_index_in_word': i,
                    'surprisal_bits':      token_surprisal,
                    'was_capped':          was_capped,
                    'model':               self.__class__.__name__,
                })

            token_surprisals.append(token_surprisal)
            token_entropies.append(token_entropy)

        # Filter NaN only — capped values are finite and must be included
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s)]
        valid_entropies = [e for e in token_entropies if not np.isnan(e)]

        total_surprisal = sum(valid_surprisals) if valid_surprisals else float('nan')
        avg_entropy = np.mean(valid_entropies) if valid_entropies else float('nan')

        return {
            'surprisal': total_surprisal,
            'probability': float('nan'),
            'entropy': avg_entropy,
            'word': target_word,
            'tokens': token_strings,
            'token_surprisals': token_surprisals,
            'num_chars': len(target_word) if target_word else 0,
            'truncation': _truncation,
            'num_valid_tokens': len(valid_surprisals)
        }


class AutoregressiveLMSurprisalCalculator:
    '''
    Class for word-level surprisal calculation using Autoregressive Language Models.
    '''

    def __init__(self, model_name: str, device: str = None):
        '''
        Initialize autoregressive LM for surprisal calculation.

        :param model_name: HuggingFace model identifier for causal LM
        :type model_name: str
        :param device: device to use (cuda/cpu)
        :type device: str
        '''

        logger.info(f"Loading autoregressive model: {model_name}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Using {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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

        self.context_clipped_count = 0
        self.low_prob_events: List[Dict] = []

        self.model.eval()
        logger.info("Autoregressive model loaded!")

    def export_low_prob_events(self, path: str) -> int:
        """Write low-probability token events to CSV and reset the accumulator.

        Args:
            path: File path for the output CSV

        Returns:
            Number of events exported
        """
        n = len(self.low_prob_events)
        if n > 0:
            pd.DataFrame(self.low_prob_events).to_csv(path, index=False)
            logger.info(f"Exported {n} low-prob events to {path}")
        self.low_prob_events = []
        return n

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

        # DEBUG: Log token counts
        logger.info(f"=== TOKEN COUNT DEBUG ===")
        logger.info(f"Context words: {len(context_words)}")
        logger.info(f"Target sentence words: {len(words)}")
        logger.info(f"Required token count: {required_token_count}/{self.max_length}")
        logger.info(f"Will clip: {required_token_count > self.max_length}")
        if required_token_count > self.max_length:
            logger.warning(f"CLIPPING! Required: {required_token_count}, Max: {self.max_length}")
        logger.info(f"========================")

        # Calculate how many tokens we can use for post-switch words
        available_for_postswitch = self.max_length - required_token_count

        _left_clipped = False
        _right_trimmed = False

        if required_token_count > self.max_length:
            # Clip context from the left to fit
            self.context_clipped_count += 1
            _left_clipped = True
            _right_trimmed = len(words) > word_index + 1  # post-switch words exist but can't be added

            preswitch_words = words[:word_index + 1]
            preswitch_text = "".join(preswitch_words)
            preswitch_encoding = self.tokenizer(preswitch_text, add_special_tokens=True)
            preswitch_token_count = len(preswitch_encoding['input_ids'])

            tokens_for_context = self.max_length - preswitch_token_count

            # Clip context words from the left
            clipped_context_words = []
            current_tokens = 0
            for word in reversed(context_words):
                word_tokens = len(self.tokenizer(word, add_special_tokens=False)['input_ids'])
                if current_tokens + word_tokens <= tokens_for_context:
                    clipped_context_words.insert(0, word)
                    current_tokens += word_tokens
                else:
                    break

            required_words = clipped_context_words + preswitch_words
            required_text = "".join(required_words)
            full_sentence_for_alignment = required_text
            adjusted_word_index = len(clipped_context_words) + word_index
            full_words_for_alignment = required_words
            available_for_postswitch = 0
        elif available_for_postswitch <= 0:
            # No room for post-switch words
            logger.debug(f"No room for post-switch words (required: {required_token_count}/{self.max_length} tokens)")
            full_sentence_for_alignment = required_text
            adjusted_word_index = len(context_words) + word_index
            full_words_for_alignment = required_words
            _right_trimmed = len(words) > word_index + 1
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

                _right_trimmed = len(postswitch_to_use) < len(postswitch_words)
                if _right_trimmed:
                    trimmed_count = len(postswitch_words) - len(postswitch_to_use)
                    logger.debug(f"Trimmed {trimmed_count}/{len(postswitch_words)} post-switch words to fit within {self.max_length} token limit")

                full_sentence_for_alignment = "".join(required_words + postswitch_to_use)
                adjusted_word_index = len(context_words) + word_index
                full_words_for_alignment = required_words + postswitch_to_use

        if _left_clipped and _right_trimmed:
            _truncation = 'both'
        elif _left_clipped:
            _truncation = 'left_clipped'
        elif _right_trimmed:
            _truncation = 'right_trimmed'
        else:
            _truncation = 'clean'

        # Align target word to tokens (using full sentence to get correct tokenization)
        token_indices, token_strings = self._align_word_to_tokens(
            full_sentence_for_alignment,
            full_words_for_alignment,
            adjusted_word_index
        )

        input_sentence = "".join(required_words[:-1])

        if not token_indices:
            logger.warning(f"Could not align word '{target_word}' to tokens")
            return {
                'surprisal': float('nan'),
                'probability': float('nan'),
                'entropy': float('nan'),
                'word': target_word,
                'tokens': [],
                'token_surprisals': [],
                'num_chars': len(target_word) if target_word else 0,
                'truncation': _truncation,
                'num_valid_tokens': 0
            }

        token_surprisals = []
        token_entropies = []

        encoding = self.tokenizer(input_sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoding['input_ids'].to(self.device)

        # Tokenize the full sentence (with target) to get actual token IDs
        full_encoding = self.tokenizer(full_sentence_for_alignment, return_tensors="pt",
                                       add_special_tokens=True)
        full_input_ids = full_encoding['input_ids'][0].tolist()

        current_input_ids = input_ids.clone()

        for i, token_idx_in_full in enumerate(token_indices):
            # Get the actual token ID from the full sequence
            actual_token_id = full_input_ids[token_idx_in_full]

            if current_input_ids.shape[1] == 0:
                logger.warning(f"Target word at first position, surprisal may be unreliable")
                token_surprisal = float('nan')
                token_entropy = float('nan')
            else:
                with torch.no_grad():
                    outputs = self.model(current_input_ids)
                    logits = outputs.logits

                # Get logits from the last position (which predicts the next token)
                last_position = current_input_ids.shape[1] - 1
                token_logits = logits[0, last_position, :]
                token_surprisal, token_entropy, was_capped = _compute_token_surprisal_and_entropy(
                    token_logits, actual_token_id
                )

                if was_capped or token_surprisal >= LOW_PROB_THRESHOLD_BITS:
                    token_str = token_strings[i] if i < len(token_strings) else ''
                    self.low_prob_events.append({
                        'sentence':            ' '.join(words),
                        'word':                target_word,
                        'word_index':          word_index,
                        'token':               token_str,
                        'token_index_in_word': i,
                        'surprisal_bits':      token_surprisal,
                        'was_capped':          was_capped,
                        'model':               self.__class__.__name__,
                    })

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
            token_entropies.append(token_entropy)

        # Filter NaN only — capped values are finite and must be included
        valid_surprisals = [s for s in token_surprisals if not np.isnan(s)]
        valid_entropies = [e for e in token_entropies if not np.isnan(e)]

        total_surprisal = sum(valid_surprisals) if valid_surprisals else float('nan')
        avg_entropy = np.mean(valid_entropies) if valid_entropies else float('nan')

        return {
            'surprisal': total_surprisal,
            'probability': float('nan'),
            'entropy': avg_entropy,
            'word': target_word,
            'tokens': token_strings,
            'token_surprisals': token_surprisals,
            'num_chars': len(target_word) if target_word else 0,
            'truncation': _truncation,
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
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'masked' or 'autoregressive'")
