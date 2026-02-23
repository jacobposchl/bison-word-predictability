"""
Tests for _compute_token_surprisal_and_entropy and related behaviour in
surprisal_calculator.py.

All tests use synthetic logit tensors — no language model is loaded.
"""

import math
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.surprisal_calculator import (
    _compute_token_surprisal_and_entropy,
    MAX_SURPRISAL_BITS,
    LOW_PROB_THRESHOLD_BITS,
    MaskedLMSurprisalCalculator,
    AutoregressiveLMSurprisalCalculator,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def logits_from_probs(probs: list) -> torch.Tensor:
    """Convert a probability vector to corresponding logits via log."""
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p)


# ---------------------------------------------------------------------------
# Test 1: Normal case — known logits produce correct surprisal
# ---------------------------------------------------------------------------

def test_surprisal_normal_case():
    """
    Given a 3-token vocabulary where token 0 has probability 0.5,
    its surprisal should be -log2(0.5) = 1.0 bit.
    """
    # Softmax of [0, -inf, -inf] → [1, 0, 0], but use a sharp distribution
    # Use log-probabilities as logits: log(0.5) ≈ -0.693, log(0.3) ≈ -1.204, log(0.2) ≈ -1.609
    probs = [0.5, 0.3, 0.2]
    logits = logits_from_probs(probs)

    surprisal, entropy, was_capped = _compute_token_surprisal_and_entropy(logits, target_token_id=0)

    assert not was_capped
    assert abs(surprisal - (-math.log2(0.5))) < 1e-4, f"Expected 1.0 bit, got {surprisal}"


# ---------------------------------------------------------------------------
# Test 2: Entropy of uniform distribution
# ---------------------------------------------------------------------------

def test_entropy_uniform_distribution():
    """
    A uniform distribution over 4 tokens has entropy = log2(4) = 2.0 bits.
    Equal logits produce a uniform softmax.
    """
    logits = torch.zeros(4)  # equal logits → uniform distribution

    surprisal, entropy, was_capped = _compute_token_surprisal_and_entropy(logits, target_token_id=0)

    assert abs(entropy - 2.0) < 1e-4, f"Expected entropy=2.0 bits, got {entropy}"


# ---------------------------------------------------------------------------
# Test 3: Entropy of near-certain distribution
# ---------------------------------------------------------------------------

def test_entropy_certain_distribution():
    """
    When one token is overwhelmingly likely, entropy approaches 0.
    logits = [100, 0, 0] → token 0 gets nearly all probability.
    """
    logits = torch.tensor([100.0, 0.0, 0.0])

    surprisal, entropy, was_capped = _compute_token_surprisal_and_entropy(logits, target_token_id=0)

    assert entropy < 0.01, f"Expected entropy ≈ 0, got {entropy}"
    assert math.isfinite(entropy) and entropy >= 0


# ---------------------------------------------------------------------------
# Test 4: No underflow on extreme logits — log_softmax stays finite
# ---------------------------------------------------------------------------

def test_no_underflow_on_extreme_logits():
    """
    With logits=[-1000.0, 0.0], target token 0 has probability ≈ 0.
    The old softmax approach returned float('inf') because prob underflowed to 0.0.
    The new log_softmax approach computes log_prob ≈ -1000 nats (finite), giving
    surprisal ≈ 1443 bits — large but finite. was_capped stays False because
    log_softmax uses the log-sum-exp trick and never hits -inf for finite inputs.
    """
    logits = torch.tensor([-1000.0, 0.0])

    surprisal, entropy, was_capped = _compute_token_surprisal_and_entropy(logits, target_token_id=0)

    # log_softmax never returns -inf for finite inputs, so was_capped stays False
    assert not was_capped, "log_softmax handles finite extreme logits without capping"
    # Surprisal is finite (not inf/nan) — this is the key improvement over softmax
    assert math.isfinite(surprisal), "Surprisal must be finite (not inf/nan)"
    # Surprisal is large, correctly indicating near-zero probability
    assert surprisal >= LOW_PROB_THRESHOLD_BITS, f"Expected surprisal >= {LOW_PROB_THRESHOLD_BITS}, got {surprisal}"

    # Verify was_capped=True only fires for literal -inf logits (genuine degenerate case)
    logits_with_neginf = torch.tensor([float('-inf'), 0.0])
    surprisal_inf, _, was_capped_inf = _compute_token_surprisal_and_entropy(
        logits_with_neginf, target_token_id=0
    )
    assert was_capped_inf, "was_capped=True should fire for -inf logit input"
    assert surprisal_inf == MAX_SURPRISAL_BITS


# ---------------------------------------------------------------------------
# Test 5: Low-probability event accumulates in low_prob_events
# ---------------------------------------------------------------------------

def test_low_prob_event_accumulates():
    """
    When a token receives surprisal >= LOW_PROB_THRESHOLD_BITS, the calculator
    should record it in self.low_prob_events regardless of was_capped.
    """
    # Build a mock MaskedLMSurprisalCalculator that bypasses model loading
    with patch.object(MaskedLMSurprisalCalculator, '__init__', lambda self, *a, **kw: None):
        calc = MaskedLMSurprisalCalculator.__new__(MaskedLMSurprisalCalculator)
        calc.low_prob_events = []
        calc.device = torch.device('cpu')

    words = ['hello', 'world']
    target_word = 'world'
    word_index = 1
    # extreme logits: token 5 has near-zero probability relative to the rest
    logits = torch.zeros(10)
    logits[5] = -1000.0
    actual_token_id = 5

    surprisal, entropy, was_capped = _compute_token_surprisal_and_entropy(logits, actual_token_id)

    # Simulate what the calculator loop does
    if was_capped or surprisal >= LOW_PROB_THRESHOLD_BITS:
        calc.low_prob_events.append({
            'sentence':            ' '.join(words),
            'word':                target_word,
            'word_index':          word_index,
            'token':               'world',
            'token_index_in_word': 0,
            'surprisal_bits':      surprisal,
            'was_capped':          was_capped,
            'model':               'MaskedLMSurprisalCalculator',
        })

    # Event IS logged because surprisal >= LOW_PROB_THRESHOLD_BITS
    assert len(calc.low_prob_events) == 1
    # log_softmax handled this gracefully without capping
    assert calc.low_prob_events[0]['was_capped'] is False
    # Surprisal is the true large-but-finite value, not an arbitrary cap
    assert calc.low_prob_events[0]['surprisal_bits'] >= LOW_PROB_THRESHOLD_BITS
    assert math.isfinite(calc.low_prob_events[0]['surprisal_bits'])


# ---------------------------------------------------------------------------
# Test 6: export_low_prob_events writes CSV and resets accumulator
# ---------------------------------------------------------------------------

def test_export_clears_events():
    """
    After calling export_low_prob_events, the accumulator should be empty
    and the CSV should contain one row per event.
    """
    with patch.object(MaskedLMSurprisalCalculator, '__init__', lambda self, *a, **kw: None):
        calc = MaskedLMSurprisalCalculator.__new__(MaskedLMSurprisalCalculator)
        calc.low_prob_events = [
            {
                'sentence': 'hello world',
                'word': 'world',
                'word_index': 1,
                'token': 'world',
                'token_index_in_word': 0,
                'surprisal_bits': MAX_SURPRISAL_BITS,
                'was_capped': True,
                'model': 'MaskedLMSurprisalCalculator',
            },
            {
                'sentence': 'foo bar',
                'word': 'bar',
                'word_index': 1,
                'token': 'bar',
                'token_index_in_word': 0,
                'surprisal_bits': 35.0,
                'was_capped': False,
                'model': 'MaskedLMSurprisalCalculator',
            },
        ]

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        tmp_path = f.name

    try:
        n = calc.export_low_prob_events(tmp_path)

        assert n == 2, f"Expected 2 events exported, got {n}"
        assert len(calc.low_prob_events) == 0, "Accumulator should be empty after export"

        import pandas as pd
        df = pd.read_csv(tmp_path)
        assert len(df) == 2
        assert list(df.columns) == [
            'sentence', 'word', 'word_index', 'token',
            'token_index_in_word', 'surprisal_bits', 'was_capped', 'model'
        ]
    finally:
        os.unlink(tmp_path)
