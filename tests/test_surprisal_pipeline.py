"""
Tests for the full surprisal pipeline.

Coverage:
  - calculate_surprisal_for_dataset: context threading, NaN for insufficient
    context, paired-context requirement, column structure, redundant-call count
  - compute_statistics: descriptive stats, paired t-test, NaN/context filtering
  - convert_surprisal_results_to_long: wide→long reshape, row count, is_switch values
  - MaskedLMSurprisalCalculator.calculate_surprisal: context prepended to model
    input (verified via mock tokenizer + model that records input lengths)
  - AutoregressiveLMSurprisalCalculator.calculate_surprisal: context prepended
    to the autoregressive prefix

All tests use mocks — no real language models are loaded.
"""

import sys
import math
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.surprisal_analysis import (
    calculate_surprisal_for_dataset,
    convert_surprisal_results_to_long,
    compute_statistics,
)
from src.experiments.surprisal_calculator import (
    MaskedLMSurprisalCalculator,
    AutoregressiveLMSurprisalCalculator,
    CONTEXT_SENTENCE_DELIMITER,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_SEP = CONTEXT_SENTENCE_DELIMITER  # ' ||| '


class MockSurprisalCalc:
    """Records every calculate_surprisal call and returns fixed dummy values."""

    def __init__(self, surprisal: float = 5.0, entropy: float = 3.0):
        self.calls: list[dict] = []
        self._surprisal = surprisal
        self._entropy = entropy
        self.context_clipped_count = 0

    def calculate_surprisal(self, word_index, words, context=None):
        self.calls.append(
            {"word_index": word_index, "words": list(words), "context": context}
        )
        return {
            "surprisal": self._surprisal,
            "probability": float("nan"),
            "entropy": self._entropy,
            "word": words[word_index],
            "tokens": [words[word_index]],
            "token_surprisals": [self._surprisal],
            "num_chars": len(words[word_index]),
            "num_valid_tokens": 1,
            "truncation": "clean",
        }


def make_analysis_df(**overrides) -> pd.DataFrame:
    """
    Build a one-row analysis DataFrame with sensible defaults.

    cs_translation  is split on spaces → cs_words.
    matched_mono    is passed to pycantonese.segment (mocked to list() in tests).
    switch_index    and matched_switch_index are 0-based.
    """
    defaults = {
        "cs_sentence": "original CS sentence",
        # 4 space-separated words; switch_index=2 targets word 'C'
        "cs_translation": "A B C D",
        # 4 chars; after mock segment → ['甲','乙','丙','丁']; matched_switch_index=2 → '丙'
        "matched_mono": "甲乙丙丁",
        "switch_index": 2,
        "matched_switch_index": 2,
        "pattern": "C2-E1-C1",
        "similarity": 0.8,
        "cs_participant": "P01",
        "matched_participant": "P02",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# pycantonese.segment is patched to split each string into individual characters.
SEG_PATCH = "src.experiments.surprisal_analysis.pycantonese.segment"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: calculate_surprisal_for_dataset — context handling
# ─────────────────────────────────────────────────────────────────────────────


@patch(SEG_PATCH, side_effect=list)
def test_context_correctly_passed_to_calculator(mock_seg):
    """
    When two context sentences exist and context_len=1, only the *last* sentence
    is passed as `context` to calculate_surprisal.
    """
    ctx = "Sentence one" + CONTEXT_SEP + "Sentence two"
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=True,
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[1]
        )

    # Calls with non-None context are the context_len=1 iterations
    context_calls = [c for c in calc.calls if c["context"] is not None]
    assert len(context_calls) == 2, "Expected one CS call and one mono call with context"

    for call in context_calls:
        assert call["context"] == "Sentence two", (
            f"Expected last sentence 'Sentence two', got {call['context']!r}"
        )


@patch(SEG_PATCH, side_effect=list)
def test_context_len_selects_last_n_sentences(mock_seg):
    """
    With three context sentences and context_len=2, only the last two are used.
    """
    ctx = "First" + CONTEXT_SEP + "Second" + CONTEXT_SEP + "Third"
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=True,
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[2]
        )

    context_calls = [c for c in calc.calls if c["context"] is not None]
    assert len(context_calls) == 2

    for call in context_calls:
        # "Second Third" joined with a space
        assert call["context"] == "Second Third", (
            f"Expected 'Second Third', got {call['context']!r}"
        )


@patch(SEG_PATCH, side_effect=list)
def test_insufficient_context_produces_nan(mock_seg):
    """
    When context_len=2 but only 1 sentence is available, both cs and mono
    surprisal for that context length must be NaN.
    """
    ctx = "Only one sentence"   # 1 sentence — not enough for context_len=2
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=True,
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[2]
        )

    assert pd.isna(result["cs_surprisal_context_2"].iloc[0]), (
        "cs_surprisal_context_2 should be NaN when context is insufficient"
    )
    assert pd.isna(result["mono_surprisal_context_2"].iloc[0]), (
        "mono_surprisal_context_2 should be NaN when context is insufficient"
    )


@patch(SEG_PATCH, side_effect=list)
def test_context_len_0_always_calculates(mock_seg):
    """
    context_len=0 always produces a real surprisal value, even when no context
    columns exist in the DataFrame.
    """
    df = make_analysis_df()   # no cs_context / mono_context columns
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[0]
        )

    assert pd.notna(result["cs_surprisal_context_0"].iloc[0]), (
        "context_len=0 should always produce a real cs surprisal"
    )
    assert pd.notna(result["mono_surprisal_context_0"].iloc[0]), (
        "context_len=0 should always produce a real mono surprisal"
    )


@patch(SEG_PATCH, side_effect=list)
def test_context_len_0_passes_none_to_calculator(mock_seg):
    """
    When context_len=0, calculate_surprisal must receive context=None
    (i.e. no context string is constructed or passed).
    """
    ctx = "Some previous sentence"
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=True,
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[0]
        )

    # Every call for context_len=0 must have context=None
    context_0_calls = [c for c in calc.calls if c["context"] is None]
    assert len(context_0_calls) >= 2, (
        "At least 2 calls with context=None expected for context_len=0 "
        "(one CS, one mono)"
    )


@patch(SEG_PATCH, side_effect=list)
def test_both_contexts_required_or_nan(mock_seg):
    """
    If cs_context is present but mono_context is absent, both surprisal values
    should be NaN for that context length (paired comparison requirement).
    """
    df = make_analysis_df(
        cs_context="Previous sentence",
        cs_context_valid=True,
        # mono_context intentionally omitted
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[1]
        )

    assert pd.isna(result["cs_surprisal_context_1"].iloc[0]), (
        "cs_surprisal_context_1 should be NaN when mono_context is missing"
    )
    assert pd.isna(result["mono_surprisal_context_1"].iloc[0]), (
        "mono_surprisal_context_1 should be NaN when mono_context is missing"
    )


@patch(SEG_PATCH, side_effect=list)
def test_invalid_context_flag_suppresses_context(mock_seg):
    """
    When cs_context_valid=False, the context should be treated as unavailable
    even if a cs_context string exists, resulting in NaN for context_len > 0.
    """
    ctx = "A valid-looking sentence"
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=False,   # <-- marked invalid
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[1]
        )

    assert pd.isna(result["cs_surprisal_context_1"].iloc[0]), (
        "cs_surprisal_context_1 should be NaN when cs_context_valid=False"
    )


@patch(SEG_PATCH, side_effect=list)
def test_output_columns_for_multiple_context_lengths(mock_seg):
    """
    For context_lengths=[0, 1, 2], output DataFrame must contain the expected
    surprisal, entropy, and difference columns for every context length.
    """
    ctx = "Sent one" + CONTEXT_SEP + "Sent two" + CONTEXT_SEP + "Sent three"
    df = make_analysis_df(
        cs_context=ctx,
        cs_context_valid=True,
        mono_context=ctx,
        mono_context_valid=True,
    )
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[0, 1, 2]
        )

    for n in [0, 1, 2]:
        assert f"cs_surprisal_context_{n}" in result.columns, (
            f"Missing cs_surprisal_context_{n}"
        )
        assert f"mono_surprisal_context_{n}" in result.columns, (
            f"Missing mono_surprisal_context_{n}"
        )
        assert f"cs_entropy_context_{n}" in result.columns, (
            f"Missing cs_entropy_context_{n}"
        )
        assert f"mono_entropy_context_{n}" in result.columns, (
            f"Missing mono_entropy_context_{n}"
        )
        assert f"surprisal_difference_context_{n}" in result.columns, (
            f"Missing surprisal_difference_context_{n}"
        )


@patch(SEG_PATCH, side_effect=list)
def test_redundant_no_context_model_calls_are_extra(mock_seg):
    """
    Documents a known performance issue: lines 161-170 of surprisal_analysis.py
    call calculate_surprisal(context=None) unconditionally to obtain word metadata,
    and then the context_len=0 loop iteration makes the *same* call again.

    For a single row with context_lengths=[0], the calculator is therefore
    invoked 4 times (2 wasted + 2 for context_len=0) rather than the minimum 2.

    This test pins the current behaviour. If the redundant calls are ever
    removed the count will drop to 2 and this test should be updated.
    """
    df = make_analysis_df()
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[0]
        )

    # 2 initial metadata calls (context=None) + 2 context_len=0 calls (context=None)
    assert len(calc.calls) == 4, (
        f"Expected 4 calls (2 metadata + 2 for context_len=0), got {len(calc.calls)}. "
        "If you fixed the redundant calls, update this count to 2."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: compute_statistics
# ─────────────────────────────────────────────────────────────────────────────


def _make_stats_df(cs_vals, mono_vals, cs_ctx=None, mono_ctx=None, ctx_len=0):
    """Build a minimal results_df for compute_statistics."""
    n = len(cs_vals)
    d = {
        f"cs_surprisal_context_{ctx_len}": cs_vals,
        f"mono_surprisal_context_{ctx_len}": mono_vals,
        f"surprisal_difference_context_{ctx_len}": [c - m for c, m in zip(cs_vals, mono_vals)],
    }
    if cs_ctx is not None:
        d["cs_context"] = cs_ctx
    if mono_ctx is not None:
        d["mono_context"] = mono_ctx
    return pd.DataFrame(d)


def test_compute_statistics_basic_math():
    """
    compute_statistics returns correct means, std, and paired t-test sign for
    a trivially constructed dataset with known values.
    """
    cs_vals  = [10.0, 12.0, 8.0]
    mono_vals = [8.0,  9.0,  7.0]   # cs always > mono → t > 0
    df = _make_stats_df(cs_vals, mono_vals, ctx_len=0)

    stats = compute_statistics(df, context_length=0)

    assert abs(stats["cs_surprisal_mean"] - np.mean(cs_vals)) < 1e-6
    assert abs(stats["mono_surprisal_mean"] - np.mean(mono_vals)) < 1e-6
    expected_diff_mean = np.mean([c - m for c, m in zip(cs_vals, mono_vals)])
    assert abs(stats["difference_mean"] - expected_diff_mean) < 1e-6
    # t-statistic sign: cs mean > mono mean → t > 0
    assert stats["ttest_statistic"] > 0


def test_compute_statistics_nan_surprisal_rows_excluded():
    """
    Rows with NaN surprisal should be excluded from statistics even when no
    context filtering is needed (context_length=0).
    """
    cs_vals  = [10.0, float("nan"), 8.0]
    mono_vals = [8.0,  float("nan"), 7.0]
    df = _make_stats_df(cs_vals, mono_vals, ctx_len=0)

    stats = compute_statistics(df, context_length=0)

    # Only 2 valid rows
    assert stats["n_valid"] == 2
    assert abs(stats["cs_surprisal_mean"] - 9.0) < 1e-6   # (10+8)/2


def test_compute_statistics_filters_na_context_for_positive_ctx_len():
    """
    For context_length > 0, rows where cs_context or mono_context is 'N/A'
    must be excluded from statistical calculations.
    """
    cs_ctx    = ["valid", "valid", "N/A"]
    mono_ctx  = ["valid", "valid", "N/A"]
    cs_vals   = [10.0, 12.0, 20.0]    # third row has 'N/A' context
    mono_vals = [8.0,  10.0, 5.0]
    df = _make_stats_df(cs_vals, mono_vals, cs_ctx=cs_ctx, mono_ctx=mono_ctx, ctx_len=1)

    stats = compute_statistics(df, context_length=1)

    # Third row excluded → n_valid == 2
    assert stats["n_valid"] == 2, (
        f"Expected 2 valid rows after excluding N/A context, got {stats['n_valid']}"
    )
    # Mean should exclude the third row's values
    assert abs(stats["cs_surprisal_mean"] - 11.0) < 1e-6   # (10+12)/2


def test_compute_statistics_context_len_0_ignores_na_context():
    """
    For context_length=0, rows with 'N/A' context should NOT be excluded
    because no context is required for the no-context condition.
    """
    cs_ctx    = ["N/A", "N/A"]
    mono_ctx  = ["N/A", "N/A"]
    cs_vals   = [10.0, 12.0]
    mono_vals = [8.0,  10.0]
    df = _make_stats_df(cs_vals, mono_vals, cs_ctx=cs_ctx, mono_ctx=mono_ctx, ctx_len=0)

    stats = compute_statistics(df, context_length=0)

    # Both rows kept — N/A context does not filter for context_len=0
    assert stats["n_valid"] == 2


def test_compute_statistics_cohens_d_sign():
    """
    Cohen's d = mean(diff) / std(diff). When CS surprisal is consistently
    higher than mono, d must be positive.
    """
    cs_vals   = [10.0, 12.0, 11.0]
    mono_vals = [8.0,   9.0,  8.0]
    df = _make_stats_df(cs_vals, mono_vals, ctx_len=0)

    stats = compute_statistics(df, context_length=0)

    assert stats["cohens_d"] > 0, (
        "Cohen's d should be positive when CS surprisal > mono surprisal"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: convert_surprisal_results_to_long
# ─────────────────────────────────────────────────────────────────────────────


def _make_wide_results(n_rows: int = 3) -> pd.DataFrame:
    """Build a minimal wide-format results DataFrame for reshape tests."""
    return pd.DataFrame({
        "sent_id":                     list(range(n_rows)),
        "cs_original_sentence":        ["orig"] * n_rows,
        "cs_translation":              ["A B C"] * n_rows,
        "matched_mono":                ["甲乙丙"] * n_rows,
        "switch_index":                [2] * n_rows,
        "matched_switch_index":        [2] * n_rows,
        "pattern":                     ["C2-E1"] * n_rows,
        "cs_participant":              ["P01"] * n_rows,
        "matched_participant":         ["P02"] * n_rows,
        "cs_surprisal_context_0":      [5.0] * n_rows,
        "cs_entropy_context_0":        [3.0] * n_rows,
        "mono_surprisal_context_0":    [4.0] * n_rows,
        "mono_entropy_context_0":      [2.5] * n_rows,
        "surprisal_difference_context_0": [1.0] * n_rows,
        "cs_word":                     ["C"] * n_rows,
        "cs_word_length":              [1] * n_rows,
        "mono_word":                   ["丙"] * n_rows,
        "mono_word_length":            [1] * n_rows,
        "cs_sent_length":              [3] * n_rows,
        "mono_sent_length":            [3] * n_rows,
        "cs_normalized_switch_point":  [2 / 3] * n_rows,
        "mono_normalized_switch_point": [2 / 3] * n_rows,
        "cs_context":                  ["N/A"] * n_rows,
        "mono_context":                ["N/A"] * n_rows,
    })


def test_long_format_doubles_row_count():
    """N wide rows → 2N long rows (one CS + one mono per pair)."""
    wide = _make_wide_results(n_rows=4)
    long_df = convert_surprisal_results_to_long(wide)
    assert len(long_df) == 8, f"Expected 8 rows, got {len(long_df)}"


def test_long_format_has_is_switch_column():
    """Long format must contain an is_switch column."""
    wide = _make_wide_results()
    long_df = convert_surprisal_results_to_long(wide)
    assert "is_switch" in long_df.columns


def test_long_format_is_switch_values_are_0_and_1():
    """Each sent_id should have exactly one is_switch=0 (mono) and one is_switch=1 (CS)."""
    wide = _make_wide_results(n_rows=3)
    long_df = convert_surprisal_results_to_long(wide)

    for sid in long_df["sent_id"].unique():
        group = long_df[long_df["sent_id"] == sid]["is_switch"].tolist()
        assert sorted(group) == [0, 1], (
            f"sent_id={sid} should have is_switch values [0,1], got {sorted(group)}"
        )


def test_long_format_surprisal_values_preserved():
    """
    The surprisal values from the wide format should appear unchanged
    in the corresponding long-format rows.
    """
    wide = _make_wide_results(n_rows=1)
    long_df = convert_surprisal_results_to_long(wide)

    cs_row   = long_df[long_df["is_switch"] == 1].iloc[0]
    mono_row = long_df[long_df["is_switch"] == 0].iloc[0]

    assert cs_row["surprisal_context_0"]   == 5.0
    assert mono_row["surprisal_context_0"] == 4.0


def test_long_format_drops_surprisal_difference_columns():
    """
    surprisal_difference_context_* columns should not appear in the long format
    (they are meaningless per-row after reshaping).
    """
    wide = _make_wide_results()
    long_df = convert_surprisal_results_to_long(wide)

    diff_cols = [c for c in long_df.columns if c.startswith("surprisal_difference_")]
    assert diff_cols == [], (
        f"Unexpected difference columns in long format: {diff_cols}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Calculator context prepending — mock tokenizer + model
#
# These tests verify that context words are physically prepended to the text
# passed to the language model, using a character-level mock tokenizer.
# ─────────────────────────────────────────────────────────────────────────────


class MockCharTokenizer:
    """
    Character-level tokenizer mock.

    Each character maps to a unique token ID: ord(char) + 10.
    The +10 offset avoids collision with special token IDs (0–3).

    Special tokens
    --------------
    CLS / BOS : 0  — decodes to ''
    SEP       : 1  — decodes to ''
    MASK      : 2  — decodes to ''
    PAD       : 3  — decodes to ''
    """

    CLS_ID  = 0
    SEP_ID  = 1
    MASK_ID = 2
    PAD_ID  = 3
    BOS_ID  = 0   # AR models share BOS with CLS for simplicity

    _SPECIAL = {0, 1, 2, 3}

    def __init__(self, model_max_length: int = 512):
        self.model_max_length = model_max_length
        self.mask_token_id    = self.MASK_ID
        self.bos_token_id     = self.BOS_ID
        self.pad_token        = "[PAD]"
        self.eos_token        = "[EOS]"
        self.pad_token_id     = self.PAD_ID

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) + 10 for c in text]
        if add_special_tokens:
            ids = [self.CLS_ID] + ids + [self.SEP_ID]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, token_ids):
        result = []
        for tid in token_ids:
            if tid in self._SPECIAL:
                continue
            char_code = tid - 10
            if 0 <= char_code <= 0x10FFFF:
                result.append(chr(char_code))
        return "".join(result)


class MockMaskedLMModel:
    """
    Returns uniform logits (vocab_size tokens equally likely) and records the
    sequence length *and full input tensor* of every call it receives.

    Note: received_inputs stores cloned tensors so callers can inspect which
    tokens the calculator placed at each position (e.g. MASK_ID).
    """

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.received_input_lengths: list[int] = []
        self.received_inputs: list[torch.Tensor] = []

    def __call__(self, input_ids):
        self.received_input_lengths.append(input_ids.shape[1])
        self.received_inputs.append(input_ids.clone())
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        result = MagicMock()
        result.logits = logits
        return result

    def to(self, device):
        return self

    def eval(self):
        return self


class MockCausalLMModel:
    """
    Returns uniform logits for autoregressive tests and records input lengths.
    """

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.received_input_lengths: list[int] = []

    def __call__(self, input_ids):
        self.received_input_lengths.append(input_ids.shape[1])
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        result = MagicMock()
        result.logits = logits
        return result

    def to(self, device):
        return self

    def eval(self):
        return self


def _make_masked_calc(model_max_length: int = 512, vocab_size: int = 300):
    """Return a MaskedLMSurprisalCalculator wired to mock tokenizer + model."""
    with patch.object(MaskedLMSurprisalCalculator, "__init__", lambda s, *a, **k: None):
        calc = MaskedLMSurprisalCalculator.__new__(MaskedLMSurprisalCalculator)
    calc.tokenizer            = MockCharTokenizer(model_max_length)
    calc.model                = MockMaskedLMModel(vocab_size)
    calc.device               = torch.device("cpu")
    calc.max_length           = model_max_length
    calc.context_clipped_count = 0
    calc.low_prob_events      = []
    return calc


def _make_ar_calc(model_max_length: int = 512, vocab_size: int = 300):
    """Return an AutoregressiveLMSurprisalCalculator wired to mock tokenizer + model."""
    with patch.object(AutoregressiveLMSurprisalCalculator, "__init__", lambda s, *a, **k: None):
        calc = AutoregressiveLMSurprisalCalculator.__new__(AutoregressiveLMSurprisalCalculator)
    calc.tokenizer            = MockCharTokenizer(model_max_length)
    calc.model                = MockCausalLMModel(vocab_size)
    calc.device               = torch.device("cpu")
    calc.max_length           = model_max_length
    calc.context_clipped_count = 0
    calc.low_prob_events      = []
    return calc


# ── Masked LM ────────────────────────────────────────────────────────────────

def test_masked_lm_model_input_grows_with_context():
    """
    When context words are provided, the masked LM should receive a *longer*
    input sequence than without context, confirming the context is prepended.

    Setup:
      words = ['A','B','C']  (3 single-char words)
      target word_index = 1 → 'B'
      context = 'XYZ'       (3 context chars split into ['X','Y','Z'])

    Without context:
      full_sentence = 'ABC'   → [CLS,A,B,C,SEP]  = 5 tokens
      masked input  = [CLS,A,MASK,C,SEP]          = 5 tokens

    With context 'XYZ':
      full_sentence = 'XYZABC' → [CLS,X,Y,Z,A,B,C,SEP] = 8 tokens
      masked input  = [CLS,X,Y,Z,A,MASK,C,SEP]          = 8 tokens
    """
    words = ["A", "B", "C"]

    calc = _make_masked_calc()
    calc.calculate_surprisal(word_index=1, words=words, context=None)
    len_no_ctx = max(calc.model.received_input_lengths)

    calc.model.received_input_lengths.clear()
    calc.calculate_surprisal(word_index=1, words=words, context="XYZ")
    len_with_ctx = max(calc.model.received_input_lengths)

    assert len_with_ctx > len_no_ctx, (
        f"Model input should be longer with context: {len_with_ctx} > {len_no_ctx}"
    )


def test_masked_lm_context_adds_exactly_n_tokens():
    """
    Adding a 3-character context ('XYZ') should increase the model input by
    exactly 3 tokens (one token per character with our char-level mock).
    """
    words = ["A", "B", "C"]
    calc  = _make_masked_calc()

    calc.calculate_surprisal(word_index=1, words=words, context=None)
    len_no_ctx = max(calc.model.received_input_lengths)

    calc.model.received_input_lengths.clear()
    calc.calculate_surprisal(word_index=1, words=words, context="XYZ")
    len_with_ctx = max(calc.model.received_input_lengths)

    assert len_with_ctx - len_no_ctx == 3, (
        f"Expected +3 tokens from 3-char context, got +{len_with_ctx - len_no_ctx}"
    )


def test_masked_lm_no_context_target_token_is_masked():
    """
    In the masked LM call, the token at the target word position must be
    replaced with MASK_ID (2 in our mock).  We verify this by inspecting
    calc.model.received_inputs, which the MockMaskedLMModel stores automatically.

    Sentence: 'ABC' (words=['A','B','C'], word_index=1 → 'B')
    Tokenised: [CLS(0), A(75), B(76), C(77), SEP(1)]
    The calculator should mask position 2 → [CLS, A, MASK(2), C, SEP].
    """
    words = ["A", "B", "C"]
    calc  = _make_masked_calc()

    calc.calculate_surprisal(word_index=1, words=words, context=None)

    assert calc.model.received_inputs, "Model was never called"
    # The last model call should have MASK_ID somewhere in the sequence
    last_input = calc.model.received_inputs[-1][0].tolist()

    assert MockCharTokenizer.MASK_ID in last_input, (
        f"MASK_ID ({MockCharTokenizer.MASK_ID}) not found in model input: {last_input}"
    )
    # Furthermore, MASK should be at position 2 (between CLS and 'C')
    assert last_input[2] == MockCharTokenizer.MASK_ID, (
        f"Expected MASK at position 2, got {last_input}"
    )


# ── Autoregressive LM ────────────────────────────────────────────────────────

def test_ar_model_prefix_grows_with_context():
    """
    The autoregressive model receives only the *prefix* (everything before the
    target word). Adding context should lengthen that prefix.

    Setup:
      words = ['A','B','C']  target word_index = 1 → 'B'
      prefix without context: 'A'    → [BOS,A,SEP] = 3 tokens
      prefix with context 'XYZ':     → [BOS,X,Y,Z,A,SEP] = 6 tokens
    """
    words = ["A", "B", "C"]

    calc = _make_ar_calc()
    calc.calculate_surprisal(word_index=1, words=words, context=None)
    len_no_ctx = min(calc.model.received_input_lengths)   # first call = shortest prefix

    calc.model.received_input_lengths.clear()
    calc.calculate_surprisal(word_index=1, words=words, context="XYZ")
    len_with_ctx = min(calc.model.received_input_lengths)

    assert len_with_ctx > len_no_ctx, (
        f"AR prefix should be longer with context: {len_with_ctx} > {len_no_ctx}"
    )


def test_ar_model_context_adds_exactly_n_tokens():
    """
    Adding a 3-character context ('XYZ') should increase the AR prefix by
    exactly 3 tokens.
    """
    words = ["A", "B", "C"]
    calc  = _make_ar_calc()

    calc.calculate_surprisal(word_index=1, words=words, context=None)
    len_no_ctx = min(calc.model.received_input_lengths)

    calc.model.received_input_lengths.clear()
    calc.calculate_surprisal(word_index=1, words=words, context="XYZ")
    len_with_ctx = min(calc.model.received_input_lengths)

    assert len_with_ctx - len_no_ctx == 3, (
        f"Expected +3 tokens from 3-char context, got +{len_with_ctx - len_no_ctx}"
    )


def test_ar_model_first_word_gets_finite_surprisal_from_bos_context():
    """
    For word_index=0 with no prior context, input_sentence is '' (empty).
    With add_special_tokens=True the tokenizer always returns at least [BOS],
    so current_input_ids.shape[1] >= 1 — the guard `shape[1] == 0` in the
    calculator is dead code that can never fire with any real tokenizer.

    The model receives [BOS] (or [BOS, SEP] with our mock) and returns logits
    for the position that predicts the first real token.  With uniform logits
    this equals log2(vocab_size) — a finite, well-defined value.

    Implication: callers should be aware that word_index=0 surprisal measures
    P(first_token | BOS), not a "no context" probability.
    """
    words = ["A", "B", "C"]
    calc  = _make_ar_calc(vocab_size=300)

    result = calc.calculate_surprisal(word_index=0, words=words, context=None)

    # Must be finite — the dead-code nan path never executes
    assert math.isfinite(result["surprisal"]), (
        f"Expected finite surprisal for word_index=0, got {result['surprisal']}"
    )
    assert result["surprisal"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Token alignment (_align_word_to_tokens) via mock tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _align(calc, sentence, words, word_index):
    return calc._align_word_to_tokens(sentence, words, word_index)


def test_masked_align_single_word_sentence():
    """
    'ABC' with words=['ABC'] and word_index=0 → all 3 character tokens found.
    Sentence tokenised: [CLS, A, B, C, SEP] = positions 0-4.
    _align skips [CLS](0) and [SEP](4) so valid range is 1-3.
    char_start=0, char_end=3 → tokens 1,2,3 all overlap.
    """
    calc = _make_masked_calc()
    sentence = "ABC"
    words    = ["ABC"]
    indices, strings = _align(calc, sentence, words, word_index=0)

    assert indices == [1, 2, 3], f"Expected [1,2,3], got {indices}"
    assert strings == ["A", "B", "C"]


def test_masked_align_second_word_of_two():
    """
    Sentence 'ABCD' split as words=['AB','CD'], word_index=1 → 'CD'.
    char_start=2, char_end=4 → tokens at positions 3,4 in [CLS,A,B,C,D,SEP].
    """
    calc = _make_masked_calc()
    sentence = "ABCD"
    words    = ["AB", "CD"]
    indices, strings = _align(calc, sentence, words, word_index=1)

    assert indices == [3, 4], f"Expected [3,4], got {indices}"
    assert strings == ["C", "D"]


def test_ar_align_skips_bos_token():
    """
    Autoregressive alignment skips the BOS token (position 0) when
    token_ids[0] == bos_token_id. Verify that char positions still map
    correctly after the skip.
    """
    calc = _make_ar_calc()
    sentence = "XY"
    words    = ["X", "Y"]
    indices, strings = _align(calc, sentence, words, word_index=1)

    # [BOS, X, Y, SEP] → start_idx=1 (BOS skipped)
    # char_start=1, char_end=2 → token 2 ('Y') overlaps
    assert 2 in indices, f"Expected token index 2 (Y) in {indices}"
    assert "Y" in strings


def test_align_returns_empty_for_out_of_range_word():
    """
    If the character span of the word has no corresponding tokens (edge case),
    _align_word_to_tokens should return empty lists rather than raise.
    """
    calc = _make_masked_calc()
    # Sentence is 'AB' but we claim word_index=1 spans chars [5,6) — impossible
    # Simulate by having very short sentence
    sentence = "A"
    words    = ["A", "B"]   # 'B' not actually in sentence
    # char_start=1, char_end=2 → beyond 'A' (len 1) → no match
    indices, strings = _align(calc, sentence, words, word_index=1)

    assert indices == [], f"Expected empty token list, got {indices}"
    assert strings == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Truncation tracking — per-row truncation categories
# ─────────────────────────────────────────────────────────────────────────────
#
# Token count reference (MockCharTokenizer, add_special_tokens=True):
#   text → [CLS] + [one token per char] + [SEP]
#   "ABC" → 5 tokens, "XYZAB" → 7 tokens
#   context='XYZ' → context_words=['XYZ'], prepended as 3 chars
# ─────────────────────────────────────────────────────────────────────────────


def test_truncation_clean():
    """
    No truncation when sentence + post-switch words fit well within max_length.

    max_length=20, words=['A','B','C','D'], word_index=2 ('C' is target).
    required_text='ABC' → 5 tokens, available_for_postswitch=15.
    'D' fits → all words included → 'clean'.
    """
    calc = _make_masked_calc(model_max_length=20)
    result = calc.calculate_surprisal(word_index=2, words=["A", "B", "C", "D"])
    assert result["truncation"] == "clean", (
        f"Expected 'clean', got {result['truncation']!r}"
    )


def test_truncation_right_trimmed_elif_branch():
    """
    When required tokens exactly fill max_length (elif branch), post-switch words
    are dropped → 'right_trimmed'.

    max_length=5, words=['A','B','C','D'], word_index=2 ('C' is target).
    required_text='ABC' → [CLS,A,B,C,SEP] = 5 tokens = max_length exactly.
    available_for_postswitch = 5 - 5 = 0 → elif branch fires.
    Post-switch 'D' exists but can't fit → 'right_trimmed'.
    """
    calc = _make_masked_calc(model_max_length=5)
    result = calc.calculate_surprisal(word_index=2, words=["A", "B", "C", "D"])
    assert result["truncation"] == "right_trimmed", (
        f"Expected 'right_trimmed' from elif branch, got {result['truncation']!r}"
    )


def test_truncation_right_trimmed_else_branch():
    """
    When some (but not all) post-switch words fit (else branch) → 'right_trimmed'.

    max_length=6, words=['A','B','C','D','E'], word_index=2 ('C' is target).
    required_text='ABC' → 5 tokens, available_for_postswitch=1.
    'D' fits (1 token ≤ 1), 'E' would need 2 total → dropped → 'right_trimmed'.
    """
    calc = _make_masked_calc(model_max_length=6)
    result = calc.calculate_surprisal(word_index=2, words=["A", "B", "C", "D", "E"])
    assert result["truncation"] == "right_trimmed", (
        f"Expected 'right_trimmed' from else branch, got {result['truncation']!r}"
    )


def test_truncation_left_clipped_no_postswitch():
    """
    Context overflow (if branch) with no post-switch words → 'left_clipped'.

    max_length=5, context='XYZ', words=['A','B'], word_index=1 ('B' is last word).
    context_words=['XYZ'], required_text='XYZAB' → 7 tokens > 5 → clips context.
    word_index=1 is the last word → no post-switch words → 'left_clipped' only.
    """
    calc = _make_masked_calc(model_max_length=5)
    result = calc.calculate_surprisal(word_index=1, words=["A", "B"], context="XYZ")
    assert result["truncation"] == "left_clipped", (
        f"Expected 'left_clipped', got {result['truncation']!r}"
    )


def test_truncation_both():
    """
    Context overflow AND post-switch words exist → 'both'.

    max_length=5, context='XYZ', words=['A','B','C'], word_index=1 ('B' is target).
    context_words=['XYZ'], required_text='XYZAB' → 7 tokens > 5 → clips context.
    Post-switch 'C' exists but available_for_postswitch=0 after clip → 'both'.
    """
    calc = _make_masked_calc(model_max_length=5)
    result = calc.calculate_surprisal(
        word_index=1, words=["A", "B", "C"], context="XYZ"
    )
    assert result["truncation"] == "both", (
        f"Expected 'both', got {result['truncation']!r}"
    )


@patch(SEG_PATCH, side_effect=list)
def test_dataset_stores_truncation_columns(mock_seg):
    """
    calculate_surprisal_for_dataset must write cs_truncation_context_{N} and
    mono_truncation_context_{N} columns. When mock returns truncation='clean',
    both columns should equal 'clean'.
    """
    df = make_analysis_df()
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[0]
        )

    assert "cs_truncation_context_0" in result.columns, (
        "Missing cs_truncation_context_0 column"
    )
    assert "mono_truncation_context_0" in result.columns, (
        "Missing mono_truncation_context_0 column"
    )
    assert result["cs_truncation_context_0"].iloc[0] == "clean", (
        f"Expected 'clean', got {result['cs_truncation_context_0'].iloc[0]!r}"
    )
    assert result["mono_truncation_context_0"].iloc[0] == "clean", (
        f"Expected 'clean', got {result['mono_truncation_context_0'].iloc[0]!r}"
    )


@patch(SEG_PATCH, side_effect=list)
def test_no_context_produces_no_context_truncation(mock_seg):
    """
    When context_len=1 but no context sentences exist, both truncation columns
    should be 'no_context' (calculator is never called for that context length).
    """
    df = make_analysis_df()  # no cs_context / mono_context columns
    calc = MockSurprisalCalc()

    with patch(SEG_PATCH, side_effect=list):
        result = calculate_surprisal_for_dataset(
            df, calc, show_progress=False, use_context=True, context_lengths=[1]
        )

    assert result["cs_truncation_context_1"].iloc[0] == "no_context", (
        f"Expected 'no_context', got {result['cs_truncation_context_1'].iloc[0]!r}"
    )
    assert result["mono_truncation_context_1"].iloc[0] == "no_context", (
        f"Expected 'no_context', got {result['mono_truncation_context_1'].iloc[0]!r}"
    )


def test_long_format_has_truncation_column():
    """
    convert_surprisal_results_to_long must carry truncation_context_{N} into the
    long format when the wide DataFrame contains cs_/mono_truncation_context_{N}.
    """
    wide = _make_wide_results(n_rows=2)
    wide["cs_truncation_context_0"]   = "clean"
    wide["mono_truncation_context_0"] = "clean"

    long_df = convert_surprisal_results_to_long(wide)

    assert "truncation_context_0" in long_df.columns, (
        f"Expected 'truncation_context_0' in long format. Columns: {list(long_df.columns)}"
    )
    assert (long_df["truncation_context_0"] == "clean").all(), (
        "All rows should have truncation_context_0 == 'clean'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: word_in_context column
# ─────────────────────────────────────────────────────────────────────────────


def _make_wide_with_context(cs_word, cs_context, mono_word, mono_context) -> pd.DataFrame:
    """Helper: wide-format df with specific word/context values."""
    wide = _make_wide_results(n_rows=1)
    wide["cs_word"]    = cs_word
    wide["mono_word"]  = mono_word
    wide["cs_context"] = cs_context
    wide["mono_context"] = mono_context
    return wide


def test_word_in_context_is_1_when_word_appears_in_context():
    """When the target word is a substring of context, word_in_context == 1."""
    wide = _make_wide_with_context(
        cs_word="hello", cs_context="I said hello yesterday",
        mono_word="你好", mono_context="佢話你好喎"
    )
    long_df = convert_surprisal_results_to_long(wide)

    cs_row   = long_df[long_df["is_switch"] == 1].iloc[0]
    mono_row = long_df[long_df["is_switch"] == 0].iloc[0]

    assert cs_row["word_in_context"] == 1, (
        f"Expected word_in_context=1 for CS row, got {cs_row['word_in_context']}"
    )
    assert mono_row["word_in_context"] == 1, (
        f"Expected word_in_context=1 for mono row, got {mono_row['word_in_context']}"
    )


def test_word_in_context_is_0_when_word_absent():
    """When the target word does not appear in context, word_in_context == 0."""
    wide = _make_wide_with_context(
        cs_word="hello", cs_context="completely different text",
        mono_word="你好", mono_context="完全唔同嘅文字"
    )
    long_df = convert_surprisal_results_to_long(wide)

    cs_row   = long_df[long_df["is_switch"] == 1].iloc[0]
    mono_row = long_df[long_df["is_switch"] == 0].iloc[0]

    assert cs_row["word_in_context"] == 0, (
        f"Expected word_in_context=0 for CS row, got {cs_row['word_in_context']}"
    )
    assert mono_row["word_in_context"] == 0, (
        f"Expected word_in_context=0 for mono row, got {mono_row['word_in_context']}"
    )


def test_word_in_context_is_0_when_context_is_na():
    """When context is 'N/A', word_in_context must be 0 regardless of word value."""
    wide = _make_wide_with_context(
        cs_word="N/A", cs_context="N/A",   # edge: word == 'N/A' but context is also 'N/A'
        mono_word="hello", mono_context="N/A"
    )
    long_df = convert_surprisal_results_to_long(wide)

    for _, row in long_df.iterrows():
        assert row["word_in_context"] == 0, (
            f"Expected word_in_context=0 when context='N/A', got {row['word_in_context']}"
        )
