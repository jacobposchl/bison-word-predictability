"""
Data loading utilities for code-switching experiments.

This module provides functions for loading code-switched and monolingual
sentences from CSV files.
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def load_code_switched_sentences(config, use_fillers: bool = False) -> pd.DataFrame:
    """
    Load code-switched sentences from CSV.
    
    Args:
        config: Config object
        use_fillers: If True, load sentences with fillers; else without
        
    Returns:
        DataFrame with code-switched sentences
    """
    if use_fillers:
        csv_path = Path(config.get_csv_with_fillers_path())
    else:
        csv_path = Path(config.get_csv_without_fillers_path())
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    logger.info(f"Loading code-switched sentences from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter to only Cantonese matrix language
    df_cantonese = df[df['matrix_language'] == 'Cantonese'].copy()
    logger.info(f"Loaded {len(df)} total sentences, {len(df_cantonese)} with Cantonese matrix language")
    
    return df_cantonese


def load_monolingual_sentences(config) -> Dict[str, pd.DataFrame]:
    """
    Load monolingual sentences from all_sentences.csv.
    
    Args:
        config: Config object
        
    Returns:
        Dictionary with 'cantonese' and 'english' DataFrames
    """
    csv_path = Path(config.get_csv_all_sentences_path())
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    logger.info(f"Loading all sentences from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract monolingual sentences based on pattern
    # Cantonese-only: patterns like "C5", "C10" (only C, no E)
    # English-only: patterns like "E3", "E7" (only E, no C)
    
    cantonese_mask = df['pattern'].str.match(r'^C\d+$', na=False)
    english_mask = df['pattern'].str.match(r'^E\d+$', na=False)
    
    cantonese_df = df[cantonese_mask].copy()
    english_df = df[english_mask].copy()
    
    logger.info(f"Found {len(cantonese_df)} monolingual Cantonese sentences")
    logger.info(f"Found {len(english_df)} monolingual English sentences")
    
    return {
        'cantonese': cantonese_df,
        'english': english_df
    }
