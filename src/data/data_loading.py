"""
Data loading utilities for code-switching experiments.

This module provides functions for loading code-switched and monolingual
sentences from CSV files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
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


def load_dataset(dataset: str = 'ALL', config=None) -> pd.DataFrame:
    """
    Load the appropriate CSV dataset for analysis.
    
    Args:
        dataset: 'ALL' (all sentences), 'WITH' (code-switched with fillers), 
                 or 'WITHOUT' (code-switched without fillers)
        config: Config object (optional, will create one if not provided)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If dataset parameter is invalid
        FileNotFoundError: If CSV file doesn't exist
    """
    if config is None:
        from src.core.config import Config
        config = Config()
    
    # Get CSV path from config
    if dataset.upper() == 'ALL':
        csv_path = config.get_csv_all_sentences_path()
    elif dataset.upper() == 'WITH':
        csv_path = config.get_csv_with_fillers_path()
    elif dataset.upper() == 'WITHOUT':
        csv_path = config.get_csv_without_fillers_path()
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'ALL', 'WITH', or 'WITHOUT'")
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Please run preprocessing first: python scripts/preprocess/preprocess.py"
        )
    
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} sentences")
    
    return df

def load_monolingual_csvs(config, use_fillers: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load pre-filtered monolingual sentences from CSV files.
    
    Args:
        config: Config object
        use_fillers: If True, load sentences with fillers; else without
        
    Returns:
        Dictionary with 'cantonese' and 'english' DataFrames
    """
    if use_fillers:
        cant_path = Path(config.get_csv_cantonese_mono_with_fillers_path())
        eng_path = Path(config.get_csv_english_mono_with_fillers_path())
    else:
        cant_path = Path(config.get_csv_cantonese_mono_without_fillers_path())
        eng_path = Path(config.get_csv_english_mono_without_fillers_path())
    
    if not cant_path.exists():
        raise FileNotFoundError(
            f"Cantonese monolingual CSV not found: {cant_path}\\n"
            f"Please run preprocessing first: python scripts/preprocess/preprocess.py"
        )
    
    if not eng_path.exists():
        raise FileNotFoundError(
            f"English monolingual CSV not found: {eng_path}\\n"
            f"Please run preprocessing first: python scripts/preprocess/preprocess.py"
        )
    
    logger.info(f"Loading monolingual sentences from CSVs...")
    cantonese_df = pd.read_csv(cant_path)
    english_df = pd.read_csv(eng_path)
    
    logger.info(f"Loaded {len(cantonese_df)} Cantonese monolingual sentences")
    logger.info(f"Loaded {len(english_df)} English monolingual sentences")
    
    return {
        'cantonese': cantonese_df,
        'english': english_df
    }