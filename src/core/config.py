"""
Configuration management for the code-switching analysis pipeline.

This module loads and validates configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing application settings."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {self.config_path}")
        
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate data path
        data_path = self.get('data.path')
        if data_path and not os.path.exists(data_path):
            logger.warning(f"Data path does not exist: {data_path}")
        
        # Validate processing settings
        min_words = self.get('processing.min_sentence_words')
        if min_words < 1:
            raise ValueError("min_sentence_words must be at least 1")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.path')
            default: Default value to return if key is not found. If None, raises KeyError.
            
        Returns:
            Configuration value or default if key not found and default is provided
            
        Raises:
            KeyError: If the key is not found in configuration and no default is provided
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                if k not in value:
                    if default is not None:
                        return default
                    raise KeyError(f"Configuration key not found: {key}")
                value = value[k]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Configuration key not found: {key}")
        
        return value
    
    def get_data_path(self) -> str:
        """Get the data directory path."""
        return self.get('data.path')
    
    def get_min_sentence_words(self) -> int:
        """Get minimum number of words required to keep a sentence."""
        return self.get('processing.min_sentence_words')
    
    def get_time_gap_threshold_ms(self) -> int:
        """Get time gap threshold in milliseconds for grouping annotations into sentences."""
        return self.get('processing.time_gap_threshold_ms')
    
    def get_results_dir(self) -> str:
        """Get base directory for all results."""
        return self.get('output.results_dir')
    
    def get_figures_dir(self) -> str:
        """Get base directory for all figures."""
        return self.get('output.figures_dir')
    
    def get_preprocessing_results_dir(self) -> str:
        """Get directory for preprocessing results (CSV files)."""
        return self.get('output.results.preprocessing_dir')
    
    def get_exploratory_results_dir(self) -> str:
        """Get directory for exploratory analysis results."""
        return self.get('output.results.exploratory_dir')
    
    def get_preprocessing_figures_dir(self) -> str:
        """Get directory for preprocessing figures."""
        return self.get('output.figures.preprocessing_dir')
    
    def get_exploratory_figures_dir(self) -> str:
        """Get directory for exploratory analysis figures."""
        return self.get('output.figures.exploratory_dir')
    
    def get_processed_data_dir(self) -> str:
        """Get output directory for processed data (CSV files)."""
        return self.get_preprocessing_results_dir()
    
    def get_csv_with_fillers_path(self) -> str:
        """Get output path for CSV with fillers."""
        filename = self.get('output.csv_with_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_without_fillers_path(self) -> str:
        """Get output path for CSV without fillers."""
        filename = self.get('output.csv_without_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_all_sentences_path(self) -> str:
        """Get output path for CSV with all sentences (monolingual + code-switched)."""
        filename = self.get('output.csv_all_sentences')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_mono_with_fillers_path(self) -> str:
        """Get output path for Cantonese monolingual sentences WITH fillers."""
        filename = self.get('output.csv_cantonese_mono_with_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_mono_without_fillers_path(self) -> str:
        """Get output path for Cantonese monolingual sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_mono_without_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_english_mono_with_fillers_path(self) -> str:
        """Get output path for English monolingual sentences WITH fillers."""
        filename = self.get('output.csv_english_mono_with_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_english_mono_without_fillers_path(self) -> str:
        """Get output path for English monolingual sentences WITHOUT fillers."""
        filename = self.get('output.csv_english_mono_without_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_translated_path(self) -> str:
        """Get output path for Cantonese-translated code-switched sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_translated')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    # Translation configuration methods
    
    def get_translation_model(self) -> str:
        """Get NLLB model name."""
        return self.get('translation.model')
    
    def get_translation_device(self) -> str:
        """Get device for NLLB (auto, cpu, cuda)."""
        return self.get('translation.device')
    
    def get_translation_use_cache(self) -> bool:
        """Get whether to use caching for translations."""
        return self.get('translation.use_cache')
    
    def get_translation_cache_dir(self) -> str:
        """Get directory for translation cache."""
        return self.get('translation.cache_dir')
    
    def get_translation_max_tokens(self) -> int:
        """Get maximum tokens for translation responses."""
        return self.get('translation.max_tokens')
    
    def get_analysis_min_cantonese_words(self) -> int:
        """Get minimum number of Cantonese words required at sentence start for analysis dataset."""
        return self.get('analysis.min_cantonese_words')
    
    def get_analysis_window_size(self) -> int:
        """Get window size for POS matching around switch points."""
        return self.get('analysis.window_size')
    
    def get_analysis_similarity_threshold(self) -> float:
        """Get minimum Levenshtein similarity threshold for matches."""
        return self.get('analysis.similarity_threshold')
