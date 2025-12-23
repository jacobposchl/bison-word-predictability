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
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config if config else self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config file: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'data': {
                'path': './data'
            },
            'processing': {
                'min_sentence_words': 2,
                'time_gap_threshold_ms': 1000
            },
            'translation': {
                'model': 'gpt-4',
                'use_cache': True,
                'cache_dir': 'cache/translations',
                'temperature': 0.3,
                'max_tokens': 200
            },
            'output': {
                'results_dir': 'results',
                'figures_dir': 'figures',
                'results': {
                    'preprocessing_dir': 'results/preprocessing',
                    'exploratory_dir': 'results/exploratory',
                    'dash_analysis_dir': 'results/dash_analysis'
                },
                'figures': {
                    'preprocessing_dir': 'figures/preprocessing',
                    'exploratory_dir': 'figures/exploratory'
                },
                'csv_with_fillers': 'code_switching_WITH_fillers.csv',
                'csv_without_fillers': 'code_switching_WITHOUT_fillers.csv',
                'csv_all_sentences': 'all_sentences.csv',
                'csv_cantonese_mono_with_fillers': 'cantonese_monolingual_WITH_fillers.csv',
                'csv_cantonese_mono_without_fillers': 'cantonese_monolingual_WITHOUT_fillers.csv',
                'csv_english_mono_with_fillers': 'english_monolingual_WITH_fillers.csv',
                'csv_english_mono_without_fillers': 'english_monolingual_WITHOUT_fillers.csv',
                'csv_cantonese_translated': 'cantonese_translated_WITHOUT_fillers.csv'
            }
        }
    
    def _validate_config(self):
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
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_data_path(self) -> str:
        """Get the data directory path."""
        return self.get('data.path', './data')
    
    def get_min_sentence_words(self) -> int:
        """Get minimum number of words required to keep a sentence."""
        return self.get('processing.min_sentence_words', 2)
    
    def get_time_gap_threshold_ms(self) -> int:
        """Get time gap threshold in milliseconds for grouping annotations into sentences."""
        return self.get('processing.time_gap_threshold_ms', 1000)
    
    def get_results_dir(self) -> str:
        """Get base directory for all results."""
        return self.get('output.results_dir', 'results')
    
    def get_figures_dir(self) -> str:
        """Get base directory for all figures."""
        return self.get('output.figures_dir', 'figures')
    
    def get_preprocessing_results_dir(self) -> str:
        """Get directory for preprocessing results (CSV files)."""
        return self.get('output.results.preprocessing_dir', 'results/preprocessing')
    
    def get_exploratory_results_dir(self) -> str:
        """Get directory for exploratory analysis results."""
        return self.get('output.results.exploratory_dir', 'results/exploratory')
    
    def get_preprocessing_figures_dir(self) -> str:
        """Get directory for preprocessing figures."""
        return self.get('output.figures.preprocessing_dir', 'figures/preprocessing')
    
    def get_exploratory_figures_dir(self) -> str:
        """Get directory for exploratory analysis figures."""
        return self.get('output.figures.exploratory_dir', 'figures/exploratory')
    
    def get_processed_data_dir(self) -> str:
        """Get output directory for processed data (CSV files). Alias for get_preprocessing_results_dir()."""
        return self.get_preprocessing_results_dir()
    
    def get_csv_with_fillers_path(self) -> str:
        """Get output path for CSV with fillers."""
        filename = self.get('output.csv_with_fillers', 'code_switching_WITH_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_without_fillers_path(self) -> str:
        """Get output path for CSV without fillers."""
        filename = self.get('output.csv_without_fillers', 'code_switching_WITHOUT_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_all_sentences_path(self) -> str:
        """Get output path for CSV with all sentences (monolingual + code-switched)."""
        filename = self.get('output.csv_all_sentences', 'all_sentences.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_mono_with_fillers_path(self) -> str:
        """Get output path for Cantonese monolingual sentences WITH fillers."""
        filename = self.get('output.csv_cantonese_mono_with_fillers', 'cantonese_monolingual_WITH_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_mono_without_fillers_path(self) -> str:
        """Get output path for Cantonese monolingual sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_mono_without_fillers', 'cantonese_monolingual_WITHOUT_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_english_mono_with_fillers_path(self) -> str:
        """Get output path for English monolingual sentences WITH fillers."""
        filename = self.get('output.csv_english_mono_with_fillers', 'english_monolingual_WITH_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_english_mono_without_fillers_path(self) -> str:
        """Get output path for English monolingual sentences WITHOUT fillers."""
        filename = self.get('output.csv_english_mono_without_fillers', 'english_monolingual_WITHOUT_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_translated_path(self) -> str:
        """Get output path for Cantonese-translated code-switched sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_translated', 'cantonese_translated_WITHOUT_fillers.csv')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    # Translation configuration methods
    # NOTE: API key is NOT stored in config for security - must be passed as argument
    
    def get_translation_model(self) -> str:
        """Get OpenAI model for translation."""
        return self.get('translation.model', 'gpt-4')
    
    def get_translation_use_cache(self) -> bool:
        """Get whether to use caching for translations."""
        return self.get('translation.use_cache', True)
    
    def get_translation_cache_dir(self) -> str:
        """Get directory for translation cache."""
        return self.get('translation.cache_dir', 'cache/translations')
    
    def get_translation_temperature(self) -> float:
        """Get temperature for translation API calls."""
        return self.get('translation.temperature', 0.3)
    
    def get_translation_max_tokens(self) -> int:
        """Get maximum tokens for translation responses."""
        return self.get('translation.max_tokens', 200)

