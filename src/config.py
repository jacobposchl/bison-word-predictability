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
            project_root = Path(__file__).parent.parent
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
                'buffer_ms': 50,
                'min_sentence_words': 2
            },
            'output': {
                'processed_data_dir': 'processed_data',
                'csv_with_fillers': 'code_switching_WITH_fillers.csv',
                'csv_without_fillers': 'code_switching_WITHOUT_fillers.csv',
                'figures_dir': 'figures'
            }
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate data path
        data_path = self.get('data.path')
        if data_path and not os.path.exists(data_path):
            logger.warning(f"Data path does not exist: {data_path}")
        
        # Validate processing settings
        buffer_ms = self.get('processing.buffer_ms')
        if buffer_ms < 0:
            raise ValueError("buffer_ms must be non-negative")
        
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
    
    def get_buffer_ms(self) -> float:
        """Get the time buffer in milliseconds for sentence overlap detection."""
        return self.get('processing.buffer_ms', 50) / 1000.0  # Convert to seconds
    
    def get_min_sentence_words(self) -> int:
        """Get minimum number of words required to keep a sentence."""
        return self.get('processing.min_sentence_words', 2)
    
    def get_processed_data_dir(self) -> str:
        """Get output directory for processed data (CSV files)."""
        return self.get('output.processed_data_dir', 'processed_data')
    
    def get_csv_with_fillers_path(self) -> str:
        """Get output path for CSV with fillers."""
        filename = self.get('output.csv_with_fillers', 'code_switching_WITH_fillers.csv')
        processed_dir = self.get_processed_data_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_without_fillers_path(self) -> str:
        """Get output path for CSV without fillers."""
        filename = self.get('output.csv_without_fillers', 'code_switching_WITHOUT_fillers.csv')
        processed_dir = self.get_processed_data_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_all_sentences_path(self) -> str:
        """Get output path for CSV with all sentences (monolingual + code-switched)."""
        filename = self.get('output.csv_all_sentences', 'all_sentences.csv')
        processed_dir = self.get_processed_data_dir()
        return os.path.join(processed_dir, filename)
    
    def get_figures_dir(self) -> str:
        """Get output directory for figures."""
        return self.get('output.figures_dir', 'figures')

