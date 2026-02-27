"""
Configuration management for the code-switching analysis pipeline.

"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing application settings."""
    
    def __init__(self):
        """
        Initialize configuration from YAML file.
        
        Loads configuration from config/config.yaml relative to project root.
        """
        
        project_root = Path(__file__).parent.parent.parent
        self.config_path = project_root / "config" / "config.yaml"
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
    
    def get_results_dir(self) -> str:
        """Get base directory for all results."""
        return self.get('output.results_dir')
    
    def get_preprocessing_results_dir(self) -> str:
        """Get directory for preprocessing results (CSV files)."""
        return self.get('output.results.preprocessing_dir')
    
    def get_matching_results_dir(self) -> str:
        """Get directory for matching analysis results."""
        return self.get('output.results.matching_dir')
    
    def get_surprisal_results_dir(self) -> str:
        """Get directory for surprisal experiment results."""
        return self.get('output.results.surprisal_dir')
    
    def get_preprocessing_figures_dir(self) -> str:
        """Get directory for preprocessing figures."""
        return self.get('output.figures.preprocessing_dir')
    
    def get_matching_figures_dir(self) -> str:
        """Get directory for matching analysis figures."""
        return self.get('output.figures.matching_dir')
    
    def get_surprisal_figures_dir(self) -> str:
        """Get directory for surprisal experiment figures."""
        return self.get('output.figures.surprisal_dir')
    
    def get_csv_all_sentences_path(self) -> str:
        """Get output path for CSV with all sentences (monolingual + code-switched)."""
        filename = self.get('output.csv_all_sentences')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_mono_without_fillers_path(self) -> str:
        """Get output path for Cantonese monolingual sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_mono_without_fillers')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_cantonese_translated_path(self) -> str:
        """Get output path for Cantonese-translated code-switched sentences WITHOUT fillers."""
        filename = self.get('output.csv_cantonese_translated')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_csv_interviewer_path(self) -> str:
        """Get output path for interviewer sentences (IR tier)."""
        filename = self.get('output.csv_interviewer')
        processed_dir = self.get_preprocessing_results_dir()
        return os.path.join(processed_dir, filename)
    
    def get_translation_model(self) -> str:
        """Get NLLB model name."""
        return self.get('translation.model')
    
    def get_translation_device(self) -> str:
        """Get device for NLLB (auto, cpu, cuda)."""
        return self.get('translation.device')
    
    def get_analysis_min_cantonese_words(self) -> int:
        """Get minimum number of Cantonese words required at sentence start for analysis dataset."""
        return self.get('analysis.min_cantonese_words')
    
    def get_analysis_window_sizes(self) -> List[int]:
        """Get list of window sizes for POS matching around switch points."""
        window_sizes = self.get('analysis.window_sizes', None)
        if window_sizes is not None:
            if isinstance(window_sizes, list):
                return window_sizes
            # If single value, convert to list
            return [window_sizes]
    
    def get_analysis_similarity_threshold(self) -> float:
        """Get minimum Levenshtein similarity threshold for matches."""
        return self.get('analysis.similarity_threshold')
    
    def get_analysis_num_workers(self) -> Optional[int]:
        """Get number of CPU cores to leave free. Returns None to use all available cores."""
        return self.get('analysis.parallel.num_workers', None)

    def get_figure_colors(self) -> Dict[str, str]:
        """Get the figure color palette as a flat dict of name → hex string."""
        return self.get('figures.colors')
    