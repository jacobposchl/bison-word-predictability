"""
Report generation functions for regression analysis.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def create_results_row(
    results: Dict[str, Dict],
    model_type: str,
    window_size: int,
    context_length: int,
    n_train: int,
    n_test: int
) -> Dict:
    """
    Create a single row of results for the consolidated results CSV.
    
    Args:
        results: Dictionary of model results
        model_type: Type of model (masked/autoregressive)
        window_size: Window size used
        context_length: Context length used
        n_train: Number of training samples
        n_test: Number of test samples
        
    Returns:
        Dictionary with all relevant results for one configuration
    """
    row = {
        'model_type': model_type,
        'window_size': window_size,
        'context_length': context_length,
        'n_train': n_train,
        'n_test': n_test,
    }
    
    # Add metrics for each model
    for model_name in ['control', 'surprisal', 'entropy', 'surprisal_entropy']:
        if model_name in results:
            result = results[model_name]
            row[f'{model_name}_auc'] = result['auc']
            row[f'{model_name}_accuracy'] = result['accuracy']
            row[f'{model_name}_precision'] = result['precision']
            row[f'{model_name}_recall'] = result['recall']
            row[f'{model_name}_f1'] = result['f1']
            row[f'{model_name}_n_features'] = len(result['features'])
            row[f'{model_name}_intercept'] = result['intercept']
    
    return row

