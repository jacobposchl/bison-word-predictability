"""
Plotting functions for regression analysis.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


def plot_roc_curves(results: Dict[str, Dict], y_test, output_dir: Path):
    """
    Create ROC curve plot for all models.
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={result['auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Code-Switch Detection Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = output_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curves to {roc_path}")


def plot_model_comparison(results: Dict[str, Dict], output_dir: Path):
    """
    Create bar chart comparing model performance.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'model': model_name,
            'AUC': result['auc'],
            'Accuracy': result['accuracy'],
            'F1': result['f1']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.set_index('model')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comp_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    comp_path = output_dir / "model_comparison.png"
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison plot to {comp_path}")


def plot_all_regression_results(results: Dict[str, Dict], y_test, output_dir: Path):
    """
    Create all regression plots.
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        output_dir: Directory to save plots
    """
    plot_roc_curves(results, y_test, output_dir)
    plot_model_comparison(results, output_dir)

