"""
Logistic Regression Analysis for Code-Switch Detection

This script performs logistic regression to test whether surprisal and entropy
improve prediction of code-switched vs monolingual sentences beyond baseline
features (frequency, word length, sentence length, POS).

Models:
1. Control model: frequency + word_length + sentence_length + position + POS
2. Surprisal model: control features + surprisal
3. Entropy model: control features + entropy
4. Surprisal + Entropy model: control features + surprisal + entropy

Usage:
    python scripts/regression/regression.py --model autoregressive
    python scripts/regression/regression.py --model masked
    python scripts/regression/regression.py --model autoregressive --results-dir results/surprisal_autoregressive
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple

from src.core.config import Config
from src.analysis.pos_tagging import pos_tag_cantonese

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Logistic regression analysis for code-switch detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['masked', 'autoregressive'],
        help='Type of model used for surprisal calculation'
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing surprisal_results.csv (default: results/surprisal/{model})"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for train/test split (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save regression results (default: results/regression_{model})"
    )
    
    return parser.parse_args()


def load_surprisal_results(results_path: Path) -> pd.DataFrame:
    """Load surprisal_results.csv."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"Surprisal results not found at: {results_path}\n"
            f"Please run surprisal calculation first: python scripts/surprisal/surprisal.py --model {results_path.parent.name.split('_')[-1]}"
        )
    
    logger.info(f"Loading surprisal results from {results_path}")
    df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(df)} rows")
    
    return df


def extract_pos_at_position(sentence: str, word_index: int) -> str:
    """
    Extract POS tag at a specific word position in a sentence.
    
    Args:
        sentence: Cantonese sentence (space-separated or unsegmented)
        word_index: 0-based index of the word position
        
    Returns:
        POS tag string, or 'UNKNOWN' if not found
    """
    try:
        tagged = pos_tag_cantonese(sentence)
        if word_index < len(tagged):
            return tagged[word_index][1]  # Return POS tag
        return 'UNKNOWN'
    except Exception as e:
        logger.debug(f"Error extracting POS for sentence at index {word_index}: {e}")
        return 'UNKNOWN'


def prepare_data_for_regression(df: pd.DataFrame, context_length: int = None) -> pd.DataFrame:
    """
    Create dataset where each row is a sentence observation.
    
    Creates two rows per original row:
    - One for CS sentence (label=1)
    - One for mono sentence (label=0)
    
    Args:
        df: DataFrame from surprisal_results.csv
        context_length: Context length to use (if None, uses old column names without context suffix)
        
    Returns:
        DataFrame with one row per sentence observation
    """
    logger.info("Preparing data for regression...")
    
    rows = []
    
    for idx, row in df.iterrows():
        # CS sentence row (label = 1)
        cs_word = row.get('cs_word', '')
        cs_sentence = row.get('cs_translation', '')
        cs_switch_idx = int(row.get('switch_index', 0))
        
        # Calculate sentence length
        cs_words_list = cs_sentence.split() if pd.notna(cs_sentence) else []
        cs_sentence_length = len(cs_words_list)
        
        # Calculate normalized position
        cs_position_norm = cs_switch_idx / cs_sentence_length if cs_sentence_length > 0 else 0.0
        
        # Extract POS tag
        cs_pos = extract_pos_at_position(cs_sentence, cs_switch_idx)
        
        # Get surprisal and entropy columns (with context length suffix if specified)
        if context_length is not None:
            cs_surprisal_col = f'cs_surprisal_context_{context_length}'
            cs_probability_col = f'cs_probability_context_{context_length}'
            cs_entropy_col = f'cs_entropy_context_{context_length}'
        else:
            cs_surprisal_col = 'cs_surprisal_total'
            cs_probability_col = 'cs_probability'
            cs_entropy_col = 'cs_entropy'
        
        rows.append({
            'is_code_switched': 1,
            'word': cs_word if pd.notna(cs_word) else '',
            'word_length': len(str(cs_word)) if pd.notna(cs_word) else 0,
            'sentence_length': cs_sentence_length,
            'position_normalized': cs_position_norm,
            'pos_tag': cs_pos,
            'word_frequency': row.get(cs_probability_col, np.nan),  # Use probability as frequency proxy
            'surprisal': row.get(cs_surprisal_col, np.nan),
            'entropy': row.get(cs_entropy_col, np.nan),
            'sentence_id': f"{idx}_cs",
            'original_row_id': idx
        })
        
        # Mono sentence row (label = 0)
        mono_word = row.get('mono_word', '')
        mono_sentence = row.get('matched_mono', '')
        mono_switch_idx = int(row.get('matched_switch_index', 0))
        
        # Calculate sentence length
        mono_words_list = mono_sentence.split() if pd.notna(mono_sentence) else []
        mono_sentence_length = len(mono_words_list)
        
        # Calculate normalized position
        mono_position_norm = mono_switch_idx / mono_sentence_length if mono_sentence_length > 0 else 0.0
        
        # Extract POS tag
        mono_pos = extract_pos_at_position(mono_sentence, mono_switch_idx)
        
        # Get surprisal and entropy columns (with context length suffix if specified)
        if context_length is not None:
            mono_surprisal_col = f'mono_surprisal_context_{context_length}'
            mono_probability_col = f'mono_probability_context_{context_length}'
            mono_entropy_col = f'mono_entropy_context_{context_length}'
        else:
            mono_surprisal_col = 'mono_surprisal_total'
            mono_probability_col = 'mono_probability'
            mono_entropy_col = 'mono_entropy'
        
        rows.append({
            'is_code_switched': 0,
            'word': mono_word if pd.notna(mono_word) else '',
            'word_length': len(str(mono_word)) if pd.notna(mono_word) else 0,
            'sentence_length': mono_sentence_length,
            'position_normalized': mono_position_norm,
            'pos_tag': mono_pos,
            'word_frequency': row.get(mono_probability_col, np.nan),  # Use probability as frequency proxy
            'surprisal': row.get(mono_surprisal_col, np.nan),
            'entropy': row.get(mono_entropy_col, np.nan),
            'sentence_id': f"{idx}_mono",
            'original_row_id': idx
        })
    
    result_df = pd.DataFrame(rows)
    
    # Remove rows with missing critical values
    n_before = len(result_df)
    result_df = result_df.dropna(subset=['word_frequency', 'surprisal', 'entropy'])
    n_after = len(result_df)
    
    if n_before > n_after:
        logger.info(f"Removed {n_before - n_after} rows with missing values")
    
    logger.info(f"Created {len(result_df)} observations ({len(result_df[result_df['is_code_switched']==1])} CS, {len(result_df[result_df['is_code_switched']==0])} mono)")
    
    return result_df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix with one-hot encoding for POS tags.
    
    Args:
        df: DataFrame from prepare_data_for_regression()
        
    Returns:
        Tuple of (feature_df, feature_names)
    """
    logger.info("Preparing features...")
    
    # Create copy
    feature_df = df.copy()
    
    # One-hot encode POS tags
    pos_dummies = pd.get_dummies(feature_df['pos_tag'], prefix='pos')
    feature_df = pd.concat([feature_df, pos_dummies], axis=1)
    
    # Get feature names
    numeric_features = [
        'word_length',
        'sentence_length', 
        'position_normalized',
        'word_frequency'
    ]
    
    pos_feature_names = [col for col in pos_dummies.columns]
    
    feature_names = numeric_features + pos_feature_names
    
    # Select only feature columns
    feature_df = feature_df[numeric_features + pos_feature_names]
    
    logger.info(f"Prepared {len(feature_names)} features")
    logger.info(f"  Numeric features: {len(numeric_features)}")
    logger.info(f"  POS features: {len(pos_feature_names)}")
    
    return feature_df, feature_names


def fit_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Fit multiple logistic regression models with different feature sets.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target labels
        feature_names: List of feature names
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with model results
    """
    logger.info("Fitting models...")
    
    # Define feature sets
    control_features = [
        'word_length', 'sentence_length', 'position_normalized', 'word_frequency'
    ] + [f for f in feature_names if f.startswith('pos_')]
    
    surprisal_features = control_features + ['surprisal']
    entropy_features = control_features + ['entropy']
    surprisal_entropy_features = control_features + ['surprisal', 'entropy']
    
    feature_sets = {
        'control': control_features,
        'surprisal': surprisal_features,
        'entropy': entropy_features,
        'surprisal_entropy': surprisal_entropy_features
    }
    
    results = {}
    
    for model_name, features in feature_sets.items():
        # Check which features are available
        available_features = [f for f in features if f in X_train.columns]
        missing_features = [f for f in features if f not in X_train.columns]
        
        if missing_features:
            logger.warning(f"Model '{model_name}': Missing features {missing_features}, using available features only")
        
        if not available_features:
            logger.error(f"Model '{model_name}': No features available, skipping")
            continue
        
        # Select features
        X_train_subset = X_train[available_features]
        X_test_subset = X_test[available_features]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        
        # Fit model
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Coefficients
        coefficients = dict(zip(available_features, model.coef_[0]))
        
        # Intercept
        intercept = model.intercept_[0]
        
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'features': available_features,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'coefficients': coefficients,
            'intercept': intercept,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"  {model_name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return results


def compare_models(results: Dict[str, Dict], y_test: pd.Series) -> pd.DataFrame:
    """
    Compare model performance and test for improvement.
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Comparing models...")
    
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'model': model_name,
            'auc': result['auc'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'n_features': len(result['features'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('auc', ascending=False)
    
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    return comparison_df


def save_results(
    results: Dict[str, Dict],
    comparison_df: pd.DataFrame,
    output_dir: Path,
    model_type: str
):
    """Save regression results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model comparison
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved model comparison to {comparison_path}")
    
    # Save detailed results for each model
    for model_name, result in results.items():
        # Save coefficients
        coeff_df = pd.DataFrame([
            {'feature': feat, 'coefficient': coeff}
            for feat, coeff in result['coefficients'].items()
        ])
        coeff_df = coeff_df.sort_values('coefficient', key=abs, ascending=False)
        
        coeff_path = output_dir / f"{model_name}_coefficients.csv"
        coeff_df.to_csv(coeff_path, index=False)
        
        # Save classification report
        report_df = pd.DataFrame(result['classification_report']).transpose()
        report_path = output_dir / f"{model_name}_classification_report.csv"
        report_df.to_csv(report_path)
    
    # Save summary text file
    summary_path = output_dir / "regression_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model type: {model_type}\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n")
        for model_name, result in results.items():
            f.write(f"\n{model_name.upper()} MODEL:\n")
            f.write(f"  AUC: {result['auc']:.4f}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1 Score: {result['f1']:.4f}\n")
            f.write(f"  Number of features: {len(result['features'])}\n")
            f.write(f"  Intercept: {result['intercept']:.4f}\n")
            f.write(f"\n  Top 10 Coefficients (by absolute value):\n")
            coeff_df = pd.DataFrame([
                {'feature': feat, 'coefficient': coeff}
                for feat, coeff in result['coefficients'].items()
            ])
            coeff_df = coeff_df.sort_values('coefficient', key=abs, ascending=False)
            for _, row in coeff_df.head(10).iterrows():
                f.write(f"    {row['feature']}: {row['coefficient']:.4f}\n")
    
    logger.info(f"Saved summary to {summary_path}")


def plot_results(results: Dict[str, Dict], y_test: pd.Series, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC curves
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
    
    # Model comparison bar chart
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


def main():
    """Main execution."""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("LOGISTIC REGRESSION ANALYSIS")
    logger.info("Code-Switch Detection")
    logger.info("="*80)
    
    # Load configuration
    config = Config()
    
    # Get window sizes and context lengths from config
    window_sizes = config.get_analysis_window_sizes()
    context_lengths = config.get('context.context_lengths', [3])
    
    if not isinstance(context_lengths, list):
        context_lengths = [context_lengths]
    
    logger.info(f"Window sizes to process: {window_sizes}")
    logger.info(f"Context lengths to process: {context_lengths}")
    
    # Determine base results directory
    if args.results_dir:
        results_base = Path(args.results_dir)
    else:
        results_base = Path(config.get_surprisal_results_dir()) / args.model
    
    # Determine base output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        results_base_dir = Path(config.get_results_dir())
        output_base = results_base_dir / f"regression_{args.model}"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each window size and context length combination
    for window_size in window_sizes:
        window_results_dir = results_base / f"window_{window_size}"
        window_results_path = window_results_dir / "surprisal_results.csv"
        
        if not window_results_path.exists():
            logger.warning(f"Results not found for window size {window_size}: {window_results_path}")
            logger.warning("Skipping this window size...")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing window size {window_size}")
        logger.info(f"{'='*80}")
        
        # Load data for this window size
        df = load_surprisal_results(window_results_path)
        
        # Process each context length
        for context_length in context_lengths:
            logger.info(f"\n{'-'*80}")
            logger.info(f"Processing context length {context_length}")
            logger.info(f"{'-'*80}")
            
            # Check if this context length exists in the data
            cs_surprisal_col = f'cs_surprisal_context_{context_length}'
            if cs_surprisal_col not in df.columns:
                logger.warning(f"Context length {context_length} not found in data. Available columns: {[c for c in df.columns if 'context_' in c]}")
                logger.warning("Skipping this context length...")
                continue
            
            # Prepare data for regression with this context length
            regression_df = prepare_data_for_regression(df, context_length=context_length)
            
            if len(regression_df) == 0:
                logger.warning(f"No valid data for window {window_size}, context {context_length}. Skipping...")
                continue
            
            # Prepare features (include surprisal and entropy in base features)
            X_base, feature_names = prepare_features(regression_df)
            
            # Add surprisal and entropy to feature matrix
            X = X_base.copy()
            X['surprisal'] = regression_df['surprisal'].values
            X['entropy'] = regression_df['entropy'].values
            feature_names = feature_names + ['surprisal', 'entropy']
            
            y = regression_df['is_code_switched']
            
            # Split data
            logger.info(f"Splitting data (test_size={args.test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=y
            )
            
            logger.info(f"  Training set: {len(X_train)} observations")
            logger.info(f"  Test set: {len(X_test)} observations")
            
            # Fit models
            results = fit_models(
                X_train, X_test, y_train, y_test,
                feature_names, random_state=args.random_state
            )
            
            # Compare models
            comparison_df = compare_models(results, y_test)
            
            # Create output directory for this combination
            output_dir = output_base / f"window_{window_size}" / f"context_{context_length}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            save_results(results, comparison_df, output_dir, args.model)
            
            # Note: Plots can be generated separately using:
            # python scripts/plots/figures.py --regression --model {args.model}
            
            logger.info(f"\nResults saved to: {output_dir}")
            logger.info(f"To generate figures, run: python scripts/plots/figures.py --regression --model {args.model}")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {output_base}")


if __name__ == "__main__":
    main()

