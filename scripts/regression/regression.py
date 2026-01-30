"""
Logistic Regression Analysis for Code-Switch Detection

This script performs logistic regression to test whether surprisal and entropy
improve prediction of code-switched vs monolingual sentences beyond baseline
features (frequency, word length, sentence length, POS, group).

Features:
- word_frequency: Retrieved from surprisal results (calculated using pycantonese corpus data)
- group: Speaker group (Heritage/Homeland/Immersed) - one-hot encoded
- word_length, sentence_length, position_normalized: Basic linguistic features
- pos_tag: Part-of-speech tags - one-hot encoded
- surprisal: Surprisal values from language models
- entropy: Entropy values from language models

Models:
1. Control model: frequency + word_length + sentence_length + position + POS + group
2. Surprisal model: control features + surprisal
3. Entropy model: control features + entropy
4. Surprisal + Entropy model: control features + surprisal + entropy

The script saves the regression datasets to:
results/regression/{model}/window_{N}/context_{M}/dataset.csv

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
from datetime import datetime
import logging
from typing import Dict, List, Tuple

from src.core.config import Config
from src.plots.regression.report_generator import create_results_row

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
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
    )
    
    return parser.parse_args()


def load_surprisal_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Surprisal results not found at: {results_path}\n"
            f"Please run surprisal calculation first: python scripts/surprisal/surprisal.py --model {results_path.parent.name.split('_')[-1]}"
        )
    
    df = pd.read_csv(results_path)
    return df




def prepare_data_for_regression(df: pd.DataFrame, context_length: int = None) -> pd.DataFrame:
    rows = []
    
    for idx, row in df.iterrows():
        # CS sentence row (label = 1)
        cs_word = row.get('cs_word', '')
        cs_sentence = row.get('cs_translation', '')
        cs_switch_idx = int(row.get('switch_index', 0))
        cs_group = row.get('cs_group', 'Unknown')
        
        cs_words_list = cs_sentence.split() if pd.notna(cs_sentence) else []
        cs_sentence_length = len(cs_words_list)
        
        # Calculate normalized position
        cs_position_norm = cs_switch_idx / cs_sentence_length if cs_sentence_length > 0 else np.nan
        
        cs_pos = row.get('cs_switch_pos', 'UNKNOWN')
        
        cs_word_frequency = row.get('cs_word_frequency', np.nan)
        
        if context_length is not None:
            cs_surprisal_col = f'cs_surprisal_context_{context_length}'
            cs_entropy_col = f'cs_entropy_context_{context_length}'
        else:
            cs_surprisal_col = 'cs_surprisal_total'
            cs_entropy_col = 'cs_entropy'
        
        rows.append({
            'is_code_switched': 1,
            'word': cs_word if pd.notna(cs_word) else '',
            'word_length': len(str(cs_word)) if pd.notna(cs_word) else 0,
            'sentence_length': cs_sentence_length,
            'position_normalized': cs_position_norm,
            'pos_tag': cs_pos,
            'word_frequency': cs_word_frequency,
            'group': cs_group if pd.notna(cs_group) else 'Unknown',
            'surprisal': row.get(cs_surprisal_col, np.nan),
            'entropy': row.get(cs_entropy_col, np.nan),
            'sentence_id': f"{idx}_cs",
            'original_row_id': idx
        })
        
        # Mono sentence row (label = 0)
        mono_word = row.get('mono_word', '')
        mono_sentence = row.get('matched_mono', '')
        mono_switch_idx = int(row.get('matched_switch_index', 0))
        mono_group = row.get('matched_group', 'Unknown')
        

        mono_words_list = mono_sentence.split() if pd.notna(mono_sentence) else []
        mono_sentence_length = len(mono_words_list)
        
        mono_position_norm = mono_switch_idx / mono_sentence_length if mono_sentence_length > 0 else np.nan
        
        mono_pos = row.get('mono_switch_pos', 'UNKNOWN')
        
        mono_word_frequency = row.get('mono_word_frequency', np.nan)
        
        if context_length is not None:
            mono_surprisal_col = f'mono_surprisal_context_{context_length}'
            mono_entropy_col = f'mono_entropy_context_{context_length}'
        else:
            mono_surprisal_col = 'mono_surprisal_total'
            mono_entropy_col = 'mono_entropy'
        
        rows.append({
            'is_code_switched': 0,
            'word': mono_word if pd.notna(mono_word) else '',
            'word_length': len(str(mono_word)) if pd.notna(mono_word) else 0,
            'sentence_length': mono_sentence_length,
            'position_normalized': mono_position_norm,
            'pos_tag': mono_pos,
            'word_frequency': mono_word_frequency,
            'group': mono_group if pd.notna(mono_group) else 'Unknown',
            'surprisal': row.get(mono_surprisal_col, np.nan),
            'entropy': row.get(mono_entropy_col, np.nan),
            'sentence_id': f"{idx}_mono",
            'original_row_id': idx
        })
    
    result_df = pd.DataFrame(rows)
    
    n_before = len(result_df)
    result_df = result_df.dropna(subset=['word_frequency', 'surprisal'])
    n_after = len(result_df)
    
    if n_before > n_after:
        logger.warning(f"Dropped {n_before - n_after} rows with missing data")

    return result_df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_df = df.copy()
    
    pos_dummies = pd.get_dummies(feature_df['pos_tag'], prefix='pos')
    feature_df = pd.concat([feature_df, pos_dummies], axis=1)
    
    group_dummies = pd.get_dummies(feature_df['group'], prefix='group')
    feature_df = pd.concat([feature_df, group_dummies], axis=1)
    
    numeric_features = [
        'word_length',
        'sentence_length', 
        'position_normalized',
        'word_frequency'
    ]
    
    pos_feature_names = [col for col in pos_dummies.columns]
    group_feature_names = [col for col in group_dummies.columns]
    
    feature_names = numeric_features + pos_feature_names + group_feature_names
    
    feature_df = feature_df[numeric_features + pos_feature_names + group_feature_names]
    
    return feature_df, feature_names


def fit_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    random_state: int = 42
) -> Dict[str, Dict]:
    
    control_features = [
        'word_length', 'sentence_length', 'position_normalized', 'word_frequency'
    ] + [f for f in feature_names if f.startswith('pos_')] + [f for f in feature_names if f.startswith('group_')]
    
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
        available_features = [f for f in features if f in X_train.columns]
        missing_features = [f for f in features if f not in X_train.columns]
        
        if missing_features:
            raise ValueError(
                f"Model '{model_name}': Missing required features {missing_features}. "
                f"Available columns: {list(X_train.columns)}"
            )
        

        X_train_subset = X_train[available_features]
        X_test_subset = X_test[available_features]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        
        # Fit
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
    
    return results


def compare_models(results: Dict[str, Dict], y_test: pd.Series) -> pd.DataFrame:
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
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"\n{'Model':<20} {'AUC':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Features':>10}")
    print("-"*80)
    for _, row in comparison_df.iterrows():
        print(f"{row['model']:<20} {row['auc']:>8.4f} {row['accuracy']:>10.4f} {row['precision']:>10.4f} {row['recall']:>8.4f} {row['f1']:>8.4f} {row['n_features']:>10}")
    print("="*80)
    
    return comparison_df

def main():
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION ANALYSIS - Code-Switch Detection")
    print("="*80)
    
    config = Config()
    
    window_sizes = config.get_analysis_window_sizes()
    context_lengths = config.get('context.context_lengths', [3])
    
    if not isinstance(context_lengths, list):
        context_lengths = [context_lengths]
    
    results_base = Path(config.get_surprisal_results_dir()) / args.model
    results_base_dir = Path(config.get_results_dir())
    output_base = results_base_dir / f"regression_{args.model}"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for window_size in window_sizes:
        window_results_dir = results_base / f"window_{window_size}"
        window_results_path = window_results_dir / "surprisal_results.csv"
        
        if not window_results_path.exists():
            logger.warning(f"Skipping window {window_size}: file not found")
            continue
        
        df = load_surprisal_results(window_results_path)
        
        for context_length in context_lengths:
            print(f"\n{'─'*80}")
            print(f"Window: {window_size} | Context: {context_length}")
            print(f"{'─'*80}")
            
            cs_surprisal_col = f'cs_surprisal_context_{context_length}'
            if cs_surprisal_col not in df.columns:
                logger.warning(f"Context {context_length} not found, skipping...")
                continue
            
            regression_df = prepare_data_for_regression(df, context_length=context_length)
            
            if len(regression_df) == 0:
                logger.warning(f"No valid data, skipping...")
                continue
            
            X_base, feature_names = prepare_features(regression_df)
            
            X = X_base.copy()
            X['surprisal'] = regression_df['surprisal'].values
            feature_names = feature_names + ['surprisal']
            
            entropy_available = regression_df['entropy'].notna().any()
            if entropy_available:
                X['entropy'] = regression_df['entropy'].values
                feature_names = feature_names + ['entropy']
            
            y = regression_df['is_code_switched']
            
            random_seed = config.get('experiment.random_seed', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=args.test_size,
                random_state=random_seed,
                stratify=y
            )
            
            print(f"Dataset: {len(X_train)} train, {len(X_test)} test | Features: {len(feature_names)}")
            
            results = fit_models(
                X_train, X_test, y_train, y_test,
                feature_names, random_state=random_seed
            )
            
            comparison_df = compare_models(results, y_test)
            
            results_row = create_results_row(
                results=results,
                model_type=args.model,
                window_size=window_size,
                context_length=context_length,
                n_train=len(X_train),
                n_test=len(X_test)
            )
            all_results.append(results_row)
            
            context_output_dir = output_base / f"window_{window_size}" / f"context_{context_length}"
            context_output_dir.mkdir(parents=True, exist_ok=True)
            
            import pickle
            results_pkl_path = context_output_dir / "model_results.pkl"
            plot_data = {
                'results': results,
                'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
            }
            with open(results_pkl_path, 'wb') as f:
                pickle.dump(plot_data, f)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = output_base / "regression_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"✓ Results saved: {results_csv_path}")
        print(f"✓ Total configurations: {len(all_results)}")
        print(f"{'='*80}\n")
    else:
        logger.warning("No results to save!")


if __name__ == "__main__":
    main()

