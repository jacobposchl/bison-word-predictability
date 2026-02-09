"""
Mixed-Effects Logistic Regression Analysis for Code-Switch Detection

This script performs mixed-effects logistic regression to test whether surprisal 
and entropy improve prediction of code-switched vs monolingual sentences beyond 
baseline features, while accounting for speaker clustering and matched pairs.

Features:
- word_length: Number of characters in word
- pos_collapsed: Part-of-speech (noun/verb/other) - one-hot encoded
- group: Speaker group (Heritage/Homeland/Immersed) - one-hot encoded
- surprisal: Surprisal values from language models (context_0, context_1, context_2, context_3)
- entropy: Entropy values from language models (context_0, context_1, context_2, context_3)

Random Effects:
- Speaker ID: Accounts for individual speaker differences

Models (fitted separately for each context length 0, 1, 2, 3):
1. Control model: word_length + pos_collapsed + group
2. Surprisal model: control features + surprisal
3. Entropy model: control features + entropy
4. Surprisal + Entropy model: control features + surprisal + entropy

Usage:
    python scripts/regression/regression.py --model autoregressive --window 1
    python scripts/regression/regression.py --model masked --window 1
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
import logging
from typing import Dict, Tuple
import warnings

# Logistic regression
from statsmodels.genmod import families
import statsmodels.formula.api as smf

from src.core.config import Config

warnings.filterwarnings('ignore', category=FutureWarning)
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
        "--window",
        type=int,
        default=None,
        help='Specific window size to analyze (default: all windows)'
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help='Specific context length to analyze (default: all context lengths)'
    )
    
    return parser.parse_args()


def load_surprisal_results(results_path: Path) -> pd.DataFrame:
    """Load surprisal results CSV."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"Surprisal results not found at: {results_path}\n"
            f"Please run surprisal calculation first"
        )
    
    df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(df)} rows from {results_path}")
    return df


def collapse_pos_tag(pos: str) -> str:
    """
    Collapse POS tags to noun, verb, or other.
    Filter out X tags by returning None.
    """
    if pd.isna(pos) or pos == 'UNKNOWN' or pos == 'X':
        return None
    
    pos_upper = str(pos).upper()
    
    # Noun tags
    if any(tag in pos_upper for tag in ['NN', 'NOUN', ' N']):
        return 'noun'
    # Verb tags
    elif any(tag in pos_upper for tag in ['VB', 'VERB', ' V']):
        return 'verb'
    else:
        return 'other'


def prepare_data_for_regression(df: pd.DataFrame, context_length: int) -> pd.DataFrame:
    """
    Prepare regression dataset from long-format surprisal results.
    
    Each row in long format represents either a CS or mono sentence.
    Extracts relevant features for regression.
    """
    rows = []
    
    for idx, row in df.iterrows():
        word = row.get('word', '')
        word_length = row.get('word_length', len(str(word)) if pd.notna(word) else 0)
        pos_tag = row.get('switch_pos', 'UNKNOWN')
        pos_collapsed = collapse_pos_tag(pos_tag)
        group = row.get('group', 'Unknown')
        participant_id = row.get('participant_id', 'Unknown')
        
        # Skip rows with X POS tags
        if pos_collapsed is None:
            continue
        
        surprisal_col = f'surprisal_context_{context_length}'
        entropy_col = f'entropy_context_{context_length}'
        
        rows.append({
            'is_code_switched': int(row.get('is_switch', 0)),
            'word_length': word_length,
            'pos_collapsed': pos_collapsed,
            'group': group if pd.notna(group) else 'Unknown',
            'participant_id': participant_id if pd.notna(participant_id) else 'Unknown',
            'surprisal': row.get(surprisal_col, np.nan),
            'entropy': row.get(entropy_col, np.nan),
            'sent_id': row.get('sent_id', idx)
        })
    
    result_df = pd.DataFrame(rows)
    
    # Filter out rows with missing data
    n_before = len(result_df)
    result_df = result_df.dropna(subset=['surprisal', 'pos_collapsed'])
    n_after = len(result_df)
    
    if n_before > n_after:
        logger.info(f"  Dropped {n_before - n_after} rows with missing data")
    
    # Ensure categorical types
    result_df['pos_collapsed'] = pd.Categorical(result_df['pos_collapsed'])
    result_df['group'] = pd.Categorical(result_df['group'])
    result_df['participant_id'] = result_df['participant_id'].astype(str)
    
    logger.info(f"  Final dataset: {len(result_df)} rows")
    logger.info(f"    CS={result_df['is_code_switched'].sum()}, Mono={(1 - result_df['is_code_switched']).sum()}")
    logger.info(f"    Speakers={result_df['participant_id'].nunique()}")
    
    return result_df


def fit_logistic_models(data: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Fit logistic regression models using GLM.
    
    Returns detailed coefficient information, effect sizes, and model comparisons.
    """
    
    # Standardize continuous variables (for effect size interpretation)
    data = data.copy()
    data['word_length_std'] = (data['word_length'] - data['word_length'].mean()) / data['word_length'].std()
    data['surprisal_std'] = (data['surprisal'] - data['surprisal'].mean()) / data['surprisal'].std()
    data['entropy_std'] = (data['entropy'] - data['entropy'].mean()) / data['entropy'].std()
    
    # Create interaction term (standardized)
    data['surprisal_x_entropy'] = data['surprisal_std'] * data['entropy_std']
    
    formulas = {
        'control': 'is_code_switched ~ word_length_std + C(pos_collapsed) + C(group)',
        'surprisal': 'is_code_switched ~ word_length_std + C(pos_collapsed) + C(group) + surprisal_std',
        'entropy': 'is_code_switched ~ word_length_std + C(pos_collapsed) + C(group) + entropy_std',
        'surprisal_entropy': 'is_code_switched ~ word_length_std + C(pos_collapsed) + C(group) + surprisal_std + entropy_std',
        'interaction': 'is_code_switched ~ word_length_std + C(pos_collapsed) + C(group) + surprisal_std + entropy_std + surprisal_x_entropy'
    }
    
    results = {}
    fitted_models = {}
    
    for model_name, formula in formulas.items():
        logger.info(f"    Fitting {model_name} model...")
        
        try:
            # Fit logistic regression using GLM
            model = smf.glm(
                formula=formula,
                data=data,
                family=families.Binomial()
            )
            
            fitted_model = model.fit()
            fitted_models[model_name] = fitted_model
            
            # Get predictions
            y_pred_proba = fitted_model.fittedvalues
            y_pred = (y_pred_proba > 0.5).astype(int)
            y_true = data['is_code_switched'].values
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_pred_proba)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Extract coefficient details
            coef_df = pd.DataFrame({
                'coefficient': fitted_model.params,
                'std_error': fitted_model.bse,
                'z_value': fitted_model.tvalues,
                'p_value': fitted_model.pvalues,
                'ci_lower': fitted_model.conf_int()[0],
                'ci_upper': fitted_model.conf_int()[1]
            })
            
            # Calculate odds ratios
            coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
            coef_df['or_ci_lower'] = np.exp(coef_df['ci_lower'])
            coef_df['or_ci_upper'] = np.exp(coef_df['ci_upper'])
            
            results[model_name] = {
                'model': fitted_model,
                'formula': formula,
                'auc': auc,
                'accuracy': accuracy,
                'coefficients_table': coef_df,
                'loglikelihood': fitted_model.llf,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'n_params': len(fitted_model.params),
                'y_pred_proba': y_pred_proba
            }
            
            logger.info(f"      {model_name}: LogLik={fitted_model.llf:.2f}, AIC={fitted_model.aic:.1f}")
            
        except Exception as e:
            logger.error(f"      Error fitting {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Likelihood ratio tests (comparing nested models)
    lr_tests = {}
    if 'control' in fitted_models and 'surprisal' in fitted_models:
        lr_stat = 2 * (fitted_models['surprisal'].llf - fitted_models['control'].llf)
        df = fitted_models['surprisal'].df_model - fitted_models['control'].df_model
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, df)
        lr_tests['surprisal_vs_control'] = {
            'chi2': lr_stat,
            'df': df,
            'p_value': p_value
        }
    
    if 'control' in fitted_models and 'entropy' in fitted_models:
        lr_stat = 2 * (fitted_models['entropy'].llf - fitted_models['control'].llf)
        df = fitted_models['entropy'].df_model - fitted_models['control'].df_model
        p_value = 1 - chi2.cdf(lr_stat, df)
        lr_tests['entropy_vs_control'] = {
            'chi2': lr_stat,
            'df': df,
            'p_value': p_value
        }
    
    if 'surprisal' in fitted_models and 'surprisal_entropy' in fitted_models:
        lr_stat = 2 * (fitted_models['surprisal_entropy'].llf - fitted_models['surprisal'].llf)
        df = fitted_models['surprisal_entropy'].df_model - fitted_models['surprisal'].df_model
        p_value = 1 - chi2.cdf(lr_stat, df)
        lr_tests['entropy_added_to_surprisal'] = {
            'chi2': lr_stat,
            'df': df,
            'p_value': p_value
        }
    
    if 'surprisal_entropy' in fitted_models and 'interaction' in fitted_models:
        lr_stat = 2 * (fitted_models['interaction'].llf - fitted_models['surprisal_entropy'].llf)
        df = fitted_models['interaction'].df_model - fitted_models['surprisal_entropy'].df_model
        p_value = 1 - chi2.cdf(lr_stat, df)
        lr_tests['interaction_added_to_main_effects'] = {
            'chi2': lr_stat,
            'df': df,
            'p_value': p_value
        }
    
    return results, lr_tests


def print_model_comparison(results: Dict, lr_tests: Dict, data: pd.DataFrame, window: int, context: int):
    """Print comprehensive model comparison including coefficients, effect sizes, and LR tests."""
    
    print("\n" + "="*120)
    print(f"LOGISTIC REGRESSION RESULTS - Window={window}, Context Length={context}")
    print("="*120)
    
    # Dataset summary
    n_total = len(data)
    n_cs = data['is_code_switched'].sum()
    n_mono = n_total - n_cs
    n_speakers = data['participant_id'].nunique()
    
    print(f"\nDataset: N={n_total:,} observations (CS={n_cs:,}, Mono={n_mono:,}), Speakers={n_speakers}")
    
    # Model fit statistics
    print("\n" + "─" * 120)
    print("TABLE 1: Model Fit Statistics")
    print("─" * 120)
    print(f"{'Model':<30} {'Log-Likelihood':>15} {'AIC':>10} {'BIC':>10} {'Parameters':>12} {'AUC':>10}")
    print("─" * 120)
    
    model_display_names = {
        'control': 'Control',
        'surprisal': 'Surprisal',
        'entropy': 'Entropy',
        'surprisal_entropy': 'Surprisal + Entropy',
        'interaction': 'Interaction (S × E)'
    }
    
    for model_name in ['control', 'surprisal', 'entropy', 'surprisal_entropy', 'interaction']:
        if model_name not in results or 'error' in results[model_name]:
            continue
        r = results[model_name]
        display_name = model_display_names.get(model_name, model_name)
        print(f"{display_name:<30} {r['loglikelihood']:>15.2f} {r['aic']:>10.1f} {r['bic']:>10.1f} "
              f"{r['n_params']:>12} {r['auc']:>10.4f}")
    
    # Likelihood ratio tests
    print("\n" + "─" * 120)
    print("TABLE 2: Likelihood Ratio Tests (Nested Model Comparisons)")
    print("─" * 120)
    print(f"{'Comparison':<50} {'χ²':>12} {'df':>8} {'p-value':>12} {'':>8}")
    print("─" * 120)
    
    for test_name, test_result in lr_tests.items():
        sig = '***' if test_result['p_value'] < 0.001 else \
              '**' if test_result['p_value'] < 0.01 else \
              '*' if test_result['p_value'] < 0.05 else 'ns'
        display_name = test_name.replace('_', ' ').title().replace('Vs', 'vs.')
        print(f"{display_name:<50} {test_result['chi2']:>12.2f} {test_result['df']:>8} "
              f"{test_result['p_value']:>12.4f} {sig:>8}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    # Key coefficients from each model
    print("\n" + "─" * 120)
    print("TABLE 3: Standardized Coefficients and Odds Ratios")
    print("─" * 120)
    
    model_display_names = {
        'control': 'Control',
        'surprisal': 'Surprisal',
        'entropy': 'Entropy',
        'surprisal_entropy': 'Surprisal + Entropy',
        'interaction': 'Interaction (Surprisal × Entropy)'
    }
    
    for model_name in ['control', 'surprisal', 'entropy', 'surprisal_entropy', 'interaction']:
        if model_name not in results or 'error' in results[model_name]:
            continue
        
        display_name = model_display_names.get(model_name, model_name)
        print(f"\n{display_name}:")
        print(f"  {'Predictor':<25} {'β':>10} {'SE':>10} {'z':>10} {'p':>10} {'':>5} {'OR':>10} {'95% CI':>20}")
        print("  " + "─" * 110)
        
        coef_df = results[model_name]['coefficients_table']
        
        # Focus on key predictors
        key_vars = ['word_length_std', 'surprisal_std', 'entropy_std', 'surprisal_x_entropy']
        for var in key_vars:
            if var in coef_df.index:
                row = coef_df.loc[var]
                sig = '***' if row['p_value'] < 0.001 else \
                      '**' if row['p_value'] < 0.01 else \
                      '*' if row['p_value'] < 0.05 else ''
                # Custom display names
                var_display_map = {
                    'word_length_std': 'Word Length',
                    'surprisal_std': 'Surprisal',
                    'entropy_std': 'Entropy',
                    'surprisal_x_entropy': 'Surprisal × Entropy'
                }
                var_display = var_display_map.get(var, var.replace('_std', '').replace('_', ' ').title())
                ci_display = f"[{row['or_ci_lower']:.2f}, {row['or_ci_upper']:.2f}]"
                print(f"  {var_display:<25} {row['coefficient']:>10.4f} {row['std_error']:>10.4f} "
                      f"{row['z_value']:>10.3f} {row['p_value']:>10.4f} {sig:>5} "
                      f"{row['odds_ratio']:>10.3f} {ci_display:>20}")
    
    # Correlation matrix
    print("\n" + "─" * 120)
    print("TABLE 4: Correlation Matrix")
    print("─" * 120)
    corr_vars = ['is_code_switched', 'word_length', 'surprisal', 'entropy']
    available_vars = [v for v in corr_vars if v in data.columns]
    if len(available_vars) > 1:
        corr_matrix = data[available_vars].corr()
        
        # Format column names
        corr_display = corr_matrix.copy()
        corr_display.index = ['CS (target)', 'Word Length', 'Surprisal', 'Entropy'][:len(corr_display)]
        corr_display.columns = ['CS', 'Length', 'Surprisal', 'Entropy'][:len(corr_display.columns)]
        
        print(corr_display.round(3).to_string())
    
    print("\n" + "="*120 + "\n")


def main():
    args = parse_arguments()
    
    print("\n" + "="*95)
    print("LOGISTIC REGRESSION ANALYSIS - Code-Switch Detection")
    print("="*95)
    
    config = Config()
    
    # Get window sizes to analyze
    if args.window is not None:
        window_sizes = [args.window]
        logger.info(f"Analyzing only window size: {args.window}")
    else:
        window_sizes = config.get_analysis_window_sizes()
        logger.info(f"Analyzing all window sizes: {window_sizes}")
    
    # Get context lengths to analyze
    if args.context_len is not None:
        context_lengths = [args.context_len]
        logger.info(f"Analyzing only context length: {args.context_len}")
    else:
        context_lengths = config.get('context.context_lengths', [0, 1, 2, 3])
        logger.info(f"Analyzing all context lengths: {context_lengths}")
    
    if not isinstance(context_lengths, list):
        context_lengths = [context_lengths]
    
    results_base = Path(config.get_surprisal_results_dir()) / args.model
    output_base = Path(config.get_results_dir()) / f"regression_{args.model}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for window_size in window_sizes:
        window_results_path = results_base / f"window_{window_size}" / "surprisal_results.csv"
        
        if not window_results_path.exists():
            logger.warning(f"Skipping window {window_size}: file not found")
            continue
        
        df = load_surprisal_results(window_results_path)
        
        for context_length in context_lengths:
            logger.info(f"\nProcessing: Window={window_size}, Context Length={context_length}")
            
            # Check if columns exist
            surprisal_col = f'surprisal_context_{context_length}'
            if surprisal_col not in df.columns:
                logger.warning(f"  Context {context_length} not found in data, skipping...")
                continue
            
            # Prepare data for this context length
            regression_df = prepare_data_for_regression(df, context_length=context_length)
            
            if len(regression_df) == 0:
                logger.warning(f"  No valid data, skipping...")
                continue
            
            # Fit models
            results, lr_tests = fit_logistic_models(regression_df)
            
            # Print comprehensive comparison
            print_model_comparison(results, lr_tests, regression_df, window_size, context_length)
            
            # Save results
            context_output_dir = output_base / f"window_{window_size}" / f"context_{context_length}"
            context_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save coefficient tables
            for model_name, result in results.items():
                if 'error' not in result and 'coefficients_table' in result:
                    coef_csv = context_output_dir / f"{model_name}_coefficients.csv"
                    result['coefficients_table'].to_csv(coef_csv)
            
            # Save LR tests
            lr_df = pd.DataFrame(lr_tests).T
            lr_df.to_csv(context_output_dir / "likelihood_ratio_tests.csv")
            
            # Save detailed results
            import pickle
            results_pkl = context_output_dir / "model_results.pkl"
            with open(results_pkl, 'wb') as f:
                pickle.dump({'results': results, 'lr_tests': lr_tests}, f)
            
            # Add to summary
            for model_name, result in results.items():
                if 'error' not in result:
                    all_results.append({
                        'model_type': args.model,
                        'window_size': window_size,
                        'context_length': context_length,
                        'regression_model': model_name,
                        'loglikelihood': result['loglikelihood'],
                        'aic': result['aic'],
                        'bic': result['bic'],
                        'auc': result['auc'],
                        'n_params': result['n_params'],
                        'n_observations': len(regression_df)
                    })
    
    # Save overall summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_base / "regression_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n{'='*95}")
        print(f"✓ Summary saved: {summary_csv}")
        print(f"✓ Total configurations: {len(all_results)}")
        print(f"{'='*95}\n")
    else:
        logger.warning("No results to save!")


if __name__ == "__main__":
    main()
