"""
Main experiment script for code-switching surprisal analysis.

This script runs the complete Calvillo-inspired methodology:
1. Load code-switched sentences and monolingual baselines
2. Translate code-switched sentences to full Cantonese
3. Find matching monolingual sentences for each code-switched sentence
4. Calculate surprisal at switch positions (translated sentences)
5. Calculate surprisal at matched positions (monolingual baselines)
6. Compare surprisal values
7. Generate results and visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.experiments.translation import CodeSwitchTranslator
from src.experiments.surprisal_calculator import CantoneseSuprisalCalculator
from src.analysis.matching_algorithm import find_matches, precompute_monolingual_pos_sequences
from src.analysis.pos_tagging import pos_tag_cantonese, extract_pos_sequence
from src.data.data_loading import load_code_switched_sentences, load_monolingual_sentences
from src.core.tokenization import segment_cantonese_sentence
from src.analysis.pattern_analysis import find_switch_positions

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_experiment(
    code_switched_df: pd.DataFrame,
    monolingual_dict: Dict[str, pd.DataFrame],
    translator: CodeSwitchTranslator,
    surprisal_calc: CantoneseSuprisalCalculator,
    output_dir: Path,
    sample_size: int = None,
    similarity_threshold: float = 0.4
) -> pd.DataFrame:
    """
    Run the full surprisal experiment.
    
    Args:
        code_switched_df: DataFrame with code-switched sentences
        monolingual_dict: Dictionary with monolingual sentences
        translator: Translator instance
        surprisal_calc: Surprisal calculator instance
        output_dir: Output directory for results
        sample_size: Optional sample size (None = use all)
        similarity_threshold: Minimum similarity for matching
        
    Returns:
        DataFrame with experimental results
    """
    logger.info("="*80)
    logger.info("RUNNING SURPRISAL EXPERIMENT")
    logger.info("="*80)
    
    # Sample if requested
    if sample_size and sample_size < len(code_switched_df):
        logger.info(f"Sampling {sample_size} sentences from {len(code_switched_df)}")
        code_switched_df = code_switched_df.sample(n=sample_size, random_state=42)
    else:
        logger.info(f"Processing all {len(code_switched_df)} code-switched sentences")
    
    # Precompute POS sequences for monolingual sentences
    logger.info("\nPrecomputing POS sequences for monolingual sentences...")
    monolingual_dicts = {
        'cantonese': monolingual_dict['cantonese'].to_dict('records'),
        'english': monolingual_dict['english'].to_dict('records')
    }
    monolingual_dicts = precompute_monolingual_pos_sequences(monolingual_dicts)
    
    results = []
    
    logger.info("\nProcessing code-switched sentences...")
    for idx, row in tqdm(code_switched_df.iterrows(), total=len(code_switched_df), desc="Analyzing"):
        try:
            # Get sentence info
            sentence = row['reconstructed_sentence']
            pattern = row['pattern']
            
            # Segment the code-switched sentence
            words = segment_cantonese_sentence(sentence)
            
            # Translate to full Cantonese
            translation_result = translator.translate_code_switched_sentence(
                sentence=sentence,
                pattern=pattern,
                words=words
            )
            translated_sentence = translation_result['translated_sentence']
            
            # Segment translated sentence
            translated_words = segment_cantonese_sentence(translated_sentence)
            
            # Find switch positions
            switch_positions = find_switch_positions(pattern)
            
            # Find matching monolingual sentences
            cs_sentence_dict = row.to_dict()
            matches = find_matches(
                cs_sentence_dict,
                monolingual_dicts,
                similarity_threshold=similarity_threshold
            )
            
            if not matches:
                logger.debug(f"No matches found for sentence: {sentence}")
                continue
            
            # For each switch position
            for switch_pos in switch_positions:
                if switch_pos >= len(translated_words):
                    logger.debug(f"Switch position {switch_pos} out of range for translated sentence")
                    continue
                
                # Calculate surprisal at switch position in translated sentence
                try:
                    cs_surprisal_result = surprisal_calc.calculate_surprisal(
                        sentence=translated_sentence,
                        word_index=switch_pos,
                        words=translated_words
                    )
                    cs_surprisal = cs_surprisal_result['surprisal']
                except Exception as e:
                    logger.debug(f"Error calculating CS surprisal: {e}")
                    continue
                
                # Calculate surprisal at matched positions in monolingual sentences
                mono_surprisals = []
                for match in matches[:5]:  # Use top 5 matches
                    try:
                        mono_sentence = match['monolingual_sentence']['reconstructed_sentence']
                        mono_words = segment_cantonese_sentence(mono_sentence)
                        
                        # Use the same relative position
                        mono_pos = min(switch_pos, len(mono_words) - 1)
                        
                        mono_surprisal_result = surprisal_calc.calculate_surprisal(
                            sentence=mono_sentence,
                            word_index=mono_pos,
                            words=mono_words
                        )
                        mono_surprisals.append(mono_surprisal_result['surprisal'])
                    except Exception as e:
                        logger.debug(f"Error calculating mono surprisal: {e}")
                        continue
                
                if not mono_surprisals:
                    continue
                
                # Store result
                results.append({
                    'sentence_id': idx,
                    'original_sentence': sentence,
                    'translated_sentence': translated_sentence,
                    'pattern': pattern,
                    'switch_position': switch_pos,
                    'cs_surprisal': cs_surprisal,
                    'mono_surprisal_mean': np.mean(mono_surprisals),
                    'mono_surprisal_std': np.std(mono_surprisals),
                    'surprisal_difference': cs_surprisal - np.mean(mono_surprisals),
                    'num_matches': len(matches),
                    'group': row.get('group', ''),
                    'participant_id': row.get('participant_id', '')
                })
        
        except Exception as e:
            logger.error(f"Error processing sentence {idx}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    logger.info(f"\nCompleted analysis of {len(results_df)} switch positions")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "surprisal_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    return results_df


def analyze_results(results_df: pd.DataFrame, output_dir: Path):
    """
    Analyze and visualize results.
    
    Args:
        results_df: DataFrame with experimental results
        output_dir: Output directory
    """
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*80)
    
    # Basic statistics
    logger.info(f"\nTotal switch positions analyzed: {len(results_df)}")
    logger.info(f"Mean CS surprisal: {results_df['cs_surprisal'].mean():.4f}")
    logger.info(f"Mean mono surprisal: {results_df['mono_surprisal_mean'].mean():.4f}")
    logger.info(f"Mean difference: {results_df['surprisal_difference'].mean():.4f}")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        results_df['cs_surprisal'],
        results_df['mono_surprisal_mean']
    )
    logger.info(f"\nPaired t-test:")
    logger.info(f"  t-statistic: {t_stat:.4f}")
    logger.info(f"  p-value: {p_value:.6f}")
    
    # Effect size (Cohen's d)
    diff = results_df['cs_surprisal'] - results_df['mono_surprisal_mean']
    cohens_d = diff.mean() / diff.std()
    logger.info(f"  Cohen's d: {cohens_d:.4f}")
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(results_df['cs_surprisal'], bins=30, alpha=0.6, label='Code-switched', color='red')
    axes[0].hist(results_df['mono_surprisal_mean'], bins=30, alpha=0.6, label='Monolingual', color='blue')
    axes[0].set_xlabel('Surprisal')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Surprisal Distribution')
    axes[0].legend()
    
    # 2. Box plot comparison
    data_to_plot = [results_df['cs_surprisal'], results_df['mono_surprisal_mean']]
    axes[1].boxplot(data_to_plot, labels=['Code-switched', 'Monolingual'])
    axes[1].set_ylabel('Surprisal')
    axes[1].set_title('Surprisal Comparison')
    
    plt.tight_layout()
    fig_path = output_dir / 'surprisal_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {fig_path}")
    plt.close()
    
    # 3. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(results_df['mono_surprisal_mean'], results_df['cs_surprisal'], alpha=0.5)
    
    # Add diagonal line
    max_val = max(results_df['mono_surprisal_mean'].max(), results_df['cs_surprisal'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='y=x')
    
    ax.set_xlabel('Monolingual Surprisal')
    ax.set_ylabel('Code-switched Surprisal')
    ax.set_title('Surprisal at Code-switched vs. Matched Positions')
    ax.legend()
    
    fig_path = output_dir / 'surprisal_scatter.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {fig_path}")
    plt.close()
    
    # Save summary statistics
    summary_path = output_dir / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write("SURPRISAL EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total switch positions analyzed: {len(results_df)}\n\n")
        f.write("Descriptive Statistics:\n")
        f.write(f"  Code-switched surprisal:  Mean={results_df['cs_surprisal'].mean():.4f}, SD={results_df['cs_surprisal'].std():.4f}\n")
        f.write(f"  Monolingual surprisal:    Mean={results_df['mono_surprisal_mean'].mean():.4f}, SD={results_df['mono_surprisal_mean'].std():.4f}\n")
        f.write(f"  Difference:               Mean={results_df['surprisal_difference'].mean():.4f}, SD={results_df['surprisal_difference'].std():.4f}\n\n")
        f.write("Paired t-test:\n")
        f.write(f"  t-statistic: {t_stat:.4f}\n")
        f.write(f"  p-value: {p_value:.6f}\n")
        f.write(f"  Cohen's d: {cohens_d:.4f}\n\n")
        if p_value < 0.05:
            f.write("Result: Code-switched positions show SIGNIFICANTLY HIGHER surprisal than matched monolingual positions.\n")
        else:
            f.write("Result: No significant difference in surprisal between code-switched and monolingual positions.\n")
    
    logger.info(f"Saved: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run surprisal experiment for code-switching analysis"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of sentences to sample (None = use all)'
    )
    parser.add_argument(
        '--use-fillers',
        action='store_true',
        help='Use dataset with fillers included'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.4,
        help='Minimum similarity threshold for matching'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/surprisal_experiment',
        help='Output directory for results'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenAI API key (REQUIRED for translation)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger.info("="*80)
    logger.info("CODE-SWITCHING SURPRISAL EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Sample size: {args.sample_size or 'ALL'}")
    logger.info(f"Use fillers: {args.use_fillers}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80 + "\n")
    
    try:
        # Load config
        config = Config(args.config)
        output_dir = Path(args.output_dir)
        
        # Load data
        logger.info("Step 1: Loading data...")
        code_switched_df = load_code_switched_sentences(config, args.use_fillers)
        monolingual_dict = load_monolingual_sentences(config)
        
        # Initialize translator and surprisal calculator
        logger.info("\nStep 2: Initializing models...")
        translator = CodeSwitchTranslator(
            api_key=args.api_key,
            model=config.get_translation_model(),
            use_cache=config.get_translation_use_cache(),
            cache_dir=config.get_translation_cache_dir(),
            temperature=config.get_translation_temperature(),
            max_tokens=config.get_translation_max_tokens()
        )
        surprisal_calc = CantoneseSuprisalCalculator()
        
        # Run experiment
        logger.info("\nStep 3: Running experiment...")
        results_df = run_experiment(
            code_switched_df=code_switched_df,
            monolingual_dict=monolingual_dict,
            translator=translator,
            surprisal_calc=surprisal_calc,
            output_dir=output_dir,
            sample_size=args.sample_size,
            similarity_threshold=args.similarity_threshold
        )
        
        # Analyze results
        logger.info("\nStep 4: Analyzing results...")
        analyze_results(results_df, output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE!")
        logger.info("="*80)
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.exception(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()