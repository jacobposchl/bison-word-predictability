"""
Report generation functions for matching analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_window_matching_report(
    window_results: Dict,
    similarity_threshold: float = 0.4,
    output_dir: str = None,
    analysis_datasets: Dict = None,
    num_cs_sentences: int = None,
    num_mono_sentences: int = None,
    context_stats_by_window: Dict = None
) -> str:
    """
    Generate comprehensive CSV reports for window matching analysis.
    
    Creates six CSV files:
    1. match_similarity_scores.csv - Statistics on similarity scores from matches only
    2. total_similarity_scores.csv - Statistics on ALL similarity scores (Stage 1)
    3. ranking_distribution.csv - Match distribution by speaker/group relationship
    4. filtering_table.csv - Shows the filtering pipeline from start to finish
    5. window_cutoffs.csv - Shows POS windows cut off due to sentence boundaries
    6. context_quality.csv - Shows context quality statistics per window
    
    Args:
        window_results: Results from analyze_window_matching()
        similarity_threshold: The similarity threshold used
        output_dir: Directory to save CSV files (if None, returns text report)
        analysis_datasets: Dict mapping window_size to DataFrame (analysis datasets)
        num_cs_sentences: Total number of CS sentences from preprocessing
        num_mono_sentences: Total number of monolingual sentences
        context_stats_by_window: Dict mapping window_size to context quality stats
        
    Returns:
        Message indicating where files were saved, or text report if no output_dir
    """
    
    if output_dir is None or analysis_datasets is None:
        # Fallback to old text report if parameters not provided
        return _generate_text_report(window_results, similarity_threshold)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate match_similarity_scores.csv (only from matches)
    logger.info("Generating match_similarity_scores.csv...")
    match_similarity_data = []
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        match_similarity_scores = results.get('match_similarity_scores', results.get('similarity_scores', []))
        
        if match_similarity_scores:
            match_similarity_data.append({
                'window_size': window_size,
                'mean': np.mean(match_similarity_scores),
                'median': np.median(match_similarity_scores),
                'std': np.std(match_similarity_scores),
                'min': np.min(match_similarity_scores),
                'max': np.max(match_similarity_scores),
                'count': len(match_similarity_scores)
            })
    
    match_similarity_df = pd.DataFrame(match_similarity_data)
    match_similarity_csv = output_path / 'match_similarity_scores.csv'
    match_similarity_df.to_csv(match_similarity_csv, index=False)
    logger.info(f"Saved {match_similarity_csv}")
    
    # 2. Generate total_similarity_scores.csv (ALL Stage 1 scores)
    logger.info("Generating total_similarity_scores.csv...")
    total_similarity_data = []
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        all_similarity_scores = results.get('all_similarity_scores', [])
        
        if all_similarity_scores:
            total_similarity_data.append({
                'window_size': window_size,
                'mean': np.mean(all_similarity_scores),
                'median': np.median(all_similarity_scores),
                'std': np.std(all_similarity_scores),
                'min': np.min(all_similarity_scores),
                'max': np.max(all_similarity_scores),
                'count': len(all_similarity_scores)
            })
    
    total_similarity_df = pd.DataFrame(total_similarity_data)
    total_similarity_csv = output_path / 'total_similarity_scores.csv'
    total_similarity_df.to_csv(total_similarity_csv, index=False)
    logger.info(f"Saved {total_similarity_csv}")
    
    # 3. Generate ranking_distribution.csv
    logger.info("Generating ranking_distribution.csv...")
    ranking_data = []
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        
        # Load the analysis dataset for this window
        if window_size in analysis_datasets:
            df = analysis_datasets[window_size]
            
            # Total matches (sum of all total_matches_above_threshold)
            total_matches = df['total_matches_above_threshold'].sum()
            
            # Same speaker matches (per sentence)
            same_speaker_values = df['matches_same_speaker'].values
            
            # Same group matches (per sentence)
            same_group_values = df['matches_same_group'].values
            
            ranking_data.append({
                'window_size': window_size,
                'total_matches': int(total_matches),
                'same_speaker_mean': np.mean(same_speaker_values),
                'same_speaker_median': np.median(same_speaker_values),
                'same_speaker_std': np.std(same_speaker_values),
                'same_speaker_min': int(np.min(same_speaker_values)),
                'same_speaker_max': int(np.max(same_speaker_values)),
                'same_group_mean': np.mean(same_group_values),
                'same_group_median': np.median(same_group_values),
                'same_group_std': np.std(same_group_values),
                'same_group_min': int(np.min(same_group_values)),
                'same_group_max': int(np.max(same_group_values))
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_csv = output_path / 'ranking_distribution.csv'
    ranking_df.to_csv(ranking_csv, index=False)
    logger.info(f"Saved {ranking_csv}")
    
    # 3. Generate filtering_table.csv
    logger.info("Generating filtering_table.csv...")
    filtering_data = []
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        
        # Stage 1: Sentences passing full sentence similarity threshold
        stage1_passed = results.get('stage1_passed', 0)
        
        # Stage 2: Matches passing exact POS window matching
        stage2_passed = results.get('stage2_passed', 0)
        
        # Final: Sentences with at least one match (from analysis dataset)
        if window_size in analysis_datasets:
            final_matched_sentences = len(analysis_datasets[window_size])
        else:
            final_matched_sentences = results['sentences_with_matches']
        
        filtering_data.append({
            'window_size': window_size,
            'total_cs_sentences': num_cs_sentences if num_cs_sentences else results['total_sentences'],
            'total_mono_sentences': num_mono_sentences,
            'stage1_passed_pairs': stage1_passed,
            'stage2_passed_matches': stage2_passed,
            'final_matched_sentences': final_matched_sentences
        })
    
    filtering_df = pd.DataFrame(filtering_data)
    filtering_csv = output_path / 'filtering_table.csv'
    filtering_df.to_csv(filtering_csv, index=False)
    logger.info(f"Saved {filtering_csv}")
    
    # 4. Generate window_cutoffs.csv
    logger.info("Generating window_cutoffs.csv...")
    cutoff_data = []
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        
        cutoff_count = results.get('cutoff_count', 0)
        cutoff_percentage = results.get('cutoff_percentage', 0.0)
        
        cutoff_data.append({
            'window_size': window_size,
            'total_sentences': results['total_sentences'],
            'cutoff_count': cutoff_count,
            'cutoff_percentage': cutoff_percentage,
            'full_window_count': results['total_sentences'] - cutoff_count
        })
    
    cutoff_df = pd.DataFrame(cutoff_data)
    cutoff_csv = output_path / 'window_cutoffs.csv'
    cutoff_df.to_csv(cutoff_csv, index=False)
    logger.info(f"Saved {cutoff_csv}")
    
    # 5. Generate context_quality.csv (if context stats available)
    csv_files = [match_similarity_csv.name, total_similarity_csv.name, ranking_csv.name, filtering_csv.name, cutoff_csv.name]
    
    if context_stats_by_window:
        logger.info("Generating context_quality.csv...")
        context_data = []
        
        for window_size in sorted(context_stats_by_window.keys()):
            stats = context_stats_by_window[window_size]
            
            row = {
                'window_size': window_size,
                'total_contexts': stats.get('total_contexts', 0),
                'cs_contexts_with_issues': stats.get('cs_contexts_with_issues', 0),
                'mono_contexts_with_issues': stats.get('mono_contexts_with_issues', 0),
            }
            
            # Add CS quality statistics
            if 'cs_quality_mean' in stats:
                row['cs_quality_mean'] = stats['cs_quality_mean']
                row['cs_quality_median'] = stats['cs_quality_median']
                row['cs_quality_std'] = stats['cs_quality_std']
                row['cs_quality_min'] = stats['cs_quality_min']
                row['cs_quality_max'] = stats['cs_quality_max']
            
            # Add mono quality statistics
            if 'mono_quality_mean' in stats:
                row['mono_quality_mean'] = stats['mono_quality_mean']
                row['mono_quality_median'] = stats['mono_quality_median']
                row['mono_quality_std'] = stats['mono_quality_std']
                row['mono_quality_min'] = stats['mono_quality_min']
                row['mono_quality_max'] = stats['mono_quality_max']
            
            context_data.append(row)
        
        if context_data:
            context_df = pd.DataFrame(context_data)
            context_csv = output_path / 'context_quality.csv'
            context_df.to_csv(context_csv, index=False)
            logger.info(f"Saved {context_csv}")
            csv_files.append(context_csv.name)
    
    return f"Generated CSV reports in: {output_path}\n" + "\n".join([f"  - {f}" for f in csv_files])


def _generate_text_report(
    window_results: Dict,
    similarity_threshold: float = 0.4
) -> str:
    """
    Generate text report (legacy format for backwards compatibility).
    
    Args:
        window_results: Results from analyze_window_matching()
        similarity_threshold: The similarity threshold used
        
    Returns:
        Formatted text report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("POS WINDOW MATCHING ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("This report analyzes how well we can match code-switched sentences")
    report_lines.append("to monolingual Cantonese sentences using POS sequence similarity")
    report_lines.append("around the switch point.")
    report_lines.append("")
    report_lines.append(f"Similarity Threshold: {similarity_threshold:.2f} (Levenshtein)")
    report_lines.append("")
    
    # Section 1: Overall Statistics
    report_lines.append("1. OVERALL MATCHING STATISTICS BY WINDOW SIZE")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Table header
    report_lines.append(f"{'Window':>8} {'Total':>8} {'Matched':>8} {'Match %':>10} "
                       f"{'Total':>10} {'All':>10} {'Avg/Sent':>10} {'Avg Sim':>10}")
    report_lines.append(f"{'Size':>8} {'Sents':>8} {'Sents':>8} {'':>10} "
                       f"{'Top-K':>10} {'Matches':>10} {'(Top-K)':>10} {'':>10}")
    report_lines.append("-" * 90)
    
    # Process each window size
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        total_sentences = results['total_sentences']
        matched_sentences = results['sentences_with_matches']
        total_matches = results['total_matches']
        total_matches_all = results.get('total_matches_all', total_matches)  # Use all matches if available
        similarity_scores = results['similarity_scores']
        
        match_rate = (matched_sentences / total_sentences * 100) if total_sentences > 0 else 0
        avg_matches_per_sent = (total_matches / total_sentences) if total_sentences > 0 else 0
        avg_similarity = (sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0
        
        report_lines.append(
            f"{window_size:>8} {total_sentences:>8} {matched_sentences:>8} "
            f"{match_rate:>9.1f}% {total_matches:>10} {total_matches_all:>10} {avg_matches_per_sent:>10.2f} "
            f"{avg_similarity:>10.3f}"
        )
    
    report_lines.append("")
    
    # Section 2: Detailed Statistics
    report_lines.append("2. DETAILED STATISTICS BY WINDOW SIZE")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        total_sentences = results['total_sentences']
        matched_sentences = results['sentences_with_matches']
        total_matches = results['total_matches']
        total_matches_all = results.get('total_matches_all', total_matches)
        similarity_scores = results['similarity_scores']
        
        report_lines.append(f"Window Size: {window_size}")
        report_lines.append(f"  Total sentences processed: {total_sentences}")
        report_lines.append(f"  Sentences with matches: {matched_sentences} "
                           f"({matched_sentences/total_sentences*100:.1f}%)")
        report_lines.append(f"  Sentences without matches: {total_sentences - matched_sentences} "
                           f"({(total_sentences-matched_sentences)/total_sentences*100:.1f}%)")
        report_lines.append(f"  Total matches found (all): {total_matches_all}")
        report_lines.append(f"  Total matches stored (top-k): {total_matches}")
        report_lines.append(f"  Average matches per sentence (all): {total_matches_all/total_sentences:.2f}")
        report_lines.append(f"  Average matches per sentence (top-k): {total_matches/total_sentences:.2f}")
        
        # NEW: Match distribution statistics
        if 'min_matches' in results:
            report_lines.append(f"  Match count distribution:")
            report_lines.append(f"    Min: {results['min_matches']}")
            report_lines.append(f"    Max: {results['max_matches']}")
            report_lines.append(f"    Median: {results['median_matches']:.2f}")
            report_lines.append(f"    Std Dev: {results['std_matches']:.2f}")
        
        # NEW: Stage-level statistics
        if 'stage1_passed' in results:
            report_lines.append(f"  Two-stage matching statistics:")
            report_lines.append(f"    Stage 1 (full sentence similarity): {results['stage1_passed']} candidates")
            report_lines.append(f"    Stage 2 (exact window match): {results['stage2_passed']} matches")
            report_lines.append(f"    Stage 1 pass rate: {results['stage1_pass_rate']*100:.2f}%")
            report_lines.append(f"    Stage 2 pass rate: {results['stage2_pass_rate']*100:.2f}%")
            report_lines.append(f"    Overall pass rate: {results['overall_pass_rate']*100:.2f}%")
        
        if similarity_scores:
            report_lines.append(f"  Similarity score statistics:")
            report_lines.append(f"    Mean: {sum(similarity_scores)/len(similarity_scores):.3f}")
            report_lines.append(f"    Median: {sorted(similarity_scores)[len(similarity_scores)//2]:.3f}")
            report_lines.append(f"    Min: {min(similarity_scores):.3f}")
            report_lines.append(f"    Max: {max(similarity_scores):.3f}")
            report_lines.append(f"    Std Dev: {pd.Series(similarity_scores).std():.3f}")
        
        # Matches above threshold
        matches_above_threshold = sum(1 for s in similarity_scores if s >= similarity_threshold)
        if similarity_scores:
            report_lines.append(f"  Matches above threshold ({similarity_threshold:.2f}): "
                               f"{matches_above_threshold} ({matches_above_threshold/len(similarity_scores)*100:.1f}%)")
        
        report_lines.append("")
    
    # Section 3: Comparison Across Window Sizes
    report_lines.append("3. COMPARISON ACROSS WINDOW SIZES")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("This section compares matching performance across different window sizes.")
    report_lines.append("")
    
    # Find best window size by match rate
    best_match_rate = 0
    best_window = None
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        total_sentences = results['total_sentences']
        matched_sentences = results['sentences_with_matches']
        match_rate = (matched_sentences / total_sentences * 100) if total_sentences > 0 else 0
        
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_window = window_size
    
    if best_window is not None:
        report_lines.append(f"Best match rate: Window size {best_window} ({best_match_rate:.1f}%)")
        report_lines.append("")
    
    # Find best window size by average similarity
    best_avg_sim = 0
    best_window_sim = None
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        similarity_scores = results['similarity_scores']
        
        if similarity_scores:
            avg_sim = sum(similarity_scores) / len(similarity_scores)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_window_sim = window_size
    
    if best_window_sim is not None:
        report_lines.append(f"Best average similarity: Window size {best_window_sim} ({best_avg_sim:.3f})")
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

