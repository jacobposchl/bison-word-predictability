"""
Report generation functions for matching analysis.
"""

import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


def generate_window_matching_report(
    window_results: Dict,
    similarity_threshold: float = 0.4
) -> str:
    """
    Generate comprehensive report for window matching analysis.
    
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
                       f"{'Total':>10} {'Avg/Sent':>10} {'Avg Sim':>10}")
    report_lines.append(f"{'Size':>8} {'Sents':>8} {'Sents':>8} {'':>10} "
                       f"{'Matches':>10} {'':>10} {'':>10}")
    report_lines.append("-" * 80)
    
    # Process each window size
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        window_size = results['window_size']
        total_sentences = results['total_sentences']
        matched_sentences = results['sentences_with_matches']
        total_matches = results['total_matches']
        similarity_scores = results['similarity_scores']
        
        match_rate = (matched_sentences / total_sentences * 100) if total_sentences > 0 else 0
        avg_matches_per_sent = (total_matches / total_sentences) if total_sentences > 0 else 0
        avg_similarity = (sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0
        
        report_lines.append(
            f"{window_size:>8} {total_sentences:>8} {matched_sentences:>8} "
            f"{match_rate:>9.1f}% {total_matches:>10} {avg_matches_per_sent:>10.2f} "
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
        similarity_scores = results['similarity_scores']
        
        report_lines.append(f"Window Size: {window_size}")
        report_lines.append(f"  Total sentences processed: {total_sentences}")
        report_lines.append(f"  Sentences with matches: {matched_sentences} "
                           f"({matched_sentences/total_sentences*100:.1f}%)")
        report_lines.append(f"  Sentences without matches: {total_sentences - matched_sentences} "
                           f"({(total_sentences-matched_sentences)/total_sentences*100:.1f}%)")
        report_lines.append(f"  Total matches found: {total_matches}")
        report_lines.append(f"  Average matches per sentence: {total_matches/total_sentences:.2f}")
        
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

