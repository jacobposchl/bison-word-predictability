"""
Report generation functions for exploratory analysis.
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
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        n = results['window_size']
        total = results['total_sentences']
        matched = results['sentences_with_matches']
        match_rate = results['match_rate'] * 100
        total_matches = results['total_matches']
        avg_matches = results['avg_matches_per_sentence']
        avg_sim = results['avg_similarity']
        
        report_lines.append(f"  n={n:>5} {total:>8} {matched:>8} {match_rate:>9.1f}% "
                          f"{total_matches:>10} {avg_matches:>10.2f} {avg_sim:>10.3f}")
    
    report_lines.append("")
    
    # Section 2: Similarity Distribution Summary
    report_lines.append("2. SIMILARITY SCORE DISTRIBUTIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        n = results['window_size']
        scores = results['similarity_scores']
        
        if scores:
            report_lines.append(f"Window Size n={n}:")
            report_lines.append(f"  Min:     {min(scores):.3f}")
            report_lines.append(f"  25th %:  {pd.Series(scores).quantile(0.25):.3f}")
            report_lines.append(f"  Median:  {pd.Series(scores).quantile(0.50):.3f}")
            report_lines.append(f"  75th %:  {pd.Series(scores).quantile(0.75):.3f}")
            report_lines.append(f"  Max:     {max(scores):.3f}")
            report_lines.append(f"  Mean:    {sum(scores)/len(scores):.3f}")
            report_lines.append(f"  StdDev:  {pd.Series(scores).std():.3f}")
            
            # Similarity buckets
            high_sim = sum(1 for s in scores if s >= 0.8)
            med_sim = sum(1 for s in scores if 0.6 <= s < 0.8)
            low_sim = sum(1 for s in scores if s < 0.6)
            total_scores = len(scores)
            
            report_lines.append(f"  High similarity (≥0.8): {high_sim}/{total_scores} ({high_sim/total_scores*100:.1f}%)")
            report_lines.append(f"  Med similarity (0.6-0.8): {med_sim}/{total_scores} ({med_sim/total_scores*100:.1f}%)")
            report_lines.append(f"  Low similarity (<0.6): {low_sim}/{total_scores} ({low_sim/total_scores*100:.1f}%)")
            report_lines.append("")
    
    # Section 3: Match Quality Analysis
    report_lines.append("3. MATCH QUALITY BREAKDOWN")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        n = results['window_size']
        detailed = results['detailed_matches']
        
        if detailed:
            # Get only rank-1 (selected) matches for statistics
            selected_matches = [m for m in detailed if m['rank'] == 1]
            
            if selected_matches:
                same_group = sum(1 for m in selected_matches if m['same_group'])
                same_speaker = sum(1 for m in selected_matches if m['same_speaker'])
                total = len(selected_matches)
                
                report_lines.append(f"Window Size n={n}:")
                report_lines.append(f"  Selected matches (best per sentence): {total}")
                report_lines.append(f"  Matches from same group:   {same_group}/{total} ({same_group/total*100:.1f}%)")
                report_lines.append(f"  Matches from same speaker: {same_speaker}/{total} ({same_speaker/total*100:.1f}%)")
                
                # Calculate average time distance (convert ms to seconds)
                avg_time_dist = sum(m['time_distance'] for m in selected_matches) / total / 1000.0
                report_lines.append(f"  Average time distance:     {avg_time_dist:.2f} seconds")
                
                # Calculate average matches per sentence (all matches above threshold)
                # Group by sentence to count matches
                sentence_match_counts = {}
                for m in detailed:
                    key = m['cs_translation']
                    sentence_match_counts[key] = sentence_match_counts.get(key, 0) + 1
                
                avg_matches_per_sent = sum(sentence_match_counts.values()) / len(sentence_match_counts) if sentence_match_counts else 0
                report_lines.append(f"  Avg matches above threshold per sentence: {avg_matches_per_sent:.2f}")
                
                # Quartiles of time distance
                time_distances = [m['time_distance'] / 1000.0 for m in selected_matches]  # Convert to seconds
                if time_distances:
                    report_lines.append(f"  Time distance quartiles:")
                    report_lines.append(f"    25th percentile: {pd.Series(time_distances).quantile(0.25):.2f}s")
                    report_lines.append(f"    Median:          {pd.Series(time_distances).quantile(0.50):.2f}s")
                    report_lines.append(f"    75th percentile: {pd.Series(time_distances).quantile(0.75):.2f}s")
                
                report_lines.append("")
    
    # Section 4: Example Matches
    report_lines.append("4. EXAMPLE MATCHES (Top 3-5 Sentences with Best Average Similarity)")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        n = results['window_size']
        examples = results['example_matches']
        
        if not examples:
            continue
        
        report_lines.append(f"Window Size n={n}:")
        report_lines.append("")
        
        for ex_idx, example in enumerate(examples[:3], 1):  # Show top 3 examples
            cs_sent = example['cs_sentence']
            cs_trans = example['cs_translation']
            matches = example['matches'][:5]  # Top 5 matches
            
            report_lines.append(f"  Example {ex_idx}:")
            report_lines.append(f"    Code-switched:  {cs_sent}")
            report_lines.append(f"    Translated:     {cs_trans}")
            report_lines.append(f"    Pattern:        {matches[0]['cs_pattern']}")
            report_lines.append(f"    Switch Index:   {matches[0]['switch_index']}")
            report_lines.append(f"    POS Window:     {matches[0]['pos_window']}")
            report_lines.append("")
            report_lines.append(f"    Top {len(matches)} Matches:")
            
            for match in matches:
                rank = match['rank']
                sim = match['similarity']
                matched_sent = match['matched_sentence']
                matched_pos = match['matched_pos']
                same_grp = "✓" if match['same_group'] else " "
                same_spk = "✓" if match['same_speaker'] else " "
                time_dist = match['time_distance']
                
                report_lines.append(f"      {rank}. [{sim:.3f}] (Grp:{same_grp} Spk:{same_spk} Time:{time_dist:>7.1f}s)")
                report_lines.append(f"         Text: {matched_sent}")
                report_lines.append(f"         POS:  {matched_pos}")
            
            report_lines.append("")
    
    # Section 5: Interpretation
    report_lines.append("5. INTERPRETATION & RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Analyze results across window sizes
    best_window = None
    best_match_rate = 0.0
    
    for window_key, results in window_results.items():
        if results['match_rate'] > best_match_rate:
            best_match_rate = results['match_rate']
            best_window = results['window_size']
    
    report_lines.append(f"Best performing window size: n={best_window}")
    report_lines.append(f"  Match rate: {best_match_rate*100:.1f}%")
    report_lines.append("")
    
    # Overall assessment
    if best_match_rate >= 0.7:
        report_lines.append("✓ EXCELLENT: High match rate indicates strong POS pattern similarity")
        report_lines.append("  The methodology is well-suited for this dataset.")
    elif best_match_rate >= 0.5:
        report_lines.append("⚠ GOOD: Moderate match rate, methodology is feasible")
        report_lines.append("  Consider adjusting similarity threshold or window size.")
    else:
        report_lines.append("⚠ CAUTION: Low match rate may indicate limited POS pattern overlap")
        report_lines.append("  May need to collect more monolingual data or adjust methodology.")
    
    report_lines.append("")
    
    # Window size recommendations
    report_lines.append("Window Size Recommendations:")
    for window_key in sorted(window_results.keys()):
        results = window_results[window_key]
        n = results['window_size']
        match_rate = results['match_rate']
        avg_sim = results['avg_similarity']
        
        if match_rate >= 0.6 and avg_sim >= 0.7:
            assessment = "✓ Recommended"
        elif match_rate >= 0.4:
            assessment = "○ Acceptable"
        else:
            assessment = "✗ Not recommended"
        
        report_lines.append(f"  n={n}: {assessment} (Match: {match_rate*100:.1f}%, Avg Sim: {avg_sim:.3f})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)
