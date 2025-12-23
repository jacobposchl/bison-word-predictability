"""
Calvillo et al. (2020) methodology feasibility analysis.

This module implements exploratory analysis to assess whether the Calvillo
matching methodology is feasible with Cantonese-English code-switching data.
"""

import logging
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from .pos_tagging import (
    pos_tag_cantonese,
    pos_tag_english,
    pos_tag_mixed_sentence,
    extract_pos_sequence,
    parse_pattern_segments
)
from .matching_algorithm import find_matches, precompute_monolingual_pos_sequences

logger = logging.getLogger(__name__)


def is_monolingual(pattern: str) -> Optional[str]:
    """
    Determine if a pattern represents a monolingual sentence.
    
    Args:
        pattern: Pattern string like "C10", "E8", or "C5-E3"
        
    Returns:
        'Cantonese' if pure Cantonese, 'English' if pure English,
        None if code-switched
    """
    segments = parse_pattern_segments(pattern)
    languages = {lang for lang, _ in segments}
    
    if len(languages) == 0:
        return None
    elif len(languages) == 1:
        lang = languages.pop()
        return 'Cantonese' if lang == 'C' else 'English'
    else:
        return None  # Code-switched


def extract_monolingual_sentences(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract and categorize monolingual sentences from the dataset.
    
    Parses the pattern column to identify:
    - Pure Cantonese sentences (patterns like "C10")
    - Pure English sentences (patterns like "E8")
    - Code-switched sentences (patterns like "C5-E3")
    
    Args:
        df: DataFrame with 'pattern' column
        
    Returns:
        Dictionary with keys 'cantonese', 'english', 'code_switched'
        Each value is a filtered DataFrame
    """
    logger.info("Extracting monolingual sentences from dataset...")
    
    # Initialize result dictionaries
    cantonese_rows = []
    english_rows = []
    code_switched_rows = []
    
    # Process each row
    for idx, row in df.iterrows():
        pattern = row.get('pattern', '')
        lang_type = is_monolingual(pattern)
        
        if lang_type == 'Cantonese':
            cantonese_rows.append(row)
        elif lang_type == 'English':
            english_rows.append(row)
        else:
            code_switched_rows.append(row)
    
    # Create DataFrames
    cantonese_df = pd.DataFrame(cantonese_rows).reset_index(drop=True)
    english_df = pd.DataFrame(english_rows).reset_index(drop=True)
    code_switched_df = pd.DataFrame(code_switched_rows).reset_index(drop=True)
    
    # Print summary statistics
    total = len(df)
    cant_count = len(cantonese_df)
    eng_count = len(english_df)
    cs_count = len(code_switched_df)
    
    logger.info("=" * 80)
    logger.info("MONOLINGUAL SENTENCE EXTRACTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total sentences: {total}")
    logger.info(f"  Pure Cantonese: {cant_count} ({cant_count/total*100:.1f}%)")
    logger.info(f"  Pure English: {eng_count} ({eng_count/total*100:.1f}%)")
    logger.info(f"  Code-switched: {cs_count} ({cs_count/total*100:.1f}%)")
    logger.info("")
    
    # Breakdown by speaker group
    if 'group' in df.columns:
        logger.info("Breakdown by Speaker Group:")
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            group_cant = len([r for _, r in group_df.iterrows() if is_monolingual(r.get('pattern', '')) == 'Cantonese'])
            group_eng = len([r for _, r in group_df.iterrows() if is_monolingual(r.get('pattern', '')) == 'English'])
            group_cs = len([r for _, r in group_df.iterrows() if is_monolingual(r.get('pattern', '')) is None])
            logger.info(f"  {group}:")
            logger.info(f"    Cantonese: {group_cant}, English: {group_eng}, Code-switched: {group_cs}")
    
    logger.info("=" * 80)
    
    return {
        'cantonese': cantonese_df,
        'english': english_df,
        'code_switched': code_switched_df
    }


def analyze_pos_tagging(
    sentences: pd.DataFrame,
    sample_size: int = 100,
    random_seed: int = 42
) -> Dict:
    """
    Analyze POS tagging on a sample of sentences.
    
    Args:
        sentences: DataFrame with sentences to analyze
        sample_size: Number of sentences to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with statistics and sample results
    """
    if sample_size == 0:
        logger.info(f"Analyzing POS tagging on FULL dataset ({len(sentences)} sentences)...")
        sample_df = sentences.copy()
    else:
        logger.info(f"Analyzing POS tagging on sample of {sample_size} sentences...")
        # Set random seed
        random.seed(random_seed)
        
        # Sample sentences
        if len(sentences) > sample_size:
            sample_df = sentences.sample(n=sample_size, random_state=random_seed).copy()
        else:
            sample_df = sentences.copy()
    
    results = []
    pos_tag_counts = Counter()
    sequence_lengths = []
    errors = 0
    
    # Use tqdm for progress bar
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="POS Tagging"):
        sentence = row.get('reconstructed_sentence', '')
        pattern = row.get('pattern', '')
        
        try:
            # Determine if monolingual or code-switched
            lang_type = is_monolingual(pattern)
            
            if lang_type == 'Cantonese':
                tagged = pos_tag_cantonese(sentence)
                pos_seq = extract_pos_sequence(tagged)
                lang = 'Cantonese'
            elif lang_type == 'English':
                tagged = pos_tag_english(sentence)
                pos_seq = extract_pos_sequence(tagged)
                lang = 'English'
            else:
                # Code-switched
                tagged = pos_tag_mixed_sentence(sentence, pattern)
                pos_seq = extract_pos_sequence(tagged)
                lang = 'Code-switched'
            
            # Collect statistics
            if pos_seq:
                sequence_lengths.append(len(pos_seq))
                for pos in pos_seq:
                    pos_tag_counts[pos] += 1
                
                results.append({
                    'sentence': sentence,
                    'pattern': pattern,
                    'lang_type': lang,
                    'pos_sequence': ' '.join(pos_seq),
                    'sequence_length': len(pos_seq),
                    'error': False
                })
            else:
                errors += 1
                results.append({
                    'sentence': sentence,
                    'pattern': pattern,
                    'lang_type': lang,
                    'pos_sequence': '',
                    'sequence_length': 0,
                    'error': True
                })
                
        except Exception as e:
            errors += 1
            logger.warning(f"Error tagging sentence '{sentence[:50]}...': {e}")
            results.append({
                'sentence': sentence,
                'pattern': pattern,
                'lang_type': 'Unknown',
                'pos_sequence': '',
                'sequence_length': 0,
                'error': True
            })
    
    # Calculate statistics
    stats = {
        'total_sampled': len(sample_df),
        'successful': len(sample_df) - errors,
        'errors': errors,
        'error_rate': errors / len(sample_df) * 100 if len(sample_df) > 0 else 0,
        'avg_sequence_length': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
        'min_sequence_length': min(sequence_lengths) if sequence_lengths else 0,
        'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0,
        'pos_tag_distribution': dict(pos_tag_counts.most_common(20)),
        'sample_results': pd.DataFrame(results)
    }
    
    # Print summary
    logger.info("=" * 80)
    logger.info("POS TAGGING ANALYSIS RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total sampled: {stats['total_sampled']}")
    logger.info(f"Successful: {stats['successful']} ({100-stats['error_rate']:.1f}%)")
    logger.info(f"Errors: {stats['errors']} ({stats['error_rate']:.1f}%)")
    logger.info(f"Average sequence length: {stats['avg_sequence_length']:.1f}")
    logger.info(f"Sequence length range: {stats['min_sequence_length']}-{stats['max_sequence_length']}")
    logger.info("")
    logger.info("Top 10 POS tags:")
    for pos, count in list(pos_tag_counts.most_common(10)):
        logger.info(f"  {pos}: {count}")
    logger.info("=" * 80)
    
    return stats


def test_matching_algorithm(
    code_switched: pd.DataFrame,
    monolingual: Dict[str, pd.DataFrame],
    sample_size: int = 100,
    similarity_threshold: float = 0.4,
    random_seed: int = 42,
    max_monolingual_per_lang: Optional[int] = None
) -> Dict:
    """
    Test the matching algorithm on a sample of code-switched sentences.
    
    Args:
        code_switched: DataFrame with code-switched sentences
        monolingual: Dictionary with 'cantonese' and 'english' DataFrames
        sample_size: Number of sentences to test
        similarity_threshold: Minimum similarity for matches (default 0.4)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with matching statistics and results
    """
    if sample_size == 0:
        logger.info(f"Testing matching algorithm on FULL dataset ({len(code_switched)} code-switched sentences)...")
        sample_df = code_switched.copy()
    else:
        logger.info(f"Testing matching algorithm on sample of {sample_size} code-switched sentences...")
        random.seed(random_seed)
        
        # Sample code-switched sentences
        if len(code_switched) > sample_size:
            sample_df = code_switched.sample(n=sample_size, random_state=random_seed).copy()
        else:
            sample_df = code_switched.copy()
    
    # Convert monolingual DataFrames to list of dicts for matching function
    monolingual_dicts = {
        'cantonese': monolingual['cantonese'].to_dict('records') if 'cantonese' in monolingual else [],
        'english': monolingual['english'].to_dict('records') if 'english' in monolingual else []
    }
    
    # Pre-compute POS sequences for all monolingual sentences (major speedup!)
    # Optionally limit to avoid comparing against too many sentences
    monolingual_dicts = precompute_monolingual_pos_sequences(
        monolingual_dicts,
        max_per_language=max_monolingual_per_lang
    )
    
    results = []
    sentences_with_matches = 0
    total_matches = 0
    similarity_scores = []
    c_to_e_matches = 0
    e_to_c_matches = 0
    c_to_e_sentences = 0
    e_to_c_sentences = 0
    
    # Use tqdm for progress bar
    for row_idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Matching"):
        
        cs_sentence = row.to_dict()
        pattern = row.get('pattern', '')
        
        # Determine switch direction
        segments = parse_pattern_segments(pattern)
        has_c_to_e = False
        has_e_to_c = False
        
        for i in range(len(segments) - 1):
            if segments[i][0] == 'C' and segments[i+1][0] == 'E':
                has_c_to_e = True
            elif segments[i][0] == 'E' and segments[i+1][0] == 'C':
                has_e_to_c = True
        
        # Count sentences (not switches) - each sentence counted once per direction
        if has_c_to_e:
            c_to_e_sentences += 1
        if has_e_to_c:
            e_to_c_sentences += 1
        
        # Find matches
        matches = find_matches(
            cs_sentence,
            monolingual_dicts,
            similarity_threshold=similarity_threshold
        )
        
        num_matches = len(matches)
        total_matches += num_matches
        
        if num_matches > 0:
            sentences_with_matches += 1
            for match in matches:
                similarity_scores.append(match['similarity'])
                if match['language'] == 'english':
                    c_to_e_matches += 1
                else:
                    e_to_c_matches += 1
        
        results.append({
            'sentence': row.get('reconstructed_sentence', ''),
            'pattern': pattern,
            'num_matches': num_matches,
            'has_match': num_matches > 0,
            'best_similarity': max([m['similarity'] for m in matches]) if matches else 0.0,
            'has_c_to_e': has_c_to_e,
            'has_e_to_c': has_e_to_c,
            'matches_detail': matches[:5]  # Keep top 5 matches for inspection
        })
    
    # Calculate statistics
    success_rate = (sentences_with_matches / len(sample_df)) * 100 if len(sample_df) > 0 else 0
    avg_matches = total_matches / len(sample_df) if len(sample_df) > 0 else 0
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    stats = {
        'total_tested': len(sample_df),
        'sentences_with_matches': sentences_with_matches,
        'success_rate': success_rate,
        'total_matches': total_matches,
        'avg_matches_per_sentence': avg_matches,
        'avg_similarity': avg_similarity,
        'min_similarity': min(similarity_scores) if similarity_scores else 0,
        'max_similarity': max(similarity_scores) if similarity_scores else 0,
        'c_to_e_sentences': c_to_e_sentences,
        'e_to_c_sentences': e_to_c_sentences,
        'c_to_e_matches': c_to_e_matches,
        'e_to_c_matches': e_to_c_matches,
        'similarity_distribution': Counter([round(s, 1) for s in similarity_scores]),
        'results': pd.DataFrame(results)
    }
    
    # Print summary
    logger.info("=" * 80)
    logger.info("MATCHING ALGORITHM TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total tested: {stats['total_tested']}")
    logger.info(f"Sentences with at least one match: {sentences_with_matches} ({success_rate:.1f}%)")
    logger.info(f"Total matches found: {total_matches}")
    logger.info(f"Average matches per sentence: {avg_matches:.2f}")
    if similarity_scores:
        logger.info(f"Average similarity score: {avg_similarity:.3f}")
        logger.info(f"Similarity range: {stats['min_similarity']:.3f} - {stats['max_similarity']:.3f}")
    logger.info("")
    logger.info(f"C→E switches: {c_to_e_sentences} sentences, {c_to_e_matches} matches")
    logger.info(f"E→C switches: {e_to_c_sentences} sentences, {e_to_c_matches} matches")
    logger.info("=" * 80)
    
    return stats


def analyze_distributions(df: pd.DataFrame) -> Dict:
    """
    Analyze distributions of code-switching patterns.
    
    Analyzes:
    - Switch direction (C→E vs E→C)
    - Switch position (beginning, middle, end)
    - Segment length (single-word vs multi-word)
    - Sentence length
    - Matrix language distribution
    
    Args:
        df: DataFrame with code-switched sentences
        
    Returns:
        Dictionary with distribution statistics
    """
    logger.info("Analyzing code-switching distributions...")
    
    switch_directions = {'C→E': 0, 'E→C': 0, 'Multiple': 0}
    switch_positions = {'beginning': 0, 'middle': 0, 'end': 0}
    segment_lengths = {'single_word': 0, 'multi_word': 0}
    sentence_lengths = []
    matrix_languages = Counter()
    
    for idx, row in df.iterrows():
        pattern = row.get('pattern', '')
        segments = parse_pattern_segments(pattern)
        
        # Sentence length
        total_words = sum(count for _, count in segments)
        sentence_lengths.append(total_words)
        
        # Matrix language
        matrix_lang = row.get('matrix_language', 'Unknown')
        matrix_languages[matrix_lang] += 1
        
        # Switch direction
        c_to_e = 0
        e_to_c = 0
        for i in range(len(segments) - 1):
            if segments[i][0] == 'C' and segments[i+1][0] == 'E':
                c_to_e += 1
            elif segments[i][0] == 'E' and segments[i+1][0] == 'C':
                e_to_c += 1
        
        if c_to_e > 0 and e_to_c > 0:
            switch_directions['Multiple'] += 1
        elif c_to_e > 0:
            switch_directions['C→E'] += 1
        elif e_to_c > 0:
            switch_directions['E→C'] += 1
        
        # Switch position
        if len(segments) > 1:
            first_segment_length = segments[0][1]
            sentence_length = total_words
            
            if first_segment_length <= sentence_length * 0.25:
                switch_positions['beginning'] += 1
            elif first_segment_length >= sentence_length * 0.75:
                switch_positions['end'] += 1
            else:
                switch_positions['middle'] += 1
        
        # Segment lengths
        for lang, count in segments:
            if count == 1:
                segment_lengths['single_word'] += 1
            else:
                segment_lengths['multi_word'] += 1
    
    stats = {
        'switch_directions': switch_directions,
        'switch_positions': switch_positions,
        'segment_lengths': segment_lengths,
        'sentence_lengths': sentence_lengths,
        'matrix_languages': dict(matrix_languages),
        'avg_sentence_length': sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
        'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
        'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0
    }
    
    # Print summary
    logger.info("=" * 80)
    logger.info("DISTRIBUTION ANALYSIS RESULTS")
    logger.info("=" * 80)
    logger.info(f"Switch Directions:")
    total_switches = sum(switch_directions.values())
    for direction, count in switch_directions.items():
        pct = (count / total_switches * 100) if total_switches > 0 else 0
        logger.info(f"  {direction}: {count} ({pct:.1f}%)")
    logger.info("")
    logger.info(f"Switch Positions:")
    total_positions = sum(switch_positions.values())
    for position, count in switch_positions.items():
        pct = (count / total_positions * 100) if total_positions > 0 else 0
        logger.info(f"  {position}: {count} ({pct:.1f}%)")
    logger.info("")
    logger.info(f"Sentence Length: avg={stats['avg_sentence_length']:.1f}, "
                f"range={stats['min_sentence_length']}-{stats['max_sentence_length']}")
    logger.info("=" * 80)
    
    return stats


def generate_report(all_results: Dict) -> str:
    """
    Generate comprehensive feasibility report.
    
    Args:
        all_results: Dictionary containing all analysis results
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CALVILLO ET AL. (2020) METHODOLOGY FEASIBILITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset Overview
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 80)
    if 'monolingual' in all_results:
        mono = all_results['monolingual']
        total = (len(mono.get('cantonese', [])) + 
                len(mono.get('english', [])) + 
                len(mono.get('code_switched', [])))
        report_lines.append(f"Total sentences: {total}")
        report_lines.append(f"  - Pure Cantonese: {len(mono.get('cantonese', []))}")
        report_lines.append(f"  - Pure English: {len(mono.get('english', []))}")
        report_lines.append(f"  - Code-switched: {len(mono.get('code_switched', []))}")
    report_lines.append("")
    
    # Monolingual Sentence Availability
    report_lines.append("2. MONOLINGUAL SENTENCE AVAILABILITY")
    report_lines.append("-" * 80)
    if 'monolingual' in all_results:
        mono = all_results['monolingual']
        cant_count = len(mono.get('cantonese', []))
        eng_count = len(mono.get('english', []))
        report_lines.append(f"Cantonese monolingual sentences: {cant_count}")
        report_lines.append(f"English monolingual sentences: {eng_count}")
        
        target = 100
        report_lines.append("")
        report_lines.append("Assessment:")
        if cant_count >= target and eng_count >= target:
            report_lines.append("  ✓ SUFFICIENT: Both languages have enough monolingual sentences")
        elif cant_count >= target or eng_count >= target:
            report_lines.append("  ⚠ PARTIAL: One language may be limited")
        else:
            report_lines.append("  ✗ LIMITED: Both languages may need more monolingual sentences")
    report_lines.append("")
    
    # POS Tagging Quality
    report_lines.append("3. POS TAGGING QUALITY")
    report_lines.append("-" * 80)
    if 'pos_tagging' in all_results:
        pos = all_results['pos_tagging']
        report_lines.append(f"Error rate: {pos.get('error_rate', 0):.1f}%")
        report_lines.append(f"Average sequence length: {pos.get('avg_sequence_length', 0):.1f}")
        report_lines.append("")
        report_lines.append("Assessment:")
        if pos.get('error_rate', 100) < 10:
            report_lines.append("  ✓ GOOD: POS tagging is reliable")
        elif pos.get('error_rate', 100) < 25:
            report_lines.append("  ⚠ ACCEPTABLE: Some tagging errors may affect matching")
        else:
            report_lines.append("  ✗ POOR: High error rate may significantly impact matching")
    report_lines.append("")
    
    # Matching Success Rates
    report_lines.append("4. MATCHING SUCCESS RATES")
    report_lines.append("-" * 80)
    if 'matching' in all_results:
        match = all_results['matching']
        report_lines.append(f"Sentences with matches: {match.get('success_rate', 0):.1f}%")
        report_lines.append(f"Average matches per sentence: {match.get('avg_matches_per_sentence', 0):.2f}")
        report_lines.append(f"Average similarity: {match.get('avg_similarity', 0):.3f}")
        report_lines.append("")
        report_lines.append("Assessment:")
        success_rate = match.get('success_rate', 0)
        if success_rate >= 50:
            report_lines.append("  ✓ FEASIBLE: Majority of sentences can find matches")
        elif success_rate >= 30:
            report_lines.append("  ⚠ MARGINAL: Significant portion may lack matches")
        else:
            report_lines.append("  ✗ PROBLEMATIC: Most sentences cannot find good matches")
    report_lines.append("")
    
    # Data Distribution Insights
    report_lines.append("5. DATA DISTRIBUTION INSIGHTS")
    report_lines.append("-" * 80)
    if 'distributions' in all_results:
        dist = all_results['distributions']
        report_lines.append(f"Switch directions: {dist.get('switch_directions', {})}")
        report_lines.append(f"Average sentence length: {dist.get('avg_sentence_length', 0):.1f} words")
    report_lines.append("")
    
    # Potential Issues
    report_lines.append("6. POTENTIAL ISSUES")
    report_lines.append("-" * 80)
    issues = []
    
    if 'monolingual' in all_results:
        mono = all_results['monolingual']
        if len(mono.get('english', [])) < 100:
            issues.append("Limited English monolingual sentences for matching")
        if len(mono.get('cantonese', [])) < 100:
            issues.append("Limited Cantonese monolingual sentences for matching")
    
    if 'pos_tagging' in all_results and all_results['pos_tagging'].get('error_rate', 0) > 15:
        issues.append("High POS tagging error rate may affect matching quality")
    
    if 'matching' in all_results and all_results['matching'].get('success_rate', 0) < 40:
        issues.append("Low matching success rate - many sentences lack matches")
    
    if issues:
        for issue in issues:
            report_lines.append(f"  - {issue}")
    else:
        report_lines.append("  No major issues identified")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("7. RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    # Determine overall feasibility
    feasible = True
    concerns = []
    
    if 'matching' in all_results:
        if all_results['matching'].get('success_rate', 0) < 50:
            feasible = False
            concerns.append("Matching success rate below 50%")
    
    if 'monolingual' in all_results:
        mono = all_results['monolingual']
        if len(mono.get('cantonese', [])) < 50 or len(mono.get('english', [])) < 50:
            feasible = False
            concerns.append("Insufficient monolingual sentences")
    
    if feasible and not concerns:
        report_lines.append("✓ PROCEED: Methodology appears feasible with current data")
        report_lines.append("")
        report_lines.append("Suggested next steps:")
        report_lines.append("  1. Implement full matching algorithm")
        report_lines.append("  2. Conduct full analysis on all code-switched sentences")
        report_lines.append("  3. Validate results with manual inspection")
    elif len(concerns) == 1:
        report_lines.append("⚠ PROCEED WITH CAUTION: One major concern identified")
        report_lines.append(f"  Concern: {concerns[0]}")
        report_lines.append("")
        report_lines.append("Suggested modifications:")
        report_lines.append("  1. Adjust similarity threshold if matching rate is low")
        report_lines.append("  2. Consider relaxing matching criteria")
        report_lines.append("  3. Focus on subset of data with better coverage")
    else:
        report_lines.append("✗ NOT RECOMMENDED: Multiple concerns identified")
        for concern in concerns:
            report_lines.append(f"  - {concern}")
        report_lines.append("")
        report_lines.append("Suggested alternatives:")
        report_lines.append("  1. Collect more monolingual sentences")
        report_lines.append("  2. Modify matching algorithm parameters")
        report_lines.append("  3. Consider alternative methodology")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)
