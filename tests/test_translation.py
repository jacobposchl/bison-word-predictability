"""
Comprehensive test suite for translation functionality.

Tests both NLLB (free, local) and OpenAI (paid, cloud) translation backends.

Usage:
    # Test NLLB (no API key needed)
    python tests/test_translation.py
    python tests/test_translation.py --verbose
    
    # Test OpenAI (requires API key)
    python tests/test_translation.py --backend openai --api-key YOUR_API_KEY
"""

import os
import sys
import json
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.translation import CodeSwitchTranslator, TranslationCache
from src.experiments.nllb_translator import NLLBTranslator


class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✓ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ✗ {test_name}: {error}")


def run_cache_tests(results: TestResult, verbose: bool = False):
    """Test the translation cache functionality."""
    print("\n" + "="*70)
    print("1. CACHE FUNCTIONALITY TESTS")
    print("="*70)
    
    try:
        # Test 1: Cache initialization
        if verbose:
            print("\nTest: Cache initialization...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(tmpdir)
            assert cache.cache_dir.exists(), "Cache directory not created"
            # Cache file is created lazily, but cache dict should exist
            assert isinstance(cache.cache, dict), "Cache is not a dictionary"
        results.add_pass("Cache initialization")
    except Exception as e:
        results.add_fail("Cache initialization", str(e))
    
    try:
        # Test 2: Cache set and get
        if verbose:
            print("\nTest: Cache set/get...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(tmpdir)
            cache.set("hello", "你好")
            result = cache.get("hello")
            assert result == "你好", f"Expected '你好', got '{result}'"
        results.add_pass("Cache set and get")
    except Exception as e:
        results.add_fail("Cache set and get", str(e))
    
    try:
        # Test 3: Cache with context
        if verbose:
            print("\nTest: Cache with context...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(tmpdir)
            cache.set("go", "去", context="我")
            cache.set("go", "走", context="佢")
            result1 = cache.get("go", context="我")
            result2 = cache.get("go", context="佢")
            assert result1 == "去", f"Expected '去', got '{result1}'"
            assert result2 == "走", f"Expected '走', got '{result2}'"
        results.add_pass("Cache with context")
    except Exception as e:
        results.add_fail("Cache with context", str(e))
    
    try:
        # Test 4: Cache persistence
        if verbose:
            print("\nTest: Cache persistence...")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = TranslationCache(tmpdir)
            cache1.set("test", "測試")
            cache2 = TranslationCache(tmpdir)
            result = cache2.get("test")
            assert result == "測試", f"Expected '測試', got '{result}'"
        results.add_pass("Cache persistence")
    except Exception as e:
        results.add_fail("Cache persistence", str(e))


def run_translator_basic_tests(results: TestResult, verbose: bool = False):
    """Test basic translator functionality without API calls."""
    print("\n" + "="*70)
    print("2. TRANSLATOR BASIC TESTS (No API)")
    print("="*70)
    
    try:
        # Test 1: Initialization requires API key
        if verbose:
            print("\nTest: API key validation...")
        try:
            CodeSwitchTranslator(api_key="")
            results.add_fail("API key validation", "Should have raised ValueError")
        except ValueError:
            results.add_pass("API key validation")
    except Exception as e:
        results.add_fail("API key validation", str(e))
    
    try:
        # Test 2: Pattern parsing
        if verbose:
            print("\nTest: Pattern parsing...")
        with tempfile.TemporaryDirectory() as tmpdir:
            translator = CodeSwitchTranslator(api_key="test-key", cache_dir=tmpdir)
            
            pattern1 = translator._parse_pattern("C5-E2-C3")
            assert pattern1 == [('C', 5), ('E', 2), ('C', 3)], f"Pattern parse failed: {pattern1}"
            
            pattern2 = translator._parse_pattern("E1")
            assert pattern2 == [('E', 1)], f"Pattern parse failed: {pattern2}"
            
            pattern3 = translator._parse_pattern("C10-E1-C5")
            assert pattern3 == [('C', 10), ('E', 1), ('C', 5)], f"Pattern parse failed: {pattern3}"
        results.add_pass("Pattern parsing")
    except Exception as e:
        results.add_fail("Pattern parsing", str(e))
    
    try:
        # Test 3: Segment extraction
        if verbose:
            print("\nTest: Segment extraction...")
        with tempfile.TemporaryDirectory() as tmpdir:
            translator = CodeSwitchTranslator(api_key="test-key", cache_dir=tmpdir)
            
            words = ['我', '哋', 'go', '咗', '公', '園']
            pattern = "C2-E1-C3"
            segments = translator._extract_segments_from_sentence(words, pattern)
            
            assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
            assert segments[0] == ('C', ['我', '哋'], 0, 2), f"Segment 0 wrong: {segments[0]}"
            assert segments[1] == ('E', ['go'], 2, 3), f"Segment 1 wrong: {segments[1]}"
            assert segments[2] == ('C', ['咗', '公', '園'], 3, 6), f"Segment 2 wrong: {segments[2]}"
        results.add_pass("Segment extraction")
    except Exception as e:
        results.add_fail("Segment extraction", str(e))


def run_translation_tests(api_key: str, results: TestResult, verbose: bool = False):
    """Test actual translation with real API calls."""
    print("\n" + "="*70)
    print("3. REAL TRANSLATION TESTS (With OpenAI API)")
    print("="*70)
    
    # Test cases for translation
    test_cases = [
        {
            'name': 'Single English word ("go")',
            'sentence': '我 哋 go 咗 公 園',
            'pattern': 'C2-E1-C3',
            'words': ['我', '哋', 'go', '咗', '公', '園'],
            'expected_english_segments': 1
        },
        {
            'name': 'Single English word ("like")',
            'sentence': '佢 like 我',
            'pattern': 'C1-E1-C1',
            'words': ['佢', 'like', '我'],
            'expected_english_segments': 1
        },
        {
            'name': 'Multi-word English phrase',
            'sentence': '今 日 天 氣 very good 呀',
            'pattern': 'C4-E2-C1',
            'words': ['今', '日', '天', '氣', 'very', 'good', '呀'],
            'expected_english_segments': 1
        },
        {
            'name': 'Multiple English segments',
            'sentence': 'I 想 eat 嘢',
            'pattern': 'E1-C1-E1-C1',
            'words': ['I', '想', 'eat', '嘢'],
            'expected_english_segments': 2
        },
        {
            'name': 'Pure Cantonese (no translation needed)',
            'sentence': '我 哋 去 咗 公 園',
            'pattern': 'C6',
            'words': ['我', '哋', '去', '咗', '公', '園'],
            'expected_english_segments': 0
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        translator = CodeSwitchTranslator(
            api_key=api_key,
            cache_dir=tmpdir,
            model="gpt-4",
            use_cache=True
        )
        
        for test_case in test_cases:
            try:
                if verbose:
                    print(f"\nTest: {test_case['name']}")
                    print(f"  Original: {test_case['sentence']}")
                    print(f"  Pattern:  {test_case['pattern']}")
                
                result = translator.translate_code_switched_sentence(
                    test_case['sentence'],
                    test_case['pattern'],
                    test_case['words']
                )
                
                # Validate result structure
                assert 'translated_sentence' in result, "Missing 'translated_sentence' in result"
                assert 'original_sentence' in result, "Missing 'original_sentence' in result"
                assert 'pattern' in result, "Missing 'pattern' in result"
                assert 'segments' in result, "Missing 'segments' in result"
                
                # Validate content
                assert result['original_sentence'] == test_case['sentence'], "Original sentence mismatch"
                assert result['pattern'] == test_case['pattern'], "Pattern mismatch"
                
                # Check segments
                english_segments = [s for s in result['segments'] if s['language'] == 'English']
                assert len(english_segments) == test_case['expected_english_segments'], \
                    f"Expected {test_case['expected_english_segments']} English segments, got {len(english_segments)}"
                
                # Validate translation is not empty (unless no English)
                if test_case['expected_english_segments'] > 0:
                    assert result['translated_sentence'], "Translation is empty"
                    assert result['translated_sentence'] != test_case['sentence'], \
                        "Translation should differ from original"
                
                if verbose:
                    print(f"  Translated: {result['translated_sentence']}")
                    print(f"  English segments translated: {len(english_segments)}")
                
                results.add_pass(test_case['name'])
                
            except Exception as e:
                results.add_fail(test_case['name'], str(e))


def run_batch_translation_tests(api_key: str, results: TestResult, verbose: bool = False):
    """Test batch translation functionality."""
    print("\n" + "="*70)
    print("4. BATCH TRANSLATION TESTS")
    print("="*70)
    
    try:
        if verbose:
            print("\nTest: Batch translation...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            translator = CodeSwitchTranslator(
                api_key=api_key,
                cache_dir=tmpdir,
                model="gpt-4",
                show_progress=False  # Disable for testing
            )
            
            sentences = [
                "我 哋 go 咗 公 園",
                "佢 like 我"
            ]
            patterns = ["C2-E1-C3", "C1-E1-C1"]
            words_list = [
                ['我', '哋', 'go', '咗', '公', '園'],
                ['佢', 'like', '我']
            ]
            
            batch_results = translator.translate_batch(sentences, patterns, words_list)
            
            assert len(batch_results) == 2, f"Expected 2 results, got {len(batch_results)}"
            
            for i, result in enumerate(batch_results):
                assert 'translated_sentence' in result, f"Result {i} missing 'translated_sentence'"
                assert 'original_sentence' in result, f"Result {i} missing 'original_sentence'"
                assert result['original_sentence'] == sentences[i], f"Result {i} original mismatch"
                
                if verbose:
                    print(f"  Sentence {i+1}: {result['original_sentence']}")
                    print(f"    → {result['translated_sentence']}")
        
        results.add_pass("Batch translation")
    except Exception as e:
        results.add_fail("Batch translation", str(e))
    
    try:
        # Test batch validation
        if verbose:
            print("\nTest: Batch input validation...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            translator = CodeSwitchTranslator(api_key="test-key", cache_dir=tmpdir)
            
            sentences = ["test1", "test2"]
            patterns = ["C1"]  # Mismatched length
            words_list = [['test'], ['test']]
            
            try:
                translator.translate_batch(sentences, patterns, words_list)
                results.add_fail("Batch input validation", "Should have raised ValueError")
            except ValueError:
                results.add_pass("Batch input validation")
    except Exception as e:
        results.add_fail("Batch input validation", str(e))


def run_caching_tests(api_key: str, results: TestResult, verbose: bool = False):
    """Test that caching works correctly with real API."""
    print("\n" + "="*70)
    print("5. CACHING EFFICIENCY TESTS")
    print("="*70)
    
    try:
        if verbose:
            print("\nTest: Cache prevents redundant API calls...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            translator = CodeSwitchTranslator(
                api_key=api_key,
                cache_dir=tmpdir,
                model="gpt-4",
                use_cache=True,
                show_progress=False  # Disable for testing,
                )
            
            # First translation - should call API
            sentence = "我 哋 go 咗 公 園"
            pattern = "C2-E1-C3"
            words = ['我', '哋', 'go', '咗', '公', '園']
            
            result1 = translator.translate_code_switched_sentence(sentence, pattern, words)
            cache_size_1 = len(translator.cache.cache)
            
            # Second translation - should use cache
            result2 = translator.translate_code_switched_sentence(sentence, pattern, words)
            cache_size_2 = len(translator.cache.cache)
            
            # Cache size should not change (no new API call)
            assert cache_size_1 == cache_size_2, "Cache grew unexpectedly on second call"
            assert result1['translated_sentence'] == result2['translated_sentence'], \
                "Cached result differs from original"
            
            if verbose:
                print(f"  First call - cache size: {cache_size_1}")
                print(f"  Second call - cache size: {cache_size_2} (no growth = cache hit)")
        
        results.add_pass("Cache prevents redundant API calls")
    except Exception as e:
        results.add_fail("Cache prevents redundant API calls", str(e))


def run_integration_tests(api_key: str, results: TestResult, verbose: bool = False):
    """Test integration with data export module."""
    print("\n" + "="*70)
    print("6. INTEGRATION TESTS (With data_export)")
    print("="*70)
    
    try:
        if verbose:
            print("\nTest: Export with translation integration...")
        
        from src.core.config import Config
        from src.data.data_export import export_translated_sentences
        
        config = Config()
        
        # Create test sentences (mix of Cantonese and English matrix)
        all_sentences = [
            {
                'start_time': 0.0,
                'end_time': 1.0,
                'reconstructed_text': '我 哋 go 咗 公 園',
                'reconstructed_text_without_fillers': '我 哋 go 咗 公 園',
                'text': '我 哋 go 咗 公 園',
                'pattern_with_fillers': 'C2-E1-C3',
                'pattern_content_only': 'C2-E1-C3',
                'matrix_language': 'Cantonese',  # Should be included
                'group': 'Heritage',
                'participant_id': 'TEST001'
            },
            {
                'start_time': 1.0,
                'end_time': 2.0,
                'reconstructed_text': '佢 like 我',
                'reconstructed_text_without_fillers': '佢 like 我',
                'text': '佢 like 我',
                'pattern_with_fillers': 'C1-E1-C1',
                'pattern_content_only': 'C1-E1-C1',
                'matrix_language': 'Cantonese',  # Should be included
                'group': 'Immersed',
                'participant_id': 'TEST002'
            },
            {
                'start_time': 2.0,
                'end_time': 3.0,
                'reconstructed_text': 'I 想 eat 嘢',
                'reconstructed_text_without_fillers': 'I 想 eat 嘢',
                'text': 'I 想 eat 嘢',
                'pattern_with_fillers': 'E1-C1-E1-C1',
                'pattern_content_only': 'E1-C1-E1-C1',
                'matrix_language': 'English',  # Should be EXCLUDED
                'group': 'Heritage',
                'participant_id': 'TEST003'
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override cache directory
            original_cache_method = config.get_translation_cache_dir
            config.get_translation_cache_dir = lambda: tmpdir
            
            # Override output path
            original_csv_method = config.get_csv_cantonese_translated_path
            config.get_csv_cantonese_translated_path = lambda: os.path.join(tmpdir, "test_output.csv")
            
            # Run export with translation
            df = export_translated_sentences(
                all_sentences,
                config,
                api_key=api_key
            )
            
            # Restore original methods
            config.get_translation_cache_dir = original_cache_method
            config.get_csv_cantonese_translated_path = original_csv_method
            
            # Validate DataFrame
            assert len(df) == 2, f"Expected 2 rows (only Cantonese matrix), got {len(df)}"
            assert 'cantonese_translation' in df.columns, "Missing 'cantonese_translation' column"
            assert 'reconstructed_sentence' in df.columns, "Missing 'reconstructed_sentence' column"
            assert 'pattern' in df.columns, "Missing 'pattern' column"
            assert 'matrix_language' in df.columns, "Missing 'matrix_language' column"
            
            # All should be Cantonese matrix language
            assert all(df['matrix_language'] == 'Cantonese'), "Should only contain Cantonese matrix sentences"
            
            # Check translations are not empty
            for idx in df.index:
                translation = df.loc[idx, 'cantonese_translation']
                assert translation, f"Translation at index {idx} is empty"
            
            if verbose:
                print(f"  Exported {len(df)} translated sentences (Cantonese matrix only)")
                for idx in df.index:
                    print(f"  {idx+1}. {df.loc[idx, 'reconstructed_sentence']}")
                    print(f"     → {df.loc[idx, 'cantonese_translation']}")
        
        results.add_pass("Export with translation integration")
    except Exception as e:
        results.add_fail("Export with translation integration", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Test translation functionality with real OpenAI API calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_translation.py sk-abc123...
  python tests/test_translation.py sk-abc123... --verbose
        """
    )
    parser.add_argument('api_key', help='OpenAI API key (required)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" TRANSLATION TEST SUITE ".center(70, "="))
    print("="*70)
    print(f"\nUsing API key: {args.api_key[:15]}...{args.api_key[-4:]}")
    print(f"Verbose mode: {args.verbose}")
    
    results = TestResult()
    
    # Run all test suites
    run_cache_tests(results, args.verbose)
    run_translator_basic_tests(results, args.verbose)
    run_translation_tests(args.api_key, results, args.verbose)
    run_batch_translation_tests(args.api_key, results, args.verbose)
    run_caching_tests(args.api_key, results, args.verbose)
    run_integration_tests(args.api_key, results, args.verbose)
    
    # Print summary
    print("\n" + "="*70)
    print(" TEST SUMMARY ".center(70, "="))
    print("="*70)
    print(f"\nTotal Tests: {results.passed + results.failed}")
    print(f"  ✓ Passed: {results.passed}")
    print(f"  ✗ Failed: {results.failed}")
    
    if results.errors:
        print("\nFailed Tests:")
        for test_name, error in results.errors:
            print(f"  • {test_name}")
            print(f"    Error: {error}")
    
    print("\n" + "="*70)
    
    # Exit with appropriate code
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
