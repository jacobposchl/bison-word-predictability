"""
Test suite for NLLB translation functionality.

Tests the free, local NLLB translator without needing an API key.

Usage:
    python tests/test_nllb_translation.py
    python tests/test_nllb_translation.py --verbose
"""

import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def run_nllb_tests(results: TestResult, verbose: bool = False):
    """Test NLLB translation."""
    print("\n" + "="*70)
    print("NLLB TRANSLATION TESTS")
    print("="*70)
    
    # Test cases
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
            'name': 'Pure Cantonese (no translation needed)',
            'sentence': '我 哋 去 咗 公 園',
            'pattern': 'C6',
            'words': ['我', '哋', '去', '咗', '公', '園'],
            'expected_english_segments': 0
        }
    ]
    
    try:
        if verbose:
            print("\nInitializing NLLB translator...")
        
        translator = NLLBTranslator(
            model_name="facebook/nllb-200-distilled-600M",
            device="auto",
            show_progress=True
        )
        
        print("\n" + "="*70)
        print("Running translation tests...")
        print("="*70)
        
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
                assert 'translated_sentence' in result, "Missing 'translated_sentence'"
                assert 'original_sentence' in result, "Missing 'original_sentence'"
                assert 'pattern' in result, "Missing 'pattern'"
                assert 'segments' in result, "Missing 'segments'"
                
                # Validate content
                assert result['original_sentence'] == test_case['sentence'], "Original mismatch"
                assert result['pattern'] == test_case['pattern'], "Pattern mismatch"
                
                # Check segments
                english_segments = [s for s in result['segments'] if s['language'] == 'English']
                assert len(english_segments) == test_case['expected_english_segments'], \
                    f"Expected {test_case['expected_english_segments']} English segments, got {len(english_segments)}"
                
                # Validate translation is not empty (unless no English)
                if test_case['expected_english_segments'] > 0:
                    assert result['translated_sentence'], "Translation is empty"
                
                if verbose:
                    print(f"  Translated: {result['translated_sentence']}")
                    if english_segments:
                        for seg in english_segments:
                            print(f"    '{seg['original']}' → '{seg['translated']}'")
                
                results.add_pass(test_case['name'])
                
            except Exception as e:
                results.add_fail(test_case['name'], str(e))
        
    except Exception as e:
        print(f"\nFailed to initialize NLLB: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install transformers sentencepiece protobuf torch")
        results.add_fail("NLLB initialization", str(e))


def run_batch_test(results: TestResult, verbose: bool = False):
    """Test batch translation."""
    print("\n" + "="*70)
    print("BATCH TRANSLATION TEST")
    print("="*70)
    
    try:
        translator = NLLBTranslator(
            model_name="facebook/nllb-200-distilled-600M",
            device="auto",
            show_progress=False  # Disable for this test
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
            assert 'translated_sentence' in result, f"Result {i} missing field"
            assert result['original_sentence'] == sentences[i], f"Result {i} mismatch"
            
            if verbose:
                print(f"  {i+1}. {result['original_sentence']} → {result['translated_sentence']}")
        
        results.add_pass("Batch translation")
        
    except Exception as e:
        results.add_fail("Batch translation", str(e))


def run_integration_test(results: TestResult, verbose: bool = False):
    """Test integration with data_export module."""
    print("\n" + "="*70)
    print("INTEGRATION TEST")
    print("="*70)
    
    try:
        from src.core.config import Config
        from src.data.data_export import export_translated_sentences
        
        config = Config()
        
        # Temporarily override config to use NLLB
        original_backend = config.get
        def mock_get(key, default=None):
            if key == 'translation.backend':
                return 'nllb'
            return original_backend(key, default)
        config.get = mock_get
        
        # Create test sentences
        all_sentences = [
            {
                'start_time': 0.0,
                'end_time': 1.0,
                'reconstructed_text': '我 哋 go 咗 公 園',
                'reconstructed_text_without_fillers': '我 哋 go 咗 公 園',
                'text': '我 哋 go 咗 公 園',
                'pattern_with_fillers': 'C2-E1-C3',
                'pattern_content_only': 'C2-E1-C3',
                'matrix_language': 'Cantonese',
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
                'matrix_language': 'Cantonese',
                'group': 'Immersed',
                'participant_id': 'TEST002'
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_csv_method = config.get_csv_cantonese_translated_path
            config.get_csv_cantonese_translated_path = lambda: str(Path(tmpdir) / "test_output.csv")
            
            df = export_translated_sentences(all_sentences, config)
            
            config.get_csv_cantonese_translated_path = original_csv_method
            
            assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
            assert 'cantonese_translation' in df.columns, "Missing translation column"
            assert all(df['matrix_language'] == 'Cantonese'), "Should only have Cantonese matrix"
            
            for idx in df.index:
                translation = df.loc[idx, 'cantonese_translation']
                assert translation, f"Translation at index {idx} is empty"
            
            if verbose:
                print(f"\n  Exported {len(df)} translated sentences:")
                for idx in df.index:
                    print(f"    {df.loc[idx, 'reconstructed_sentence']}")
                    print(f"    → {df.loc[idx, 'cantonese_translation']}")
        
        results.add_pass("Integration with data_export")
        
    except Exception as e:
        results.add_fail("Integration with data_export", str(e))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test NLLB translation functionality"
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" NLLB TRANSLATION TEST SUITE ".center(70, "="))
    print("="*70)
    print("\nThis tests the free, local NLLB translation (no API key needed)")
    print("The model will be downloaded on first run (~2.4GB)")
    
    results = TestResult()
    
    # Run all test suites
    run_nllb_tests(results, args.verbose)
    run_batch_test(results, args.verbose)
    run_integration_test(results, args.verbose)
    
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
    
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
