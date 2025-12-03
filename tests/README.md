# Test Suite

This directory contains validation and testing scripts for the code-switching predictability analysis.

## Test Files

### `test_cantonese_segmentation.py`
Comprehensive test for Cantonese segmentation and Levenshtein similarity:
- Tests Cantonese text segmentation using pycantonese
- Verifies POS tagging works correctly with proper segmentation
- Tests Levenshtein similarity calculation
- Validates end-to-end pipeline

**Usage:**
```bash
python tests/test_cantonese_segmentation.py
```

### `deep_validation.py`
Deep validation of the matching algorithm and similarity calculations:
- Tests Levenshtein similarity calculation correctness
- Validates POS window extraction
- Tests Cantonese segmentation
- Checks matching logic and statistics consistency

**Usage:**
```bash
python tests/deep_validation.py
```

### `validate_analysis.py`
Comprehensive validation of exploratory analysis results:
- Validates pattern parsing
- Checks data consistency across files
- Validates matching results
- Verifies switch point detection
- Validates POS tagging results
- Checks for potential confounds

**Usage:**
```bash
python tests/validate_analysis.py
```

## Running All Tests

To run all validation scripts:
```bash
python tests/deep_validation.py
python tests/validate_analysis.py
python tests/test_cantonese_segmentation.py
```

## Requirements

All tests require:
- Processed data files in `../processed_data/`
- Exploratory results in `../exploratory_results/` (for some tests)
- All dependencies from `requirements.txt`

## Notes

- Tests use relative paths from the project root
- Some tests may skip if required data files are not found
- All tests print detailed results to console

