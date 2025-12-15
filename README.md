# Code-Switching Predictability Analysis

A Python package for analyzing code-switching patterns in Cantonese-English bilingual speech data from ELAN Annotation Format (EAF) files.

## Overview

This package preprocesses raw EAF annotation files into processed data for code-switching analysis:
- Extracts and cleans bilingual speech annotations from EAF files
- Identifies code-switching patterns (transitions between Cantonese and English)
- Determines matrix language (dominant language) for each sentence
- Analyzes the impact of filler words on code-switching patterns
- Generates visualizations and exports processed data to CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

**Virtual Environment is recommended**

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The required packages are:
- `pympi-ling` - For processing EAF files
- `pycantonese` - For Cantonese word segmentation
- `pandas` - For data manipulation
- `numpy` - For numerical operations
- `matplotlib` - For visualization
- `pyyaml` - For configuration file parsing

## Configuration

Edit `config/config.yaml` to configure the analysis:

```yaml
data:
  path: "./data"  # Path to directory containing EAF files

processing:
  buffer_ms: 50  # Time buffer for sentence overlap detection -> implemented for small displacements in the EAF files transcriptions
  min_sentence_words: 2  # Minimum words to keep a sentence

output:
  processed_data_dir: "processed_data"  # Directory for saving CSV files
  csv_with_fillers: "code_switching_WITH_fillers.csv"
  csv_without_fillers: "code_switching_WITHOUT_fillers.csv"
  figures_dir: "figures"
```

### Data Format Requirements

Your EAF files should have the following tier structure:
- Main tier: Participant ID (e.g., `ACH2004`, `ACHE2001`, `ACI2003`)
  - `ACH*`: Homeland speakers
  - `ACHE*`: Heritage speakers
  - `ACI*`: Immersed speakers
- Cantonese tier: `{ParticipantID}-Cantonese-Spaced`
- English tier: `{ParticipantID}-English`

## Main Scripts

This package has two main scripts that should be run in sequence:

### 1. `src.preprocess` - Data Preprocessing

**What it does:**
- Loads raw EAF annotation files from `raw_data/` directory
- Extracts and cleans bilingual speech annotations
- Identifies code-switching patterns (e.g., `C5-E3-C2`)
- Determines matrix language (dominant language) for each sentence
- Analyzes impact of filler words
- Generates visualizations
- Exports processed data to CSV files in `processed_data/`

**When to run:** First step - processes raw EAF files into analyzable CSV data.

**Basic usage:**
```bash
python -m src.preprocess
```

**Command-line options:**
```bash
python -m src.preprocess [OPTIONS]
```

Options:
- `--config PATH`: Path to configuration YAML file (default: `config/config.yaml`)
- `--data-path PATH`: Override data path from config file
- `--output-dir PATH`: Override output directory from config file
- `--no-plots`: Skip generating visualization plots
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Use default settings
python -m src.preprocess

# Use custom config file
python -m src.preprocess --config my_config.yaml

# Override data path
python -m src.preprocess --data-path /path/to/eaf/files

# Skip visualizations (faster processing)
python -m src.preprocess --no-plots

# Verbose output for debugging
python -m src.preprocess --verbose
```

### 2. `scripts/exploratory_analysis.py` - Feasibility Analysis

**What it does:**
- Loads processed CSV data from `results/preprocessing/`
- Extracts monolingual sentences (pure Cantonese and pure English)
- Analyzes POS tagging quality and accuracy
- Tests matching algorithm (finds similar monolingual sentences for code-switched sentences)
- Analyzes code-switching distributions
- Generates comprehensive feasibility report
- Saves results to `results/exploratory/`

**When to run:** Second step - analyzes processed data to assess methodology feasibility.

**Basic usage:**
```bash
python scripts/exploratory_analysis.py
```

**Command-line options:**
```bash
python scripts/exploratory_analysis.py [OPTIONS]
```

Options:
- `--dataset {ALL,WITH,WITHOUT}`: Which dataset to analyze (default: `ALL`)
  - `ALL`: All sentences (monolingual + code-switched)
  - `WITH`: Code-switched sentences with fillers
  - `WITHOUT`: Code-switched sentences without fillers
- `--sample-size N`: Number of sentences to sample for testing (default: 100, use 0 for full dataset)
- `--full-dataset`: Process full dataset instead of sampling (overrides `--sample-size`)
- `--output-dir PATH`: Override output directory from config file (default: from config)
- `--config PATH`: Path to configuration YAML file (default: `config/config.yaml`)
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Quick analysis with sample (default)
python scripts/exploratory_analysis.py

# Analyze full dataset (may take longer)
python scripts/exploratory_analysis.py --full-dataset

# Analyze only code-switched sentences without fillers
python scripts/exploratory_analysis.py --dataset WITHOUT

# Custom sample size
python scripts/exploratory_analysis.py --sample-size 500

# Verbose output
python scripts/exploratory_analysis.py --verbose
```

## Workflow

The typical workflow is:

1. **Preprocess raw data:**
   ```bash
   python -m src.preprocess
   ```
   This creates CSV files in `processed_data/` directory.

2. **Run feasibility analysis:**
   ```bash
   python scripts/exploratory_analysis.py
   ```
   This analyzes the processed data and generates a feasibility report in `results/exploratory/`.

## Output Files

### Preprocessing Output (`results/preprocessing/`)

The preprocessing script generates:

#### CSV Files

Saved to the `results/preprocessing/` directory:

1. **`code_switching_WITH_fillers.csv`**: Code-switching sentences with filler words included in pattern analysis
2. **`code_switching_WITHOUT_fillers.csv`**: Code-switching sentences with filler words excluded
3. **`all_sentences.csv`**: All sentences (monolingual + code-switched) for exploratory analysis

Both code-switching CSV files contain:
- `reconstructed_sentence`: Cleaned sentence text
- `sentence_original`: Original sentence from EAF file
- `pattern`: Code-switching pattern (e.g., `E3-C5-E2`)
- `matrix_language`: Dominant language (Cantonese/English/Equal)
- `group_code`: Speaker group code (H/HE/I)
- `group`: Speaker group name (Homeland/Heritage/Immersed)
- `participant_id`: Participant identifier
- `filler_count`: Number of filler words detected
- `has_fillers`: Boolean indicating presence of fillers

#### Preprocessing Visualizations

Saved to the `figures/preprocessing/` directory:

1. **`matrix_language_distribution.png`**: Stacked bar charts showing matrix language distribution across speaker groups
2. **`equal_matrix_cases.png`**: Analysis of equal matrix language cases
3. **`equal_matrix_prevalence.png`**: Prevalence of equal matrix cases across groups
4. **`filler_impact.png`**: Impact of filler removal on matrix language percentages

### Exploratory Analysis Output (`results/exploratory/`)

The exploratory analysis script generates:

1. **`monolingual_sentences.csv`**: Extracted monolingual sentences (pure Cantonese and English)
2. **`pos_tagged_sample.csv`**: Sample of POS-tagged sentences with sequences
3. **`matching_results_sample.csv`**: Results of matching algorithm tests
4. **`feasibility_report.txt`**: Comprehensive feasibility assessment report

Figures (if any) are saved to `figures/exploratory/`.

## Project Structure

```
code-switch-predictability-uc-irvine/
├── scripts/                       # Executable scripts
│   ├── exploratory_analysis.py   # Feasibility analysis script
│   └── analyze_dash_splitting.py  # Dash splitting analysis script
├── src/                           # Source modules
│   ├── __init__.py
│   ├── preprocess.py              # Data preprocessing module
│   ├── calvillo_feasibility.py    # Calvillo methodology implementation
│   ├── matching_algorithm.py     # Matching algorithm for code-switching
│   ├── pos_tagging.py            # POS tagging utilities
│   ├── config.py                 # Configuration management
│   ├── eaf_processor.py          # EAF file processing
│   ├── text_cleaning.py          # Text cleaning utilities
│   ├── tokenization.py           # Tokenization functions
│   ├── pattern_analysis.py       # Pattern building and analysis
│   ├── data_export.py            # CSV export
│   └── visualization.py         # Plotting functions
├── tests/                        # Test and validation scripts
│   ├── test_cantonese_segmentation.py
│   ├── deep_validation.py
│   └── validate_analysis.py
├── config/
│   └── config.yaml               # Configuration file
├── raw_data/                     # INPUT: EAF annotation files
├── results/                      # OUTPUT: All analysis results
│   ├── preprocessing/            # CSV files from preprocessing
│   ├── exploratory/              # Results from exploratory analysis
│   └── dash_analysis/            # Results from dash splitting analysis
├── figures/                      # OUTPUT: All visualization plots
│   ├── preprocessing/            # Figures from preprocessing
│   └── exploratory/              # Figures from exploratory analysis
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How It Works

1. **File Loading**: Scans the data directory for EAF files and loads them using `pympi-ling`

2. **Tier Extraction**: Extracts annotations from the main tier, Cantonese tier, and English tier

3. **Text Cleaning**: 
   - Removes annotation markers (X, XX, XXX)
   - Filters punctuation-only annotations
   - Normalizes Unicode dashes
   - Identifies and handles filler words

4. **Tokenization**:
   - Cantonese: Uses `pycantonese` for word segmentation
   - English: Uses whitespace splitting
   - Assigns per-word timestamps

5. **Pattern Building**:
   - Groups consecutive words by language
   - Creates patterns like `E3-C5-E2` (3 English, 5 Cantonese, 2 English)
   - Builds patterns both with and without fillers

6. **Matrix Language Identification**:
   - Determines dominant language based on word count
   - Returns "Equal" if counts are equal

7. **Analysis & Export**:
   - Filters for actual code-switching sentences
   - Generates statistics and visualizations
   - Exports to CSV

## Citations

`pympi-ling` citation:

```
@misc{pympi-1.71,
	author={Lubbers, Mart and Torreira, Francisco},
	title={pympi-ling: a {Python} module for processing {ELAN}s {EAF} and {Praat}s {TextGrid} annotation files.},
	howpublished={\url{https://pypi.python.org/pypi/pympi-ling}},
	year={2013-2025},
	note={Version 1.71}
}
```

`pycantonese` citation:
```
@inproceedings{lee-etal-2022-pycantonese,
   title = "PyCantonese: Cantonese Linguistics and NLP in Python",
   author = "Lee, Jackson L.  and
      Chen, Litong  and
      Lam, Charles  and
      Lau, Chaak Ming  and
      Tsui, Tsz-Him",
   booktitle = "Proceedings of The 13th Language Resources and Evaluation Conference",
   month = june,
   year = "2022",
   publisher = "European Language Resources Association",
   language = "English",
}
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that your data path in `config.yaml` is correct
2. **No participant tier found**: Ensure your EAF files follow the naming convention (ACH*, ACHE*, ACI*)
3. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Getting Help

- Check that your EAF files have the required tier structure
- Enable verbose logging with `--verbose` flag for detailed error messages
- Verify that the data path in your config file is correct

## License

This project is for research purposes. Please ensure you have appropriate permissions to use the data files.
