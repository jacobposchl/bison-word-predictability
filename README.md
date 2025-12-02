# Code-Switching Predictability Analysis

A Python package for analyzing code-switching patterns in Cantonese-English bilingual speech data from ELAN Annotation Format (EAF) files.

## Overview

This package preprocesses raw EAF annotation files into processed data for code-switching analysis:
- Extracts and cleans bilingual speech annotations from EAF files
- Identifies code-switching patterns (transitions between Cantonese and English)
- Determines matrix language (dominant language) for each sentence
- Analyzes the impact of filler words on code-switching patterns
- Generates visualizations and exports processed data to CSV

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for processing, analysis, and visualization
- **Configuration Management**: YAML-based configuration for easy customization
- **Filler Detection**: Analyzes code-switching patterns both with and without filler words
- **Comprehensive Analysis**: Matrix language identification, pattern building, and statistical summaries
- **Visualization**: Multiple plots showing distribution and impact analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

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
  buffer_ms: 50  # Time buffer for sentence overlap detection
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

## Usage

### Basic Usage

Run the analysis with default configuration using either method:

**Option 1: As a module (recommended)**
```bash
python -m src.preprocess
```

### Command-Line Options

```bash
python -m src.preprocess [OPTIONS]
```

Options:
- `--config PATH`: Path to configuration YAML file (default: `config/config.yaml`)
- `--data-path PATH`: Override data path from config file
- `--output-dir PATH`: Override output directory from config file
- `--no-plots`: Skip generating visualization plots
- `--verbose`: Enable verbose logging

### Examples

```bash
# Use custom config file
python -m src.preprocess --config my_config.yaml

# Override data path
python -m src.preprocess --data-path /path/to/eaf/files

# Skip visualizations
python -m src.preprocess --no-plots

# Verbose output
python -m src.preprocess --verbose
```

## Output

The analysis generates:

### CSV Files

Saved to the `processed_data/` directory (or custom output directory):

1. **`code_switching_WITH_fillers.csv`**: Code-switching sentences with filler words included in pattern analysis
2. **`code_switching_WITHOUT_fillers.csv`**: Code-switching sentences with filler words excluded

Both CSV files contain:
- `reconstructed_sentence`: Cleaned sentence text
- `sentence_original`: Original sentence from EAF file
- `pattern`: Code-switching pattern (e.g., `E3-C5-E2`)
- `matrix_language`: Dominant language (Cantonese/English/Equal)
- `group_code`: Speaker group code (H/HE/I)
- `group`: Speaker group name (Homeland/Heritage/Immersed)
- `participant_id`: Participant identifier
- `filler_count`: Number of filler words detected
- `has_fillers`: Boolean indicating presence of fillers

### Visualizations

Saved to the `figures/` directory (or custom output directory):

1. **`matrix_language_distribution.png`**: Stacked bar charts showing matrix language distribution across speaker groups
2. **`equal_matrix_cases.png`**: Analysis of equal matrix language cases
3. **`equal_matrix_prevalence.png`**: Prevalence of equal matrix cases across groups
4. **`filler_impact.png`**: Impact of filler removal on matrix language percentages

### Console Output

The script prints detailed analysis summaries including:
- Dataset statistics
- Group distributions
- Matrix language distributions
- Impact of filler removal

## Project Structure

```
code-switch-predictability-uc-irvine/
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Data preprocessing script
│   ├── config.py                  # Configuration management
│   ├── eaf_processor.py           # EAF file processing
│   ├── text_cleaning.py           # Text cleaning utilities
│   ├── tokenization.py            # Tokenization functions
│   ├── pattern_analysis.py        # Pattern building and analysis
│   ├── data_export.py              # CSV export
│   └── visualization.py           # Plotting functions
├── config/
│   └── config.yaml                # Configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── notebook/                      # Original Jupyter notebook
    └── word_predictability_data_analysis (1).ipynb
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

## Citation

If you use `pympi-ling` in your research, please cite:

```
@misc{pympi-1.71,
	author={Lubbers, Mart and Torreira, Francisco},
	title={pympi-ling: a {Python} module for processing {ELAN}s {EAF} and {Praat}s {TextGrid} annotation files.},
	howpublished={\url{https://pypi.python.org/pypi/pympi-ling}},
	year={2013-2025},
	note={Version 1.71}
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

