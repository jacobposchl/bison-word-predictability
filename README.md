# Code-Switching Predictability Analysis

## Overview

This package analyzes code-switching predictability by comparing surprisal values at code-switch points between code-switched and monolingual sentences. The analysis is performed in three stages:

**STAGE 1: Data Preprocessing**
- Extracts and cleans bilingual speech annotations from EAF files
- Identifies code-switching patterns and matrix language for each sentence
- Translates code-switched sentences to full Cantonese using NLLB
- Exports processed datasets to CSV files

**STAGE 2: Window Matching Analysis**
- Matches translated code-switched sentences with similar monolingual Cantonese sentences
- Uses POS window matching to find the best matches
- Creates the final analysis dataset with matched sentence pairs

**STAGE 3: Surprisal Comparison Experiment**
- Calculates surprisal values at code-switch points for both code-switched and monolingual sentences
- Supports discourse context using previous sentences from the same speaker
- Compares surprisal values and generates statistical summaries and visualizations


## Configuration

Edit `config/config.yaml` to configure the analysis:

```yaml
data:
  path: "./data"  # Path to directory containing EAF files

processing:
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

This package has three main scripts that should be run in sequence:

### 1. `scripts/preprocess/preprocess.py` - Data Preprocessing

**What it does:**
- Loads raw EAF annotation files from `raw_data/` directory
- Extracts and cleans bilingual speech annotations
- Identifies code-switching patterns (e.g., `C5-E3-C2`)
- Determines matrix language (dominant language) for each sentence
- Analyzes impact of filler words
- Translates code-switched sentences to full Cantonese using NLLB
- Generates visualizations
- Exports processed data to CSV files in `results/preprocessing/`

**When to run:** First step - processes raw EAF files into analyzable CSV data.

**Basic usage:**
```bash
python scripts/preprocess/preprocess.py
```

**Command-line options:**
```bash
python scripts/preprocess/preprocess.py [OPTIONS]
```

Options:
- `--no-plots`: Skip generating visualization plots
- `--no-translation`: Skip translation process (faster, but needed for later steps)
- `--verbose`: Enable verbose logging

**Examples:**
```bash
# Use default settings
python scripts/preprocess/preprocess.py

# Skip visualizations (faster processing)
python scripts/preprocess/preprocess.py --no-plots

# Skip translation (if already done)
python scripts/preprocess/preprocess.py --no-translation

# Verbose output for debugging
python scripts/preprocess/preprocess.py --verbose
```

### 2. `scripts/exploratory/exploratory_analysis.py` - Window Matching Analysis

**What it does:**
- Loads translated code-switched sentences and monolingual Cantonese sentences from preprocessing
- Performs POS window matching to find similar monolingual sentences for each code-switched sentence
- Analyzes similarity distributions between matched sentences
- Creates the final analysis dataset with matched sentence pairs
- Generates similarity distribution plots and matching reports
- Saves results to `results/exploratory/`

**When to run:** Second step - creates matched sentence pairs for surprisal analysis.

**Basic usage:**
```bash
python scripts/exploratory/exploratory_analysis.py
```

**Command-line options:**
```bash
python scripts/exploratory/exploratory_analysis.py [OPTIONS]
```

Options:
- `--sample-size N`: Number of sentences to process (default: all sentences)

**Examples:**
```bash
# Process all sentences (default)
python scripts/exploratory/exploratory_analysis.py

# Process a sample for testing
python scripts/exploratory/exploratory_analysis.py --sample-size 100
```

### 3. `scripts/main_experiment/run_surprisal_experiment.py` - Surprisal Comparison Experiment

**What it does:**
- Loads the analysis dataset with matched sentence pairs
- Calculates surprisal values at code-switch points for both:
  - Code-switched sentences (translated to full Cantonese)
  - Matched monolingual Cantonese baseline sentences
- Supports discourse context using previous sentences from the same speaker
- Compares surprisal values between code-switched and monolingual sentences
- Generates statistical summaries and visualizations
- Saves results to `results/main_experiment_{model_type}/`

**When to run:** Third step - performs the main surprisal analysis experiment.

**Basic usage:**
```bash
python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive
```

**Command-line options:**
```bash
python scripts/main_experiment/run_surprisal_experiment.py [OPTIONS]
```

Required arguments:
- `--model {masked,autoregressive}`: Type of model to use
  - `masked`: BERT-style masked language model
  - `autoregressive`: GPT-style autoregressive language model

Optional arguments:
- `--sample-size N`: Number of sentences to process (default: all)
- `--no-context`: Disable discourse context (use only current sentence)
- `--compare-context`: Run both with and without context for comparison

**Examples:**
```bash
# Run with autoregressive model and context (default)
python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive

# Run with masked model
python scripts/main_experiment/run_surprisal_experiment.py --model masked

# Run without discourse context
python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive --no-context

# Compare both context modes
python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive --compare-context

# Process a sample for testing
python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive --sample-size 50
```

## Workflow

The typical workflow is:

1. **Preprocess raw data:**
   ```bash
   python scripts/preprocess/preprocess.py
   ```
   This processes EAF files and creates CSV files in `results/preprocessing/` directory, including translated sentences.

2. **Run window matching analysis:**
   ```bash
   python scripts/exploratory/exploratory_analysis.py
   ```
   This matches code-switched sentences with similar monolingual sentences and creates the analysis dataset in `results/exploratory/`.

3. **Run surprisal experiment:**
   ```bash
   python scripts/main_experiment/run_surprisal_experiment.py --model autoregressive
   ```
   This calculates and compares surprisal values at code-switch points, generating results in `results/main_experiment_autoregressive/` (or `results/main_experiment_masked/` for masked models).

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

1. **`analysis_dataset.csv`**: Final analysis dataset with code-switched sentences, their translations, matched monolingual sentences, and matching statistics
2. **`window_matching_report.txt`**: Report on window matching performance and similarity distributions

Figures are saved to `figures/exploratory/`:
- **`window_matching_similarity_distributions.png`**: Distribution of similarity scores between matched sentences

### Main Experiment Output (`results/main_experiment_{model_type}/`)

The main experiment script generates:

1. **`surprisal_results.csv`**: Detailed surprisal values for each sentence pair at code-switch points
2. **`statistics_summary.txt`**: Statistical summary including means, medians, t-tests, and effect sizes

Figures are saved to `figures/main_experiment_{model_type}/`:
- **`surprisal_distributions.png`**: Distribution of surprisal values for code-switched vs. monolingual sentences
- **`surprisal_scatter.png`**: Scatter plot comparing surprisal values
- **`surprisal_differences.png`**: Histogram of differences between code-switched and monolingual surprisal
- **`surprisal_summary.png`**: Summary statistics visualization

## Project Structure

```
code-switch-predictability-uc-irvine/
├── scripts/                       # Executable scripts
│   ├── preprocess/               # Preprocessing script
│   │   └── preprocess.py         # Data preprocessing script
│   ├── exploratory/              # Exploratory analysis script
│   │   └── exploratory_analysis.py  # Window matching analysis script
│   └── main_experiment/          # Main experiment script
│       └── run_surprisal_experiment.py  # Surprisal comparison experiment
├── src/                           # Source modules
│   ├── __init__.py
│   ├── core/                     # Core functionality (config, etc.)
│   ├── data/                     # Data processing modules
│   ├── analysis/                 # Analysis modules
│   ├── experiments/              # Experiment modules
│   └── plots/                    # Plotting modules
├── tests/                        # Test and validation scripts
│   ├── cantonese_segmentation_test.py
│   ├── similarity_matching_test.py
│   ├── test_monolingual_datasets.py
│   └── test_nllb_translation.py
├── config/
│   └── config.yaml               # Configuration file
├── raw_data/                     # INPUT: EAF annotation files
├── results/                      # OUTPUT: All analysis results
│   ├── preprocessing/            # CSV files from preprocessing
│   ├── exploratory/              # Results from exploratory analysis
│   ├── main_experiment_autoregressive/  # Results from main experiment (autoregressive)
│   └── main_experiment_masked/   # Results from main experiment (masked)
├── figures/                      # OUTPUT: All visualization plots
│   ├── preprocessing/            # Figures from preprocessing
│   ├── exploratory/              # Figures from exploratory analysis
│   ├── main_experiment_autoregressive/  # Figures from main experiment (autoregressive)
│   └── main_experiment_masked/   # Figures from main experiment (masked)
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
