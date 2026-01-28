# Preprocessing Methodology

This document outlines the methodology used to preprocess raw EAF (ELAN Annotation Format) files into structured datasets for code-switching analysis.

## Overview

The preprocessing pipeline transforms bilingual speech annotations from EAF files into structured CSV datasets with code-switching patterns, language labels, and metadata. The process involves tokenization, language identification, filler removal, pattern generation, and optional translation.

## Data Input

**Source Format:** ELAN Annotation Format (EAF) files containing bilingual speech annotations.

**Tier Structure:**
- Main participant tier: Contains sentence-level annotations (tier names: `ACH*`, `ACHE*`, `ACI*`)
- Participant groups: Homeland (`ACH*`), Heritage (`ACHE*`), Immersed (`ACI*`)

**Extracted Information:**
- Sentence text from main tier annotations
- Start and end timestamps
- Participant ID and group classification

## Processing Steps

### 1. Tokenization and Language Identification

**Script-Based Segmentation:**
- Sentences are segmented by script boundaries (CJK characters vs. ASCII)
- CJK characters → Cantonese (`C`)
- ASCII alphabetic characters → English (`E`)

**Language-Specific Tokenization:**
- **Cantonese:** PyCantonese word segmentation (`pycantonese.segment()`)
- **English:** Whitespace-based tokenization

**Text Cleaning:**
- Splitting on ellipses (`...`, `…`)
- Splitting on commas
- Splitting on internal dashes
- Removal of punctuation-only tokens
- Normalization of whitespace

**Output:** List of `(word, language)` tuples for each sentence.

### 2. Filler Detection and Removal

**Filler Identification:**
- Language-specific filler word lists (English: "um", "uh", "like", etc.; Cantonese: "啊", "呢", "咁", etc.)
- Words are checked against filler lists in both languages to handle cross-language fillers

**Filler Removal:**
- Fillers are identified and excluded from pattern generation
- Content words (non-fillers) are retained for analysis
- Filler metadata is preserved for potential later analysis

**Rationale:** Fillers are hesitation markers, not genuine code-switches, and are excluded to ensure patterns reflect actual linguistic code-switching behavior.

### 3. Pattern Generation

**Pattern Format:**
- Sequential representation of language segments (e.g., `C5-E3-C2`)
- Numbers indicate consecutive word counts in each language
- Generated from content words only (fillers excluded)

**Pattern Construction:**
1. Iterate through content word sequence
2. Count consecutive words in the same language
3. Create segments: `{language}{count}` (e.g., `C5`, `E3`)
4. Join segments with hyphens: `C5-E3-C2`

**Special Cases:**
- `FILLER_ONLY`: Sentence contains only fillers, no content words
- Monolingual patterns: `C10` (Cantonese only) or `E8` (English only)

### 4. Matrix Language Identification

**Method:** Matrix Language Framework (MLF) model

**Criteria:**
- Count words in each language (content words only, fillers excluded)
- Matrix language = language with higher word count
- If equal counts → `"Equal"`

**Output:** `'Cantonese'`, `'English'`, or `'Equal'`

### 5. Filtering Criteria

**Minimum Word Count Filter:**
- Sentences must contain at least `min_sentence_words` content words (default: 2)
- Word count calculated from pattern (sum of numbers in pattern string)

**Code-Switching Filter:**
- Pattern must contain both `'C'` and `'E'` characters
- Excludes `'FILLER_ONLY'` patterns

**Translation Dataset Filtering (for translated sentences):**
- Matrix language must be `'Cantonese'`
- Pattern must start with at least `min_cantonese_words` Cantonese words followed by English (e.g., `C5-E2` passes if `min_cantonese_words ≤ 5`)
- Ensures sufficient Cantonese context before code-switch point

### 6. Translation (Optional)

**Model:** Meta's NLLB (No Language Left Behind) model

**Process:**
- English segments translated to Cantonese
- Cantonese segments preserved as-is
- Full sentence reconstructed in Cantonese

**Verification:**
- Translation must be fully Cantonese (no English words in Cantonese portion)
- English words allowed only after the code-switch point (in original English segments)
- Invalid translations are excluded from final dataset

**POS Tagging:**
- Part-of-speech tags assigned to translated sentences
- Mixed-language handling: English words tagged as `'ENG'`, Cantonese words tagged with actual POS tags

## Output Files

### 1. `all_sentences.csv`
**Content:** All processed sentences (monolingual + code-switched)

**Columns:**
- `start_time`, `end_time`: Temporal boundaries
- `reconstructed_sentence`: Text without fillers
- `sentence_original`: Original annotation text
- `pattern`: Code-switching pattern (e.g., `C5-E3-C2`)
- `matrix_language`: Dominant language (`'Cantonese'`, `'English'`, or `'Equal'`)
- `group`: Speaker group (`'Homeland'`, `'Heritage'`, `'Immersed'`)
- `participant_id`: Participant identifier

### 2. `cantonese_monolingual_WITHOUT_fillers.csv`
**Content:** Cantonese monolingual sentences only (pattern contains `'C'` but not `'E'`)

**Additional Columns:**
- `pos`: Part-of-speech tag sequence

**Purpose:** Used for matching algorithm to find similar monolingual sentences for code-switched sentences.

### 3. `cantonese_translated_WITHOUT_fillers.csv`
**Content:** Code-switched sentences with Cantonese matrix language, translated to full Cantonese

**Additional Columns:**
- `code_switch_original`: Original code-switched sentence
- `cantonese_translation`: Full Cantonese translation
- `translated_pos`: POS tags for translated sentence
- `switch_index`: Word index where English segment begins (0-based)

**Purpose:** Primary dataset for surprisal analysis comparing code-switched vs. monolingual sentences.

### 4. `preprocessing_report.csv`
**Content:** Summary statistics of preprocessing pipeline

**Sections:**
- Preprocessing: Total sentences, filtering steps, sentence type breakdowns
- Monolingual Export: Cantonese monolingual filtering statistics
- Translation: Filtering stages, retention rates, translation success rates, POS coverage

## Key Design Decisions

1. **Filler Exclusion:** Fillers are excluded from pattern generation to ensure patterns reflect genuine code-switching, not hesitation markers.

2. **Content-Only Patterns:** Patterns are generated from content words only, ensuring consistency between pattern representation and actual linguistic content.

3. **Matrix Language from Content:** Matrix language is determined from content words only, excluding fillers, to accurately reflect the dominant language of meaningful content.

4. **Script-Based Language Identification:** Language is identified by character script (CJK vs. ASCII), which is robust for Cantonese-English code-switching.

5. **Translation Verification:** Translations are verified to ensure English words do not appear in the Cantonese portion, maintaining language boundaries for analysis.

## Reproducibility Notes

- **Configuration:** All parameters (minimum word counts, filler lists, etc.) are defined in `config/config.yaml`
- **Dependencies:** PyCantonese for Cantonese segmentation, NLLB for translation
- **Deterministic Processing:** Same input files produce same output (translation may vary slightly due to model stochasticity)
- **Progress Tracking:** Processing progress displayed via tqdm progress bars

