# Surprisal Analysis Pipeline Methodology

## Overview

The surprisal analysis pipeline measures word predictability at code-switch points by comparing surprisal values between code-switched sentences (translated to Cantonese) and matched monolingual Cantonese baseline sentences. This allows us to quantify whether words at switch points are more or less predictable than structurally similar words in purely monolingual contexts.

**Script Location:** [`scripts/surprisal/surprisal.py`](../scripts/surprisal/surprisal.py)

**Research Question:** Are words at code-switch boundaries less predictable (higher surprisal) than comparable words in monolingual sentences with similar syntactic structure?

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Pipeline Overview](#pipeline-overview)
3. [Model Types](#model-types)
4. [Configuration](#configuration)
5. [Detailed Methodology](#detailed-methodology)
6. [Output Files and Interpretation](#output-files-and-interpretation)
7. [Running the Pipeline](#running-the-pipeline)
8. [Statistical Analysis](#statistical-analysis)

---

## Theoretical Background

### Surprisal as a Measure of Predictability

**Surprisal** quantifies how unexpected a word is in context. It is defined as the negative log probability of a word given its context:

$$\text{Surprisal}(w_i) = -\log_2 P(w_i \mid \text{context})$$

**Key properties:**
- **Higher surprisal** = less predictable word (unexpected)
- **Lower surprisal** = more predictable word (expected)
- Measured in **bits** (log base 2)
- Range: [0, ∞), where 0 = completely certain (P=1.0)

**Example:**
```
Context: "我想去"
Word: "學校" (school)

If P(學校 | 我想去) = 0.25
Surprisal = -log₂(0.25) = 2.0 bits

If P(學校 | 我想去) = 0.5
Surprisal = -log₂(0.5) = 1.0 bits
```

### Code-Switching Predictability Hypothesis

**Research Hypothesis:** Words at code-switch boundaries may have:
1. **Higher surprisal** if code-switching disrupts prediction
2. **Lower surprisal** if code-switches occur at highly predictable positions

**Comparison Strategy:**
- **CS Translation:** Measure surprisal at the position corresponding to the first English word (switch point)
- **Monolingual Baseline:** Measure surprisal at matched position in structurally similar monolingual sentence
- **Difference:** CS surprisal - Mono surprisal indicates code-switching effect

---

## Pipeline Overview

```
Input:
  ├─ Analysis datasets from matching pipeline (one per window size)
  └─ Pre-trained language models (BERT or GPT-style)

Pipeline:
  1. Load matched sentence pairs
  2. Initialize language model
  3. For each sentence pair:
     ├─ Extract switch word position
     ├─ Segment sentences into words
     ├─ Calculate surprisal at switch position (with/without context)
     └─ Calculate surprisal at matched position
  4. Compute statistical comparisons
  5. Generate reports

Output:
  ├─ Surprisal values for each sentence pair
  ├─ Statistical summary reports
  └─ Long-format data for regression analysis
```

---

## Model Types

The pipeline supports two types of language models, each with different characteristics:

### 1. Masked Language Models (BERT-style)

**Model:** `hon9kon9ize/bert-large-cantonese`

**Architecture:**
- Bidirectional transformer (sees context on both sides)
- Trained with masked language modeling objective
- Uses `[MASK]` token for prediction

**Surprisal Calculation:**
```
For word at position i:
1. Replace word with [MASK] token
2. Get probability distribution over vocabulary
3. Extract probability of original word
4. Surprisal = -log₂(P(word))
```

**Context Usage:**
- **Left context:** All words before target
- **Right context:** All words after target (within model limits)
- **Bidirectional:** Uses future context for prediction

**Advantages:**
- Natural for cloze-style prediction
- Uses full sentence context
- Well-suited for comparing specific word positions

**Limitations:**
- Not fully autoregressive (different from human reading)
- May not reflect left-to-right processing

**Example:**
```
Sentence: "我 想 去 學校 讀書"
Target: "學校" (position 3)

Masked: "我 想 去 [MASK] 讀書"
Model predicts: P(學校 | "我 想 去 _ 讀書")
```

### 2. Autoregressive Language Models (GPT-style)

**Model:** `hon9kon9ize/CantoneseLLMChat-v1.0-7B`

**Architecture:**
- Unidirectional transformer (left-to-right)
- Trained to predict next token
- Causal attention (can't see future)

**Surprisal Calculation:**
```
For word at position i:
1. Feed all words before position i as context
2. Get probability distribution for next token
3. If word has multiple tokens, calculate surprisal sequentially:
   - Token 1: P(token₁ | context)
   - Token 2: P(token₂ | context + token₁)
   - Token 3: P(token₃ | context + token₁ + token₂)
4. Sum surprisal across all tokens
```

**Context Usage:**
- **Left context only:** Words before target
- **No right context:** Cannot see future words
- **Sequential:** Predicts one token at a time

**Advantages:**
- Models natural left-to-right reading process
- Autoregressive prediction matches human processing
- Reflects incremental comprehension

**Limitations:**
- Computationally expensive (sequential token prediction)
- Requires 4-bit quantization for memory efficiency
- No access to right context

**Example:**
```
Sentence: "我 想 去 學校 讀書"
Target: "學校" (position 3)

Context: "我 想 去"
Model predicts: P(學校 | "我 想 去")

If "學校" tokenizes to ["學", "校"]:
  P₁ = P(學 | "我 想 去")
  P₂ = P(校 | "我 想 去 學")
  Total surprisal = -log₂(P₁) + -log₂(P₂)
```

### Model Comparison

| Feature | Masked (BERT) | Autoregressive (GPT) |
|---------|---------------|----------------------|
| Context | Bidirectional | Left-only |
| Prediction | Cloze-style | Next-token |
| Human processing | Less natural | More natural |
| Computation | Fast (single pass) | Slow (sequential) |
| Memory | Moderate | High (7B params) |
| Best for | Controlled comparisons | Natural reading |

---

## Configuration

### Model Selection

```yaml
experiment:
  # Masked model (BERT-style)
  masked_model: "hon9kon9ize/bert-large-cantonese"
  
  # Autoregressive model (GPT-style)
  autoregressive_model: "hon9kon9ize/CantoneseLLMChat-v1.0-7B"
  
  # Device (GPU strongly recommended)
  device: "cuda"
```

### Context Settings

```yaml
context:
  # Number of previous sentences to include
  # Creates separate columns for each length
  context_lengths: [1, 2, 3]
  
  # Minimum translation quality for context
  min_translation_quality: 0.3
```

**Context Lengths:** The pipeline calculates surprisal for each context length independently:
- `context_lengths: [1, 2, 3]` produces:
  - `cs_surprisal_context_1`: With 1 previous sentence
  - `cs_surprisal_context_2`: With 2 previous sentences
  - `cs_surprisal_context_3`: With 3 previous sentences

**Why Multiple Lengths?** Allows analysis of how discourse context length affects predictability.

---

## Detailed Methodology

### Step 1: Load Matched Sentence Pairs

**Input:** Analysis datasets from matching pipeline

```python
# Load dataset for specific window size
df = pd.read_csv('results/matching/analysis_dataset_window_2.csv')

# Required columns:
# - cs_translation: Cantonese translation of CS sentence
# - matched_mono: Matched monolingual sentence
# - switch_index: Position of first English word (in translation)
# - matched_switch_index: Corresponding position in mono sentence
# - cs_context, mono_context: Previous sentences (optional)
```

### Step 2: Initialize Language Model

**Masked Model:**
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "hon9kon9ize/bert-large-cantonese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.to('cuda')
model.eval()
```

**Autoregressive Model:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "hon9kon9ize/CantoneseLLMChat-v1.0-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use 4-bit quantization to reduce memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

**Why Quantization?** 7B parameter model requires ~28GB in full precision, but only ~7GB with 4-bit quantization.

### Step 3: Word Segmentation

**CS Translation:** Already segmented (space-separated)
```python
cs_words = row['cs_translation'].split()
# Example: ['我', '係', '香港', '人', '但係', '我', '鍾意', '佢']
```

**Monolingual Sentence:** Use PyCantonese for proper segmentation
```python
import pycantonese
mono_words = pycantonese.segment(row['matched_mono'])
# Example: ['我', '係', '本地', '人', '不過', '我', '鍾意', '佢']
```

**Why Different Approaches?**
- CS translations are pre-segmented during translation
- Monolingual sentences need segmentation to find word boundaries

### Step 4: Token Alignment

Language models use subword tokenization (one word → multiple tokens). We must align words to their token sequences.

**Example:**
```
Word: "學校" (school)
Tokens: ["學", "校"] (2 tokens)

Must calculate: Surprisal(學) + Surprisal(校)
```

**Alignment Process:**
```python
def _align_word_to_tokens(sentence, words, word_index):
    target_word = words[word_index]
    
    # Calculate character positions
    char_start = sum(len(w) for w in words[:word_index])
    char_end = char_start + len(target_word)
    
    # Tokenize sentence
    encoding = tokenizer(sentence)
    token_ids = encoding['input_ids']
    
    # Find tokens that overlap with target word
    token_indices = []
    current_char_pos = 0
    
    for i, token_id in enumerate(token_ids):
        token_str = tokenizer.decode([token_id])
        token_len = len(token_str)
        token_start = current_char_pos
        token_end = current_char_pos + token_len
        
        # Check overlap
        if token_start < char_end and token_end > char_start:
            token_indices.append(i)
        
        current_char_pos += token_len
    
    return token_indices
```

### Step 5: Context Handling

**Context Format:** Previous sentences joined with delimiter `' ||| '`
```
Example with 2 previous sentences:
cs_context = "我 去 學校 ||| 今日 天氣 好"
```

**Context Integration:**
```python
if context:
    # Remove delimiter, join into single string
    context_clean = context.replace(' ||| ', ' ')
    context_words = context_clean.split()
    
    # Prepend to sentence
    full_words = context_words + sentence_words
    
    # Adjust target index
    adjusted_index = len(context_words) + original_index
```

**Maximum Length Handling:**

Models have token limits (BERT: 512, GPT: 2048). If context + sentence exceeds limit:

```python
# Priority: Keep all pre-switch words + target word
required_words = context_words + words[:switch_index + 1]
required_tokens = tokenizer.encode(required_words)

if len(required_tokens) > max_length:
    # Clip context from the left (oldest sentences removed first)
    # Keep as much recent context as fits
    
    # Calculate tokens needed for pre-switch + target
    essential_tokens = tokenizer.encode(words[:switch_index + 1])
    available_for_context = max_length - len(essential_tokens)
    
    # Add context words from right to left until full
    clipped_context = []
    current_tokens = 0
    for word in reversed(context_words):
        word_tokens = len(tokenizer.encode(word))
        if current_tokens + word_tokens <= available_for_context:
            clipped_context.insert(0, word)
            current_tokens += word_tokens
        else:
            break
```

**Logging:** Tracks how many sentences had context clipped for quality control.

### Step 6: Surprisal Calculation

#### Masked Model (BERT)

For each token in the target word:

```python
# 1. Create masked input
masked_input = input_ids.copy()
masked_input[token_idx] = tokenizer.mask_token_id

# Example: "我 想 去 [MASK] 讀書"

# 2. Get predictions
outputs = model(masked_input)
logits = outputs.logits[0, token_idx, :]  # Predictions for masked position

# 3. Calculate probability distribution
probs = softmax(logits)

# 4. Get probability of original token
token_prob = probs[original_token_id]

# 5. Calculate surprisal
token_surprisal = -log₂(token_prob)
```

**Multi-token words:** Sum surprisal across all tokens
```python
word_surprisal = sum(token_surprisals)
```

#### Autoregressive Model (GPT)

For each token in the target word sequentially:

```python
# Start with context only
current_input = context_tokens

for token in target_word_tokens:
    # 1. Get predictions for next token
    outputs = model(current_input)
    logits = outputs.logits[0, -1, :]  # Last position predicts next
    
    # 2. Calculate probability distribution
    probs = softmax(logits)
    
    # 3. Get probability of this token
    token_prob = probs[token]
    
    # 4. Calculate surprisal
    token_surprisal = -log₂(token_prob)
    
    # 5. Append token to input for next iteration
    current_input = concatenate(current_input, [token])
```

**Multi-token words:** Sum sequential surprisal
```python
word_surprisal = Σ -log₂(P(token_i | context + token₁...token_{i-1}))
```

**Key Difference:**
- **Masked:** Each token predicted independently (in parallel)
- **Autoregressive:** Each token predicted sequentially (conditioned on previous)

### Step 7: Calculate for Both Sentences

For each matched pair:

```python
# CS translation surprisal at switch point
cs_result = calculate_surprisal(
    word_index=switch_index,
    words=cs_words,
    context=cs_context  # k previous sentences
)

# Monolingual surprisal at matched position
mono_result = calculate_surprisal(
    word_index=matched_switch_index,
    words=mono_words,
    context=mono_context  # k previous sentences
)

# Store results
results.append({
    'cs_surprisal': cs_result['surprisal'],
    'mono_surprisal': mono_result['surprisal'],
    'surprisal_difference': cs_result['surprisal'] - mono_result['surprisal'],
    'cs_word': cs_result['word'],
    'mono_word': mono_result['word'],
    # ... additional metadata
})
```

### Step 8: Multiple Context Lengths

For each context length in `[1, 2, 3]`:

```python
for context_len in [1, 2, 3]:
    # Extract last N sentences from context
    if len(cs_context_sentences) >= context_len:
        cs_context = ' '.join(cs_context_sentences[-context_len:])
    
    if len(mono_context_sentences) >= context_len:
        mono_context = ' '.join(mono_context_sentences[-context_len:])
    
    # Calculate surprisal with this context
    cs_result = calculate_surprisal(..., context=cs_context)
    mono_result = calculate_surprisal(..., context=mono_context)
    
    # Store with context length suffix
    results[f'cs_surprisal_context_{context_len}'] = cs_result['surprisal']
    results[f'mono_surprisal_context_{context_len}'] = mono_result['surprisal']
```

**Result:** Each sentence has surprisal values for all requested context lengths.

### Step 9: Word-Level Metadata

Additional information extracted for each word:

**Word Length:**
```python
word_length = len(word)  # Number of characters
```

**Word Frequency:** Model-based frequency estimation
```python
# Use neutral contexts to estimate word frequency
neutral_contexts = ['係', '的', '我', '你', '在', '有', '是', '了']

log_probs = []
for context in neutral_contexts:
    # Calculate: P(word | context)
    prob = model.predict(context + " " + word)
    log_probs.append(log(prob))

word_frequency = mean(log_probs)
```

**Why Model-Based?** Corpus frequencies may not reflect model's internal representations.

---

## Output Files and Interpretation

### Surprisal Results (Long Format)

**File:** `results/surprisal/{model_type}/window_{N}/{mode}/surprisal_results.csv`

**Structure:** Two rows per sentence pair (one for CS, one for mono)

**Key Columns:**

| Column | Description |
|--------|-------------|
| `sent_id` | Unique sentence pair identifier |
| `is_switch` | 1 = CS translation, 0 = monolingual |
| `surprisal_context_1` | Surprisal with 1 previous sentence |
| `surprisal_context_2` | Surprisal with 2 previous sentences |
| `surprisal_context_3` | Surprisal with 3 previous sentences |
| `entropy_context_N` | Entropy of probability distribution |
| `word` | The word being measured |
| `word_length` | Number of characters |
| `word_frequency` | Model-based frequency estimate |
| `sent_length` | Total sentence length |
| `switch_index` | Position of measured word |
| `normalized_switch_point` | Position / sentence_length |
| `sentence` | Full sentence text |
| `context` | Discourse context used |

**Example Rows:**
```csv
sent_id,is_switch,surprisal_context_2,word,word_length,sentence
1,1,8.34,但係,2,我 係 香港 人 但係 我 鍾意 佢
1,0,6.12,不過,2,我 係 本地 人 不過 我 鍾意 佢
```

**Interpretation:**
- Row 1: CS word "但係" has surprisal 8.34 bits
- Row 2: Mono word "不過" has surprisal 6.12 bits
- Difference: 8.34 - 6.12 = 2.22 bits higher for CS

### Statistics Summary Report

**File:** `results/surprisal/{model_type}/window_{N}/{mode}/statistics_summary.txt`

**Contents:**

```
================================================================================
SURPRISAL COMPARISON STATISTICS
================================================================================

Sample Size:
  Total comparisons: 423
  Valid calculations: 385
  Success rate: 91.0%

Context Usage:
  Context length: 2 sentences
  Calculations with context: 385
  Calculations without context: 38

Code-Switched Translation Surprisal:
  Mean:   7.834
  Median: 7.521
  Std:    2.145

Monolingual Baseline Surprisal:
  Mean:   6.912
  Median: 6.734
  Std:    1.987

Difference (CS - Monolingual):
  Mean:   0.922
  Median: 0.845
  Std:    1.234

Paired t-test:
  t-statistic: 14.657
  p-value:     0.000001
  Significance: ***

Effect Size:
  Cohen's d: 0.747
```

**Interpretation:**
- **Mean difference > 0:** CS words are less predictable (higher surprisal)
- **p-value < 0.001:** Highly significant difference
- **Cohen's d = 0.747:** Medium-to-large effect size
- **Conclusion:** Code-switching significantly reduces word predictability

---

## Running the Pipeline

### Prerequisites

1. **Complete matching pipeline first:**
   ```bash
   python scripts/matching/matching.py
   ```

2. **GPU required** (CPU extremely slow for 7B model):
   - CUDA-capable GPU with ≥16GB VRAM (24GB recommended for autoregressive)
   - cuDNN and PyTorch with CUDA support

3. **Install dependencies:**
   ```bash
   pip install transformers torch accelerate bitsandbytes
   ```

### Basic Usage

**Masked model (BERT-style):**
```bash
python scripts/surprisal/surprisal.py --model masked
```

**Autoregressive model (GPT-style):**
```bash
python scripts/surprisal/surprisal.py --model autoregressive
```

### Advanced Options

**Without discourse context:**
```bash
python scripts/surprisal/surprisal.py --model masked --no-context
```

**Compare with/without context:**
```bash
python scripts/surprisal/surprisal.py --model masked --compare-context
```
This creates two output directories:
- `with_context/`: Results using discourse context
- `without_context/`: Results without discourse context

### Expected Runtime

**Masked model (BERT):**
- ~5-10 minutes per window size (450 sentences)
- Faster due to parallel token prediction

**Autoregressive model (GPT):**
- ~30-60 minutes per window size (450 sentences)
- Slower due to sequential token prediction
- Memory: ~12GB VRAM with 4-bit quantization

### Output Directory Structure

```
results/surprisal/
├── masked/                          # BERT-style results
│   ├── window_1/
│   │   ├── surprisal_results.csv
│   │   └── statistics_summary.txt
│   ├── window_2/
│   │   ├── surprisal_results.csv
│   │   └── statistics_summary.txt
│   └── window_3/
│       ├── surprisal_results.csv
│       └── statistics_summary.txt
│
└── autoregressive/                  # GPT-style results
    ├── window_1/
    │   ├── surprisal_results.csv
    │   └── statistics_summary.txt
    ├── window_2/
    │   ├── surprisal_results.csv
    │   └── statistics_summary.txt
    └── window_3/
        ├── surprisal_results.csv
        └── statistics_summary.txt
```

---

## Statistical Analysis

### Paired t-test

**Why paired?** Each CS sentence is matched with a structurally similar monolingual sentence, creating natural pairs.

**Null hypothesis (H₀):** Mean difference = 0 (no effect of code-switching)

**Alternative hypothesis (H₁):** Mean difference ≠ 0 (code-switching affects predictability)

**Test:**
```python
from scipy import stats

# Extract paired surprisal values
cs_surprisals = df[df['is_switch'] == 1]['surprisal_context_2']
mono_surprisals = df[df['is_switch'] == 0]['surprisal_context_2']

# Paired t-test
t_stat, p_value = stats.ttest_rel(cs_surprisals, mono_surprisals)
```

**Interpretation:**
- **p < 0.05:** Significant difference
- **p < 0.01:** Highly significant
- **p < 0.001:** Very highly significant

### Effect Size (Cohen's d)

**Formula:**
```
d = (Mean_CS - Mean_Mono) / SD_difference
```

**Interpretation:**
- **d < 0.2:** Small effect
- **0.2 ≤ d < 0.5:** Small-to-medium effect
- **0.5 ≤ d < 0.8:** Medium-to-large effect
- **d ≥ 0.8:** Large effect

**Why important?** Statistical significance doesn't indicate practical importance. Effect size shows magnitude.

### Context Length Comparison

Compare results across different context lengths:

```python
for ctx_len in [1, 2, 3]:
    cs = df[f'cs_surprisal_context_{ctx_len}']
    mono = df[f'mono_surprisal_context_{ctx_len}']
    
    diff = (cs - mono).mean()
    print(f"Context {ctx_len}: Mean difference = {diff:.3f}")
```

**Expected Pattern:**
- More context → better predictions → lower surprisal
- Effect may be larger or smaller depending on context length

### Regression Analysis

For more sophisticated analysis, use long-format data:

```python
import statsmodels.formula.api as smf

# Load long-format data
df = pd.read_csv('results/surprisal/masked/window_2/surprisal_results.csv')

# Mixed-effects regression
model = smf.mixedlm(
    "surprisal_context_2 ~ is_switch + word_length + normalized_switch_point",
    data=df,
    groups=df["sent_id"]
)
result = model.fit()
print(result.summary())
```

**Controls for:**
- Word length (longer words may have different surprisal)
- Position in sentence (early vs late words)
- Sentence-level random effects

---

## Key Methodological Considerations

### 1. Word Tokenization

**Challenge:** Different tokenizers segment differently

**Solution:**
- Use PyCantonese for consistent Cantonese segmentation
- CS translations pre-segmented during translation (consistent)
- Ensure alignment between word-level and token-level

### 2. Multi-token Words

**Issue:** Many Cantonese words split into multiple tokens

**Handling:**
- Calculate surprisal for each token separately
- Sum across all tokens in the word
- This matches how models process text incrementally

### 3. Context Length Limits

**Problem:** Models have maximum sequence lengths

**Solution:**
- Prioritize keeping all pre-switch words + target
- Clip context from the left (remove oldest sentences)
- Log when clipping occurs for transparency

### 4. Entropy vs Surprisal

**Entropy:** Uncertainty in the probability distribution
```
H = -Σ P(w) × log₂ P(w)
```

**Surprisal:** Unexpectedness of specific word
```
S = -log₂ P(w)
```

**Relationship:**
- High entropy distribution → predictions uncertain
- High surprisal word → specific word unexpected
- Both metrics provide complementary information

### 5. Matching Quality

**Quality Control:**
- Only analyze sentence pairs with valid matches (from matching pipeline)
- Similarity threshold ensures syntactic comparability
- Context quality flags identify potentially problematic cases

---

## Summary

The surprisal analysis pipeline:

1. **Quantifies word predictability** at code-switch points using state-of-the-art language models
2. **Controls for syntactic structure** by comparing to matched monolingual sentences
3. **Incorporates discourse context** with configurable context lengths
4. **Supports two model types** (masked and autoregressive) for methodological robustness
5. **Provides statistical validation** through paired tests and effect sizes
6. **Produces publication-ready output** with comprehensive reporting

**Research Output:** Determines whether code-switching affects word predictability, controlling for syntactic structure and discourse context—a key question in bilingual language processing.
