# NLLB Translation Setup Guide

## Overview

Your translation system now supports **NLLB (No Language Left Behind)** - a free, open-source translation model from Meta that runs locally without any API costs!

## Quick Start

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` - Hugging Face library for NLLB
- `sentencepiece` - Tokenization for NLLB
- `protobuf` - Protocol buffers for model format
- `torch` - PyTorch (already installed)

### 2. Configuration

Your `config.yaml` is already set to use NLLB:

```yaml
translation:
  backend: "nllb"  # Use free NLLB (or "openai" for GPT)
  
  nllb:
    model: "facebook/nllb-200-distilled-600M"  # 2.4GB, good balance
    device: "auto"  # Auto-detect GPU/CPU
```

### 3. Run Tests

```bash
# Test NLLB translation (no API key needed!)
python tests/test_nllb_translation.py

# With verbose output
python tests/test_nllb_translation.py --verbose
```

**Note:** First run will download the model (~2.4GB). This is a one-time download.

### 4. Run Preprocessing with Translation

```bash
# No API key needed with NLLB!
python -m scripts.preprocess.preprocess
```

## Model Options

NLLB comes in different sizes. Choose based on your needs:

| Model | Size | Speed | Quality | Config Value |
|-------|------|-------|---------|--------------|
| **600M (default)** | 2.4GB | Fast | Good | `facebook/nllb-200-distilled-600M` |
| 1.3B | 5GB | Medium | Better | `facebook/nllb-200-1.3B` |
| 3.3B | 13GB | Slow | Best | `facebook/nllb-200-3.3B` |

To change model, edit `config/config.yaml`:

```yaml
nllb:
  model: "facebook/nllb-200-1.3B"  # For better quality
```

## GPU vs CPU

NLLB can use GPU for faster translation:

- **Auto-detect** (recommended): `device: "auto"`
- **Force GPU**: `device: "cuda"` (requires CUDA-capable GPU)
- **Force CPU**: `device: "cpu"` (slower but works everywhere)

## Switching Back to OpenAI

If you prefer GPT-4 or GPT-3.5, edit `config.yaml`:

```yaml
translation:
  backend: "openai"  # Switch to OpenAI
  
  openai:
    model: "gpt-3.5-turbo"  # Cheaper than gpt-4
```

Then provide API key:

```bash
python -m scripts.preprocess.preprocess --api-key YOUR_KEY
```

## Performance Comparison

### NLLB (600M model)
- ✅ **Cost:** FREE
- ✅ **Privacy:** Runs locally, data never leaves your machine
- ✅ **Speed:** ~2-5 sentences/second (CPU), ~10-20 sentences/second (GPU)
- ✅ **Offline:** Works without internet
- ⚠️ **Quality:** Good for Cantonese, but may be less natural than GPT-4

### GPT-4
- ⚠️ **Cost:** ~$0.03 per 1K tokens (~$3-5 per 1000 sentences)
- ⚠️ **Privacy:** Data sent to OpenAI
- ✅ **Speed:** ~1-2 sentences/second (API rate limited)
- ❌ **Offline:** Requires internet
- ✅ **Quality:** Very natural, colloquial translations

### GPT-3.5-Turbo
- ✅ **Cost:** ~$0.0015 per 1K tokens (~20x cheaper than GPT-4)
- ⚠️ **Privacy:** Data sent to OpenAI
- ✅ **Speed:** ~2-3 sentences/second
- ❌ **Offline:** Requires internet
- ✅ **Quality:** Good, between NLLB and GPT-4

## Troubleshooting

### Model Download Fails
```bash
# Set HuggingFace cache directory if needed
set HF_HOME=C:\path\to\cache
python tests/test_nllb_translation.py
```

### Out of Memory
If you get OOM errors, try:
1. Use smaller model: `facebook/nllb-200-distilled-600M`
2. Force CPU: `device: "cpu"`
3. Close other applications

### Slow Translation
- Check if GPU is being used (should see "Device: cuda" in logs)
- Consider upgrading to GPU if available
- Use smaller model for faster speed

### Poor Translation Quality
- Try larger model: `facebook/nllb-200-1.3B` or `3.3B`
- Or switch to `gpt-3.5-turbo` for better quality at low cost
- Check that input text is properly segmented

## Files Changed

1. ✅ `requirements.txt` - Added `sentencepiece` and `protobuf`
2. ✅ `config/config.yaml` - Updated translation settings
3. ✅ `src/experiments/nllb_translator.py` - New NLLB translator
4. ✅ `src/core/config.py` - Added NLLB config methods
5. ✅ `src/data/data_export.py` - Support both backends
6. ✅ `tests/test_nllb_translation.py` - New test suite

## What Gets Translated

The system translates code-switched sentences where:
- ✅ Cantonese is the matrix (dominant) language
- ✅ Contains embedded English segments
- ✅ Fillers are removed
- ✅ Results in full Cantonese sentence

Output: `results/preprocessing/cantonese_translated_WITHOUT_fillers.csv`

## Example Output

**Original:** `我 哋 go 咗 公 園`  
**Pattern:** `C2-E1-C3`  
**Translation (NLLB):** `我哋去咗公園`

The English word "go" is translated to Cantonese "去".
