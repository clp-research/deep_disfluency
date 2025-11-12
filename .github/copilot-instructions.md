# AI Agent Instructions for deep_disfluency

This repository contains code for deep learning-driven incremental disfluency detection and dialogue processing tasks.

## Project Overview

The deep_disfluency project implements incremental disfluency detection using deep learning, with support for both transcribed text and live ASR input. The system processes words (with optional POS tags and timings) incrementally to output XML-style disfluency tags.

### Key Components

- `deep_disfluency/tagger/`: Core disfluency tagging implementation
- `deep_disfluency/corpus/`: Corpus creation and management tools
- `deep_disfluency/evaluation/`: Evaluation metrics and utilities 
- `deep_disfluency/feature_extraction/`: Feature extraction from raw data
- `deep_disfluency/decoder/`: HMM and noisy channel decoders
- `deep_disfluency/embeddings/`: Word embedding models
- `deep_disfluency/asr/`: ASR integration (IBM Watson)

## Important Patterns and Conventions

### Disfluency Tag Format
The system uses XML-style tags to mark disfluencies:
```
<e/> - Edit term
<rms id="N"/> - Reparandum start 
<rm id="N"/> - Mid-reparandum
<i id="N"/> - Interregnum
<rps id="N"/> - Repair onset
<rp id="N"/> - Mid-repair
<rpn id="N"/> - Repair end
<rpndel id="N"/> - Delete repair end
<f/> - Fluent word
```

### Evaluation Methods
Key evaluation metrics are implemented in `evaluation/disf_evaluation.py`:
- Word-level and interval-level accuracy 
- Incremental metrics (edit overhead, timing metrics)
- Speaker rate analysis
- Error analysis

### Data Flow
1. Raw corpus data -> Corpus creation -> Feature extraction
2. Features -> Model training -> Saved models
3. Live input -> Feature extraction -> Incremental tagging

## Critical Workflows

### Setup
```bash
# Setup with pip
pip install deep_disfluency

# Development setup 
pip install -r requirements.txt
```

### Live ASR Integration
1. Install PortAudio
2. Configure IBM Watson credentials
3. Use `demos/asr.py` for live processing

### Testing
- Use `evaluation/disf_evaluation.py` for full evaluation
- Key test data in `data/disfluency_detection/switchboard/`

## Key Integration Points

- ASR Integration: `asr/ibm_watson.py` 
- Feature Extraction: `feature_extraction/extract_features.py`
- Model Interface: `tagger/deep_tagger.py`

## Common Pitfalls

1. Always configure Python path when developing without installation
2. Ensure proper model files are in place before running tagger
3. Watch for partial word handling in corpus creation
4. Be careful with repair tag IDs in evaluation