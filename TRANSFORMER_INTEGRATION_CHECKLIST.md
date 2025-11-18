# Transformer Tagger - Quick Integration Checklist

## Pre-Integration Steps (Do These First)

- [ ] **Install dependencies**
  ```bash
  pip install torch>=1.9.0 transformers>=4.20.0 seqeval>=1.2.2
  ```

- [ ] **Verify installations**
  ```python
  python -c "import torch; print(torch.__version__)"
  python -c "import transformers; print(transformers.__version__)"
  python -c "from seqeval.metrics import f1_score"
  ```

- [ ] **Run test suite**
  ```bash
  cd deep_disfluency/tagger
  python test_transformer_tagger.py
  ```
  Expected: 5/5 tests pass

---

## Integration Points

### 1. Evaluation Module Integration

**File**: `deep_disfluency/evaluation/disf_evaluation.py`

**Task**: Update to use new tagger

```python
# OLD (Theano)
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger
tagger = DeepDisfluencyTagger(model_config_path)

# NEW (Transformer)
from deep_disfluency.tagger.transformer_integration import create_compatible_tagger_wrapper
tagger = create_compatible_tagger_wrapper(device="cpu")  # or "cuda"
```

**Metrics**:
```python
# OLD
# Custom scoring

# NEW  
from deep_disfluency.tagger.transformer_integration import SeqevalIntegration
metrics = SeqevalIntegration.compute_metrics(predicted, gold)
```

### 2. Corpus Loading

**File**: `deep_disfluency/corpus/load.py`

**Task**: Load corpus data for testing

```python
# Example corpus loader
from pathlib import Path
import json

def load_test_corpus():
    """Load Switchboard test set"""
    corpus_path = Path("deep_disfluency/data/disfluency_detection/switchboard")
    
    # Load test data
    test_data = []
    test_file = corpus_path / "swbd_disf_test_data_timings.csv"
    
    # Parse CSV and create (words, tags) tuples
    # See corpus_util.py for parsing utilities
    
    return test_data
```

### 3. Experiment Scripts

**File**: `deep_disfluency/experiments/experiment_configs.csv`

**Task**: Add transformer tagger configurations

```csv
experiment_id,tagger_type,model_name,device,context_window
TRANSFORMER_001,transformer,distilbert-base-uncased,cpu,10
TRANSFORMER_002,transformer,distilbert-base-uncased,cuda,10
TRANSFORMER_FINETUNED,transformer,./saved_models/distilbert-disfluency,cpu,10
```

### 4. Demo Scripts

**File**: `demos/demo.py`

**Task**: Add transformer tagger demo

```python
from deep_disfluency.tagger.transformer_integration import create_compatible_tagger_wrapper
from deep_disfluency.tagger.incremental_transformer_pipeline import IncrementalTransformerPipeline

def demo_transformer_tagger():
    """Demo incremental disfluency detection"""
    tagger = create_compatible_tagger_wrapper(device="cpu")
    pipeline = IncrementalTransformerPipeline(tagger)
    
    # Example conversation
    dialogue = [
        ("I", "PRP"),
        ("want", "VB"),
        ("to", "TO"),
        ("buy", "VB"),
        ("a", "DT"),
        ("ticket", "NN"),
    ]
    
    print("Incremental Disfluency Detection:")
    for word, pos in dialogue:
        result = pipeline.process_word(word, pos)
        print(f"  {word} -> {result['tag']}")
    
    stats = pipeline.get_statistics()
    print(f"\nStats: {stats['num_words']} words, " 
          f"{stats['num_disfluencies']} disfluencies")
```

---

## Testing Integration

### Unit Test

Create `test_integration.py`:

```python
from deep_disfluency.tagger.transformer_integration import (
    create_compatible_tagger_wrapper,
    SeqevalIntegration,
    evaluate_on_corpus
)

def test_full_pipeline():
    """Test complete pipeline"""
    adapter = create_compatible_tagger_wrapper(device="cpu")
    
    # Test sequence
    words = ["I", "like", "apples"]
    predicted_tags = adapter.tag_sequence(words)
    
    # Should get tags
    assert len(predicted_tags) == 3
    assert all(tag in ['<f/>', '<e/>', '<rms/>', '<rm/>', '<i/>', 
                       '<rps/>', '<rp/>', '<rpn/>', '<rpndel/>'] 
               for tag in predicted_tags)
    
    # Test evaluation
    gold_tags = ['<f/>', '<f/>', '<f/>']
    metrics = SeqevalIntegration.compute_metrics(predicted_tags, gold_tags)
    
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    
    print("✓ Integration test passed")

if __name__ == "__main__":
    test_full_pipeline()
```

### Integration Test with Corpus

Create `test_corpus_evaluation.py`:

```python
from pathlib import Path
from deep_disfluency.corpus.load import load_corpus  # Adapt to actual loader
from deep_disfluency.tagger.transformer_integration import evaluate_on_corpus, create_compatible_tagger_wrapper

def test_on_real_corpus():
    """Evaluate on actual Switchboard data"""
    adapter = create_compatible_tagger_wrapper(device="cpu")
    
    # Load test corpus (adjust path as needed)
    corpus_path = Path("deep_disfluency/data/disfluency_detection/switchboard")
    
    # TODO: Load corpus data
    # test_data = load_corpus(corpus_path / "swbd_disf_test_data_timings.csv")
    
    # Evaluate
    # results = evaluate_on_corpus(adapter, test_data, verbose=True)
    
    # Print results
    # print(f"Overall F1: {results['overall_metrics']['f1']:.3f}")
    # print(f"Precision: {results['overall_metrics']['precision']:.3f}")
    # print(f"Recall: {results['overall_metrics']['recall']:.3f}")

if __name__ == "__main__":
    test_on_real_corpus()
```

---

## Rollout Steps

### Phase 1: Isolated Testing ✅
- [x] Create transformer tagger module
- [x] Create incremental pipeline
- [x] Create test suite
- [x] Document API

### Phase 2: Integration Testing (CURRENT)
- [ ] Run test_transformer_tagger.py
- [ ] Create integration test
- [ ] Test on sample corpus
- [ ] Verify metrics match expected

### Phase 3: Full Evaluation (NEXT)
- [ ] Integrate with disf_evaluation.py
- [ ] Run on full test set
- [ ] Compare metrics with baseline
- [ ] Document results

### Phase 4: Deployment
- [ ] Update experiment scripts
- [ ] Update demos
- [ ] Deprecate Theano tagger (keep for reference)
- [ ] Update documentation

---

## Metrics to Track

Before/After comparison table:

| Metric | Theano LSTM | Transformer | Delta |
|--------|------------|-------------|-------|
| **Tag Accuracy** | TBD | TBD | TBD |
| **Precision** | TBD | TBD | TBD |
| **Recall** | TBD | TBD | TBD |
| **F1 Score** | TBD | TBD | TBD |
| **Model Size** | ~1GB | ~250MB | -75% |
| **Per-word Time** | 50-100ms | 10-50ms | -50% |
| **Initialization** | 10-20s | ~5s | -50% |

---

## Common Integration Issues & Solutions

### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**:
```bash
pip install torch transformers seqeval
```

### Issue 2: Version Conflicts
```
ImportError: cannot import name 'AutoTokenizer'
```
**Solution**:
```bash
pip install --upgrade transformers
```

### Issue 3: Device Issues
```
RuntimeError: CUDA out of memory
```
**Solution**:
```python
adapter = create_compatible_tagger_wrapper(device="cpu")
```

### Issue 4: Corpus Format Mismatch
```
ValueError: Cannot parse corpus data
```
**Solution**:
- Check corpus format matches expected (words, pos, tags)
- Use corpus_util.py parsing functions
- See TRANSFORMER_TAGGER_README.md for format details

---

## Command Reference

### Install & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or individually
pip install torch>=1.9.0 transformers>=4.20.0 seqeval>=1.2.2
```

### Testing
```bash
# Basic transformer tagger tests
python deep_disfluency/tagger/test_transformer_tagger.py

# Integration tests (create these)
python test_integration.py
python test_corpus_evaluation.py
```

### Quick Demo
```python
from deep_disfluency.tagger.transformer_integration import create_compatible_tagger_wrapper
from deep_disfluency.tagger.incremental_transformer_pipeline import IncrementalTransformerPipeline

adapter = create_compatible_tagger_wrapper(device="cpu")
words = ["I", "like", "apples"]
tags = adapter.tag_sequence(words)
print(" ".join(f"{w} {t}" for w, t in zip(words, tags)))
```

### Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "from seqeval.metrics import f1_score; print('seqeval: OK')"
```

---

## Rollback Plan

If issues encountered:

1. **Theano tagger still available**:
   ```python
   from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger
   ```

2. **No changes to master branch** - all work on feature branch

3. **Quick fallback**:
   ```bash
   git checkout master
   pip install Theano  # If needed for old tagger
   ```

---

## Success Criteria

✅ All of these should be true:

- [ ] test_transformer_tagger.py passes all 5 tests
- [ ] Transformer tagger produces valid disfluency tags
- [ ] Evaluation metrics computed correctly via seqeval
- [ ] Incremental tagging works (word-by-word)
- [ ] Rollback mechanism functional (ASR corrections)
- [ ] Model can be saved/loaded
- [ ] Integration with evaluation module smooth
- [ ] No errors in backwards-compatible adapter
- [ ] Documentation complete and clear

---

## Next: Run Test Suite

```bash
cd /Users/aida/Desktop/code/deep_disfluency/deep_disfluency/tagger
python test_transformer_tagger.py
```

Expected output: **5/5 tests passed** ✅

Then proceed to corpus evaluation and integration testing.

---

**Status**: Ready for integration testing  
**Branch**: port/python3-applied (do NOT commit to master)  
**Responsibility**: User to run tests and confirm integration
