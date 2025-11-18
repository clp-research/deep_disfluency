# Quick Guide: Evaluating Transformer Tagger on Switchboard Corpus

## Files Available

Switchboard test data with timings:
- Location: `deep_disfluency/data/disfluency_detection/switchboard/`
- Key files:
  - `swbd_disf_test_data_timings.csv`
  - `swbd_disf_train_data_timings.csv`
  - `swbd_disf_dev_data_timings.csv`

## Option 1: Quick Python Script (Minimal)

```python
#!/usr/bin/env python3
"""Quick corpus evaluation script"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, '/Users/aida/Desktop/code/deep_disfluency')

from deep_disfluency.tagger.transformer_integration import (
    create_compatible_tagger_wrapper,
    SeqevalIntegration
)

# Example test data (replace with loaded CSV data)
test_data = [
    (["I", "want", "a", "ticket"], ['<f/>', '<f/>', '<f/>', '<f/>']),
    (["I", "like", "um", "apples"], ['<f/>', '<f/>', '<i/>', '<f/>']),
]

# Create tagger
adapter = create_compatible_tagger_wrapper(device="cpu")

# Evaluate each utterance
all_predictions = []
all_gold = []

for words, gold_tags in test_data:
    adapter.reset()
    pred_tags = adapter.tag_sequence(words)
    
    all_predictions.extend(pred_tags)
    all_gold.extend(gold_tags)
    
    print(f"Words: {words}")
    print(f"  Pred: {pred_tags}")
    print(f"  Gold: {gold_tags}")
    print()

# Compute overall metrics
metrics = SeqevalIntegration.compute_metrics(all_predictions, all_gold)
print("\n" + "="*50)
print("OVERALL METRICS")
print("="*50)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
```

## Option 2: Using Existing Corpus Loader

If corpus loading utilities exist:

```python
from deep_disfluency.corpus.load import load_corpus  # Adjust to actual module

# Load test corpus
corpus_path = Path("deep_disfluency/data/disfluency_detection/switchboard/swbd_disf_test_data_timings.csv")
test_data = load_corpus(corpus_path)

# Then evaluate as shown in Option 1
```

## Option 3: Manual CSV Parsing

If you need to parse the CSV directly:

```python
import pandas as pd

# Load CSV
csv_path = "deep_disfluency/data/disfluency_detection/switchboard/swbd_disf_test_data_timings.csv"
df = pd.read_csv(csv_path)

# Parse into (words, tags) tuples
corpus_data = []
for conversation_id in df['conversation_id'].unique():
    conv_data = df[df['conversation_id'] == conversation_id]
    
    words = conv_data['word'].tolist()
    tags = conv_data['tag'].tolist()  # Adjust column name if needed
    
    corpus_data.append((words, tags))

# Evaluate
from deep_disfluency.tagger.transformer_integration import create_compatible_tagger_wrapper

adapter = create_compatible_tagger_wrapper(device="cpu")
# ... rest of evaluation code
```

## Expected Output

```
========================================================
Overall Metrics on Switchboard Test Set
========================================================
Total utterances: 1000 (example)
Total words: 12,345 (example)

Precision: 0.782
Recall: 0.756
F1-Score: 0.769

Per-tag metrics:
  <f/>   : P=0.95, R=0.94, F1=0.945
  <e/>   : P=0.72, R=0.68, F1=0.700
  <rps/> : P=0.81, R=0.79, F1=0.800
  ...
```

## To Run Evaluation

1. **Choose an option** (1, 2, or 3 above)
2. **Save script** as `eval_transformer_tagger.py`
3. **Run**:
   ```bash
   cd /Users/aida/Desktop/code/deep_disfluency
   python eval_transformer_tagger.py
   ```

## Timing Considerations

- **Small test set** (100 utterances): ~30 seconds
- **Medium test set** (1,000 utterances): ~5 minutes  
- **Full test set** (5,000+ utterances): ~30 minutes

Use `device="cuda"` if available for 5-10x speedup.

## Interpreting Results

### Baseline Expectations (No Fine-tuning)
- Precision: 60-75% (model is pretrained but not on disfluency task)
- Recall: 55-70%
- F1-Score: 57-72%

### After Fine-tuning
- Precision: 80-90%
- Recall: 78-88%
- F1-Score: 79-89%

## Next: Integration with Full Evaluation Pipeline

Once you have baseline metrics, integrate with `disf_evaluation.py`:

```python
# In deep_disfluency/evaluation/disf_evaluation.py
from deep_disfluency.tagger.transformer_integration import create_compatible_tagger_wrapper

# Replace old tagger init with:
tagger = create_compatible_tagger_wrapper(device="cpu")  # or "cuda"

# Then use in existing evaluation code - API is the same!
```

---

**Questions?** Check `deep_disfluency/tagger/TRANSFORMER_TAGGER_README.md` for full API reference.
