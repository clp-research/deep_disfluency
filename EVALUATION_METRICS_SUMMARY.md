# Deep Disfluency Evaluation Metrics Summary

## Overview

The deep_disfluency system calculates a comprehensive suite of evaluation metrics across multiple dimensions:
- **Tag-level metrics** (individual disfluency types)
- **Correlation metrics** (speaker behavior analysis)
- **Timing metrics** (incremental processing)
- **Segmentation metrics** (utterance boundaries)
- **Error analysis** (repair type classification)

---

## 1. **Core Tag-Level Metrics**

### Precision, Recall, F1-Score

**Function:** `p_r_f(tps, fps, fns)` in `eval_utils.py`

Calculated for each disfluency tag:

```
Precision = TPs / (TPs + FPs)
Recall    = TPs / (TPs + FNs)
F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
```

**Applied to all tags:**
- `<rms` - Reparandum start
- `<rm` - Mid-reparandum
- `<i` - Interregnum
- `<e` - Edit term
- `<rps` - Repair onset (most important)
- `<rp` - Mid-repair
- `<rpn` - Repair end
- `<rpnrep` - Repair end (repetition type)
- `<rpnsub` - Repair end (substitution type)
- `<rpndel` - Repair end (deletion type)
- `t/>` - Turn/utterance end (when utt_eval=True)

**Output format:**
- `p_<TAG>_word` / `p_<TAG>_interval` - Precision per word or 10-second window
- `r_<TAG>_word` / `r_<TAG>_interval` - Recall per word or 10-second window
- `f1_<TAG>_word` / `f1_<TAG>_interval` - F1-Score per word or 10-second window

---

## 2. **Combined/Aggregate Metrics**

### Multi-Tag Combinations

Combined accuracy for tags working together in repair structure:

- **`<rm.<i.<rp`** - Combined repair structure
  - Precision: `p_<rm.<i.<rp_word` / `p_<rm.<i.<rp_interval`
  - Recall: `r_<rm.<i.<rp_word` / `r_<rm.<i.<rp_interval`
  - F1: `f1_<rm.<i.<rp_word` / `f1_<rm.<i.<rp_interval`

---

## 3. **Relaxed Metrics**

### Lenient Evaluation (for robustness)

Relaxed versions of key repair-related tags that allow partial matches:

**Relaxed tags:**
- `<rps_relaxed` - Repair onset (lenient)
- `<e_relaxed` - Edit term (lenient)
- `t/>_relaxed` - Turn boundary (lenient)

**Output format:**
- `p_<TAG>_relaxed_word` / `p_<TAG>_relaxed_interval`
- `r_<TAG>_relaxed_word` / `r_<TAG>_relaxed_interval`
- `f1_<TAG>_relaxed_word` / `f1_<TAG>_relaxed_interval`

---

## 4. **Segmentation Metrics**

### NIST SU (Segmentation Unit) Error Rate

**Function:** `NIST_SU(results)` in `eval_utils.py`

```
NIST_SU = ((False Negatives + False Positives) / (TPs + FNs)) * 100
```

Measures segmentation errors as percentage of reference segments. Lower is better.

**Calculated for:**
- Word-level evaluation: `NIST_SU_word`
- Interval-level evaluation: `NIST_SU_interval`

---

### DSER (Dialogue Segmentation Error Rate)

**Function:** `DSER(results)` in `eval_utils.py`

```
DSER = ((Total Segments - Correctly Segmented) / Total Segments) * 100
```

Percentage of dialogue segments incorrectly segmented. Used when `utt_eval=True`.

**Calculated for:**
- Word-level: `DSER_word`
- Interval-level: `DSER_interval`

---

### SegER (Segmentation Error Rate)

**Function:** `SegER(results)` in `eval_utils.py`

```
SegER = (Edit Distance between sequences / Total Segments) * 100
```

Edit distance between reference and hypothesis position sequences. Currently commented out in evaluation.

---

## 5. **Levenshtein Distance Metrics** ⭐ (ALREADY IMPLEMENTED)

### Character/Sequence-Level Edit Distance

**Function:** `alignment_cost(r, h, subcost=1)` in `eval_utils.py`

```python
def alignment_cost(r, h, subcost=1):
    """Calculation of Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time and space complexity.
    """
    # Dynamic programming implementation
    # Returns minimum edit distance (insertions, deletions, substitutions)
```

**Used for:**
- Sequence alignment between reference and hypothesis
- Token-level word sequence distance
- Already integrated into segmentation error calculations

**Example usage:**
```python
alignment_cost("who is there".split(), "is there".split())  # Returns 1
alignment_cost("who is there".split(), "".split())          # Returns 3
```

---

## 6. **Correlation Metrics** (Speaker Behavior Analysis)

### Pearson Correlation

Measures linear relationship between predicted and actual repair rates per speaker.

**Metrics:**
- `pearson_r_correl_rps_number` - Correlation of repair count
- `pearson_r_p_value_rps_number` - P-value for correlation
- `pearson_r_correl_rps_rate_per_word` - Correlation of repairs/word
- `pearson_r_p_value_rps_rate_per_word` - P-value
- `pearson_r_correl_rps_rate_per_utt` - Correlation of repairs/utterance
- `pearson_r_p_value_rps_rate_per_utt` - P-value

### Spearman's Rank Correlation

Non-parametric rank-based correlation (robust to outliers).

**Metrics:**
- `spearman_rank_correl_rps_number` - Rank correlation of repair count
- `spearman_rank_p_value_rps_number` - P-value
- `spearman_rank_correl_rps_rate_per_word` - Rank correlation of repairs/word
- `spearman_rank_p_value_rps_rate_per_word` - P-value
- `spearman_rank_correl_rps_rate_per_utt` - Rank correlation of repairs/utt
- `spearman_rank_p_value_rps_rate_per_utt` - P-value

**Implementation:**
```python
from scipy.stats import pearsonr, spearmanr

# Correlation between predicted and gold repair rates per speaker
pearson_r, pearson_p = pearsonr(gold_rates, hypothesis_rates)
spearman_r, spearman_p = spearmanr(gold_rates, hypothesis_rates)
```

---

## 7. **Incremental/Timing Metrics**

### Delayed Accuracy (Edit Overhead)

Measures accuracy degradation when processing incrementally with limited lookahead.

**Metrics:**
- `delayed_acc_<rm_1_word` - Accuracy with 1-word delay
- `delayed_acc_<rm_2_word` - Accuracy with 2-word delay
- ...
- `delayed_acc_<rm_6_word` - Accuracy with 6-word delay
- `delayed_acc_<rm_mean_word` - Mean delayed accuracy

Also available for `interval` mode.

### Time-to-Detection (TTD)

Measures how quickly after a disfluency occurrence the system detects it.

**Tags measured:**
- `t_t_detection_<rms_word` - Detection latency for reparandum start
- `t_t_detection_<rps_word` - Detection latency for repair onset
- `t_t_detection_<e_word` - Detection latency for edit term
- `t_t_detection_t/>_word` - Detection latency for utterance end
- `t_t_detection_final_t/>_word` - Final utterance end detection

### Edit Overhead

Measures additional processing burden of incremental vs. batch processing.

**Metrics:**
- `edit_overhead_rel_word` - Relative edit overhead (word-level)
- `edit_overhead_rel_interval` - Relative edit overhead (interval-level)
- `edit_overhead_rel_<rm` - Edit overhead specific to reparandum

```
edit_overhead_rel = 100 * ((tag_dict["edit_overhead"][0] / tag_dict["edit_overhead"][1]) - 1)
```

### Processing Overhead

Computational cost of incremental updates.

**Metric:** `processing_overhead_word`

---

## 8. **Speaker/Conversation Rate Metrics**

### Speaker-Level Statistics

Captured separately in `SPEAKER_RATE_HEADER`:

```
corpus, conversation_no, speaker,
total_turns, total_words,
rps_hyp, rps_gold,
rps_rate_per_utt_hyp, rps_rate_per_utt_gold,
rps_rate_words_hyp, rps_rate_words_gold
```

**Measures:**
- `rps_hyp` / `rps_gold` - Absolute repair count (hypothesis vs. reference)
- `rps_rate_per_utt_hyp` / `rps_rate_per_utt_gold` - Repairs per utterance
- `rps_rate_words_hyp` / `rps_rate_words_gold` - Repairs per word

---

## 9. **Error Analysis Metrics**

### Repair Type Classification

Detailed breakdown of repair subtypes for deeper error analysis:

**For each repair type (rep/sub/del):**
- Precision: `p_rps_rep`, `p_rps_sub`, `p_rps_del`
- Recall: `r_rps_rep`, `r_rps_sub`, `r_rps_del`
- F1: `f1_rps_rep`, `f1_rps_sub`, `f1_rps_del`

**In-training vs. Novel repairs:**
- `p_rps_rep_in_training` - Repairs seen during training
- `p_rps_rep_novel` - Repairs not seen during training
- (And recall/F1 variants for each)

---

## 10. **Evaluation Levels**

All metrics are computed at two levels:

### Word-Level (`_word`)
- Evaluation per word token
- Fine-grained, word-by-word accuracy
- Catches local misclassifications

### Interval-Level (`_interval`)
- Evaluation per 10-second time window
- Coarser-grained, temporal window accuracy
- Useful for real-time streaming scenarios
- Better for timing-based metrics

---

## 11. **Evaluation Modes**

### Final Output Evaluation
- Single pass through entire dialogue
- Batch processing scenario
- Outputs final tagging accuracy

### Incremental Evaluation
- Online, real-time processing
- Word-by-word updates
- Includes time-to-detection metrics
- Includes edit overhead metrics

---

## 12. **Summary of All Metrics**

| Category | Examples | Count |
|----------|----------|-------|
| **Tag-Level P/R/F1** | `p_<rps_word`, `r_<rm_interval`, `f1_<e_word` | 11 tags × 3 metrics × 2 levels = 66 |
| **Combined Tags** | `p_<rm.<i.<rp_word` | 3 metrics × 2 levels = 6 |
| **Relaxed Tags** | `p_<rps_relaxed_word` | 3 tags × 3 metrics × 2 levels = 18 |
| **Segmentation** | `NIST_SU_word`, `DSER_word`, `SegER_word` | 3 × 2 levels = 6 |
| **Levenshtein** | `alignment_cost()` | 1 function (used in SegER) |
| **Correlation** | `pearson_r_correl_rps_rate_per_utt`, `spearman_rank_correl_*` | 12 metrics |
| **Incremental** | `delayed_acc_<rm_*`, `t_t_detection_*`, `edit_overhead_*` | 20+ metrics |
| **Speaker Rates** | `rps_rate_per_utt_hyp`, `rps_rate_words_gold` | 6 metrics |
| **Error Analysis** | `p_rps_rep_novel`, `r_rps_sub_in_training` | 30+ metrics |
| **TOTAL** | | **150+ distinct metrics** |

---

## 13. **Key Data Structures**

### Tag Dictionary (tag_dict)

```python
tag_dict = {
    tag_name: [TPs, FPs, FNs]  # For accuracy calculation
    for tag in ACC_TAGS + COMBINED_ACC_TAGS + RELAXED_TAGS
}

# Additional metrics tracked:
tag_dict["NIST_SU"] = [TPs, FPs, FNs]
tag_dict["DSER"] = [CorrectSegs, TotalSegs]
tag_dict["edit_overhead"] = [overhead_count, total_count]
tag_dict["t_t_detection_<rms_word"] = [latency_values]
```

### Results Dictionary

```python
results = {
    "f1_<rps_word": 0.85,
    "p_<rps_word": 0.82,
    "r_<rps_word": 0.88,
    "NIST_SU_word": 5.2,
    "pearson_r_correl_rps_rate_per_utt": 0.91,
    "spearman_rank_p_value_rps_number": 0.003,
    "edit_overhead_rel_word": 15.3,
    # ... 150+ more metrics
}
```

---

## 14. **Integration with Transformer Tagger**

The new PyTorch transformer tagger integrates with these metrics via:

### `SeqevalIntegration` class (transformer_integration.py)

```python
from seqeval.metrics import precision_score, recall_score, f1_score

# Convert transformer outputs (IOB format) to disfluency tags
predictions_tags = convert_to_disfluency_tags(predictions)
gold_tags = convert_to_disfluency_tags(gold_sequence)

# Calculate NER-style metrics
precision = precision_score([gold_tags], [predictions_tags])
recall = recall_score([gold_tags], [predictions_tags])
f1 = f1_score([gold_tags], [predictions_tags])
```

### `IncrementalEvaluationMetrics` class

```python
# Tracks metrics in real-time
metrics = IncrementalEvaluationMetrics()
for word, pred_tag, gold_tag in stream:
    metrics.update(word, pred_tag, gold_tag)
    
# Get incremental results
results = metrics.get_results()  # F1, precision, recall, TTD
```

---

## 15. **Usage Example**

### Running Full Evaluation

```python
from deep_disfluency.evaluation.disf_evaluation import final_output_disfluency_eval

results = final_output_disfluency_eval(
    prediction_speakers_dict=predicted_data,
    gold_speakers_dict=reference_data,
    utt_eval=True,          # Include utterance boundary metrics
    error_analysis=True,    # Include repair type breakdown
    word=True,              # Word-level metrics
    interval=True,          # 10-second window metrics
    outputfilename="eval_results.csv"
)

# Results contains 150+ metrics
print(f"F1 for <rps: {results['f1_<rps_word']}")
print(f"Segmentation errors: {results['NIST_SU_word']}%")
print(f"Speaker correlation: {results['pearson_r_correl_rps_rate_per_utt']}")
```

---

## Summary

**In addition to Levenshtein distance metrics**, the deep_disfluency system calculates:

1. ✅ **Core metrics** (Precision, Recall, F1) for 11+ disfluency tag types
2. ✅ **Combined metrics** for tag sequences (<rm.<i.<rp)
3. ✅ **Relaxed metrics** for lenient evaluation
4. ✅ **Segmentation metrics** (NIST_SU, DSER, SegER)
5. ✅ **Levenshtein distance** for sequence alignment (alignment_cost function)
6. ✅ **Correlation metrics** (Pearson & Spearman) for speaker behavior
7. ✅ **Incremental metrics** (delayed accuracy, time-to-detection, edit overhead)
8. ✅ **Speaker rate metrics** (repairs per word, per utterance)
9. ✅ **Error analysis** (repair type classification with in-training/novel breakdown)

**Total: 150+ distinct evaluation metrics** computed at word-level, interval-level, and speaker-level granularities.
