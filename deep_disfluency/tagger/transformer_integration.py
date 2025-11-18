"""
Integration Module for Transformer Tagger with Evaluation Pipeline

Provides adapters and utilities to integrate TransformerDisfluencyTagger
with the existing evaluation framework (disf_evaluation.py, seqeval metrics).
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Map internal tag format to IOB format for seqeval
TAG_TO_IOB = {
    '<f/>': 'O',           # Outside (fluent)
    '<e/>': 'B-E',         # Edit term
    '<rms/>': 'B-RMS',     # Reparandum start
    '<rm/>': 'I-RMS',      # Mid-reparandum
    '<i/>': 'B-I',         # Interregnum
    '<rps/>': 'B-RPS',     # Repair start
    '<rp/>': 'I-RPS',      # Mid-repair
    '<rpn/>': 'B-RPN',     # Repair end
    '<rpndel/>': 'B-RPND', # Delete repair end
}

IOB_TO_TAG = {v: k for k, v in TAG_TO_IOB.items()}


class TransformerTaggerAdapter:
    """
    Adapter to make TransformerDisfluencyTagger compatible with
    existing evaluation pipelines expecting the old Theano tagger.
    """
    
    def __init__(self, transformer_tagger: 'TransformerDisfluencyTagger'):
        """
        Initialize adapter.
        
        Args:
            transformer_tagger: Instance of TransformerDisfluencyTagger
        """
        self.tagger = transformer_tagger
        self.word_buffer = []
        self.tag_buffer = []
    
    def tag_sequence(self, words: List[str]) -> List[str]:
        """
        Tag an entire sequence (for compatibility with old interface).
        
        Args:
            words: List of word tokens
            
        Returns:
            List of tags in same XML format as old tagger
        """
        self.reset()
        tags = []
        
        for word in words:
            tag = self.tagger.tag_new_word(word, pos=None, timing=None)
            tags.append(tag)
            self.word_buffer.append(word)
            self.tag_buffer.append(tag)
        
        return tags
    
    def tag_sequence_incremental(
        self,
        words: List[str],
        rollback_pos: Optional[int] = None,
    ) -> List[str]:
        """
        Tag with rollback support (for ASR corrections).
        
        Args:
            words: List of word tokens
            rollback_pos: Position to rollback to (if None, no rollback)
            
        Returns:
            List of tags after rollback and re-tagging
        """
        rollback_amount = 0
        
        if rollback_pos is not None:
            rollback_amount = len(self.word_buffer) - rollback_pos
            if rollback_amount > 0:
                self.word_buffer = self.word_buffer[:rollback_pos]
                self.tag_buffer = self.tag_buffer[:rollback_pos]
        
        tags = []
        for word in words:
            tag = self.tagger.tag_new_word(
                word, pos=None, timing=None, rollback=rollback_amount
            )
            tags.append(tag)
            self.word_buffer.append(word)
            self.tag_buffer.append(tag)
            rollback_amount = 0  # Only apply to first word
        
        return self.tag_buffer.copy()
    
    def reset(self):
        """Clear state for new utterance."""
        self.tagger.reset()
        self.word_buffer = []
        self.tag_buffer = []
    
    def get_current_sequence(self) -> Tuple[List[str], List[str]]:
        """
        Get current word and tag sequences.
        
        Returns:
            Tuple of (words, tags)
        """
        return (self.word_buffer.copy(), self.tag_buffer.copy())


class SeqevalIntegration:
    """
    Integration with seqeval for standard NER/token classification metrics.
    """
    
    @staticmethod
    def convert_to_iob(tags: List[str]) -> List[str]:
        """
        Convert internal tag format to IOB format for seqeval.
        
        Args:
            tags: List of internal tags (e.g., '<f/>', '<e/>')
            
        Returns:
            List of IOB tags (e.g., 'O', 'B-E')
        """
        iob_tags = []
        for tag in tags:
            iob_tag = TAG_TO_IOB.get(tag, 'O')
            iob_tags.append(iob_tag)
        return iob_tags
    
    @staticmethod
    def convert_from_iob(iob_tags: List[str]) -> List[str]:
        """
        Convert IOB tags back to internal format.
        
        Args:
            iob_tags: List of IOB tags
            
        Returns:
            List of internal tags
        """
        tags = []
        for iob_tag in iob_tags:
            tag = IOB_TO_TAG.get(iob_tag, '<f/>')
            tags.append(tag)
        return tags
    
    @staticmethod
    def compute_metrics(
        predicted_tags: List[str],
        gold_tags: List[str],
    ) -> Dict[str, float]:
        """
        Compute standard token classification metrics using seqeval.
        
        Args:
            predicted_tags: Predicted tags (internal format)
            gold_tags: Gold standard tags (internal format)
            
        Returns:
            Dict with precision, recall, F1 scores
        """
        try:
            from seqeval.metrics import precision_score, recall_score, f1_score
        except ImportError:
            logger.warning("seqeval not installed; returning placeholder metrics")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        
        # Convert to IOB format for seqeval
        pred_iob = SeqevalIntegration.convert_to_iob(predicted_tags)
        gold_iob = SeqevalIntegration.convert_to_iob(gold_tags)
        
        # seqeval expects lists of lists (sentences)
        pred_sequences = [pred_iob]
        gold_sequences = [gold_iob]
        
        precision = precision_score(gold_sequences, pred_sequences, average='weighted')
        recall = recall_score(gold_sequences, pred_sequences, average='weighted')
        f1 = f1_score(gold_sequences, pred_sequences, average='weighted')
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    
    @staticmethod
    def compute_per_label_metrics(
        predicted_tags: List[str],
        gold_tags: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-label metrics (precision, recall, F1 for each tag).
        
        Args:
            predicted_tags: Predicted tags
            gold_tags: Gold tags
            
        Returns:
            Dict mapping tag names to metric dicts
        """
        try:
            from seqeval.metrics import classification_report
        except ImportError:
            logger.warning("seqeval not installed; cannot compute per-label metrics")
            return {}
        
        pred_iob = SeqevalIntegration.convert_to_iob(predicted_tags)
        gold_iob = SeqevalIntegration.convert_to_iob(gold_tags)
        
        report = classification_report(
            [gold_iob], [pred_iob], output_dict=True
        )
        
        return report


class IncrementalEvaluationMetrics:
    """
    Incremental evaluation metrics (edit overhead, online accuracy, etc.).
    """
    
    def __init__(self):
        """Initialize metric tracker."""
        self.correct_predictions = 0
        self.total_predictions = 0
        self.edit_overhead = 0  # Number of words that were errors
        self.corrections = 0     # Number of rollback corrections
    
    def update(
        self,
        predicted_tag: str,
        gold_tag: str,
        was_correction: bool = False,
    ):
        """
        Update metrics with single prediction.
        
        Args:
            predicted_tag: Tag predicted by tagger
            gold_tag: Gold standard tag
            was_correction: Whether this was a rollback correction
        """
        self.total_predictions += 1
        
        if predicted_tag == gold_tag:
            self.correct_predictions += 1
        else:
            self.edit_overhead += 1
        
        if was_correction:
            self.corrections += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics.
        
        Returns:
            Dict with accuracy, edit overhead, etc.
        """
        accuracy = (
            self.correct_predictions / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        
        overhead_rate = (
            self.edit_overhead / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        
        return {
            'accuracy': accuracy,
            'edit_overhead': self.edit_overhead,
            'overhead_rate': overhead_rate,
            'corrections': self.corrections,
            'total_predictions': self.total_predictions,
        }
    
    def reset(self):
        """Reset for new utterance."""
        self.correct_predictions = 0
        self.total_predictions = 0
        self.edit_overhead = 0
        self.corrections = 0


def create_compatible_tagger_wrapper(
    model_name: str = "distilbert-base-uncased",
    device: str = "cpu",
) -> 'TransformerTaggerAdapter':
    """
    Factory function to create a fully initialized and compatible tagger.
    
    Args:
        model_name: HuggingFace model identifier
        device: 'cpu' or 'cuda'
        
    Returns:
        TransformerTaggerAdapter ready for use
    """
    try:
        from transformer_tagger import TransformerDisfluencyTagger
        
        tagger = TransformerDisfluencyTagger(
            model_name=model_name,
            num_labels=9,
            context_window=10,
            device=device,
        )
        
        adapter = TransformerTaggerAdapter(tagger)
        return adapter
    except ImportError as e:
        logger.error(f"Cannot import transformer_tagger: {e}")
        raise


def evaluate_on_corpus(
    adapter: 'TransformerTaggerAdapter',
    corpus_data: List[Tuple[List[str], List[str]]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate transformer tagger on entire corpus.
    
    Args:
        adapter: TransformerTaggerAdapter instance
        corpus_data: List of (words, gold_tags) tuples
        verbose: Whether to print per-utterance results
        
    Returns:
        Dict with overall metrics
    """
    all_predictions = []
    all_gold_tags = []
    utterance_results = []
    
    for utterance_idx, (words, gold_tags) in enumerate(corpus_data):
        adapter.reset()
        pred_tags = adapter.tag_sequence(words)
        
        all_predictions.extend(pred_tags)
        all_gold_tags.extend(gold_tags)
        
        # Per-utterance metrics
        correct = sum(1 for p, g in zip(pred_tags, gold_tags) if p == g)
        accuracy = correct / len(gold_tags) if gold_tags else 0.0
        
        utterance_results.append({
            'utterance_id': utterance_idx,
            'num_words': len(words),
            'accuracy': accuracy,
        })
        
        if verbose and utterance_idx < 3:  # Print first 3
            logger.info(f"\nUtterance {utterance_idx}:")
            logger.info(f"  Words: {' '.join(words)}")
            logger.info(f"  Predicted: {' '.join(pred_tags)}")
            logger.info(f"  Gold: {' '.join(gold_tags)}")
            logger.info(f"  Accuracy: {accuracy:.1%}")
    
    # Overall metrics
    seqeval_metrics = SeqevalIntegration.compute_metrics(
        all_predictions, all_gold_tags
    )
    
    return {
        'overall_metrics': seqeval_metrics,
        'utterance_metrics': utterance_results,
        'total_utterances': len(corpus_data),
        'total_words': len(all_predictions),
    }
