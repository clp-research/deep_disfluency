"""
Incremental Transformer Tagger Pipeline
Wrapper that manages stateful, word-by-word incremental tagging with proper context handling.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class IncrementalTransformerPipeline:
    """
    Incremental pipeline for word-by-word disfluency tagging.
    
    Handles:
    - Context window management
    - Rolling buffer with lookback
    - Smooth integration with evaluation metrics
    - Support for ASR corrections (rollback)
    """
    
    def __init__(
        self,
        transformer_tagger: 'TransformerDisfluencyTagger',
        context_window: int = 10,
        use_confidence_scores: bool = False,
    ):
        """
        Initialize the incremental pipeline.
        
        Args:
            transformer_tagger: Instance of TransformerDisfluencyTagger
            context_window: How many previous words to consider
            use_confidence_scores: Whether to track softmax confidence
        """
        self.tagger = transformer_tagger
        self.context_window = context_window
        self.use_confidence_scores = use_confidence_scores
        
        # State tracking
        self.word_history = []
        self.tag_history = []
        self.confidence_history = []
        self.timing_history = []
        
    def reset(self):
        """Reset for new utterance."""
        self.tagger.reset()
        self.word_history = []
        self.tag_history = []
        self.confidence_history = []
        self.timing_history = []
    
    def process_word(
        self,
        word: str,
        pos: Optional[str] = None,
        timing: Optional[float] = None,
        rollback: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a single incoming word incrementally.
        
        Args:
            word: The word token
            pos: POS tag (optional)
            timing: Duration in seconds (optional)
            rollback: Number of previous words to revise (for ASR corrections)
            
        Returns:
            Dict with:
                - 'tag': predicted disfluency tag
                - 'word': the input word
                - 'pos': POS tag if provided
                - 'timing': duration if provided
                - 'full_sequence': all tags up to current position
        """
        # Handle corrections
        if rollback > 0:
            self.word_history = self.word_history[:-rollback]
            self.tag_history = self.tag_history[:-rollback]
            self.confidence_history = self.confidence_history[:-rollback]
            self.timing_history = self.timing_history[:-rollback]
        
        # Predict tag for new word
        predicted_tag = self.tagger.tag_new_word(word, pos, timing, rollback=rollback)
        
        # Store history
        self.word_history.append(word)
        self.tag_history.append(predicted_tag)
        self.timing_history.append(timing)
        
        # Optionally compute confidence (placeholder for now)
        confidence = 0.95  # TODO: extract from model softmax
        self.confidence_history.append(confidence)
        
        return {
            'word': word,
            'tag': predicted_tag,
            'pos': pos,
            'timing': timing,
            'confidence': confidence,
            'full_sequence': self.tag_history.copy(),
            'turn_number': len(self.word_history),
        }
    
    def process_sequence(
        self,
        sequence: List[Tuple[str, Optional[str], Optional[float]]],
        rollback: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple words (e.g., after ASR correction of entire prefix).
        
        Args:
            sequence: List of (word, pos, timing) tuples
            rollback: Words to revise before processing
            
        Returns:
            List of result dicts (one per word)
        """
        results = []
        for word, pos, timing in sequence:
            result = self.process_word(word, pos, timing, rollback=rollback)
            results.append(result)
            rollback = 0  # Only apply rollback to first word
        
        return results
    
    def get_incremental_output(self) -> str:
        """
        Get XML-style incremental output string.
        
        Returns:
            Space-separated tags as XML string
            Example: "the <f/> cat <f/> sat <f/> down <f/>"
        """
        output_parts = []
        for word, tag in zip(self.word_history, self.tag_history):
            output_parts.append(f"{word} {tag}")
        
        return " ".join(output_parts)
    
    def get_final_output(self) -> str:
        """
        Get final output (same as incremental in this case).
        
        Returns:
            Space-separated tags as string
        """
        return self.get_incremental_output()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current tagging run.
        
        Returns:
            Dict with:
                - num_words: total words processed
                - num_disfluencies: estimated count of non-fluent tags
                - avg_confidence: average model confidence
                - error_rate: placeholder
        """
        non_fluent_count = sum(1 for tag in self.tag_history if tag != '<f/>')
        avg_conf = np.mean(self.confidence_history) if self.confidence_history else 0.0
        
        return {
            'num_words': len(self.word_history),
            'num_disfluencies': non_fluent_count,
            'avg_confidence': float(avg_conf),
            'words_per_second': self._compute_speech_rate(),
        }
    
    def _compute_speech_rate(self) -> float:
        """Compute words per second (if timing data available)."""
        if not self.timing_history or all(t is None for t in self.timing_history):
            return 0.0
        
        total_time = sum(t for t in self.timing_history if t is not None)
        word_count = len([t for t in self.timing_history if t is not None])
        
        if total_time > 0:
            return word_count / total_time
        return 0.0
    
    def to_eval_format(self) -> Tuple[List[str], List[str]]:
        """
        Convert to format suitable for seqeval evaluation.
        
        Returns:
            Tuple of (words, tags) lists
        """
        return (self.word_history.copy(), self.tag_history.copy())


class IncrementalTaggingDemo:
    """Demo class showing incremental tagging workflow."""
    
    @staticmethod
    def example_workflow(tagger, pipeline):
        """
        Example of incremental tagging in action.
        
        Args:
            tagger: TransformerDisfluencyTagger instance
            pipeline: IncrementalTransformerPipeline instance
        """
        # Sample sentence with disfluency
        sentence = [
            ("I", None, 0.1),
            ("want", None, 0.15),
            ("to", None, 0.1),
            ("go", None, 0.12),  # Disfluency: edit term
            ("uh", None, 0.1),   # Disfluency: reparandum
            ("buy", None, 0.13), # Repair
            ("a", None, 0.08),
            ("ticket", None, 0.14),
        ]
        
        print("=" * 60)
        print("INCREMENTAL DISFLUENCY TAGGING DEMO")
        print("=" * 60)
        
        results = []
        for i, (word, pos, timing) in enumerate(sentence, 1):
            result = pipeline.process_word(word, pos, timing)
            print(f"\nWord {i}: {word}")
            print(f"  Predicted Tag: {result['tag']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Sequence So Far: {' '.join(result['full_sequence'])}")
            results.append(result)
        
        stats = pipeline.get_statistics()
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total Words: {stats['num_words']}")
        print(f"Disfluencies Detected: {stats['num_disfluencies']}")
        print(f"Avg Confidence: {stats['avg_confidence']:.2%}")
        print(f"Speech Rate: {stats['words_per_second']:.1f} wps")
        
        print("\nFinal Output:")
        print(pipeline.get_final_output())
        
        return results
