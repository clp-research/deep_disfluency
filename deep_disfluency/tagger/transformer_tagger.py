"""
PyTorch-based Transformer Tagger for Incremental Disfluency Detection

Replaces Theano-based RNN/LSTM with DistilBERT + token classification.
Supports incremental tagging with sliding-window context.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TransformerDisfluencyTagger:
    """
    A PyTorch Transformer-based tagger for incremental disfluency detection.
    
    Supports the same tag set as the original Theano tagger:
    - <f/> : fluent word
    - <e/> : edit term
    - <rms id="N"/> : reparandum start
    - <rm id="N"/> : mid-reparandum
    - <i id="N"/> : interregnum
    - <rps id="N"/> : repair onset
    - <rp id="N"/> : mid-repair
    - <rpn id="N"/> : repair end
    - <rpnDel id="N"/> : delete repair end
    - And utterance segmentation tags if applicable
    """
    
    # Map disfluency tags to label indices
    TAG_TO_ID = {
        '<f/>': 0,
        '<e/>': 1,
        '<rms/>': 2,
        '<rm/>': 3,
        '<i/>': 4,
        '<rps/>': 5,
        '<rp/>': 6,
        '<rpn/>': 7,
        '<rpnDel/>': 8,
        'O': 0,  # No disfluency (same as fluent)
    }
    
    ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 9,
        device: str = None,
        max_seq_length: int = 512,
        context_window: int = 10,
    ):
        """
        Initialize the transformer tagger.
        
        Args:
            model_name: HuggingFace model ID (default: distilbert-base-uncased)
            num_labels: Number of output tags (default: 9 for disfluency tags)
            device: torch device (auto-detect if None)
            max_seq_length: Maximum sequence length (BERT default: 512)
            context_window: Number of previous words to use as context
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.context_window = context_window
        
        # Load pretrained model and tokenizer
        logger.info(f"Loading {model_name} from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Incremental state
        self.word_buffer = []  # List of (word, subword_tokens, pred_tag)
        self.output_tags = []  # Predicted tags for each word
        self.state_cache = None
        
    def reset(self):
        """Reset incremental state for a new utterance."""
        self.word_buffer = []
        self.output_tags = []
        self.state_cache = None
        
    def tag_new_word(
        self,
        word: str,
        pos: Optional[str] = None,
        timing: Optional[float] = None,
        rollback: int = 0,
    ) -> str:
        """
        Tag a new incoming word incrementally.
        
        Args:
            word: The word to tag
            pos: POS tag (optional, not used in transformer baseline but kept for API compat)
            timing: Word duration (optional)
            rollback: Number of words to rollback (for ASR corrections)
            
        Returns:
            Predicted disfluency tag for the word
        """
        # Handle rollback (revise recent predictions)
        if rollback > 0:
            self.word_buffer = self.word_buffer[:-rollback]
            self.output_tags = self.output_tags[:-rollback]
        
        # Add word to buffer
        self.word_buffer.append(word)
        
        # Get context: use last context_window words + current word
        context_start = max(0, len(self.word_buffer) - self.context_window - 1)
        context_words = self.word_buffer[context_start:]
        
        # Predict tags for context (only return tag for current word)
        pred_tags = self._predict_tags_for_sequence(context_words)
        
        # The tag for the new word is the last predicted tag
        new_tag = pred_tags[-1]
        self.output_tags.append(new_tag)
        
        return new_tag
    
    def tag_new_prefix(
        self,
        prefix: List[Tuple[str, Optional[str], Optional[float]]],
        rollback: int = 0,
    ) -> List[str]:
        """
        Tag a sequence of words (e.g., entire prefix after ASR correction).
        
        Args:
            prefix: List of (word, pos, timing) tuples
            rollback: Number of words to rollback before processing
            
        Returns:
            List of predicted tags
        """
        if rollback > 0:
            self.word_buffer = self.word_buffer[:-rollback]
            self.output_tags = self.output_tags[:-rollback]
        
        tags = []
        for word, pos, timing in prefix:
            tag = self.tag_new_word(word, pos, timing, rollback=0)
            tags.append(tag)
        
        return tags
    
    def _predict_tags_for_sequence(self, words: List[str]) -> List[str]:
        """
        Predict tags for a sequence of words using the model.
        
        Args:
            words: List of words
            
        Returns:
            List of predicted tags (same length as words)
        """
        # Tokenize with word boundaries preserved
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=True,
        )
        
        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            logits = outputs.logits
        
        # Map predictions back to words (handle subword tokens)
        predictions = torch.argmax(logits, dim=2)
        
        # Convert token predictions to word predictions
        word_ids = encoding.word_ids()
        word_preds = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                if word_idx not in word_preds:
                    word_preds[word_idx] = predictions[0, token_idx].item()
        
        # Map label IDs back to tag strings
        tags = []
        for i in range(len(words)):
            label_id = word_preds.get(i, 0)
            tag = self.ID_TO_TAG.get(label_id, '<f/>')
            tags.append(tag)
        
        return tags
    
    def get_best_tag_sequence(self) -> List[str]:
        """
        Return the current best tag sequence.
        
        Returns:
            List of predicted tags
        """
        return self.output_tags.copy()
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def get_state(self) -> Dict:
        """Get current state (for debugging/analysis)."""
        return {
            'word_buffer': self.word_buffer.copy(),
            'output_tags': self.output_tags.copy(),
            'model_name': self.model_name,
            'device': str(self.device),
        }
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        logger.info(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load model and tokenizer from saved path."""
        logger.info(f"Loading model from {path}")
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()


# Alias for compatibility with existing code
DeepDisfluencyTransformerTagger = TransformerDisfluencyTagger
