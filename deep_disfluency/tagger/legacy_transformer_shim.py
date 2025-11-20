"""Legacy shim to allow DeepDisfluencyTagger to call transformer model
using the older soft_max_return_hidden_layer interface.

This implements a minimal `soft_max_return_hidden_layer` method that
returns (hidden, softmax) where softmax is a numpy array compatible
with the expected downstream decoder.
"""
from typing import List, Tuple
import numpy as np

class LegacyTransformerShim(object):
    def __init__(self, transformer_tagger):
        self.tagger = transformer_tagger

    def soft_max_return_hidden_layer(self, words_idx_batch, pos_idx_batch=None):
        """Accepts lists like the old API: words_idx_batch is a list of word-index lists.
        We'll convert words back to tokens via the tagger's tokenizer and run the model.
        Returns: (None, softmax_np) where softmax_np is shape (T, C)
        """
        # reconstruct words as a single sequence (best-effort); the transformer API
        # expects word strings; the adapter should expose `original_words_buffer` or
        # accept words directly. We'll try to use the tagger's last input buffer.
        # Fallback: join input tokens in the tokenizer if available.
        try:
            words = getattr(self.tagger, 'last_input_words', None)
            if words is None:
                # fallback: try to read from tagger internal buffer
                words = getattr(self.tagger, 'word_buffer', None)
            if words is None:
                # Last resort: create a dummy single-token input
                words = ["the"]
        except Exception:
            words = ["the"]

        # use tagger to get predictions for the sequence
        # We'll call tagger.model (TransformerDisfluencyTagger) predict routine
        preds = []
        for w in words:
            t = self.tagger.tag_new_word(w)
            if isinstance(t, list):
                preds.append(t[-1])
            else:
                preds.append(t)

        # convert predicted XML tags to indices using tagger's TAG_TO_ID mapping if available
        tag_to_id = getattr(self.tagger, 'TAG_TO_ID', None)
        if tag_to_id:
            soft = np.zeros((len(preds), len(tag_to_id)), dtype=float)
            for i, pt in enumerate(preds):
                idx = tag_to_id.get(pt, 0)
                soft[i, idx] = 1.0
        else:
            # fallback: single-class
            soft = np.ones((len(preds), 1), dtype=float)

        return None, soft

    # provide save/load hooks expected by older code
    def save(self, path):
        if hasattr(self.tagger, 'save_model'):
            return self.tagger.save_model(path)

    def load_weights_from_folder(self, model_folder):
        if hasattr(self.tagger, 'load_model'):
            return self.tagger.load_model(model_folder)
