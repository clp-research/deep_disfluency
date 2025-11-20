"""
Quick test to verify that DeepDisfluencyTagger can accept a Transformer tagger
as its backend and tag words incrementally.

This test is intentionally lightweight and does not require a full experiment setup.
"""

import sys
sys.path.insert(0, '/Users/aida/Desktop/code/deep_disfluency')

import torch
from deep_disfluency.tagger.transformer_tagger import TransformerDisfluencyTagger
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger

def test_deep_tagger_with_transformer():
    """Test that DeepDisfluencyTagger accepts and uses a Transformer tagger."""
    print("=" * 60)
    print("TEST: DeepDisfluencyTagger with Transformer Backend")
    print("=" * 60)
    
    # Create a Transformer tagger
    print("\n1. Initializing Transformer tagger...")
    try:
        transformer_tagger = TransformerDisfluencyTagger(
            model_name="distilbert-base-uncased",
            num_labels=9,
            device="cpu",
            context_window=5
        )
        print("   ✓ Transformer tagger created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create transformer tagger: {e}")
        return False
    
    # Create DeepDisfluencyTagger with the Transformer backend
    print("\n2. Initializing DeepDisfluencyTagger with Transformer backend...")
    try:
        deep_tagger = DeepDisfluencyTagger(
            config_file=None,
            config_number=None,
            saved_model_dir=None,
            use_decoder=False,  # Disable decoder for simplicity in this test
            transformer_tagger=transformer_tagger
        )
        print("   ✓ DeepDisfluencyTagger initialized with Transformer backend")
        print(f"   Model type: {deep_tagger.model_type}")
    except Exception as e:
        print(f"   ✗ Failed to initialize deep_tagger with transformer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test incremental tagging
    print("\n3. Testing incremental tagging with transformer backend...")
    test_words = ["the", "cat", "is", "on", "the", "mat"]
    try:
        deep_tagger.reset()
        tags = []
        for i, word in enumerate(test_words):
            result = deep_tagger.tag_new_word(word)
            if isinstance(result, list):
                tag = result[-1] if result else '<f/>'
            else:
                tag = result
            tags.append(tag)
            print(f"   Word {i+1}: '{word}' → {tag}")
        
        print(f"\n   ✓ Tagged {len(test_words)} words successfully")
        print(f"   Output tags: {tags}")
    except Exception as e:
        print(f"   ✗ Failed during incremental tagging: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_deep_tagger_with_transformer()
    sys.exit(0 if success else 1)
