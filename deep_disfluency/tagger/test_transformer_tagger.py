#!/usr/bin/env python3
"""
Test and Demo Script for Transformer-Based Disfluency Tagger

Demonstrates:
1. Loading DistilBERT transformer tagger
2. Incremental tagging of sample sentences
3. Batch processing with ASR rollback
4. Evaluation with seqeval metrics
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_transformer_tagger():
    """Test basic transformer tagger functionality."""
    logger.info("=" * 70)
    logger.info("TEST 1: Transformer Tagger Initialization")
    logger.info("=" * 70)
    
    try:
        from transformer_tagger import TransformerDisfluencyTagger
        
        logger.info("Creating TransformerDisfluencyTagger instance...")
        tagger = TransformerDisfluencyTagger(
            model_name="distilbert-base-uncased",
            num_labels=9,
            context_window=10,
            device="cpu"  # Use CPU for testing
        )
        logger.info("✓ Tagger initialized successfully")
        logger.info(f"  - Model: {tagger.model_name}")
        logger.info(f"  - Device: {tagger.device}")
        logger.info(f"  - Context window: {tagger.context_window}")
        
        return tagger
    except Exception as e:
        logger.error(f"✗ Failed to initialize tagger: {e}")
        return None


def test_incremental_pipeline(tagger):
    """Test incremental tagging pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Incremental Tagging Pipeline")
    logger.info("=" * 70)
    
    try:
        from incremental_transformer_pipeline import IncrementalTransformerPipeline
        
        pipeline = IncrementalTransformerPipeline(
            transformer_tagger=tagger,
            context_window=10,
            use_confidence_scores=True
        )
        logger.info("✓ Pipeline initialized successfully")
        
        # Sample sentence with annotated disfluencies
        # Format: (word, pos_tag, duration_seconds)
        test_sentence = [
            ("I", "PRP", 0.10),
            ("want", "VB", 0.15),
            ("to", "TO", 0.08),
            ("go", "VB", 0.12),     # Disfluency: edit
            ("uh", "UH", 0.08),     # Disfluency: reparandum
            ("buy", "VB", 0.13),    # Repair
            ("a", "DT", 0.06),
            ("ticket", "NN", 0.14),
        ]
        
        logger.info(f"\nProcessing {len(test_sentence)} words incrementally...")
        for i, (word, pos, timing) in enumerate(test_sentence, 1):
            result = pipeline.process_word(word, pos, timing)
            logger.info(
                f"  Word {i}: '{word}' (POS: {pos}) -> Tag: {result['tag']} "
                f"(conf: {result['confidence']:.1%})"
            )
        
        stats = pipeline.get_statistics()
        logger.info("\nIncremental Tagging Statistics:")
        logger.info(f"  - Total words: {stats['num_words']}")
        logger.info(f"  - Disfluencies: {stats['num_disfluencies']}")
        logger.info(f"  - Avg confidence: {stats['avg_confidence']:.1%}")
        logger.info(f"  - Speech rate: {stats['words_per_second']:.1f} wps")
        
        logger.info(f"\nFinal output:\n  {pipeline.get_final_output()}")
        
        return pipeline
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rollback(tagger):
    """Test ASR correction with rollback."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: ASR Correction with Rollback")
    logger.info("=" * 70)
    
    try:
        from incremental_transformer_pipeline import IncrementalTransformerPipeline
        
        pipeline = IncrementalTransformerPipeline(transformer_tagger=tagger)
        
        # Simulate ASR correcting last 2 words
        words_sequence = [
            ("I", None, 0.1),
            ("want", None, 0.15),
            ("to", None, 0.1),
            ("gooo", None, 0.12),      # ASR error
            ("buy", None, 0.13),       # Initial tagging
        ]
        
        logger.info("Initial transcription:")
        for word, _, _ in words_sequence:
            result = pipeline.process_word(word, None, None)
            logger.info(f"  {word} -> {result['tag']}")
        
        logger.info("\nASR correction: revise last 2 words")
        correction = [
            ("go", None, 0.12),        # Corrected word
            ("buy", None, 0.13),       # Correction
        ]
        
        results = pipeline.process_sequence(correction, rollback=2)
        logger.info("After correction:")
        for result in results:
            logger.info(f"  {result['word']} -> {result['tag']}")
        
        logger.info("✓ Rollback test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Rollback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing(tagger):
    """Test batch processing of multiple sequences."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Batch Processing")
    logger.info("=" * 70)
    
    try:
        from incremental_transformer_pipeline import IncrementalTransformerPipeline
        
        # Multiple test sentences
        test_sequences = [
            [
                ("I", None, 0.1),
                ("like", None, 0.12),
                ("apples", None, 0.15),
            ],
            [
                ("she", None, 0.08),
                ("went", None, 0.14),
                ("to", None, 0.09),
                ("the", None, 0.08),
                ("store", None, 0.13),
            ],
        ]
        
        for seq_idx, sequence in enumerate(test_sequences, 1):
            pipeline = IncrementalTransformerPipeline(transformer_tagger=tagger)
            logger.info(f"\nSequence {seq_idx}:")
            
            for word, _, _ in sequence:
                result = pipeline.process_word(word)
                logger.info(f"  {word} -> {result['tag']}")
            
            pipeline.reset()
        
        logger.info("✓ Batch processing test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load(tagger):
    """Test model saving and loading."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Model Save/Load")
    logger.info("=" * 70)
    
    try:
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            logger.info(f"Saving model to {save_path}...")
            tagger.save_model(str(save_path))
            logger.info("✓ Model saved")
            
            logger.info("Loading model from disk...")
            tagger.load_model(str(save_path))
            logger.info("✓ Model loaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("TRANSFORMER DISFLUENCY TAGGER - TEST SUITE")
    logger.info("=" * 70 + "\n")
    
    results = {}
    
    # Test 1: Initialization
    tagger = test_transformer_tagger()
    results['initialization'] = tagger is not None
    
    if tagger is None:
        logger.error("\n✗ Cannot proceed without tagger initialization")
        logger.info("\nIMPORTANT: Ensure torch and transformers are installed:")
        logger.info("  pip install torch transformers seqeval")
        return 1
    
    # Test 2: Incremental pipeline
    pipeline = test_incremental_pipeline(tagger)
    results['incremental_pipeline'] = pipeline is not None
    
    # Test 3: Rollback
    results['rollback'] = test_rollback(tagger)
    
    # Test 4: Batch processing
    results['batch_processing'] = test_batch_processing(tagger)
    
    # Test 5: Save/load
    results['save_load'] = test_model_save_load(tagger)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASSED" if passed_flag else "✗ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
