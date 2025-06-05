#!/usr/bin/env python3
"""
Switch to a different embedding model.
"""

import sys
from src.core.embeddings import embedding_manager, ProviderConfig

def switch_to_qa_model():
    """Switch to Q&A optimized model."""
    print("Switching to Q&A optimized model...")
    
    # Test the model first
    config = ProviderConfig(
        provider_type='sentence_transformers',
        model_name='multi-qa-mpnet-base-dot-v1',
        dimension=768,
        batch_size=32
    )
    
    provider = embedding_manager.get_provider('qa_model', config)
    test_result = provider.embed_texts(["Test policy document"])
    
    print(f"✓ Model loaded: {provider.model_name}")
    print(f"✓ Dimension: {len(test_result.embeddings[0])}")
    print(f"✓ Processing time: {test_result.processing_time_ms}ms")
    print()
    print("To use this model, update your vector collection:")
    print("uv run python -m src.cli index create sentence_transformers_qa")
    
if __name__ == "__main__":
    switch_to_qa_model()