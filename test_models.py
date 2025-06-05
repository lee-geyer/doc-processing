#!/usr/bin/env python3
"""
Test different embedding models for comparison.
"""

from src.core.embeddings import embedding_manager, ProviderConfig

def test_current_model():
    """Test the current sentence transformers model."""
    print("=== Current Model ===")
    provider = embedding_manager.get_provider('sentence_transformers')
    print(f"Model: {provider.model_name}")
    print(f"Dimension: {provider.dimension}")
    print(f"Provider: {provider.provider_type}")
    
    # Test embedding
    test_text = ["Policy compliance requirements for infection control procedures"]
    result = provider.embed_texts(test_text)
    print(f"Processing time: {result.processing_time_ms}ms")
    print(f"Actual dimension: {len(result.embeddings[0])}")
    print()

def test_qa_optimized_model():
    """Test Q&A optimized model."""
    print("=== Q&A Optimized Model ===")
    try:
        config = ProviderConfig(
            provider_type='sentence_transformers',
            model_name='multi-qa-mpnet-base-dot-v1',
            dimension=768,
            batch_size=32
        )
        provider = embedding_manager.get_provider('qa_model', config)
        print(f"Model: {provider.model_name}")
        print(f"Dimension: {provider.dimension}")
        
        # Test embedding
        test_text = ["Policy compliance requirements for infection control procedures"]
        result = provider.embed_texts(test_text)
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Actual dimension: {len(result.embeddings[0])}")
        print("✓ Q&A model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Q&A model: {e}")
    print()

def test_faster_model():
    """Test faster, smaller model."""
    print("=== Faster Model (MiniLM) ===")
    try:
        config = ProviderConfig(
            provider_type='sentence_transformers',
            model_name='all-MiniLM-L6-v2',
            dimension=384,
            batch_size=32
        )
        provider = embedding_manager.get_provider('fast_model', config)
        print(f"Model: {provider.model_name}")
        print(f"Dimension: {provider.dimension}")
        
        # Test embedding
        test_text = ["Policy compliance requirements for infection control procedures"]
        result = provider.embed_texts(test_text)
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Actual dimension: {len(result.embeddings[0])}")
        print("✓ Fast model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading fast model: {e}")
    print()

def compare_models():
    """Compare models on policy-relevant text."""
    print("=== Model Comparison ===")
    test_texts = [
        "Infection control hand hygiene requirements",
        "Medication administration safety protocols", 
        "Emergency response procedures and documentation"
    ]
    
    models_to_test = [
        ('current', 'sentence_transformers'),
        # Add API models if keys are available
    ]
    
    # Test if API keys exist
    try:
        from src.config.settings import settings
        if settings.openai_api_key:
            models_to_test.append(('openai_small', 'openai_small'))
        if settings.cohere_api_key:
            models_to_test.append(('cohere', 'cohere'))
    except:
        pass
    
    print(f"Testing {len(test_texts)} sample texts...")
    
    for model_name, provider_name in models_to_test:
        try:
            provider = embedding_manager.get_provider(provider_name)
            result = provider.embed_texts(test_texts)
            print(f"{model_name:12}: {result.processing_time_ms:4}ms, "
                  f"dim={result.dimension:4}, cost=${result.cost_estimate or 0:.4f}")
        except Exception as e:
            print(f"{model_name:12}: ERROR - {e}")

def list_available_models():
    """List all available models."""
    print("=== Available Models ===")
    
    print("\nSentence Transformers (Free, Local):")
    st_models = [
        "all-mpnet-base-v2 (current)",
        "multi-qa-mpnet-base-dot-v1 (Q&A optimized)",
        "all-MiniLM-L6-v2 (fast, 384-dim)",
        "all-MiniLM-L12-v2 (balanced, 384-dim)",
        "paraphrase-mpnet-base-v2 (paraphrasing)",
        "sentence-t5-base (T5-based)"
    ]
    for model in st_models:
        print(f"  • {model}")
    
    print("\nAPI Models (Premium):")
    api_models = [
        "OpenAI text-embedding-3-small (1536-dim, $0.02/1M tokens)",
        "OpenAI text-embedding-3-large (3072-dim, $0.13/1M tokens)",
        "Cohere embed-english-v3.0 (1024-dim, $0.10/1M tokens)"
    ]
    for model in api_models:
        print(f"  • {model}")

if __name__ == "__main__":
    print("🔍 Embedding Model Testing\n")
    
    # Test current model
    test_current_model()
    
    # Test alternatives
    test_qa_optimized_model()
    test_faster_model()
    
    # Compare performance
    compare_models()
    
    # List all options
    list_available_models()