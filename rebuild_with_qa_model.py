#!/usr/bin/env python3
"""
Rebuild vector database with Q&A optimized embedding model.
"""

from src.core.vector_store import PolicyVectorStore
from src.core.embeddings import embedding_manager, ProviderConfig
from src.core.sync_manager import SyncManager

def create_qa_collection():
    """Create new collection with Q&A optimized model."""
    print("🚀 Creating new collection with Q&A optimized model...")
    
    # Configure Q&A optimized model
    qa_config = ProviderConfig(
        provider_type='sentence_transformers',
        model_name='multi-qa-mpnet-base-dot-v1',
        dimension=768,
        batch_size=32,
        cost_per_1k_tokens=0.0
    )
    
    # Get the embedding provider
    embedding_provider = embedding_manager.get_provider('qa_model', qa_config)
    print(f"✓ Loaded embedding model: {embedding_provider.model_name}")
    print(f"✓ Model dimension: {embedding_provider.dimension}")
    
    # Create vector store
    vector_store = PolicyVectorStore()
    
    # Create collection for Q&A model
    collection_name = "policy_docs_qa_optimized"
    success = vector_store.create_collection(
        collection_name=collection_name,
        dimension=embedding_provider.dimension,
        distance_metric="cosine"
    )
    
    if success:
        print(f"✓ Created new Qdrant collection: {collection_name}")
    else:
        print(f"⚠ Collection {collection_name} already exists")
    
    return vector_store, embedding_provider, collection_name

def start_processing():
    """Start processing all files with the new model."""
    print("\n📚 Starting document processing with Q&A model...")
    
    # Create the collection first
    vector_store, embedding_provider, collection_name = create_qa_collection()
    
    print("✓ Q&A optimized collection ready")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Model: {embedding_provider.model_name}")
    print(f"✓ Ready to process 857 documents")
    
    return vector_store, embedding_provider, collection_name

if __name__ == "__main__":
    try:
        vector_store, embedding_provider, collection_name = start_processing()
        print("\n🎯 Next steps:")
        print("1. Start processing: uv run python process_remaining.py")
        print("2. Monitor progress: uv run python monitor_completion.py")
        print("3. The new model is optimized for Q&A retrieval!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise