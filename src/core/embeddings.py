import time
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: List[List[float]]
    input_texts: List[str]
    model_name: str
    provider: str
    dimension: int
    processing_time_ms: int
    token_count: int
    cost_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding operation."""
    results: List[EmbeddingResult]
    total_embeddings: int
    total_tokens: int
    total_cost: float
    total_time_ms: int
    success_rate: float
    failed_batches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Configuration for embedding provider."""
    provider_type: str
    model_name: str
    api_key: Optional[str] = None
    dimension: Optional[int] = None
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
    batch_size: int = 100
    rate_limit_rpm: Optional[int] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_type = config.provider_type
        self.model_name = config.model_name
        self.dimension = config.dimension
        self.max_tokens = config.max_tokens
        self.cost_per_1k_tokens = config.cost_per_1k_tokens
        
        # Initialize provider-specific client
        self._initialize_client()
        
        # Get actual dimension if not provided
        if not self.dimension:
            self.dimension = self._get_model_dimension()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        pass
    
    @abstractmethod
    def _get_model_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        pass
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Provider-specific implementation."""
        pass
    
    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> EmbeddingResult:
        """
        Embed texts using this provider.
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                input_texts=[],
                model_name=self.model_name,
                provider=self.provider_type,
                dimension=self.dimension or 0,
                processing_time_ms=0,
                token_count=0
            )
        
        start_time = time.time()
        
        # Use provided batch size or default
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Process in batches if needed
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting if specified
                if self.config.rate_limit_rpm and len(texts) > batch_size:
                    time.sleep(60.0 / self.config.rate_limit_rpm)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Estimate token count and cost
            token_count = self._estimate_token_count(texts)
            cost_estimate = self._calculate_cost(token_count) if self.cost_per_1k_tokens else None
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                input_texts=texts,
                model_name=self.model_name,
                provider=self.provider_type,
                dimension=len(all_embeddings[0]) if all_embeddings else self.dimension or 0,
                processing_time_ms=processing_time,
                token_count=token_count,
                cost_estimate=cost_estimate,
                metadata={
                    "batch_size": batch_size,
                    "batches_processed": len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error embedding texts with {self.provider_type}: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        result = self.embed_texts([text])
        return result.embeddings[0] if result.embeddings else []
    
    def _estimate_token_count(self, texts: List[str]) -> int:
        """Estimate token count for texts."""
        # Simple estimation: ~0.75 tokens per word
        total_words = sum(len(text.split()) for text in texts)
        return int(total_words * 0.75)
    
    def _calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count."""
        if not self.cost_per_1k_tokens:
            return 0.0
        return (token_count / 1000) * self.cost_per_1k_tokens
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "batch_size": self.config.batch_size
        }


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider (local)."""
    
    def _initialize_client(self):
        """Initialize Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized Sentence Transformers model: {self.model_name}")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: uv add sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def _get_model_dimension(self) -> int:
        """Get model dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Sentence Transformers."""
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        return embeddings.tolist()


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            
            api_key = self.config.api_key or settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client for model: {self.model_name}")
        except ImportError:
            raise ImportError("openai not installed. Run: uv add openai")
    
    def _get_model_dimension(self) -> int:
        """Get model dimension based on model name."""
        # OpenAI embedding dimensions (as of 2024)
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model_name, 1536)
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI API."""
        try:
            # OpenAI has a limit on batch size
            if len(texts) > 2048:
                raise ValueError("OpenAI batch size limit is 2048 texts")
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                timeout=self.config.timeout_seconds
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class CohereProvider(EmbeddingProvider):
    """Cohere embedding provider."""
    
    def _initialize_client(self):
        """Initialize Cohere client."""
        try:
            import cohere
            
            api_key = self.config.api_key or settings.cohere_api_key
            if not api_key:
                raise ValueError("Cohere API key not provided")
            
            self.client = cohere.Client(api_key)
            logger.info(f"Initialized Cohere client for model: {self.model_name}")
        except ImportError:
            raise ImportError("cohere not installed. Run: uv add cohere")
    
    def _get_model_dimension(self) -> int:
        """Get model dimension based on model name."""
        # Cohere embedding dimensions
        dimensions = {
            "embed-english-v3.0": 1024,
            "embed-english-v2.0": 4096,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-v2.0": 768
        }
        return dimensions.get(self.model_name, 1024)
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using Cohere API."""
        try:
            # Cohere has different batch size limits
            if len(texts) > 96:
                raise ValueError("Cohere batch size limit is 96 texts")
            
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"  # For retrieval use case
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise


class EmbeddingManager:
    """Manages multiple embedding providers and their configurations."""
    
    def __init__(self):
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.default_configs = self._get_default_configs()
    
    def _get_default_configs(self) -> Dict[str, ProviderConfig]:
        """Get default configurations for all providers."""
        return {
            "sentence_transformers": ProviderConfig(
                provider_type="sentence_transformers",
                model_name="all-mpnet-base-v2",
                dimension=768,
                batch_size=32,
                cost_per_1k_tokens=0.0  # Free local model
            ),
            "openai_small": ProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
                batch_size=100,
                cost_per_1k_tokens=0.00002,  # $0.02 per 1M tokens
                rate_limit_rpm=3000
            ),
            "openai_large": ProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-large",
                dimension=3072,
                batch_size=100,
                cost_per_1k_tokens=0.00013,  # $0.13 per 1M tokens
                rate_limit_rpm=3000
            ),
            "cohere": ProviderConfig(
                provider_type="cohere",
                model_name="embed-english-v3.0",
                dimension=1024,
                batch_size=96,
                cost_per_1k_tokens=0.0001,  # $0.10 per 1M tokens
                rate_limit_rpm=1000
            )
        }
    
    def get_provider(self, provider_name: str, config: Optional[ProviderConfig] = None) -> EmbeddingProvider:
        """Get or create an embedding provider."""
        if provider_name in self.providers:
            return self.providers[provider_name]
        
        # Use provided config or default
        provider_config = config or self.default_configs.get(provider_name)
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")
        
        # Create provider based on type
        if provider_config.provider_type == "sentence_transformers":
            provider = SentenceTransformersProvider(provider_config)
        elif provider_config.provider_type == "openai":
            provider = OpenAIProvider(provider_config)
        elif provider_config.provider_type == "cohere":
            provider = CohereProvider(provider_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_config.provider_type}")
        
        self.providers[provider_name] = provider
        return provider
    
    def list_providers(self) -> List[str]:
        """List available provider names."""
        return list(self.default_configs.keys())
    
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a provider."""
        if provider_name in self.providers:
            return self.providers[provider_name].get_info()
        
        config = self.default_configs.get(provider_name)
        if config:
            return {
                "provider_type": config.provider_type,
                "model_name": config.model_name,
                "dimension": config.dimension,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "batch_size": config.batch_size,
                "status": "not_initialized"
            }
        
        raise ValueError(f"Unknown provider: {provider_name}")
    
    def compare_providers(
        self, 
        texts: List[str], 
        provider_names: Optional[List[str]] = None
    ) -> Dict[str, EmbeddingResult]:
        """Compare multiple providers on the same texts."""
        provider_names = provider_names or list(self.default_configs.keys())
        results = {}
        
        for provider_name in provider_names:
            try:
                provider = self.get_provider(provider_name)
                result = provider.embed_texts(texts)
                results[provider_name] = result
                logger.info(f"Embedded {len(texts)} texts with {provider_name} "
                           f"in {result.processing_time_ms}ms")
            except Exception as e:
                logger.error(f"Failed to embed with {provider_name}: {e}")
                results[provider_name] = None
        
        return results
    
    def batch_embed_documents(
        self, 
        texts: List[str], 
        provider_name: str,
        batch_size: Optional[int] = None
    ) -> EmbeddingBatchResult:
        """Embed large number of documents in batches."""
        provider = self.get_provider(provider_name)
        batch_size = batch_size or provider.config.batch_size
        
        start_time = time.time()
        results = []
        failed_batches = []
        total_cost = 0.0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                result = provider.embed_texts(batch_texts, batch_size=len(batch_texts))
                results.append(result)
                total_cost += result.cost_estimate or 0.0
                
                logger.info(f"Processed batch {batch_num}/{(len(texts) - 1) // batch_size + 1} "
                           f"({len(batch_texts)} texts)")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                failed_batches.append({
                    "batch_num": batch_num,
                    "start_index": i,
                    "text_count": len(batch_texts),
                    "error": str(e)
                })
        
        total_time = int((time.time() - start_time) * 1000)
        total_embeddings = sum(len(r.embeddings) for r in results)
        total_tokens = sum(r.token_count for r in results)
        success_rate = len(results) / ((len(texts) - 1) // batch_size + 1) if texts else 0.0
        
        return EmbeddingBatchResult(
            results=results,
            total_embeddings=total_embeddings,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_time_ms=total_time,
            success_rate=success_rate,
            failed_batches=failed_batches
        )


# Global embedding manager instance
embedding_manager = EmbeddingManager()


def get_embedding_provider(provider_name: str) -> EmbeddingProvider:
    """Convenience function to get an embedding provider."""
    return embedding_manager.get_provider(provider_name)