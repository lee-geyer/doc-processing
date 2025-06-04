import uuid
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    MatchValue, SearchRequest, SearchParams, CollectionInfo,
    UpdateResult, OptimizersConfigDiff, HnswConfigDiff
)

from src.core.embeddings import EmbeddingProvider, EmbeddingResult
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorPoint:
    """Represents a vector point with metadata."""
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class SearchResult:
    """Result of vector search."""
    points: List[VectorPoint]
    query_vector: List[float]
    search_params: Dict[str, Any]
    processing_time_ms: int
    total_results: int


@dataclass
class CollectionStats:
    """Statistics for a vector collection."""
    name: str
    vectors_count: int
    dimension: int
    distance_metric: str
    index_size_mb: float
    ram_usage_mb: float
    disk_usage_mb: float
    optimization_status: str
    last_updated: datetime


class VectorStore:
    """Manages vector operations with Qdrant."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=60.0
        )
        
        # Test connection
        self._test_connection()
        
        logger.info(f"Initialized VectorStore with Qdrant at {self.url}")
    
    def _test_connection(self):
        """Test connection to Qdrant."""
        try:
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at {self.url}: {e}")
    
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance_metric: str = "cosine",
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new vector collection.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, dot, euclidean)
            optimization_config: Optional optimization settings
            
        Returns:
            True if created successfully
        """
        try:
            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidean": Distance.EUCLID
            }
            
            distance = distance_map.get(distance_metric.lower(), Distance.COSINE)
            
            # Default optimization for policy documents
            default_config = OptimizersConfigDiff(
                default_segment_number=2,
                max_segment_size=200000,
                indexing_threshold=10000,
                flush_interval_sec=30,
                max_optimization_threads=2
            )
            
            # HNSW index configuration for fast search
            hnsw_config = HnswConfigDiff(
                m=16,  # Number of bi-directional links for every new element
                ef_construct=100,  # Size of the dynamic candidate list
                full_scan_threshold=10000  # Use full scan for small collections
            )
            
            # Create collection
            result = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance,
                    hnsw_config=hnsw_config
                ),
                optimizers_config=optimization_config or default_config
            )
            
            logger.info(f"Created collection '{collection_name}' with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to collection.
        
        Args:
            collection_name: Target collection
            vectors: List of vector embeddings
            payloads: List of metadata for each vector
            ids: Optional custom IDs (generated if None)
            
        Returns:
            List of vector IDs that were added
        """
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors must match number of payloads")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        elif len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors")
        
        try:
            # Create points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Upload to Qdrant
            result = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(vectors)} vectors to collection '{collection_name}'")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding vectors to '{collection_name}': {e}")
            raise
    
    def update_vectors(
        self,
        collection_name: str,
        vector_ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Update existing vectors."""
        try:
            if vectors and payloads:
                # Update both vectors and payloads
                points = [
                    PointStruct(id=vid, vector=vec, payload=pay)
                    for vid, vec, pay in zip(vector_ids, vectors, payloads)
                ]
                self.client.upsert(collection_name=collection_name, points=points)
            elif payloads:
                # Update only payloads
                for vid, payload in zip(vector_ids, payloads):
                    self.client.set_payload(
                        collection_name=collection_name,
                        payload=payload,
                        points=[vid]
                    )
            else:
                raise ValueError("Must provide either vectors or payloads to update")
            
            logger.info(f"Updated {len(vector_ids)} vectors in collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating vectors: {e}")
            return False
    
    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=vector_ids
            )
            logger.info(f"Deleted {len(vector_ids)} vectors from collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_payload: bool = True,
        include_vector: bool = False
    ) -> SearchResult:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search in
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filters
            include_payload: Include metadata in results
            include_vector: Include vectors in results
            
        Returns:
            Search results with points and metadata
        """
        start_time = time.time()
        
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            # Perform search
            search_params = SearchParams(
                hnsw_ef=128,  # Size of the dynamic candidate list
                exact=False   # Use approximate search for speed
            )
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                search_params=search_params,
                score_threshold=score_threshold,
                with_payload=include_payload,
                with_vectors=include_vector
            )
            
            # Convert results to VectorPoint objects
            points = []
            for result in results:
                point = VectorPoint(
                    id=str(result.id),
                    vector=result.vector if include_vector else [],
                    payload=result.payload or {},
                    score=result.score
                )
                points.append(point)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                points=points,
                query_vector=query_vector,
                search_params={
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "filter_conditions": filter_conditions
                },
                processing_time_ms=processing_time,
                total_results=len(points)
            )
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}")
            raise
    
    def search_with_context(
        self,
        collection_name: str,
        query_vector: List[float],
        policy_acronym: Optional[str] = None,
        document_type: Optional[str] = None,
        section: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> SearchResult:
        """Search with policy-specific context filters."""
        filter_conditions = {}
        
        if policy_acronym:
            filter_conditions["policy_acronym"] = policy_acronym
        if document_type:
            filter_conditions["document_type"] = document_type
        if section:
            filter_conditions["section"] = section
        
        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            include_payload=True
        )
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions."""
        must_conditions = []
        
        for field, value in conditions.items():
            if isinstance(value, (str, int, float, bool)):
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
                must_conditions.append(condition)
            elif isinstance(value, list):
                # Handle list of values (OR condition)
                for val in value:
                    condition = FieldCondition(
                        key=field,
                        match=MatchValue(value=val)
                    )
                    must_conditions.append(condition)
        
        return Filter(must=must_conditions)
    
    def get_collection_stats(self, collection_name: str) -> Optional[CollectionStats]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            
            # Get additional stats
            vectors_count = info.vectors_count or 0
            
            return CollectionStats(
                name=collection_name,
                vectors_count=vectors_count,
                dimension=info.config.params.vectors.size,
                distance_metric=info.config.params.vectors.distance.name.lower(),
                index_size_mb=0.0,  # Would need more detailed API
                ram_usage_mb=0.0,   # Would need more detailed API
                disk_usage_mb=0.0,  # Would need more detailed API
                optimization_status="unknown",
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def count_vectors(self, collection_name: str) -> int:
        """Count vectors in collection."""
        try:
            info = self.client.get_collection(collection_name)
            return info.vectors_count or 0
        except Exception as e:
            logger.error(f"Error counting vectors: {e}")
            return 0
    
    def optimize_collection(self, collection_name: str) -> bool:
        """Optimize collection for better performance."""
        try:
            # Trigger optimization
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Force reindexing
                )
            )
            logger.info(f"Triggered optimization for collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
            return False
    
    def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """Create a backup snapshot of collection."""
        try:
            # Note: This would typically use Qdrant's snapshot API
            # For now, we'll implement a simple export
            logger.warning("Backup functionality would require Qdrant snapshot API")
            return False
        except Exception as e:
            logger.error(f"Error backing up collection: {e}")
            return False


class PolicyVectorStore(VectorStore):
    """Specialized vector store for policy documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collection_prefix = settings.qdrant_collection_prefix
    
    def get_collection_name(self, provider_name: str) -> str:
        """Get standardized collection name for provider."""
        return f"{self.collection_prefix}_{provider_name}"
    
    def create_policy_collection(
        self,
        provider_name: str,
        dimension: int,
        distance_metric: str = "cosine"
    ) -> bool:
        """Create collection optimized for policy documents."""
        collection_name = self.get_collection_name(provider_name)
        
        # Policy-specific optimization
        optimization_config = OptimizersConfigDiff(
            default_segment_number=2,
            max_segment_size=100000,  # Smaller segments for policy docs
            indexing_threshold=5000,
            flush_interval_sec=60,
            max_optimization_threads=1
        )
        
        return self.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
            optimization_config=optimization_config
        )
    
    def add_document_chunks(
        self,
        provider_name: str,
        chunk_data: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add document chunks with policy-specific metadata."""
        collection_name = self.get_collection_name(provider_name)
        
        # Ensure collection exists
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        # Prepare payloads with standardized metadata
        payloads = []
        for chunk in chunk_data:
            payload = {
                # Core identifiers
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_index": chunk.get("document_index"),
                
                # Content
                "text": chunk["text"][:1000],  # Limit text size in payload
                "chunk_index": chunk.get("chunk_index", 0),
                
                # Policy context
                "policy_acronym": chunk.get("policy_acronym"),
                "policy_manual": chunk.get("policy_manual"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "document_type": chunk.get("document_type"),
                
                # Source attribution
                "file_path": chunk.get("file_path"),
                "page_numbers": chunk.get("page_numbers", []),
                
                # Additional metadata
                "created_at": datetime.utcnow().isoformat(),
                "embedding_model": provider_name
            }
            payloads.append(payload)
        
        # Generate UUID-based IDs (Qdrant requirement)
        ids = [str(uuid.uuid4()) for _ in chunk_data]
        
        return self.add_vectors(collection_name, embeddings, payloads, ids)
    
    def search_policy_documents(
        self,
        provider_name: str,
        query_vector: List[float],
        policy_filter: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.6
    ) -> SearchResult:
        """Search policy documents with specialized filtering."""
        collection_name = self.get_collection_name(provider_name)
        
        filter_conditions = {}
        if policy_filter:
            filter_conditions["policy_acronym"] = policy_filter
        if document_type_filter:
            filter_conditions["document_type"] = document_type_filter
        
        return self.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            include_payload=True
        )
    
    def get_similar_chunks(
        self,
        provider_name: str,
        vector_id: str,
        limit: int = 5
    ) -> SearchResult:
        """Find chunks similar to a given chunk by vector ID."""
        collection_name = self.get_collection_name(provider_name)
        
        try:
            # Get the chunk vector
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_vectors=True
            )[0]
            
            # Search for similar vectors
            return self.search(
                collection_name=collection_name,
                query_vector=point.vector,
                limit=limit + 1,  # +1 to exclude the source chunk
                include_payload=True
            )
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            raise


# Global instances
vector_store = VectorStore()
policy_vector_store = PolicyVectorStore()


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    return vector_store


def get_policy_vector_store() -> PolicyVectorStore:
    """Get the global policy vector store instance."""
    return policy_vector_store