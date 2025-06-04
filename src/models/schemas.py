from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class FileSyncStateBase(BaseModel):
    file_path: str
    file_hash: str
    file_size_bytes: int
    last_modified: datetime
    sync_status: str = 'pending'


class FileSyncStateCreate(FileSyncStateBase):
    pass


class FileSyncStateUpdate(BaseModel):
    sync_status: Optional[str] = None
    last_sync_at: Optional[datetime] = None
    sync_error_count: Optional[int] = None
    last_error_message: Optional[str] = None


class FileSyncState(FileSyncStateBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    last_sync_at: Optional[datetime] = None
    sync_error_count: int = 0
    last_error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentBase(BaseModel):
    file_path: str
    document_index: Optional[str] = None
    file_hash: str
    original_filename: str
    file_type: str
    file_size_bytes: int


class DocumentCreate(DocumentBase):
    parsing_method: str
    parsing_success: bool
    parsing_error_message: Optional[str] = None
    markdown_content: Optional[str] = None
    cleaned_content: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    document_metadata: Optional[Dict[str, Any]] = None
    context_hierarchy: Optional[Dict[str, Any]] = None


class Document(DocumentBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    parsing_method: str
    parsing_success: bool
    parsing_error_message: Optional[str] = None
    markdown_content: Optional[str] = None
    cleaned_content: Optional[str] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    document_metadata: Optional[Dict[str, Any]] = None
    context_hierarchy: Optional[Dict[str, Any]] = None
    total_chunks: int = 0
    total_tokens: int = 0
    processing_duration_ms: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class ChunkBase(BaseModel):
    chunk_text: str
    chunk_markdown: Optional[str] = None
    chunk_tokens: int
    chunk_hash: str
    page_numbers: Optional[List[int]] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    context_metadata: Dict[str, Any]
    surrounding_context: Optional[str] = None
    document_position: Optional[float] = None


class ChunkCreate(ChunkBase):
    document_id: int
    chunk_index: int


class DocumentChunk(ChunkBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    document_id: int
    chunk_index: int
    vector_id: Optional[UUID] = None
    embedding_model_id: Optional[int] = None
    vector_collection_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class EmbeddingModelBase(BaseModel):
    provider: str
    model_name: str
    dimension: int
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
    is_active: bool = True


class EmbeddingModelCreate(EmbeddingModelBase):
    pass


class EmbeddingModel(EmbeddingModelBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    avg_retrieval_accuracy: Optional[float] = None
    avg_processing_speed_ms: Optional[float] = None
    total_cost_usd: float = 0
    created_at: datetime


class VectorCollectionBase(BaseModel):
    collection_name: str
    embedding_model_id: int
    dimension: int
    distance_metric: str = 'cosine'
    index_config: Optional[Dict[str, Any]] = None


class VectorCollectionCreate(VectorCollectionBase):
    pass


class VectorCollection(VectorCollectionBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    total_vectors: int = 0
    last_sync_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class VectorSyncOperationBase(BaseModel):
    operation_type: str
    file_path: Optional[str] = None
    document_id: Optional[int] = None
    chunks_affected: int = 0
    vector_ids: Optional[List[UUID]] = None
    collection_id: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0


class VectorSyncOperationCreate(VectorSyncOperationBase):
    pass


class VectorSyncOperation(VectorSyncOperationBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    created_at: datetime


class EmbeddingEvaluationBase(BaseModel):
    evaluation_name: str
    embedding_model_id: int
    test_query_count: int
    evaluation_date: datetime


class EmbeddingEvaluationCreate(EmbeddingEvaluationBase):
    precision_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    avg_response_time_ms: Optional[float] = None
    semantic_clustering_score: Optional[float] = None
    cross_document_score: Optional[float] = None
    total_cost_usd: Optional[float] = None
    cost_per_query: Optional[float] = None
    detailed_results: Optional[Dict[str, Any]] = None


class EmbeddingEvaluation(EmbeddingEvaluationBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    precision_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    avg_response_time_ms: Optional[float] = None
    semantic_clustering_score: Optional[float] = None
    cross_document_score: Optional[float] = None
    total_cost_usd: Optional[float] = None
    cost_per_query: Optional[float] = None
    detailed_results: Optional[Dict[str, Any]] = None
    created_at: datetime


# Response models for API
class FileMonitorStatus(BaseModel):
    is_running: bool
    monitored_directory: str
    total_files: int
    pending_files: int
    processing_files: int
    synced_files: int
    error_files: int
    last_scan: Optional[datetime] = None


class ProcessingStats(BaseModel):
    total_documents: int
    successfully_parsed: int
    failed_parsing: int
    total_chunks: int
    avg_chunks_per_document: float
    total_tokens: int
    avg_tokens_per_chunk: float