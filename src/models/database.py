from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, ARRAY, JSON, BigInteger, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.config.settings import settings

# Create database engine
engine = create_engine(str(settings.database_url))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class FileSyncState(Base):
    __tablename__ = "file_sync_state"
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False, unique=True)
    file_hash = Column(String(64), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    last_modified = Column(DateTime, nullable=False)
    sync_status = Column(String(20), nullable=False, default='pending')  # pending, processing, synced, error
    last_sync_at = Column(DateTime)
    sync_error_count = Column(Integer, default=0)
    last_error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_file_sync_state_path', 'file_path'),
        Index('idx_file_sync_state_status', 'sync_status'),
        Index('idx_file_sync_state_modified', 'last_modified'),
    )


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False, unique=True)
    document_index = Column(String(100))  # From doc-sync system
    file_hash = Column(String(64), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, doc
    file_size_bytes = Column(BigInteger, nullable=False)
    
    # Parsing information
    parsing_method = Column(String(50), nullable=False)  # pymupdf, python_docx, etc.
    parsing_success = Column(Boolean, nullable=False)
    parsing_error_message = Column(Text)
    
    # Document content and structure
    markdown_content = Column(Text)  # Full parsed markdown
    cleaned_content = Column(Text)   # After noise removal
    word_count = Column(Integer)
    page_count = Column(Integer)
    
    # Document metadata from parsing
    document_metadata = Column(JSON)  # Title, author, creation date, etc.
    context_hierarchy = Column(JSON)  # Policy manual > section structure
    
    # Processing statistics
    total_chunks = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    processing_duration_ms = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_path', 'file_path'),
        Index('idx_documents_hash', 'file_hash'),
        Index('idx_documents_index', 'document_index'),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position within document
    
    # Chunk content
    chunk_text = Column(Text, nullable=False)  # Clean text for embedding
    chunk_markdown = Column(Text)               # Original markdown with formatting
    chunk_tokens = Column(Integer, nullable=False)
    chunk_hash = Column(String(64), nullable=False)  # For change detection
    
    # Source attribution for citations
    page_numbers = Column(ARRAY(Integer))  # Array of page numbers
    section_title = Column(String(255))    # Immediate section heading
    subsection_title = Column(String(255)) # Subsection if applicable
    
    # Contextual metadata for RAG
    context_metadata = Column(JSON, nullable=False)  # Policy manual, section, document type, etc.
    surrounding_context = Column(Text)                # Text before and after for context
    document_position = Column(Float)                 # Relative position (0.0 to 1.0)
    
    # Vector storage references
    vector_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
    embedding_model_id = Column(Integer, ForeignKey('embedding_models.id'))
    vector_collection_id = Column(Integer, ForeignKey('vector_collections.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embedding_model = relationship("EmbeddingModel")
    vector_collection = relationship("VectorCollection")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('document_id', 'chunk_index'),
        Index('idx_chunks_document', 'document_id'),
        Index('idx_chunks_vector', 'vector_id'),
        Index('idx_chunks_hash', 'chunk_hash'),
    )


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(50), nullable=False)      # sentence_transformers, openai, cohere
    model_name = Column(String(100), nullable=False)   # Model identifier
    dimension = Column(Integer, nullable=False)        # Vector dimension
    max_tokens = Column(Integer)                        # Maximum input tokens
    cost_per_1k_tokens = Column(Float)                  # Cost tracking
    is_active = Column(Boolean, default=True)
    
    # Evaluation metrics
    avg_retrieval_accuracy = Column(Float)
    avg_processing_speed_ms = Column(Float)
    total_cost_usd = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('provider', 'model_name'),
    )


class VectorCollection(Base):
    __tablename__ = "vector_collections"
    
    id = Column(Integer, primary_key=True)
    collection_name = Column(String(100), nullable=False, unique=True)
    embedding_model_id = Column(Integer, ForeignKey('embedding_models.id'), nullable=False)
    dimension = Column(Integer, nullable=False)
    
    # Collection statistics
    total_vectors = Column(Integer, default=0)
    last_sync_at = Column(DateTime)
    
    # Configuration
    distance_metric = Column(String(20), default='cosine')  # cosine, dot, euclidean
    index_config = Column(JSON)                              # Qdrant index configuration
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    embedding_model = relationship("EmbeddingModel")


class VectorSyncOperation(Base):
    __tablename__ = "vector_sync_operations"
    
    id = Column(Integer, primary_key=True)
    operation_type = Column(String(20), nullable=False)  # add, update, delete, batch_update
    file_path = Column(String(500))
    document_id = Column(Integer, ForeignKey('documents.id'))
    
    # Operation details
    chunks_affected = Column(Integer, default=0)
    vector_ids = Column(ARRAY(UUID(as_uuid=True)))  # Qdrant point IDs affected
    collection_id = Column(Integer, ForeignKey('vector_collections.id'))
    
    # Operation results
    status = Column(String(20), nullable=False)  # success, failed, partial
    error_message = Column(Text)
    execution_time_ms = Column(Integer)
    retry_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document")
    collection = relationship("VectorCollection")
    
    # Indexes
    __table_args__ = (
        Index('idx_sync_ops_status', 'status'),
        Index('idx_sync_ops_created', 'created_at'),
    )


class EmbeddingEvaluation(Base):
    __tablename__ = "embedding_evaluations"
    
    id = Column(Integer, primary_key=True)
    evaluation_name = Column(String(100), nullable=False)
    embedding_model_id = Column(Integer, ForeignKey('embedding_models.id'), nullable=False)
    
    # Test configuration
    test_query_count = Column(Integer, nullable=False)
    evaluation_date = Column(DateTime, nullable=False)
    
    # Evaluation metrics
    precision_at_5 = Column(Float)
    recall_at_10 = Column(Float)
    avg_response_time_ms = Column(Float)
    semantic_clustering_score = Column(Float)
    cross_document_score = Column(Float)
    
    # Cost analysis
    total_cost_usd = Column(Float)
    cost_per_query = Column(Float)
    
    # Detailed results
    detailed_results = Column(JSON)  # Full evaluation data
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    embedding_model = relationship("EmbeddingModel")


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()