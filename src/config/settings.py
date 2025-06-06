from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database Configuration
    database_url: PostgresDsn = Field(
        default="postgresql://user:password@localhost/doc_processing",
        description="PostgreSQL database URL"
    )
    
    # Mirror Directory (from doc-sync project)
    mirror_directory: str = Field(
        default="/Users/leegeyer/Library/CloudStorage/OneDrive-Extendicare(Canada)Inc/INTEGRATION FILES TARGETED",
        description="Path to the doc-sync mirror directory"
    )
    
    # Doc-Sync Database Integration
    doc_sync_database_url: Optional[PostgresDsn] = Field(
        default=None,
        description="Doc-sync database URL for document index integration"
    )
    
    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (optional for local instance)"
    )
    qdrant_collection_prefix: str = Field(
        default="policy_docs",
        description="Prefix for Qdrant collections"
    )
    
    # Embedding Providers
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for embeddings"
    )
    cohere_api_key: Optional[str] = Field(
        default=None,
        description="Cohere API key for embeddings"
    )
    
    # Processing Configuration
    max_concurrent_files: int = Field(
        default=5,
        description="Maximum number of files to process concurrently"
    )
    
    # Chunking Configuration (Optimized for sentence transformers 512 token limit)
    chunk_size: int = Field(
        default=400,
        description="Default chunk size in tokens (conservative limit for 512 token model)"
    )
    chunk_overlap: int = Field(
        default=60,
        description="Default chunk overlap in tokens (reduced redundancy)"
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in tokens"
    )
    max_chunk_size: int = Field(
        default=2000,
        description="Maximum chunk size in tokens"
    )
    respect_sentence_boundaries: bool = Field(
        default=True,
        description="Avoid splitting sentences when chunking"
    )
    respect_paragraph_boundaries: bool = Field(
        default=True,
        description="Prefer to break chunks at paragraph boundaries"
    )
    preserve_table_structure: bool = Field(
        default=True,
        description="Keep tables intact in single chunks when possible"
    )
    include_surrounding_context: bool = Field(
        default=True,
        description="Include context from adjacent chunks"
    )
    surrounding_context_tokens: int = Field(
        default=50,
        description="Number of tokens of surrounding context to include"
    )
    
    # Metadata Configuration
    extract_section_hierarchy: bool = Field(
        default=True,
        description="Extract section and subsection titles for chunks"
    )
    include_page_numbers: bool = Field(
        default=True,
        description="Track page numbers for each chunk"
    )
    generate_chunk_summaries: bool = Field(
        default=False,
        description="Generate AI summaries for each chunk"
    )
    
    default_embedding_model: str = Field(
        default="sentence_transformers",
        description="Default embedding provider"
    )
    
    # File Monitoring
    monitor_poll_interval: float = Field(
        default=1.0,
        description="File monitoring poll interval in seconds"
    )
    batch_processing_size: int = Field(
        default=10,
        description="Number of files to process in a batch"
    )
    retry_max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        description="Exponential backoff factor for retries"
    )
    
    # Text Cleaning
    enable_disclaimer_removal: bool = Field(
        default=True,
        description="Enable automatic disclaimer text removal"
    )
    enable_noise_detection: bool = Field(
        default=True,
        description="Enable frequency-based noise detection"
    )
    noise_frequency_threshold: float = Field(
        default=0.8,
        description="Remove text appearing in more than this fraction of documents"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="logs/doc_processing.log",
        description="Log file path"
    )
    
    # Development/Testing
    pytest_args: str = Field(
        default="-v --tb=short",
        description="Default pytest arguments"
    )
    jupyter_port: int = Field(
        default=8888,
        description="Jupyter notebook server port"
    )


# Global settings instance
settings = Settings()