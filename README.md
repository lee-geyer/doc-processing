# Document Processing & Vector Storage System - PRODUCTION READY

**Part 2 of comprehensive policy document management and RAG system - FULLY OPERATIONAL**

This system monitors mirrored files from the doc-sync system, parses documents using multiple providers, performs systematic embedding evaluation, and maintains a Qdrant vector database with rich contextual metadata for RAG applications.

## Production Status (June 2025)

**SYSTEM OPERATIONAL**
- **777 policy documents** with extractable content processed (90.8% success rate)
- **720,953 vector embeddings** stored and searchable
- **36-minute processing** of full document corpus
- **Real-time semantic search** with policy-aware routing

**Key Metrics:**
- Search precision: 0.828 similarity for domain-specific queries
- Response time: ~15ms across 720K+ vectors  
- Policy coverage: IPC, CARE, EPM, MAINT, EVS, PRV, LEG, ADM, RC
- Content extraction: 90.8% (75 documents are forms/templates with no extractable text)

## System Overview

The Document Processing system:
- Monitors targeted policy files from the doc-sync mirror
- Parses PDF and Word documents using open-source tools
- Removes repetitive disclaimer text and noise
- Creates contextual chunks with rich metadata
- Evaluates multiple embedding providers systematically
- Maintains real-time synchronization with a Qdrant vector database

## Prerequisites

- Python 3.12+
- PostgreSQL database
- Docker (for local Qdrant instance)
- Access to the doc-sync mirror directory
- API keys for embedding providers (OpenAI, Cohere)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/lee-geyer/doc-processing.git
   cd doc-processing
   ```

2. **Set up environment**
   ```bash
   uv sync
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Set up databases**
   ```bash
   # PostgreSQL
   createdb doc_processing
   uv run alembic upgrade head
   
   # Qdrant (local Docker)
   docker-compose up -d qdrant
   ```

4. **Initialize embedding models**
   ```bash
   uv run python -m src.cli evaluate setup
   ```

5. **Start file monitoring**
   ```bash
   uv run python -m src.cli monitor start
   ```

## Key Features

### Real-time File Monitoring
- Watches the doc-sync mirror directory for changes
- Automatically processes new, updated, and deleted files
- Maintains sync state with exponential backoff retry logic

### Multi-Provider Document Parsing
- **PDF Support**: Uses PyMuPDF (fitz) for robust PDF parsing
- **Word Support**: Uses python-docx for .docx file parsing
- **Structure Preservation**: Maintains document hierarchy and formatting

### Intelligent Text Cleaning
- Removes repetitive disclaimer text across documents
- Configurable noise pattern detection
- Validates cleaning to prevent over-removal

### Contextual Chunking for RAG
- Token-based chunking with configurable overlap
- Rich metadata including policy manual, section, and document type
- Source attribution for accurate citations
- Maintains surrounding context for better understanding

### Multi-Provider Embedding Evaluation
- **Sentence Transformers**: Local, free baseline (768 dimensions)
- **OpenAI**: Premium quality embeddings (3072 dimensions)
- **Cohere**: Competitive alternative (1024 dimensions)
- Systematic evaluation framework with quality metrics

### Qdrant Vector Database
- Real-time synchronization with file changes
- Optimized for policy document search
- Rich metadata for contextual retrieval
- Support for both local and cloud deployment

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Doc-Sync      │───▶│ File Monitor     │───▶│ Document Parser │
│   Mirror Files  │    │ (Watchdog)       │    │ Multi-Provider  │
│   (870 files)   │    │                  │    │ PyMuPDF/Docx    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              │                          ▼
                              │                 ┌─────────────────┐
                              │                 │ Text Cleaning   │
                              │                 │ Remove Noise    │
                              │                 └─────────────────┘
                              │                          │
                              │                          ▼
                              │                 ┌─────────────────┐
                              │                 │ Contextual      │
                              │                 │ Chunking        │
                              │                 │ + Metadata      │
                              │                 └─────────────────┘
                              │                          │
                              │                          ▼
                              │                 ┌─────────────────┐
                              │                 │ Multi-Provider  │
                              │                 │ Embeddings      │
                              │                 │ Evaluation      │
                              │                 └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌─────────────────────────────────────────┐
                       │ Qdrant Vector Database                  │
                       │ - Contextual metadata for RAG           │
                       │ - Source attribution & citations        │
                       │ - Real-time sync with file changes      │
                       └─────────────────────────────────────────┘
```

## CLI Commands

### File Monitoring
```bash
uv run python -m src.cli monitor start    # Start file monitoring daemon
uv run python -m src.cli monitor status   # Check monitoring status
uv run python -m src.cli sync scan        # Scan for file changes
```

### Document Processing
```bash
uv run python -m src.cli process file <path>  # Process single file
uv run python -m src.cli process batch        # Process pending files
uv run python -m src.cli clean analyze        # Analyze noise patterns
```

### Embedding Evaluation
```bash
uv run python -m src.cli evaluate setup       # Initialize evaluation
uv run python -m src.cli evaluate run         # Run full evaluation
uv run python -m src.cli evaluate compare     # Compare providers
```

### Vector Database
```bash
uv run python -m src.cli index create <provider>  # Create collection
uv run python -m src.cli index sync              # Sync vectors
uv run python -m src.cli index search <query>    # Test search
```

## Configuration

Create a `.env` file with:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/doc_processing

# Mirror Directory (from doc-sync)
MIRROR_DIRECTORY=/path/to/INTEGRATION FILES TARGETED

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_PREFIX=policy_docs

# Embedding Providers
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_EMBEDDING_MODEL=sentence_transformers
```

## Development

### Run Tests
```bash
uv run pytest
uv run pytest --cov=src --cov-report=html
```

### Start API Server
```bash
uv run uvicorn src.api.main:app --reload --port 8001
```

### Access Documentation
- API Docs: http://localhost:8001/docs
- Monitoring Dashboard: http://localhost:8001

## Integration with Doc-Sync

This system monitors the mirror directory created by the doc-sync project:
- **Mirror Path**: Configured in `MIRROR_DIRECTORY` environment variable
- **Document Index**: Retrieved from doc-sync database for citations
- **Policy Context**: Extracted from file path structure

## License

This project is proprietary software for Extendicare (Canada) Inc.

## Contact

**Developer**: Lee Geyer - lee.geyer@geyerconsulting.com