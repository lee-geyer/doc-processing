# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Document Processing & Vector Storage System** - Part 2 of a comprehensive policy document management and RAG system. This project monitors mirrored files from the doc-sync system, parses documents, and maintains a Qdrant vector database with contextual metadata for RAG applications.

**Mirror Directory:** `/Users/leegeyer/Library/CloudStorage/OneDrive-Extendicare(Canada)Inc/INTEGRATION FILES TARGETED`

## Production Status (June 2025)

**CRITICAL ISSUE:** 28.6% of files missing from vector database
- **245 files (28.6%) failed to parse** - no vector representation
- **612 files successfully processed** (71.4%) with vector representation
- **Immediate action required:** Implement fallback synthetic descriptions for parsing failures

## Technology Stack

- Python 3.12+ with uv for package management
- Qdrant vector database (local Docker)
- PyMuPDF (fitz) and python-docx for document parsing
- Sentence Transformers for embeddings (with OpenAI/Cohere options)
- PostgreSQL for metadata storage
- FastAPI for web interface, Typer for CLI
- Watchdog for real-time file monitoring

## Key Components

### File Monitoring and Processing Pipeline

- **Mirror monitoring:** Watchdog-based real-time file change detection
- **Document parsing:** PyMuPDF for PDFs, python-docx for Word documents
- **Text cleaning:** Remove disclaimer patterns and noise text
- **Contextual chunking:** Token-based with rich metadata (1000 tokens, 200 overlap)
- **Vector sync:** Real-time Qdrant updates with policy-aware metadata

### Common Text Noise Patterns

Remove these repetitive elements during processing:
- `*This document is uncontrolled when printed.*`
- `*Extendicare (Canada) Inc. will provide, on request...*`
- `Helping people live better`
- `Page X of Y`, `Effective Date: MM/DD/YYYY`

### Synthetic Description System

Generates searchable descriptions for forms/templates with no extractable text:
- **Auto-detects:** checklist, template, form, audit, inventory, poster
- **Creates descriptions:** Policy-aware natural language descriptions with keywords
- **Ensures coverage:** 100% document searchability including empty forms

## Essential CLI Commands

```bash
# Setup and database
uv run alembic upgrade head
docker-compose up -d qdrant

# File monitoring and processing  
uv run python -m src.cli monitor start    # Start file monitoring
uv run python -m src.cli process batch    # Process all pending files

# Vector database operations
uv run python -m src.cli index sync       # Sync vectors with documents
uv run python -m src.cli index search <query>  # Test search

# Development server
uv run uvicorn src.api.main:app --reload --port 8001
```

## Important Implementation Notes

### Critical Issue to Fix
- **28.6% parsing failures:** Need fallback synthetic descriptions for failed files
- **Processing pipeline gap:** Parse failures occur before synthetic description system
- **Priority fix:** Implement try/catch around parsing with synthetic content generation

### Architecture Notes
- **Read-only mirror access:** Never modify source OneDrive files  
- **Real-time sync:** Watchdog monitoring with vector database updates
- **Policy-aware metadata:** Each vector includes manual, section, document type
- **Robust error handling:** Graceful failure handling with detailed logging

## Environment Configuration

Key environment variables:
```env
DATABASE_URL=postgresql://user:password@localhost/doc_processing
MIRROR_DIRECTORY=/Users/leegeyer/Library/CloudStorage/OneDrive-Extendicare(Canada)Inc/INTEGRATION FILES TARGETED
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_key_here  # Optional
COHERE_API_KEY=your_key_here  # Optional
```

## Contact and Support

- **Developer**: Lee Geyer - lee.geyer@geyerconsulting.com
- **Integration**: Builds on doc-sync system for file discovery and targeting