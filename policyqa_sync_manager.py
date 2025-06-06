#!/usr/bin/env python3
"""
PolicyQA Sync Manager - Updated for new naming convention.
"""

import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
import uuid

from src.config.settings import settings
from src.models.database import SessionLocal, FileSyncState, Document, DocumentChunk, VectorSyncOperation
from src.core.document_parser import DocumentParser, DocumentParsingError
from src.core.text_cleaner import TextCleaner
from src.core.chunking import DocumentChunker
from src.core.embeddings import embedding_manager, ProviderConfig
from src.core.vector_store import PolicyVectorStore
from src.utils.context_utils import ContextExtractor
from src.utils.file_utils import get_file_info, calculate_file_hash
from src.utils.logging import get_logger
from collection_manager import PolicyQACollectionManager

logger = get_logger(__name__)


class PolicyQASyncManager:
    """
    Sync manager for policyQA collections with new naming convention.
    """
    
    def __init__(self, model_key: str = "dense_qa768", max_workers: Optional[int] = None):
        self.max_workers = max_workers or settings.max_concurrent_files
        self.batch_size = settings.batch_processing_size
        self.model_key = model_key
        
        # Initialize collection manager
        self.collection_manager = PolicyQACollectionManager()
        
        if model_key not in self.collection_manager.models:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.collection_manager.models.keys())}")
        
        self.model_config = self.collection_manager.models[model_key]
        self.collection_name = self.model_config.name
        
        # Initialize processing components
        self.document_parser = DocumentParser()
        self.text_cleaner = TextCleaner()
        self.document_chunker = DocumentChunker()
        self.context_extractor = ContextExtractor()
        
        # Initialize embedding provider
        provider_config = ProviderConfig(
            provider_type=self.model_config.provider_type,
            model_name=self.model_config.model_name,
            dimension=self.model_config.dimension,
            batch_size=self.model_config.batch_size,
            cost_per_1k_tokens=self.model_config.cost_per_1k_tokens
        )
        
        self.embedding_provider = embedding_manager.get_provider(f"policyqa_{model_key}", provider_config)
        self.vector_store = PolicyVectorStore()
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"Initialized PolicyQA Sync Manager")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Model: {self.model_config.model_name}")
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        if not self.vector_store.collection_exists(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")
            success = self.vector_store.create_collection(
                collection_name=self.collection_name,
                dimension=self.model_config.dimension,
                distance_metric="cosine"
            )
            if not success:
                raise RuntimeError(f"Failed to create collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")
    
    def process_pending_files(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Process pending files in batches.
        
        Args:
            limit: Maximum number of files to process
            
        Returns:
            Dictionary with processing statistics
        """
        db = SessionLocal()
        stats = {
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'skipped': 0
        }
        
        try:
            # Get pending files
            query = db.query(FileSyncState).filter_by(sync_status='pending')
            if limit:
                query = query.limit(limit)
            
            pending_files = query.all()
            
            if not pending_files:
                return stats
            
            logger.info(f"Processing {len(pending_files)} pending files with {self.model_config.model_name}")
            
            # Process files concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file, file_state): file_state
                    for file_state in pending_files
                }
                
                for future in as_completed(future_to_file):
                    file_state = future_to_file[future]
                    try:
                        result = future.result()
                        stats['processed'] += 1
                        
                        if result['success']:
                            stats['succeeded'] += 1
                        else:
                            stats['failed'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_state.file_path}: {e}")
                        stats['failed'] += 1
                        stats['processed'] += 1
            
            return stats
            
        finally:
            db.close()
    
    def _process_single_file(self, file_state: FileSyncState) -> Dict[str, Any]:
        """Process a single file."""
        start_time = time.time()
        file_path = file_state.file_path
        
        db = SessionLocal()
        try:
            # Mark as processing
            file_state.sync_status = 'processing'
            db.merge(file_state)
            db.commit()
            
            # Parse document
            try:
                parsed_doc = self.document_parser.parse_document(file_path)
                if not parsed_doc.parsing_success:
                    raise DocumentParsingError(parsed_doc.parsing_error_message or "Unknown parsing error")
            except Exception as e:
                return self._handle_error(db, file_state, f"Parsing failed: {e}")
            
            # Clean text
            try:
                cleaning_result = self.text_cleaner.clean_text(parsed_doc.raw_text or "")
                cleaned_text = cleaning_result['cleaned_text']
            except Exception as e:
                return self._handle_error(db, file_state, f"Text cleaning failed: {e}")
            
            # Extract context
            try:
                context = self.context_extractor.extract_context(file_path)
            except Exception as e:
                return self._handle_error(db, file_state, f"Context extraction failed: {e}")
            
            # Create/update document record
            try:
                document = self._create_or_update_document(db, file_state, parsed_doc, cleaned_text, context)
            except Exception as e:
                return self._handle_error(db, file_state, f"Document creation failed: {e}")
            
            # Chunk document
            try:
                chunking_result = self.document_chunker.chunk_document(
                    text=cleaned_text,
                    markdown=parsed_doc.markdown_content or cleaned_text,
                    document_context=context,
                    file_size_bytes=file_state.file_size_bytes
                )
                chunks = chunking_result.chunks
            except Exception as e:
                return self._handle_error(db, file_state, f"Chunking failed: {e}")
            
            # Generate embeddings and store vectors
            try:
                vector_count = self._process_chunks(db, document, chunks)
            except Exception as e:
                return self._handle_error(db, file_state, f"Vector processing failed: {e}")
            
            # Mark as successful
            processing_time = int((time.time() - start_time) * 1000)
            file_state.sync_status = 'synced'
            file_state.last_sync_at = datetime.utcnow()
            file_state.sync_error_count = 0
            file_state.last_error_message = None
            
            db.merge(file_state)
            
            # Record sync operation
            sync_op = VectorSyncOperation(
                operation_type='add',
                file_path=file_path,
                document_id=document.id,
                chunks_affected=vector_count,
                status='success',
                execution_time_ms=processing_time
            )
            db.add(sync_op)
            db.commit()
            
            logger.info(f"Successfully processed {file_path}: {vector_count} vectors created")
            return {
                'success': True,
                'vectors_created': vector_count,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            return self._handle_error(db, file_state, f"Unexpected error: {e}")
        finally:
            db.close()
    
    def _handle_error(self, db: Session, file_state: FileSyncState, error_message: str) -> Dict[str, Any]:
        """Handle processing error."""
        file_state.sync_status = 'error'
        file_state.sync_error_count += 1
        file_state.last_error_message = error_message
        
        db.merge(file_state)
        
        # Record failed sync operation
        sync_op = VectorSyncOperation(
            operation_type='add',
            file_path=file_state.file_path,
            chunks_affected=0,
            status='failed',
            error_message=error_message,
            execution_time_ms=0
        )
        db.add(sync_op)
        db.commit()
        
        logger.error(f"Failed to process {file_state.file_path}: {error_message}")
        return {
            'success': False,
            'error': error_message
        }
    
    def _create_or_update_document(self, db: Session, file_state: FileSyncState, parsed_doc, cleaned_text: str, context) -> Document:
        """Create or update document record."""
        # Check if document exists
        existing_doc = db.query(Document).filter_by(file_path=file_state.file_path).first()
        
        if existing_doc:
            # Update existing document
            document = existing_doc
        else:
            # Create new document
            document = Document()
            document.file_path = file_state.file_path
        
        # Update document fields
        document.document_index = context.document_index
        document.file_hash = file_state.file_hash
        document.original_filename = file_state.file_path.split('/')[-1]
        document.file_type = parsed_doc.file_type
        document.file_size_bytes = file_state.file_size_bytes
        document.parsing_method = parsed_doc.parsing_method
        document.parsing_success = parsed_doc.parsing_success
        document.parsing_error_message = parsed_doc.parsing_error_message
        document.markdown_content = parsed_doc.markdown_content
        document.cleaned_content = cleaned_text
        document.word_count = parsed_doc.word_count
        document.page_count = parsed_doc.page_count
        document.processing_duration_ms = parsed_doc.parsing_duration_ms
        
        # Context metadata
        document.context_hierarchy = {
            'policy_manual': context.policy_manual,
            'policy_acronym': context.policy_acronym,
            'section': context.section,
            'document_type': context.document_type,
            'is_additional_resource': context.is_additional_resource
        }
        
        if not existing_doc:
            db.add(document)
        else:
            db.merge(document)
        
        db.flush()  # Get the ID
        return document
    
    def _process_chunks(self, db: Session, document: Document, chunks: List) -> int:
        """Process chunks - create embeddings and store vectors."""
        if not chunks:
            return 0
        
        # Delete existing chunks for this document
        db.query(DocumentChunk).filter_by(document_id=document.id).delete()
        
        # Prepare texts for embedding
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embedding_result = self.embedding_provider.embed_texts(chunk_texts)
        
        # Store chunks and vectors
        vector_points = []
        chunk_records = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embedding_result.embeddings)):
            # Create chunk record
            chunk_record = DocumentChunk(
                document_id=document.id,
                chunk_index=i,
                chunk_text=chunk.text,
                chunk_markdown=chunk.markdown,
                chunk_tokens=chunk.token_count,
                chunk_hash=chunk.hash,
                page_numbers=chunk.page_numbers,
                section_title=chunk.section_title,
                subsection_title=chunk.subsection_title,
                context_metadata=chunk.context_metadata,
                surrounding_context=chunk.surrounding_context,
                document_position=chunk.document_position,
                vector_id=uuid.uuid4()
            )
            
            chunk_records.append(chunk_record)
            
            # Prepare vector point for Qdrant (with full content for visualization)
            vector_points.append({
                'id': str(chunk_record.vector_id),
                'vector': embedding,
                'payload': {
                    'chunk_id': chunk_record.id,
                    'document_id': document.id,
                    'file_path': document.file_path,
                    'document_index': document.document_index,
                    'section_title': chunk_record.section_title,
                    'context_metadata': chunk.context_metadata,
                    'page_numbers': chunk_record.page_numbers or [],
                    # Include actual content for visualization
                    'chunk_text': chunk.text,
                    'chunk_markdown': chunk.markdown,
                    'token_count': chunk.token_count,
                    'document_position': chunk.document_position
                }
            })
        
        # Store chunks in database
        db.add_all(chunk_records)
        db.flush()  # Get IDs
        
        # Update vector point payloads with chunk IDs
        for chunk_record, vector_point in zip(chunk_records, vector_points):
            vector_point['payload']['chunk_id'] = chunk_record.id
        
        # Store vectors in Qdrant
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=point['id'],
                vector=point['vector'],
                payload=point['payload']
            )
            for point in vector_points
        ]
        
        self.vector_store.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Update document stats
        document.total_chunks = len(chunks)
        document.total_tokens = sum(chunk.token_count for chunk in chunks)
        db.merge(document)
        
        return len(chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        db = SessionLocal()
        try:
            total_files = db.query(FileSyncState).count()
            pending = db.query(FileSyncState).filter_by(sync_status='pending').count()
            processing = db.query(FileSyncState).filter_by(sync_status='processing').count()
            synced = db.query(FileSyncState).filter_by(sync_status='synced').count()
            error = db.query(FileSyncState).filter_by(sync_status='error').count()
            
            return {
                'collection_name': self.collection_name,
                'model_name': self.model_config.model_name,
                'total_files': total_files,
                'pending': pending,
                'processing': processing, 
                'synced': synced,
                'error': error,
                'progress_pct': (synced / total_files * 100) if total_files > 0 else 0
            }
        finally:
            db.close()