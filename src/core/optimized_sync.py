"""
Optimized sync manager for faster document processing.

Key optimizations:
1. Batch embedding generation for multiple documents
2. Async vector storage operations
3. Smaller chunk sizes for large documents
4. Memory-efficient processing
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.models.database import SessionLocal, FileSyncState, Document, DocumentChunk, VectorSyncOperation
from src.core.document_parser import DocumentParser
from src.core.text_cleaner import TextCleaner
from src.core.chunking import DocumentChunker
from src.core.embeddings import embedding_manager
from src.core.vector_store import policy_vector_store
from src.utils.context_utils import ContextExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizedSyncManager:
    """Optimized sync manager for faster processing."""
    
    def __init__(self, embedding_provider: str = None):
        self.embedding_provider_name = embedding_provider or settings.default_embedding_model
        
        # Initialize components
        self.document_parser = DocumentParser()
        self.text_cleaner = TextCleaner()
        self.document_chunker = DocumentChunker()
        self.context_extractor = ContextExtractor()
        
        # Initialize embedding provider
        self.embedding_provider = embedding_manager.get_provider(self.embedding_provider_name)
        self.vector_store = policy_vector_store
        
        # Optimization settings
        self.batch_embedding_size = 50  # Process multiple chunks at once
        self.max_chunk_size = 800  # Smaller chunks for faster processing
        self.vector_batch_size = 1000  # Larger batches for vector storage
        
        self._ensure_vector_collection()
    
    def _ensure_vector_collection(self):
        """Ensure vector collection exists."""
        collection_name = self.vector_store.get_collection_name(self.embedding_provider_name)
        if not self.vector_store.collection_exists(collection_name):
            self.vector_store.create_policy_collection(
                self.embedding_provider_name,
                self.embedding_provider.dimension,
                "cosine"
            )
    
    def process_files_optimized(self, limit: int = 50) -> Dict[str, int]:
        """Process files with optimizations for speed."""
        db = SessionLocal()
        stats = {'processed': 0, 'succeeded': 0, 'failed': 0}
        
        try:
            # Get pending files
            pending_files = db.query(FileSyncState).filter_by(
                sync_status='pending'
            ).limit(limit).all()
            
            if not pending_files:
                return stats
            
            logger.info(f"Processing {len(pending_files)} files with optimizations")
            
            # Mark as processing
            file_ids = [f.id for f in pending_files]
            db.query(FileSyncState).filter(FileSyncState.id.in_(file_ids)).update(
                {'sync_status': 'processing'}, synchronize_session=False
            )
            db.commit()
            
            # Process in batches for efficiency
            batch_size = 5
            for i in range(0, len(pending_files), batch_size):
                batch = pending_files[i:i + batch_size]
                self._process_batch_optimized(db, batch, stats)
            
            logger.info(f"Optimized processing complete: {stats}")
            return stats
            
        finally:
            db.close()
    
    def _process_batch_optimized(self, db: Session, files: List[FileSyncState], stats: Dict):
        """Process a batch of files with optimizations."""
        # Collect all chunks from batch
        all_chunks_data = []
        file_chunk_mapping = {}
        
        for file_state in files:
            try:
                # Process file
                result = self._process_file_fast(db, file_state)
                
                if result['success']:
                    stats['succeeded'] += 1
                    # Collect chunks for batch embedding
                    all_chunks_data.extend(result['chunks'])
                    file_chunk_mapping[file_state.id] = {
                        'document': result['document'],
                        'chunk_records': result['chunk_records'],
                        'chunk_start_idx': len(all_chunks_data) - len(result['chunks'])
                    }
                else:
                    stats['failed'] += 1
                    
                stats['processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_state.id}: {e}")
                stats['failed'] += 1
                stats['processed'] += 1
                self._mark_file_error(db, file_state, str(e))
        
        # Batch generate embeddings for all chunks
        if all_chunks_data:
            self._batch_embed_and_store(db, all_chunks_data, file_chunk_mapping)
    
    def _process_file_fast(self, db: Session, file_state: FileSyncState) -> Dict[str, Any]:
        """Process single file quickly."""
        start_time = time.time()
        
        try:
            # Parse document
            parsed_doc = self.document_parser.parse_document(file_state.file_path)
            if not parsed_doc.parsing_success:
                raise Exception(f"Parsing failed: {parsed_doc.parsing_error_message}")
            
            # Clean text (simplified for speed)
            cleaned_text = self.text_cleaner.clean_text_fast(parsed_doc.raw_text or "")
            
            # Extract context
            context = self.context_extractor.extract_context(file_state.file_path)
            context_metadata = self.context_extractor.create_context_metadata(context)
            
            # Create/update document
            document = self._create_or_update_document(
                db, file_state, parsed_doc, cleaned_text, context, context_metadata
            )
            
            # Create chunks with optimized size
            chunk_size = min(self.max_chunk_size, settings.chunk_size)
            chunking_result = self.document_chunker.chunk_document(
                text=cleaned_text,
                markdown=parsed_doc.markdown_content or "",
                document_context=context,
                chunk_size=chunk_size,
                chunk_overlap=150,  # Smaller overlap for speed
                file_size_bytes=file_state.file_size_bytes
            )
            
            # Create chunk records
            chunk_records = []
            chunks_data = []
            
            for chunk in chunking_result.chunks:
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    chunk_text=chunk.text,
                    chunk_markdown=chunk.markdown,
                    chunk_tokens=chunk.token_count,
                    chunk_hash=chunk.hash,
                    page_numbers=chunk.page_numbers,
                    section_title=chunk.section_title,
                    subsection_title=chunk.subsection_title,
                    context_metadata=chunk.context_metadata,
                    surrounding_context=chunk.surrounding_context,
                    document_position=chunk.document_position
                )
                chunk_records.append(chunk_record)
                db.add(chunk_record)
                
                # Prepare chunk data for embedding
                chunks_data.append({
                    'text': chunk.text,
                    'metadata': {
                        'chunk_id': None,  # Will be set after commit
                        'document_id': document.id,
                        'document_index': document.document_index,
                        'policy_acronym': chunk.context_metadata.get('policy_acronym'),
                        'file_path': document.file_path,
                        'synthetic_description': chunk.context_metadata.get('synthetic_description', False)
                    }
                })
            
            # Update document stats
            document.total_chunks = len(chunk_records)
            document.total_tokens = chunking_result.total_tokens
            document.processing_duration_ms = int((time.time() - start_time) * 1000)
            
            db.commit()
            
            # Update chunk IDs in data
            for i, chunk_record in enumerate(chunk_records):
                db.refresh(chunk_record)
                chunks_data[i]['metadata']['chunk_id'] = chunk_record.id
            
            return {
                'success': True,
                'document': document,
                'chunk_records': chunk_records,
                'chunks': chunks_data
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error in fast processing: {e}")
            return {'success': False, 'error': str(e)}
    
    def _batch_embed_and_store(self, db: Session, all_chunks: List[Dict], file_mapping: Dict):
        """Batch embed and store all chunks efficiently."""
        try:
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in all_chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks in batch")
            
            # Batch generate embeddings
            if len(texts) > self.batch_embedding_size:
                # Process in sub-batches for memory efficiency
                all_embeddings = []
                for i in range(0, len(texts), self.batch_embedding_size):
                    batch_texts = texts[i:i + self.batch_embedding_size]
                    result = self.embedding_provider.embed_texts(batch_texts)
                    all_embeddings.extend(result.embeddings)
            else:
                result = self.embedding_provider.embed_texts(texts)
                all_embeddings = result.embeddings
            
            # Prepare vector data in batches
            vector_data = []
            for i, (chunk, embedding) in enumerate(zip(all_chunks, all_embeddings)):
                vector_data.append({
                    'chunk_data': chunk['metadata'],
                    'embedding': embedding
                })
            
            # Store vectors in large batches
            logger.info(f"Storing {len(vector_data)} vectors in batches")
            
            for i in range(0, len(vector_data), self.vector_batch_size):
                batch = vector_data[i:i + self.vector_batch_size]
                
                chunk_data_list = [v['chunk_data'] for v in batch]
                embeddings_list = [v['embedding'] for v in batch]
                
                vector_ids = self.vector_store.add_document_chunks(
                    self.embedding_provider_name,
                    chunk_data_list,
                    embeddings_list
                )
                
                # Update chunk records with vector IDs
                for j, vector_id in enumerate(vector_ids):
                    chunk_id = chunk_data_list[j]['chunk_id']
                    if chunk_id:
                        db.query(DocumentChunk).filter_by(id=chunk_id).update(
                            {'vector_id': vector_id}
                        )
            
            db.commit()
            
            # Mark files as synced
            for file_id, file_data in file_mapping.items():
                file_state = db.query(FileSyncState).filter_by(id=file_id).first()
                if file_state:
                    file_state.sync_status = 'synced'
                    file_state.last_sync_at = datetime.utcnow()
                    file_state.sync_error_count = 0
                    file_state.last_error_message = None
                    
                    # Log success
                    sync_op = VectorSyncOperation(
                        operation_type='add',
                        file_path=file_state.file_path,
                        document_id=file_data['document'].id,
                        chunks_affected=len(file_data['chunk_records']),
                        status='success',
                        execution_time_ms=file_data['document'].processing_duration_ms
                    )
                    db.add(sync_op)
            
            db.commit()
            logger.info("Batch embedding and storage complete")
            
        except Exception as e:
            logger.error(f"Error in batch embed and store: {e}")
            db.rollback()
            raise
    
    def _create_or_update_document(self, db, file_state, parsed_doc, cleaned_text, context, context_metadata):
        """Create or update document record."""
        existing_doc = db.query(Document).filter_by(file_path=file_state.file_path).first()
        
        if existing_doc:
            # Update existing
            existing_doc.file_hash = file_state.file_hash
            existing_doc.file_size_bytes = file_state.file_size_bytes
            existing_doc.parsing_method = parsed_doc.parsing_method
            existing_doc.parsing_success = parsed_doc.parsing_success
            existing_doc.markdown_content = parsed_doc.markdown_content
            existing_doc.cleaned_content = cleaned_text
            existing_doc.word_count = parsed_doc.word_count
            existing_doc.page_count = parsed_doc.page_count
            existing_doc.document_metadata = parsed_doc.document_metadata
            existing_doc.context_hierarchy = context_metadata
            existing_doc.updated_at = datetime.utcnow()
            
            # Delete old chunks
            db.query(DocumentChunk).filter_by(document_id=existing_doc.id).delete()
            
            return existing_doc
        else:
            # Create new
            document = Document(
                file_path=file_state.file_path,
                document_index=context.document_index,
                file_hash=file_state.file_hash,
                original_filename=context.file_name,
                file_type=parsed_doc.file_type,
                file_size_bytes=file_state.file_size_bytes,
                parsing_method=parsed_doc.parsing_method,
                parsing_success=parsed_doc.parsing_success,
                markdown_content=parsed_doc.markdown_content,
                cleaned_content=cleaned_text,
                word_count=parsed_doc.word_count,
                page_count=parsed_doc.page_count,
                document_metadata=parsed_doc.document_metadata,
                context_hierarchy=context_metadata
            )
            db.add(document)
            return document
    
    def _mark_file_error(self, db, file_state, error_message):
        """Mark file as error."""
        file_state.sync_status = 'error'
        file_state.sync_error_count += 1
        file_state.last_error_message = error_message
        db.commit()