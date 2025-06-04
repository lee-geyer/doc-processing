import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.models.database import SessionLocal, FileSyncState, Document, DocumentChunk, VectorSyncOperation
from src.core.document_parser import DocumentParser, DocumentParsingError
from src.core.text_cleaner import TextCleaner
from src.core.chunking import DocumentChunker
from src.core.embeddings import embedding_manager
from src.core.vector_store import policy_vector_store
from src.utils.context_utils import ContextExtractor
from src.utils.file_utils import get_file_info, calculate_file_hash
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SyncManager:
    """
    Manage synchronization of files to vector database.
    """
    
    def __init__(self, max_workers: Optional[int] = None, embedding_provider: str = None):
        self.max_workers = max_workers or settings.max_concurrent_files
        self.batch_size = settings.batch_processing_size
        self.embedding_provider_name = embedding_provider or settings.default_embedding_model
        
        # Initialize processing components
        self.document_parser = DocumentParser()
        self.text_cleaner = TextCleaner()
        self.document_chunker = DocumentChunker()
        self.context_extractor = ContextExtractor()
        
        # Initialize embedding and vector components
        self.embedding_provider = embedding_manager.get_provider(self.embedding_provider_name)
        self.vector_store = policy_vector_store
        
        # Ensure vector collection exists
        self._ensure_vector_collection()
    
    def _ensure_vector_collection(self):
        """Ensure the vector collection exists for the current embedding provider."""
        collection_name = self.vector_store.get_collection_name(self.embedding_provider_name)
        
        if not self.vector_store.collection_exists(collection_name):
            logger.info(f"Creating vector collection: {collection_name}")
            success = self.vector_store.create_policy_collection(
                self.embedding_provider_name,
                self.embedding_provider.dimension,
                "cosine"
            )
            if not success:
                raise RuntimeError(f"Failed to create vector collection: {collection_name}")
        else:
            logger.info(f"Using existing vector collection: {collection_name}")
    
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
                logger.info("No pending files to process")
                return stats
            
            logger.info(f"Found {len(pending_files)} pending files to process")
            
            # Mark files as processing
            file_ids = [f.id for f in pending_files]
            db.query(FileSyncState).filter(FileSyncState.id.in_(file_ids)).update(
                {'sync_status': 'processing'},
                synchronize_session=False
            )
            db.commit()
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit processing tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file.id): file
                    for file in pending_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        stats['processed'] += 1
                        if result['success']:
                            stats['succeeded'] += 1
                        else:
                            stats['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file.file_path}: {e}", exc_info=True)
                        stats['failed'] += 1
            
            logger.info(f"Processing complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            raise
        finally:
            db.close()
    
    def _process_single_file(self, file_id: int) -> Dict[str, Any]:
        """
        Process a single file.
        
        Args:
            file_id: ID of the FileSyncState record
            
        Returns:
            Processing result dictionary
        """
        db = SessionLocal()
        start_time = time.time()
        
        try:
            # Get file sync state
            file_state = db.query(FileSyncState).filter_by(id=file_id).first()
            if not file_state:
                raise ValueError(f"File sync state not found: {file_id}")
            
            logger.info(f"Processing file: {file_state.file_path}")
            
            # Step 1: Parse document
            parsed_doc = self.document_parser.parse_document(file_state.file_path)
            
            if not parsed_doc.parsing_success:
                raise Exception(f"Document parsing failed: {parsed_doc.parsing_error_message}")
            
            # Step 2: Clean text
            cleaning_result = self.text_cleaner.clean_text(
                parsed_doc.raw_text or "",
                document_id=str(file_id)
            )
            
            # Validate cleaning didn't remove too much content
            cleaning_validation = self.text_cleaner.validate_cleaning(
                parsed_doc.raw_text or "",
                cleaning_result['cleaned_text']
            )
            
            if not cleaning_validation['is_valid']:
                logger.warning(f"Text cleaning validation failed: {cleaning_validation['reason']}")
            
            # Step 3: Extract context
            context = self.context_extractor.extract_context(file_state.file_path)
            context_metadata = self.context_extractor.create_context_metadata(context)
            
            # Step 4: Create or update document record
            existing_doc = db.query(Document).filter_by(file_path=file_state.file_path).first()
            
            if existing_doc:
                # Update existing document
                existing_doc.file_hash = file_state.file_hash
                existing_doc.file_size_bytes = file_state.file_size_bytes
                existing_doc.parsing_method = parsed_doc.parsing_method
                existing_doc.parsing_success = parsed_doc.parsing_success
                existing_doc.markdown_content = parsed_doc.markdown_content
                existing_doc.cleaned_content = cleaning_result['cleaned_text']
                existing_doc.word_count = parsed_doc.word_count
                existing_doc.page_count = parsed_doc.page_count
                existing_doc.document_metadata = parsed_doc.document_metadata
                existing_doc.context_hierarchy = context_metadata
                existing_doc.processing_duration_ms = parsed_doc.parsing_duration_ms
                existing_doc.updated_at = datetime.utcnow()
                
                document = existing_doc
            else:
                # Create new document record
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
                    cleaned_content=cleaning_result['cleaned_text'],
                    word_count=parsed_doc.word_count,
                    page_count=parsed_doc.page_count,
                    document_metadata=parsed_doc.document_metadata,
                    context_hierarchy=context_metadata,
                    processing_duration_ms=parsed_doc.parsing_duration_ms
                )
                db.add(document)
            
            # Step 5: Create chunks
            chunking_result = self.document_chunker.chunk_document(
                text=cleaning_result['cleaned_text'],
                markdown=parsed_doc.markdown_content or "",
                document_context=context
            )
            
            # Remove existing chunks if updating
            if existing_doc:
                db.query(DocumentChunk).filter_by(document_id=existing_doc.id).delete()
            
            # Commit document first to get ID
            db.commit()
            db.refresh(document)
            
            # Create chunk records
            chunk_records = []
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
            
            # Update document with chunk statistics
            document.total_chunks = len(chunk_records)
            document.total_tokens = chunking_result.total_tokens
            
            # Commit chunks to get IDs
            db.commit()
            for chunk_record in chunk_records:
                db.refresh(chunk_record)
            
            # Step 6: Generate embeddings and store in vector database
            vector_sync_result = self._sync_chunks_to_vectors(db, document, chunk_records)
            
            # Update sync state
            file_state.sync_status = 'synced'
            file_state.last_sync_at = datetime.utcnow()
            file_state.sync_error_count = 0
            file_state.last_error_message = None
            
            # Create sync operation record
            sync_op = VectorSyncOperation(
                operation_type='add',
                file_path=file_state.file_path,
                status='success',
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            db.add(sync_op)
            
            db.commit()
            
            return {
                'success': True,
                'file_path': file_state.file_path,
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}", exc_info=True)
            
            # Update error state
            if file_state:
                file_state.sync_status = 'error'
                file_state.sync_error_count += 1
                file_state.last_error_message = str(e)
                
                # Create failed sync operation record
                sync_op = VectorSyncOperation(
                    operation_type='add',
                    file_path=file_state.file_path,
                    status='failed',
                    error_message=str(e),
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    retry_count=file_state.sync_error_count
                )
                db.add(sync_op)
                
                db.commit()
            
            return {
                'success': False,
                'file_path': file_state.file_path if file_state else None,
                'error': str(e),
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
            
        finally:
            db.close()
    
    def _sync_chunks_to_vectors(self, db: Session, document: Document, chunk_records: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Generate embeddings for chunks and store in vector database.
        
        Args:
            db: Database session
            document: Document record
            chunk_records: List of chunk records
            
        Returns:
            Sync result dictionary
        """
        try:
            if not chunk_records:
                return {'success': True, 'vector_count': 0}
            
            # Prepare chunk data for embedding
            chunk_texts = [chunk.chunk_text for chunk in chunk_records]
            chunk_data = []
            
            for chunk in chunk_records:
                chunk_info = {
                    'chunk_id': chunk.id,
                    'document_id': document.id,
                    'text': chunk.chunk_text,
                    'chunk_index': chunk.chunk_index,
                    'document_index': document.document_index,
                    'policy_acronym': chunk.context_metadata.get('policy_acronym'),
                    'policy_manual': chunk.context_metadata.get('policy_manual'),
                    'section': chunk.context_metadata.get('section'),
                    'subsection': chunk.context_metadata.get('subsection'), 
                    'document_type': chunk.context_metadata.get('document_type'),
                    'file_path': document.file_path,
                    'page_numbers': chunk.page_numbers
                }
                chunk_data.append(chunk_info)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks using {self.embedding_provider_name}")
            embedding_result = self.embedding_provider.embed_texts(chunk_texts)
            
            # Store vectors in Qdrant
            logger.info(f"Storing {len(embedding_result.embeddings)} vectors in collection")
            vector_ids = self.vector_store.add_document_chunks(
                self.embedding_provider_name,
                chunk_data,
                embedding_result.embeddings
            )
            
            # Update chunk records with vector IDs
            for chunk_record, vector_id in zip(chunk_records, vector_ids):
                chunk_record.vector_id = vector_id
                # Note: embedding_model_id and vector_collection_id would be set
                # if we had those tables fully implemented
            
            db.commit()
            
            logger.info(f"Successfully synced {len(vector_ids)} vectors for document {document.file_path}")
            
            return {
                'success': True,
                'vector_count': len(vector_ids),
                'embedding_time_ms': embedding_result.processing_time_ms,
                'vector_ids': vector_ids
            }
            
        except Exception as e:
            logger.error(f"Error syncing chunks to vectors: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'vector_count': 0
            }
    
    def retry_failed_files(self, max_retries: Optional[int] = None) -> Dict[str, int]:
        """
        Retry processing failed files.
        
        Args:
            max_retries: Maximum number of retries per file
            
        Returns:
            Dictionary with retry statistics
        """
        max_retries = max_retries or settings.retry_max_attempts
        
        db = SessionLocal()
        try:
            # Get failed files that haven't exceeded retry limit
            failed_files = db.query(FileSyncState).filter(
                FileSyncState.sync_status == 'error',
                FileSyncState.sync_error_count < max_retries
            ).all()
            
            if not failed_files:
                logger.info("No failed files to retry")
                return {'total': 0, 'retried': 0}
            
            logger.info(f"Found {len(failed_files)} failed files to retry")
            
            # Reset status to pending for retry
            for file in failed_files:
                file.sync_status = 'pending'
            
            db.commit()
            
            # Process the files
            stats = self.process_pending_files(limit=len(failed_files))
            
            return {
                'total': len(failed_files),
                'retried': stats['processed'],
                'succeeded': stats['succeeded'],
                'failed': stats['failed']
            }
            
        except Exception as e:
            logger.error(f"Error retrying failed files: {e}", exc_info=True)
            raise
        finally:
            db.close()
    
    def cleanup_deleted_files(self) -> int:
        """
        Clean up vector entries for deleted files.
        
        Returns:
            Number of cleaned up files
        """
        db = SessionLocal()
        try:
            # Get deleted files
            deleted_files = db.query(FileSyncState).filter_by(sync_status='deleted').all()
            
            if not deleted_files:
                logger.info("No deleted files to clean up")
                return 0
            
            logger.info(f"Cleaning up {len(deleted_files)} deleted files")
            
            for file in deleted_files:
                try:
                    # Get document and its chunks to find vector IDs
                    document = db.query(Document).filter_by(file_path=file.file_path).first()
                    if document:
                        chunks = db.query(DocumentChunk).filter_by(document_id=document.id).all()
                        
                        # Collect vector IDs to delete
                        vector_ids = [chunk.vector_id for chunk in chunks if chunk.vector_id]
                        
                        if vector_ids:
                            # Remove vectors from Qdrant
                            collection_name = self.vector_store.get_collection_name(self.embedding_provider_name)
                            success = self.vector_store.delete_vectors(collection_name, vector_ids)
                            
                            if success:
                                logger.info(f"Deleted {len(vector_ids)} vectors for {file.file_path}")
                            else:
                                logger.warning(f"Failed to delete vectors for {file.file_path}")
                        
                        # Remove document and chunks from database
                        db.query(DocumentChunk).filter_by(document_id=document.id).delete()
                        db.delete(document)
                
                except Exception as e:
                    logger.error(f"Error cleaning up file {file.file_path}: {e}")
                    continue
                # Create deletion sync operation
                sync_op = VectorSyncOperation(
                    operation_type='delete',
                    file_path=file.file_path,
                    status='success'
                )
                db.add(sync_op)
                
                # Remove file sync state
                db.delete(file)
            
            db.commit()
            
            logger.info(f"Cleaned up {len(deleted_files)} deleted files")
            return len(deleted_files)
            
        except Exception as e:
            logger.error(f"Error cleaning up deleted files: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()