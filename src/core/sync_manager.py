import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.models.database import SessionLocal, FileSyncState, Document, VectorSyncOperation
from src.core.document_parser import DocumentParser, DocumentParsingError
from src.core.text_cleaner import TextCleaner
from src.utils.context_utils import ContextExtractor
from src.utils.file_utils import get_file_info, calculate_file_hash
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SyncManager:
    """
    Manage synchronization of files to vector database.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or settings.max_concurrent_files
        self.batch_size = settings.batch_processing_size
        
        # Initialize processing components
        self.document_parser = DocumentParser()
        self.text_cleaner = TextCleaner()
        self.context_extractor = ContextExtractor()
    
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
            
            # TODO: Step 5: Create chunks (will implement in Phase 3)
            # TODO: Step 6: Generate embeddings (will implement in Phase 4)
            # TODO: Step 7: Store in vector database (will implement in Phase 5)
            
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
            
            # TODO: Remove vectors from Qdrant
            # For now, just remove from database
            
            for file in deleted_files:
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