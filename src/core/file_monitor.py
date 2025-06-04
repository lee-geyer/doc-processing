import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Dict, Any
from queue import Queue, Empty
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.models.database import SessionLocal, FileSyncState
from src.models.schemas import FileSyncStateCreate, FileSyncStateUpdate
from src.utils.file_utils import calculate_file_hash, get_file_info, is_supported_file, scan_directory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """
    Handle file system events for document files.
    """
    
    def __init__(self, queue: Queue, supported_extensions: Set[str] = None):
        super().__init__()
        self.queue = queue
        self.supported_extensions = supported_extensions or {'.pdf', '.docx', '.doc'}
        
    def _is_relevant_event(self, event: FileSystemEvent) -> bool:
        """Check if event is relevant for processing."""
        if event.is_directory:
            return False
            
        path = Path(event.src_path)
        return path.suffix.lower() in self.supported_extensions
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if self._is_relevant_event(event):
            logger.info(f"New file detected: {event.src_path}")
            self.queue.put({
                'event_type': 'created',
                'file_path': event.src_path,
                'timestamp': datetime.utcnow()
            })
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if self._is_relevant_event(event):
            logger.info(f"File modified: {event.src_path}")
            self.queue.put({
                'event_type': 'modified',
                'file_path': event.src_path,
                'timestamp': datetime.utcnow()
            })
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion events."""
        if self._is_relevant_event(event):
            logger.info(f"File deleted: {event.src_path}")
            self.queue.put({
                'event_type': 'deleted',
                'file_path': event.src_path,
                'timestamp': datetime.utcnow()
            })
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename events."""
        if self._is_relevant_event(event):
            logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            self.queue.put({
                'event_type': 'moved',
                'file_path': event.src_path,
                'dest_path': event.dest_path,
                'timestamp': datetime.utcnow()
            })


class FileMonitor:
    """
    Monitor file system for changes in the mirror directory.
    """
    
    def __init__(self, mirror_directory: Optional[str] = None):
        self.mirror_directory = mirror_directory or settings.mirror_directory
        self.event_queue = Queue()
        self.observer = Observer()
        self.event_handler = FileChangeHandler(self.event_queue)
        self.is_running = False
        self._monitor_thread = None
        self._processor_thread = None
        self._stop_event = threading.Event()
        
        # Validate mirror directory exists
        if not Path(self.mirror_directory).exists():
            raise ValueError(f"Mirror directory does not exist: {self.mirror_directory}")
    
    def start(self):
        """Start file monitoring."""
        if self.is_running:
            logger.warning("File monitor is already running")
            return
        
        logger.info(f"Starting file monitor for: {self.mirror_directory}")
        
        # Schedule the observer
        self.observer.schedule(
            self.event_handler,
            self.mirror_directory,
            recursive=True
        )
        
        # Start the observer
        self.observer.start()
        self.is_running = True
        
        # Start event processor thread
        self._stop_event.clear()
        self._processor_thread = threading.Thread(target=self._process_events)
        self._processor_thread.daemon = True
        self._processor_thread.start()
        
        logger.info("File monitor started successfully")
    
    def stop(self):
        """Stop file monitoring."""
        if not self.is_running:
            logger.warning("File monitor is not running")
            return
        
        logger.info("Stopping file monitor...")
        
        # Stop the observer
        self.observer.stop()
        self.observer.join()
        
        # Stop the processor thread
        self._stop_event.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=5)
        
        self.is_running = False
        logger.info("File monitor stopped")
    
    def _process_events(self):
        """Process file events from the queue."""
        logger.info("Event processor started")
        
        while not self._stop_event.is_set():
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1)
                self._handle_event(event)
            except Empty:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
        
        logger.info("Event processor stopped")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle a single file event."""
        event_type = event['event_type']
        file_path = event['file_path']
        
        db = SessionLocal()
        try:
            if event_type == 'created':
                self._handle_file_created(db, file_path)
            elif event_type == 'modified':
                self._handle_file_modified(db, file_path)
            elif event_type == 'deleted':
                self._handle_file_deleted(db, file_path)
            elif event_type == 'moved':
                self._handle_file_moved(db, file_path, event.get('dest_path'))
            
            db.commit()
        except Exception as e:
            logger.error(f"Error handling {event_type} event for {file_path}: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()
    
    def _handle_file_created(self, db: Session, file_path: str):
        """Handle file creation."""
        # Check if file already exists in database
        existing = db.query(FileSyncState).filter_by(file_path=file_path).first()
        if existing:
            logger.warning(f"File already exists in database: {file_path}")
            return
        
        # Get file info
        file_info = get_file_info(file_path)
        file_hash = calculate_file_hash(file_path)
        
        # Create sync state record
        sync_state = FileSyncState(
            file_path=file_path,
            file_hash=file_hash,
            file_size_bytes=file_info['file_size_bytes'],
            last_modified=datetime.fromtimestamp(file_info['last_modified']),
            sync_status='pending'
        )
        
        db.add(sync_state)
        logger.info(f"Added new file to sync queue: {file_path}")
    
    def _handle_file_modified(self, db: Session, file_path: str):
        """Handle file modification."""
        # Get existing sync state
        sync_state = db.query(FileSyncState).filter_by(file_path=file_path).first()
        
        # Get current file info
        file_info = get_file_info(file_path)
        new_hash = calculate_file_hash(file_path)
        
        if sync_state:
            # Check if file actually changed
            if sync_state.file_hash != new_hash:
                sync_state.file_hash = new_hash
                sync_state.file_size_bytes = file_info['file_size_bytes']
                sync_state.last_modified = datetime.fromtimestamp(file_info['last_modified'])
                sync_state.sync_status = 'pending'
                sync_state.updated_at = datetime.utcnow()
                logger.info(f"File modified, marked for re-sync: {file_path}")
            else:
                logger.debug(f"File modified but content unchanged: {file_path}")
        else:
            # File not in database, treat as new
            self._handle_file_created(db, file_path)
    
    def _handle_file_deleted(self, db: Session, file_path: str):
        """Handle file deletion."""
        sync_state = db.query(FileSyncState).filter_by(file_path=file_path).first()
        if sync_state:
            # Mark for deletion in vector database
            sync_state.sync_status = 'deleted'
            sync_state.updated_at = datetime.utcnow()
            logger.info(f"File marked for deletion: {file_path}")
        else:
            logger.warning(f"Deleted file not found in database: {file_path}")
    
    def _handle_file_moved(self, db: Session, src_path: str, dest_path: str):
        """Handle file move/rename."""
        sync_state = db.query(FileSyncState).filter_by(file_path=src_path).first()
        if sync_state:
            # Update file path
            sync_state.file_path = dest_path
            sync_state.sync_status = 'pending'  # Re-sync to update metadata
            sync_state.updated_at = datetime.utcnow()
            logger.info(f"File moved: {src_path} -> {dest_path}")
        else:
            # Treat as new file
            self._handle_file_created(db, dest_path)
    
    def scan_directory(self, force_rescan: bool = False) -> Dict[str, int]:
        """
        Scan the mirror directory and update sync state.
        
        Args:
            force_rescan: Force rescan all files even if unchanged
            
        Returns:
            Dictionary with scan statistics
        """
        logger.info(f"Scanning directory: {self.mirror_directory}")
        
        # Get all supported files in directory
        current_files = set(scan_directory(self.mirror_directory))
        
        db = SessionLocal()
        stats = {
            'total_files': len(current_files),
            'new_files': 0,
            'modified_files': 0,
            'deleted_files': 0,
            'unchanged_files': 0
        }
        
        try:
            # Get all files in database
            db_files = {fs.file_path: fs for fs in db.query(FileSyncState).all()}
            
            # Check for new and modified files
            for file_path in current_files:
                file_info = get_file_info(file_path)
                file_hash = calculate_file_hash(file_path)
                
                if file_path in db_files:
                    sync_state = db_files[file_path]
                    
                    if force_rescan or sync_state.file_hash != file_hash:
                        # File modified
                        sync_state.file_hash = file_hash
                        sync_state.file_size_bytes = file_info['file_size_bytes']
                        sync_state.last_modified = datetime.fromtimestamp(file_info['last_modified'])
                        sync_state.sync_status = 'pending'
                        sync_state.updated_at = datetime.utcnow()
                        stats['modified_files'] += 1
                    else:
                        stats['unchanged_files'] += 1
                else:
                    # New file
                    sync_state = FileSyncState(
                        file_path=file_path,
                        file_hash=file_hash,
                        file_size_bytes=file_info['file_size_bytes'],
                        last_modified=datetime.fromtimestamp(file_info['last_modified']),
                        sync_status='pending'
                    )
                    db.add(sync_state)
                    stats['new_files'] += 1
            
            # Check for deleted files
            for file_path, sync_state in db_files.items():
                if file_path not in current_files and sync_state.sync_status != 'deleted':
                    sync_state.sync_status = 'deleted'
                    sync_state.updated_at = datetime.utcnow()
                    stats['deleted_files'] += 1
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error during directory scan: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()
        
        logger.info(f"Directory scan complete: {stats}")
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        db = SessionLocal()
        try:
            # Count files by sync status
            status_counts = {}
            for status in ['pending', 'processing', 'synced', 'error', 'deleted']:
                count = db.query(FileSyncState).filter_by(sync_status=status).count()
                status_counts[status] = count
            
            # Get last scan time
            last_scan = db.query(FileSyncState).order_by(
                FileSyncState.created_at.desc()
            ).first()
            
            return {
                'is_running': self.is_running,
                'monitored_directory': self.mirror_directory,
                'total_files': sum(status_counts.values()),
                'pending_files': status_counts.get('pending', 0),
                'processing_files': status_counts.get('processing', 0),
                'synced_files': status_counts.get('synced', 0),
                'error_files': status_counts.get('error', 0),
                'deleted_files': status_counts.get('deleted', 0),
                'last_scan': last_scan.created_at if last_scan else None,
                'queue_size': self.event_queue.qsize()
            }
        finally:
            db.close()