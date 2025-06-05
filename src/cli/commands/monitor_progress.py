"""
Real-time monitoring command for document processing progress.

Provides live updates on:
- Processing progress and ETA
- Current files being processed
- Error tracking and details
- Processing speed metrics
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.align import Align
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.models.database import SessionLocal, FileSyncState, VectorSyncOperation, Document
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


class ProcessingMonitor:
    """Monitor document processing progress in real-time."""
    
    def __init__(self, db: Session):
        self.db = db
        self.start_time = datetime.now()
        self.last_synced_count = 0
        self.processing_rates = []  # Track processing speed over time
        
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        # Get file counts by status
        status_counts = dict(
            self.db.query(FileSyncState.sync_status, func.count(FileSyncState.id))
            .group_by(FileSyncState.sync_status)
            .all()
        )
        
        synced = status_counts.get('synced', 0)
        processing = status_counts.get('processing', 0)
        pending = status_counts.get('pending', 0)
        error = status_counts.get('error', 0)
        total = synced + processing + pending + error
        
        # Calculate processing rate
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0 and synced > self.last_synced_count:
            rate = (synced - self.last_synced_count) / elapsed * 3600  # files per hour
            self.processing_rates.append(rate)
            if len(self.processing_rates) > 10:
                self.processing_rates.pop(0)
        
        avg_rate = sum(self.processing_rates) / len(self.processing_rates) if self.processing_rates else 0
        
        # Estimate time remaining
        if avg_rate > 0:
            hours_remaining = pending / avg_rate
            eta = datetime.now() + timedelta(hours=hours_remaining)
        else:
            eta = None
            
        return {
            'synced': synced,
            'processing': processing,
            'pending': pending,
            'error': error,
            'total': total,
            'progress_pct': (synced / total * 100) if total > 0 else 0,
            'processing_rate': avg_rate,
            'eta': eta,
            'elapsed': elapsed
        }
    
    def get_current_files(self, limit: int = 5) -> List[Dict]:
        """Get files currently being processed."""
        processing_files = self.db.query(FileSyncState).filter_by(
            sync_status='processing'
        ).order_by(desc(FileSyncState.updated_at)).limit(limit).all()
        
        return [{
            'filename': file.file_path.split('/')[-1],
            'path': file.file_path,
            'updated': file.updated_at
        } for file in processing_files]
    
    def get_recent_errors(self, limit: int = 5) -> List[Dict]:
        """Get recent error files with details."""
        error_files = self.db.query(FileSyncState).filter_by(
            sync_status='error'
        ).order_by(desc(FileSyncState.updated_at)).limit(limit).all()
        
        errors = []
        for file in error_files:
            errors.append({
                'filename': file.file_path.split('/')[-1],
                'path': file.file_path,
                'error': file.last_error_message,
                'attempts': file.sync_error_count,
                'updated': file.updated_at
            })
        return errors
    
    def get_recent_completions(self, limit: int = 10) -> List[Dict]:
        """Get recently completed files."""
        # Get recent successful sync operations
        recent_ops = self.db.query(VectorSyncOperation).filter_by(
            operation_type='add',
            status='success'
        ).order_by(desc(VectorSyncOperation.created_at)).limit(limit).all()
        
        completions = []
        for op in recent_ops:
            # Get document info
            doc = self.db.query(Document).filter_by(file_path=op.file_path).first()
            if doc:
                completions.append({
                    'filename': op.file_path.split('/')[-1],
                    'chunks': doc.total_chunks,
                    'tokens': doc.total_tokens,
                    'time_ms': op.execution_time_ms,
                    'completed': op.created_at
                })
        return completions
    
    def get_vector_stats(self) -> Dict:
        """Get vector database statistics."""
        total_chunks = self.db.query(func.sum(Document.total_chunks)).scalar() or 0
        total_tokens = self.db.query(func.sum(Document.total_tokens)).scalar() or 0
        avg_chunks = self.db.query(func.avg(Document.total_chunks)).scalar() or 0
        
        return {
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'avg_chunks_per_doc': avg_chunks
        }


def create_progress_display(stats: Dict, current_files: List[Dict], 
                          errors: List[Dict], completions: List[Dict],
                          vector_stats: Dict) -> Layout:
    """Create the monitoring display layout."""
    layout = Layout()
    
    # Create main sections
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=6),
        Layout(name="main", size=20),
        Layout(name="footer", size=3)
    )
    
    # Header
    header_text = f"[bold cyan]Document Processing Monitor[/bold cyan] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    layout["header"].update(Panel(Align.center(header_text)))
    
    # Progress section
    progress_table = Table(show_header=False, box=None)
    progress_table.add_column("Label", style="cyan", width=20)
    progress_table.add_column("Value", style="white")
    
    progress_table.add_row("Total Files:", f"{stats['total']:,}")
    progress_table.add_row("Completed:", f"[green]{stats['synced']:,}[/green] ({stats['progress_pct']:.1f}%)")
    progress_table.add_row("Processing:", f"[yellow]{stats['processing']:,}[/yellow]")
    progress_table.add_row("Pending:", f"{stats['pending']:,}")
    progress_table.add_row("Errors:", f"[red]{stats['error']:,}[/red]")
    progress_table.add_row("Processing Rate:", f"{stats['processing_rate']:.1f} files/hour")
    if stats['eta']:
        progress_table.add_row("Estimated Completion:", stats['eta'].strftime('%Y-%m-%d %H:%M'))
    
    layout["progress"].update(Panel(progress_table, title="Progress Overview"))
    
    # Main section - split into columns
    layout["main"].split_row(
        Layout(name="current", ratio=1),
        Layout(name="errors", ratio=1),
        Layout(name="recent", ratio=1)
    )
    
    # Current processing
    current_table = Table(show_header=True)
    current_table.add_column("File", style="yellow", overflow="ellipsis")
    current_table.add_column("Started", style="dim")
    
    for file in current_files:
        elapsed = (datetime.now() - file['updated']).total_seconds()
        current_table.add_row(
            file['filename'][:40],
            f"{elapsed:.0f}s ago"
        )
    
    layout["current"].update(Panel(current_table, title="Currently Processing"))
    
    # Errors
    error_table = Table(show_header=True)
    error_table.add_column("File", style="red", overflow="ellipsis", width=25)
    error_table.add_column("Error", style="dim", overflow="ellipsis", width=35)
    error_table.add_column("Attempts", style="red")
    
    for error in errors:
        error_msg = error['error'][:60] + "..." if len(error['error']) > 60 else error['error']
        error_table.add_row(
            error['filename'][:25],
            error_msg,
            str(error['attempts'])
        )
    
    layout["errors"].update(Panel(error_table, title="Recent Errors"))
    
    # Recent completions
    recent_table = Table(show_header=True)
    recent_table.add_column("File", style="green", overflow="ellipsis", width=30)
    recent_table.add_column("Chunks", style="cyan", justify="right")
    recent_table.add_column("Time", style="dim", justify="right")
    
    for comp in completions:
        recent_table.add_row(
            comp['filename'][:30],
            str(comp['chunks']),
            f"{comp['time_ms']/1000:.1f}s"
        )
    
    layout["recent"].update(Panel(recent_table, title="Recently Completed"))
    
    # Footer with vector stats
    footer_text = (f"Vectors: {vector_stats['total_chunks']:,} chunks | "
                  f"{vector_stats['total_tokens']:,} tokens | "
                  f"Avg {vector_stats['avg_chunks_per_doc']:.0f} chunks/doc")
    layout["footer"].update(Panel(footer_text))
    
    return layout


def monitor_progress(refresh_interval: int = 5):
    """Monitor processing progress with live updates."""
    db = SessionLocal()
    monitor = ProcessingMonitor(db)
    
    try:
        with Live(refresh_per_second=1, screen=True) as live:
            while True:
                # Get current stats
                stats = monitor.get_stats()
                current_files = monitor.get_current_files()
                errors = monitor.get_recent_errors()
                completions = monitor.get_recent_completions()
                vector_stats = monitor.get_vector_stats()
                
                # Create display
                display = create_progress_display(
                    stats, current_files, errors, completions, vector_stats
                )
                
                live.update(display)
                
                # Check if processing is complete
                if stats['pending'] == 0 and stats['processing'] == 0:
                    console.print("\n[bold green]✓ Processing complete![/bold green]")
                    console.print(f"Total files processed: {stats['synced']}")
                    console.print(f"Total errors: {stats['error']}")
                    break
                
                time.sleep(refresh_interval)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    finally:
        db.close()


def print_detailed_report():
    """Print a detailed report of processing status."""
    db = SessionLocal()
    
    try:
        # Overall statistics
        console.print("\n[bold cyan]Document Processing Detailed Report[/bold cyan]")
        console.print("=" * 80)
        
        # File status breakdown
        status_counts = dict(
            db.query(FileSyncState.sync_status, func.count(FileSyncState.id))
            .group_by(FileSyncState.sync_status)
            .all()
        )
        
        console.print("\n[bold]File Status Summary:[/bold]")
        total = sum(status_counts.values())
        for status, count in status_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            console.print(f"  {status.capitalize()}: {count} ({pct:.1f}%)")
        
        # Error details
        error_files = db.query(FileSyncState).filter_by(sync_status='error').all()
        if error_files:
            console.print(f"\n[bold red]Error Files ({len(error_files)} total):[/bold red]")
            
            # Group errors by type
            error_types = {}
            for file in error_files:
                error_key = file.last_error_message.split(':')[0] if ':' in file.last_error_message else 'Unknown'
                if error_key not in error_types:
                    error_types[error_key] = []
                error_types[error_key].append(file)
            
            for error_type, files in error_types.items():
                console.print(f"\n  [yellow]{error_type}[/yellow] ({len(files)} files):")
                for file in files[:3]:  # Show first 3 examples
                    console.print(f"    - {file.file_path.split('/')[-1]}")
                if len(files) > 3:
                    console.print(f"    ... and {len(files) - 3} more")
        
        # Processing statistics
        successful_ops = db.query(VectorSyncOperation).filter_by(
            operation_type='add',
            status='success'
        ).all()
        
        if successful_ops:
            avg_time = sum(op.execution_time_ms for op in successful_ops) / len(successful_ops)
            console.print(f"\n[bold]Processing Performance:[/bold]")
            console.print(f"  Average processing time: {avg_time/1000:.1f} seconds/file")
            console.print(f"  Total successful operations: {len(successful_ops)}")
        
        # Document statistics
        docs = db.query(Document).all()
        if docs:
            total_chunks = sum(doc.total_chunks for doc in docs)
            total_tokens = sum(doc.total_tokens for doc in docs)
            console.print(f"\n[bold]Document Statistics:[/bold]")
            console.print(f"  Total documents: {len(docs)}")
            console.print(f"  Total chunks: {total_chunks:,}")
            console.print(f"  Total tokens: {total_tokens:,}")
            console.print(f"  Average chunks per document: {total_chunks/len(docs):.1f}")
            console.print(f"  Average tokens per chunk: {total_tokens/total_chunks:.1f}")
        
    finally:
        db.close()