"""
Robust batch processing command that continues until all files are processed.

Features:
- Automatic retry on failures
- Progress monitoring
- Handles stuck files
- Continues from interruptions
"""

import time
from datetime import datetime, timedelta
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from sqlalchemy import func

from src.models.database import SessionLocal, FileSyncState
from src.core.sync_manager import SyncManager
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def reset_stuck_files(db, stuck_threshold_minutes: int = 30) -> int:
    """Reset files that have been stuck in processing for too long."""
    threshold = datetime.now() - timedelta(minutes=stuck_threshold_minutes)
    
    stuck_files = db.query(FileSyncState).filter(
        FileSyncState.sync_status == 'processing',
        FileSyncState.updated_at < threshold
    ).all()
    
    if stuck_files:
        console.print(f"\n[yellow]Found {len(stuck_files)} stuck files. Resetting...[/yellow]")
        for file in stuck_files:
            file.sync_status = 'pending'
            console.print(f"  - Reset: {file.file_path.split('/')[-1]}")
        db.commit()
    
    return len(stuck_files)


def get_processing_stats(db) -> Dict:
    """Get current processing statistics."""
    status_counts = dict(
        db.query(FileSyncState.sync_status, func.count(FileSyncState.id))
        .group_by(FileSyncState.sync_status)
        .all()
    )
    
    synced = status_counts.get('synced', 0)
    processing = status_counts.get('processing', 0)
    pending = status_counts.get('pending', 0)
    error = status_counts.get('error', 0)
    total = synced + processing + pending + error
    
    return {
        'synced': synced,
        'processing': processing,
        'pending': pending,
        'error': error,
        'total': total,
        'progress_pct': (synced / total * 100) if total > 0 else 0
    }


def process_all_files(batch_size: int = 10, max_retries: int = 3):
    """Process all pending files with robust error handling."""
    db = SessionLocal()
    sync_manager = SyncManager()
    
    start_time = datetime.now()
    total_processed = 0
    consecutive_failures = 0
    
    console.print("[bold cyan]Starting comprehensive document processing[/bold cyan]")
    console.print(f"Batch size: {batch_size} files")
    console.print(f"Max retries: {max_retries}")
    console.print("[dim]Press Ctrl+C to stop gracefully[/dim]\n")
    
    try:
        while True:
            # Reset any stuck files
            reset_stuck_files(db)
            
            # Get current stats
            stats = get_processing_stats(db)
            
            # Display progress
            progress_table = Table(show_header=False, box=None)
            progress_table.add_column("Metric", style="cyan")
            progress_table.add_column("Value", style="white")
            
            progress_table.add_row("Progress:", f"{stats['synced']}/{stats['total']} ({stats['progress_pct']:.1f}%)")
            progress_table.add_row("Pending:", str(stats['pending']))
            progress_table.add_row("Errors:", f"[red]{stats['error']}[/red]" if stats['error'] > 0 else "0")
            progress_table.add_row("Processing Rate:", f"{total_processed / (datetime.now() - start_time).total_seconds() * 60:.1f} files/min")
            
            console.print(Panel(progress_table, title="Processing Status"))
            
            # Check if we're done
            if stats['pending'] == 0:
                console.print("\n[bold green]✓ All files processed![/bold green]")
                break
            
            # Process next batch
            console.print(f"\n[cyan]Processing batch of {batch_size} files...[/cyan]")
            
            try:
                result = sync_manager.process_pending_files(limit=batch_size)
                
                if result['processed'] > 0:
                    total_processed += result['processed']
                    consecutive_failures = 0
                    
                    console.print(f"[green]✓[/green] Processed {result['processed']} files")
                    console.print(f"  - Succeeded: {result['succeeded']}")
                    if result['failed'] > 0:
                        console.print(f"  - Failed: [red]{result['failed']}[/red]")
                else:
                    # No files processed, might be stuck
                    consecutive_failures += 1
                    console.print("[yellow]⚠ No files processed in this batch[/yellow]")
                    
                    if consecutive_failures >= 3:
                        console.print("[red]Multiple consecutive failures. Checking for issues...[/red]")
                        time.sleep(10)  # Wait before retrying
                
            except Exception as e:
                console.print(f"[red]Error during batch processing: {e}[/red]")
                consecutive_failures += 1
                
                if consecutive_failures >= max_retries:
                    console.print("[red]Too many consecutive failures. Stopping.[/red]")
                    break
                
                console.print(f"[yellow]Waiting 30 seconds before retry...[/yellow]")
                time.sleep(30)
            
            # Brief pause between batches
            time.sleep(2)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    
    finally:
        # Final report
        final_stats = get_processing_stats(db)
        elapsed = datetime.now() - start_time
        
        console.print("\n[bold]Final Processing Report[/bold]")
        console.print("=" * 50)
        console.print(f"Total time: {elapsed}")
        console.print(f"Files processed: {total_processed}")
        console.print(f"Final status:")
        console.print(f"  - Synced: {final_stats['synced']} ({final_stats['synced']/final_stats['total']*100:.1f}%)")
        console.print(f"  - Pending: {final_stats['pending']}")
        console.print(f"  - Errors: {final_stats['error']}")
        
        if total_processed > 0:
            console.print(f"Average processing time: {elapsed.total_seconds() / total_processed:.1f} seconds/file")
        
        db.close()


if __name__ == "__main__":
    process_all_files()