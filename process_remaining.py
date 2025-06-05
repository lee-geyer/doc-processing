#!/usr/bin/env python3
"""
Process all remaining files efficiently in background.
This script will continue until all files are processed.
"""

import time
import signal
import sys
from datetime import datetime
from rich.console import Console

from src.core.sync_manager import SyncManager
from src.models.database import SessionLocal, FileSyncState

console = Console()
sync_manager = SyncManager()

# Handle graceful shutdown
def signal_handler(sig, frame):
    console.print('\n[yellow]Graceful shutdown requested...[/yellow]')
    console.print('[dim]Processing will stop after current batch[/dim]')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def reset_stuck_files():
    """Reset any files stuck in processing."""
    db = SessionLocal()
    try:
        stuck_files = db.query(FileSyncState).filter_by(sync_status='processing').all()
        if stuck_files:
            console.print(f"[yellow]Resetting {len(stuck_files)} stuck files...[/yellow]")
            for file in stuck_files:
                file.sync_status = 'pending'
            db.commit()
    finally:
        db.close()

def get_pending_count():
    """Get count of pending files."""
    db = SessionLocal()
    try:
        return db.query(FileSyncState).filter_by(sync_status='pending').count()
    finally:
        db.close()

def main():
    """Main processing loop."""
    console.print("[bold cyan]🚀 Background Processing Started[/bold cyan]")
    console.print("[dim]Press Ctrl+C for graceful shutdown[/dim]\n")
    
    # Reset any stuck files from previous runs
    reset_stuck_files()
    
    batch_size = 20  # Conservative batch size
    consecutive_failures = 0
    total_processed = 0
    start_time = datetime.now()
    
    while True:
        pending_count = get_pending_count()
        
        if pending_count == 0:
            console.print("[bold green]✅ All files processed![/bold green]")
            break
        
        console.print(f"[cyan]Processing batch (remaining: {pending_count})[/cyan]")
        
        try:
            result = sync_manager.process_pending_files(limit=batch_size)
            
            if result['processed'] > 0:
                total_processed += result['processed']
                consecutive_failures = 0
                
                # Calculate rate
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_processed / elapsed * 3600 if elapsed > 0 else 0
                
                console.print(f"✓ Batch: {result['succeeded']} succeeded, {result['failed']} failed")
                console.print(f"📊 Rate: {rate:.0f} files/hour | Total: {total_processed}")
                
            else:
                consecutive_failures += 1
                console.print("[yellow]⚠ No files processed[/yellow]")
                
                if consecutive_failures >= 3:
                    console.print("[red]Multiple failures, checking for stuck files...[/red]")
                    reset_stuck_files()
                    consecutive_failures = 0
                    time.sleep(30)  # Wait before retry
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            consecutive_failures += 1
            time.sleep(10)
        
        # Brief pause between batches
        time.sleep(1)
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"\n[bold]📈 Processing Summary:[/bold]")
    console.print(f"  Total processed: {total_processed} files")
    console.print(f"  Runtime: {elapsed/3600:.1f} hours")
    console.print(f"  Average rate: {total_processed/elapsed*3600:.0f} files/hour")

if __name__ == "__main__":
    main()