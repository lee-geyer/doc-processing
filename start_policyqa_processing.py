#!/usr/bin/env python3
"""
Start processing with policyQA Q&A optimized model.
"""

import time
import signal
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from policyqa_sync_manager import PolicyQASyncManager

console = Console()

# Handle graceful shutdown
def signal_handler(sig, frame):
    console.print('\n[yellow]Graceful shutdown requested...[/yellow]')
    console.print('[dim]Processing will stop after current batch[/dim]')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    """Start processing with policyQA Q&A model."""
    console.print(Panel.fit(
        "[bold cyan]🚀 policyQA Q&A Processing[/bold cyan]\n"
        "Using multi-qa-mpnet-base-dot-v1 model\n"
        "Collection: policyQA_dense_qa768\n"
        "[dim]Press Ctrl+C for graceful shutdown[/dim]",
        title="Document Processing"
    ))
    
    try:
        # Initialize sync manager with Q&A model
        sync_manager = PolicyQASyncManager(model_key="dense_qa768")
        
        console.print(f"✓ Initialized sync manager")
        console.print(f"✓ Collection: [cyan]{sync_manager.collection_name}[/cyan]")
        console.print(f"✓ Model: [green]{sync_manager.model_config.model_name}[/green]")
        
        # Get initial stats
        stats = sync_manager.get_stats()
        console.print(f"✓ Files to process: [yellow]{stats['pending']}[/yellow]")
        
        if stats['pending'] == 0:
            console.print("\n[bold green]✅ No files to process![/bold green]")
            return
        
        # Start processing
        batch_size = 20
        total_processed = 0
        start_time = datetime.now()
        
        console.print(f"\n[cyan]Starting processing in batches of {batch_size}...[/cyan]\n")
        
        while True:
            current_stats = sync_manager.get_stats()
            if current_stats['pending'] == 0:
                break
            
            console.print(f"[cyan]Processing batch (remaining: {current_stats['pending']})[/cyan]")
            
            # Process batch
            result = sync_manager.process_pending_files(limit=batch_size)
            
            if result['processed'] > 0:
                total_processed += result['processed']
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_processed / elapsed * 3600 if elapsed > 0 else 0
                
                console.print(f"✓ Batch: {result['succeeded']} succeeded, {result['failed']} failed")
                console.print(f"📊 Rate: {rate:.0f} files/hour | Total: {total_processed}")
                
                # Show progress
                progress_pct = current_stats['progress_pct']
                console.print(f"📈 Progress: {progress_pct:.1f}% complete")
            else:
                console.print("[yellow]⚠ No files processed in this batch[/yellow]")
                time.sleep(5)  # Wait before retry
            
            time.sleep(1)  # Brief pause between batches
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        final_stats = sync_manager.get_stats()
        
        console.print(Panel.fit(
            f"[bold green]🎉 Processing Complete![/bold green]\n\n"
            f"Collection: {sync_manager.collection_name}\n"
            f"Model: {sync_manager.model_config.model_name}\n"
            f"Files processed: {total_processed}\n"
            f"Success rate: {final_stats['synced']}/{final_stats['total_files']} "
            f"({final_stats['progress_pct']:.1f}%)\n"
            f"Runtime: {elapsed/3600:.1f} hours\n"
            f"Average rate: {total_processed/elapsed*3600:.0f} files/hour" if elapsed > 0 else "",
            title="Processing Summary"
        ))
        
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        raise

if __name__ == "__main__":
    main()