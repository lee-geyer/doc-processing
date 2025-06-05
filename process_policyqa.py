#!/usr/bin/env python3
"""
Process all documents with policyQA collections.
Uses the new naming convention and Q&A optimized model.
"""

import time
import signal
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from src.models.database import SessionLocal, FileSyncState
from collection_manager import PolicyQACollectionManager

console = Console()

# Handle graceful shutdown
def signal_handler(sig, frame):
    console.print('\n[yellow]Graceful shutdown requested...[/yellow]')
    console.print('[dim]Processing will stop after current batch[/dim]')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_pending_count():
    """Get count of pending files."""
    db = SessionLocal()
    try:
        return db.query(FileSyncState).filter_by(sync_status='pending').count()
    finally:
        db.close()

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

def main():
    """Process all files with policyQA Q&A model."""
    console.print(Panel.fit(
        "[bold cyan]🚀 policyQA Processing[/bold cyan]\n"
        "Using Q&A optimized embedding model\n"
        "[dim]Press Ctrl+C for graceful shutdown[/dim]",
        title="Document Processing"
    ))
    
    # Initialize collection manager
    manager = PolicyQACollectionManager()
    
    # Verify Q&A collection exists
    qa_config = manager.models['dense_qa768']
    collection_name = qa_config.name
    
    if not manager.vector_store.collection_exists(collection_name):
        console.print(f"[red]Collection {collection_name} does not exist![/red]")
        console.print("Create it with: uv run python collection_manager.py create dense_qa768")
        return
    
    console.print(f"✓ Using collection: [cyan]{collection_name}[/cyan]")
    console.print(f"✓ Model: [green]{qa_config.model_name}[/green]")
    console.print(f"✓ Description: {qa_config.description}")
    
    # Reset any stuck files from previous runs
    reset_stuck_files()
    
    # Check pending files
    pending_count = get_pending_count()
    if pending_count == 0:
        console.print("\n[bold green]✅ No files to process![/bold green]")
        return
    
    console.print(f"✓ Files to process: [yellow]{pending_count}[/yellow]")
    
    # Import sync manager here to avoid issues
    from src.core.sync_manager import SyncManager
    
    # The sync manager will use the default embedding model name to create collection
    # But our collection has a different name, so we need to patch this
    console.print("\n[yellow]⚠ Note: Current sync manager uses default naming convention[/yellow]")
    console.print("[yellow]For now, we'll use the existing process_remaining.py script[/yellow]")
    console.print("[yellow]In the future, we'll update sync manager for policyQA naming[/yellow]")
    
    console.print(f"\n[bold]📋 Next Steps:[/bold]")
    console.print("1. The Q&A collection is ready: policyQA_dense_qa768")
    console.print("2. We need to update the sync manager to use this collection")
    console.print("3. Or create a temporary collection with the expected name")
    
    console.print(f"\n[bold cyan]🔧 Quick Fix Options:[/bold cyan]")
    console.print("A. Create temporary collection with expected name:")
    console.print("   curl -X PUT http://localhost:6333/collections/policy_docs_sentence_transformers \\")
    console.print("        -H 'Content-Type: application/json' \\")
    console.print("        -d '{\"vectors\": {\"size\": 768, \"distance\": \"Cosine\"}}'")
    console.print("\nB. Update sync manager to use policyQA naming (better long-term)")

if __name__ == "__main__":
    main()