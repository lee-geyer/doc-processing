#!/usr/bin/env python3
"""
Process all documents with the new Q&A optimized model.
"""

import time
from datetime import datetime
from src.core.sync_manager import SyncManager
from src.core.embeddings import embedding_manager, ProviderConfig
from src.core.vector_store import PolicyVectorStore
from src.models.database import SessionLocal, FileSyncState
from rich.console import Console

console = Console()

def get_pending_count():
    """Get count of pending files."""
    db = SessionLocal()
    try:
        return db.query(FileSyncState).filter_by(sync_status='pending').count()
    finally:
        db.close()

def main():
    """Process all files with Q&A model."""
    console.print("[bold cyan]🚀 Processing with Q&A Optimized Model[/bold cyan]")
    
    # Configure Q&A model
    qa_config = ProviderConfig(
        provider_type='sentence_transformers',
        model_name='multi-qa-mpnet-base-dot-v1',
        dimension=768,
        batch_size=32
    )
    
    # Get embedding provider
    embedding_provider = embedding_manager.get_provider('qa_model', qa_config)
    console.print(f"✓ Using model: {embedding_provider.model_name}")
    
    # Create vector store
    vector_store = PolicyVectorStore()
    collection_name = "policy_docs_qa_optimized"
    
    console.print(f"✓ Target collection: {collection_name}")
    
    # Create sync manager
    sync_manager = SyncManager()
    
    # Check pending files
    pending_count = get_pending_count()
    console.print(f"✓ Files to process: {pending_count}")
    
    if pending_count == 0:
        console.print("[green]No files to process![/green]")
        return
    
    # Process files
    total_processed = 0
    batch_size = 20
    start_time = datetime.now()
    
    console.print(f"\n[cyan]Starting processing in batches of {batch_size}...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop gracefully[/dim]\n")
    
    try:
        while True:
            pending = get_pending_count()
            if pending == 0:
                break
            
            console.print(f"[cyan]Processing batch (remaining: {pending})[/cyan]")
            
            # Process batch
            result = sync_manager.process_pending_files(limit=batch_size)
            
            if result['processed'] > 0:
                total_processed += result['processed']
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_processed / elapsed * 3600 if elapsed > 0 else 0
                
                console.print(f"✓ Batch: {result['succeeded']} succeeded, {result['failed']} failed")
                console.print(f"📊 Rate: {rate:.0f} files/hour | Total: {total_processed}")
            else:
                console.print("[yellow]⚠ No files processed in this batch[/yellow]")
                time.sleep(5)  # Wait before retry
            
            time.sleep(1)  # Brief pause between batches
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Graceful shutdown requested...[/yellow]")
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"\n[bold]📈 Processing Summary:[/bold]")
    console.print(f"  Model: multi-qa-mpnet-base-dot-v1 (Q&A optimized)")
    console.print(f"  Total processed: {total_processed} files")
    console.print(f"  Runtime: {elapsed/3600:.1f} hours")
    if elapsed > 0:
        console.print(f"  Average rate: {total_processed/elapsed*3600:.0f} files/hour")

if __name__ == "__main__":
    main()