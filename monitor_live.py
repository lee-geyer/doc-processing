#!/usr/bin/env python3
"""
Live monitoring script for document processing.
Run this in a separate terminal to watch progress.
"""

import time
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from sqlalchemy import func, desc

from src.models.database import SessionLocal, FileSyncState, VectorSyncOperation
from src.core.vector_store import policy_vector_store

console = Console()


def get_stats():
    """Get current processing statistics."""
    db = SessionLocal()
    try:
        # File counts
        status_counts = dict(
            db.query(FileSyncState.sync_status, func.count(FileSyncState.id))
            .group_by(FileSyncState.sync_status)
            .all()
        )
        
        # Recent operations
        recent_ops = db.query(VectorSyncOperation).filter_by(
            status='success'
        ).order_by(desc(VectorSyncOperation.created_at)).limit(5).all()
        
        # Error files
        error_files = db.query(FileSyncState).filter_by(
            sync_status='error'
        ).order_by(desc(FileSyncState.updated_at)).limit(5).all()
        
        return {
            'counts': status_counts,
            'recent_ops': recent_ops,
            'error_files': error_files
        }
    finally:
        db.close()


def get_vector_count():
    """Get current vector count from Qdrant."""
    try:
        # Use the policyQA collection
        collection_name = "policyQA_dense_qa768"
        import requests
        response = requests.get(f'http://localhost:6333/collections/{collection_name}')
        if response.status_code == 200:
            return response.json()['result']['points_count']
        return 0
    except:
        return 0


def create_display():
    """Create the monitoring display."""
    stats = get_stats()
    vector_count = get_vector_count()
    
    # Calculate totals
    synced = stats['counts'].get('synced', 0)
    pending = stats['counts'].get('pending', 0)
    processing = stats['counts'].get('processing', 0)
    error = stats['counts'].get('error', 0)
    total = synced + pending + processing + error
    progress_pct = (synced / total * 100) if total > 0 else 0
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="stats", size=8),
        Layout(name="activity", size=10),
        Layout(name="errors", size=8)
    )
    
    # Header
    layout["header"].update(
        Panel(f"[bold cyan]Document Processing Monitor[/bold cyan] - {datetime.now().strftime('%H:%M:%S')}")
    )
    
    # Stats table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Progress:", f"[green]{synced}[/green]/{total} ({progress_pct:.1f}%)")
    stats_table.add_row("Processing:", f"[yellow]{processing}[/yellow]")
    stats_table.add_row("Pending:", str(pending))
    stats_table.add_row("Errors:", f"[red]{error}[/red]" if error > 0 else "0")
    stats_table.add_row("Vectors:", f"{vector_count:,}")
    
    # Calculate ETA
    if stats['recent_ops'] and pending > 0:
        recent_times = [op.execution_time_ms for op in stats['recent_ops'][:10]]
        avg_time = sum(recent_times) / len(recent_times) / 1000  # seconds
        eta_seconds = pending * avg_time
        eta_hours = eta_seconds / 3600
        stats_table.add_row("Est. Time:", f"{eta_hours:.1f} hours")
    
    layout["stats"].update(Panel(stats_table, title="Statistics"))
    
    # Recent activity
    activity_table = Table(show_header=True)
    activity_table.add_column("Time", style="dim", width=8)
    activity_table.add_column("File", style="green", overflow="ellipsis")
    activity_table.add_column("Time", style="cyan", width=8)
    
    for op in stats['recent_ops']:
        age = (datetime.now() - op.created_at).total_seconds()
        if age < 60:
            time_str = f"{age:.0f}s ago"
        else:
            time_str = f"{age/60:.0f}m ago"
        
        filename = op.file_path.split('/')[-1] if op.file_path else "Unknown"
        exec_time = f"{op.execution_time_ms/1000:.1f}s"
        
        activity_table.add_row(time_str, filename[:50], exec_time)
    
    layout["activity"].update(Panel(activity_table, title="Recent Completions"))
    
    # Errors
    if stats['error_files']:
        error_table = Table(show_header=True)
        error_table.add_column("File", style="red", overflow="ellipsis")
        error_table.add_column("Error", style="dim", overflow="ellipsis")
        
        for file in stats['error_files']:
            filename = file.file_path.split('/')[-1]
            error_msg = file.last_error_message[:50] + "..." if file.last_error_message else "Unknown"
            error_table.add_row(filename[:40], error_msg)
        
        layout["errors"].update(Panel(error_table, title="Recent Errors"))
    else:
        layout["errors"].update(Panel("[green]No errors[/green]", title="Errors"))
    
    return layout


def main():
    """Run the live monitor."""
    console.print("[cyan]Starting live monitor...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    try:
        with Live(create_display(), refresh_per_second=0.5) as live:
            while True:
                live.update(create_display())
                time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()