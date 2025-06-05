#!/usr/bin/env python3
"""
Comprehensive monitoring script for document processing completion.
Run this to track progress while processing continues in background.
"""

import time
import sys
import json
import requests
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from sqlalchemy import func, desc

from src.models.database import SessionLocal, FileSyncState, VectorSyncOperation, Document
from src.core.vector_store import policy_vector_store

console = Console()


class ProcessingTracker:
    """Track processing progress and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.checkpoints = []
        self.last_synced = 0
        
    def get_stats(self):
        """Get comprehensive processing statistics."""
        db = SessionLocal()
        try:
            # File counts
            status_counts = dict(
                db.query(FileSyncState.sync_status, func.count(FileSyncState.id))
                .group_by(FileSyncState.sync_status)
                .all()
            )
            
            synced = status_counts.get('synced', 0)
            pending = status_counts.get('pending', 0)
            processing = status_counts.get('processing', 0)
            error = status_counts.get('error', 0)
            total = synced + pending + processing + error
            
            # Calculate processing rate
            current_time = datetime.now()
            elapsed_seconds = (current_time - self.start_time).total_seconds()
            
            if synced > self.last_synced:
                files_processed = synced - self.last_synced
                self.checkpoints.append({
                    'time': current_time,
                    'synced': synced,
                    'processed_since_last': files_processed
                })
                self.last_synced = synced
                
                # Keep only last 10 checkpoints
                if len(self.checkpoints) > 10:
                    self.checkpoints.pop(0)
            
            # Calculate rate from recent checkpoints
            if len(self.checkpoints) >= 2:
                recent_start = self.checkpoints[-5]['time'] if len(self.checkpoints) >= 5 else self.checkpoints[0]['time']
                recent_processed = synced - (self.checkpoints[-5]['synced'] if len(self.checkpoints) >= 5 else self.checkpoints[0]['synced'])
                recent_elapsed = (current_time - recent_start).total_seconds()
                
                if recent_elapsed > 0:
                    files_per_second = recent_processed / recent_elapsed
                    files_per_minute = files_per_second * 60
                    files_per_hour = files_per_second * 3600
                else:
                    files_per_minute = files_per_hour = 0
            else:
                files_per_minute = files_per_hour = 0
            
            # Estimate completion time
            if files_per_hour > 0 and pending > 0:
                hours_remaining = pending / files_per_hour
                eta = current_time + timedelta(hours=hours_remaining)
            else:
                eta = None
            
            # Vector count
            try:
                vector_count = self._get_vector_count()
            except:
                vector_count = 0
            
            # Recent completions
            recent_ops = db.query(VectorSyncOperation).filter_by(
                status='success'
            ).order_by(desc(VectorSyncOperation.created_at)).limit(5).all()
            
            # Error breakdown
            error_files = db.query(FileSyncState).filter_by(
                sync_status='error'
            ).order_by(desc(FileSyncState.updated_at)).limit(3).all()
            
            return {
                'synced': synced,
                'pending': pending,
                'processing': processing,
                'error': error,
                'total': total,
                'progress_pct': (synced / total * 100) if total > 0 else 0,
                'files_per_minute': files_per_minute,
                'files_per_hour': files_per_hour,
                'eta': eta,
                'elapsed_hours': elapsed_seconds / 3600,
                'vector_count': vector_count,
                'recent_ops': recent_ops,
                'error_files': error_files
            }
        finally:
            db.close()
    
    def _get_vector_count(self):
        """Get current vector count."""
        try:
            response = requests.get('http://localhost:6333/collections/policy_docs_sentence_transformers', timeout=5)
            data = response.json()
            return data['result']['points_count']
        except:
            return 0


def create_monitoring_display(tracker):
    """Create comprehensive monitoring display."""
    stats = tracker.get_stats()
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=18),
        Layout(name="footer", size=3)
    )
    
    # Header
    header_text = Text(f"🚀 Document Processing Monitor - {datetime.now().strftime('%H:%M:%S')}", style="bold cyan")
    layout["header"].update(Panel(Align.center(header_text)))
    
    # Main section
    layout["main"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="activity", ratio=3)
    )
    
    # Progress panel
    progress_table = Table(show_header=False, box=None, padding=(0, 1))
    progress_table.add_column("Metric", style="cyan", width=18)
    progress_table.add_column("Value", style="white")
    
    # Progress info
    progress_table.add_row("Files Processed:", f"[green]{stats['synced']:,}[/green] / {stats['total']:,}")
    progress_table.add_row("Progress:", f"[bold]{stats['progress_pct']:.1f}%[/bold]")
    progress_table.add_row("Remaining:", f"{stats['pending']:,} files")
    progress_table.add_row("Processing:", f"[yellow]{stats['processing']:,}[/yellow]")
    progress_table.add_row("Errors:", f"[red]{stats['error']:,}[/red]" if stats['error'] > 0 else "0")
    progress_table.add_row("", "")  # Spacer
    
    # Performance metrics
    progress_table.add_row("Current Rate:", f"{stats['files_per_minute']:.1f} files/min")
    progress_table.add_row("Hourly Rate:", f"{stats['files_per_hour']:.0f} files/hour")
    progress_table.add_row("Runtime:", f"{stats['elapsed_hours']:.1f} hours")
    
    if stats['eta']:
        progress_table.add_row("Est. Completion:", stats['eta'].strftime('%H:%M'))
        hours_left = (stats['eta'] - datetime.now()).total_seconds() / 3600
        progress_table.add_row("Time Remaining:", f"{hours_left:.1f} hours")
    
    progress_table.add_row("", "")  # Spacer
    progress_table.add_row("Vectors Created:", f"{stats['vector_count']:,}")
    
    if stats['vector_count'] > 0 and stats['synced'] > 0:
        avg_vectors = stats['vector_count'] / stats['synced']
        total_estimate = avg_vectors * stats['total']
        progress_table.add_row("Est. Total Vectors:", f"{total_estimate:,.0f}")
    
    layout["progress"].update(Panel(progress_table, title="Processing Statistics"))
    
    # Activity section
    layout["activity"].split_column(
        Layout(name="recent", ratio=1),
        Layout(name="errors", ratio=1)
    )
    
    # Recent completions
    if stats['recent_ops']:
        recent_table = Table(show_header=True)
        recent_table.add_column("File", style="green", overflow="ellipsis", width=35)
        recent_table.add_column("Time", style="cyan", width=8)
        recent_table.add_column("Age", style="dim", width=8)
        
        for op in stats['recent_ops']:
            filename = op.file_path.split('/')[-1] if op.file_path else "Unknown"
            exec_time = f"{op.execution_time_ms/1000:.1f}s"
            age = (datetime.now() - op.created_at).total_seconds()
            
            if age < 60:
                age_str = f"{age:.0f}s"
            elif age < 3600:
                age_str = f"{age/60:.0f}m"
            else:
                age_str = f"{age/3600:.1f}h"
            
            recent_table.add_row(filename[:35], exec_time, age_str)
        
        layout["recent"].update(Panel(recent_table, title="Recent Completions"))
    else:
        layout["recent"].update(Panel("[dim]No recent activity[/dim]", title="Recent Completions"))
    
    # Error details
    if stats['error_files']:
        error_table = Table(show_header=True)
        error_table.add_column("File", style="red", overflow="ellipsis", width=25)
        error_table.add_column("Error", style="dim", overflow="ellipsis")
        
        for file in stats['error_files']:
            filename = file.file_path.split('/')[-1]
            error_msg = file.last_error_message
            if error_msg and len(error_msg) > 40:
                error_msg = error_msg[:37] + "..."
            error_table.add_row(filename[:25], error_msg or "Unknown")
        
        layout["errors"].update(Panel(error_table, title="Recent Errors"))
    else:
        layout["errors"].update(Panel("[green]No errors[/green]", title="Errors"))
    
    # Footer with completion estimate
    if stats['eta']:
        footer_text = f"📅 Estimated completion: {stats['eta'].strftime('%Y-%m-%d %H:%M')} | " \
                     f"🎯 Final count: ~{stats['vector_count'] / max(stats['synced'], 1) * stats['total']:,.0f} vectors"
    else:
        footer_text = f"⏱️ Runtime: {stats['elapsed_hours']:.1f} hours | 📊 {stats['vector_count']:,} vectors created"
    
    layout["footer"].update(Panel(footer_text))
    
    return layout


def main():
    """Run the comprehensive monitor."""
    console.print("[bold cyan]🔍 Comprehensive Processing Monitor[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
    
    tracker = ProcessingTracker()
    
    try:
        with Live(refresh_per_second=0.5) as live:
            while True:
                display = create_monitoring_display(tracker)
                live.update(display)
                
                # Check if processing is complete
                stats = tracker.get_stats()
                if stats['pending'] == 0 and stats['processing'] == 0:
                    console.print("\n[bold green]🎉 Processing Complete![/bold green]")
                    console.print(f"✅ Total files processed: {stats['synced']:,}")
                    console.print(f"🔢 Total vectors created: {stats['vector_count']:,}")
                    console.print(f"⏱️ Total time: {stats['elapsed_hours']:.1f} hours")
                    if stats['files_per_hour'] > 0:
                        console.print(f"⚡ Average rate: {stats['files_per_hour']:.0f} files/hour")
                    break
                
                time.sleep(2)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]📊 Final Status:[/yellow]")
        final_stats = tracker.get_stats()
        console.print(f"  Files processed: {final_stats['synced']:,}/{final_stats['total']:,} ({final_stats['progress_pct']:.1f}%)")
        console.print(f"  Vectors created: {final_stats['vector_count']:,}")
        console.print(f"  Runtime: {final_stats['elapsed_hours']:.1f} hours")
        console.print("\n[dim]Processing continues in background...[/dim]")


if __name__ == "__main__":
    main()