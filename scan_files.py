#!/usr/bin/env python3
"""
Scan mirror directory and populate database with file records.
"""

import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models.database import SessionLocal, FileSyncState
from src.utils.file_utils import calculate_file_hash

console = Console()

def scan_mirror_directory():
    """Scan mirror directory and add files to database."""
    mirror_path = '/Users/leegeyer/Library/CloudStorage/OneDrive-Extendicare(Canada)Inc/INTEGRATION FILES TARGETED'
    
    console.print(f"[cyan]Scanning mirror directory...[/cyan]")
    console.print(f"Path: {mirror_path}")
    
    if not os.path.exists(mirror_path):
        console.print(f"[red]❌ Mirror directory does not exist![/red]")
        return
    
    # Count files first
    console.print("[dim]Counting files...[/dim]")
    file_count = 0
    for root, dirs, files in os.walk(mirror_path):
        for file in files:
            if not file.startswith('.') and file.lower().endswith(('.pdf', '.docx', '.doc')):
                file_count += 1
    
    console.print(f"Found [yellow]{file_count}[/yellow] document files")
    
    if file_count == 0:
        console.print("[yellow]No files to process[/yellow]")
        return
    
    # Add files to database
    db = SessionLocal()
    try:
        added = 0
        skipped = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Adding files to database...", total=file_count)
            
            for root, dirs, files in os.walk(mirror_path):
                for file in files:
                    if not file.startswith('.') and file.lower().endswith(('.pdf', '.docx', '.doc')):
                        file_path = os.path.join(root, file)
                        
                        # Check if already exists
                        existing = db.query(FileSyncState).filter_by(file_path=file_path).first()
                        if existing:
                            skipped += 1
                        else:
                            try:
                                file_stats = os.stat(file_path)
                                file_hash = calculate_file_hash(file_path)
                                
                                file_state = FileSyncState(
                                    file_path=file_path,
                                    file_hash=file_hash,
                                    file_size_bytes=file_stats.st_size,
                                    last_modified=datetime.fromtimestamp(file_stats.st_mtime),
                                    sync_status='pending'
                                )
                                db.add(file_state)
                                added += 1
                                
                                if added % 50 == 0:
                                    db.commit()
                                    
                            except Exception as e:
                                console.print(f"[red]Error adding {file}: {e}[/red]")
                                continue
                        
                        progress.advance(task)
        
        db.commit()
        
        # Final statistics
        total = db.query(FileSyncState).count()
        pending = db.query(FileSyncState).filter_by(sync_status='pending').count()
        
        console.print(f"\n[green]✓ Scan complete![/green]")
        console.print(f"Added: [green]{added}[/green] new files")
        console.print(f"Skipped: [yellow]{skipped}[/yellow] existing files") 
        console.print(f"Database total: [cyan]{total}[/cyan] files")
        console.print(f"Ready to process: [yellow]{pending}[/yellow] pending files")
        
    except Exception as e:
        console.print(f"[red]❌ Database error: {e}[/red]")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    scan_mirror_directory()