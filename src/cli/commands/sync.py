import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from pathlib import Path

from src.core.file_monitor import FileMonitor
from src.core.sync_manager import SyncManager
from src.models.database import SessionLocal, FileSyncState
from src.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def scan(
    force: bool = typer.Option(False, "--force", "-f", help="Force rescan all files"),
):
    """Scan directory for file changes."""
    try:
        monitor = FileMonitor()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning directory...", total=None)
            
            stats = monitor.scan_directory(force_rescan=force)
            
            progress.update(task, completed=True)
        
        # Display results
        table = Table(title="Scan Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Total Files", str(stats['total_files']))
        table.add_row("New Files", str(stats['new_files']))
        table.add_row("Modified Files", str(stats['modified_files']))
        table.add_row("Deleted Files", str(stats['deleted_files']))
        table.add_row("Unchanged Files", str(stats['unchanged_files']))
        
        console.print(table)
        
        if stats['new_files'] + stats['modified_files'] > 0:
            console.print(f"\n[green]✓[/green] Found {stats['new_files'] + stats['modified_files']} files to process")
        else:
            console.print("\n[dim]No new or modified files found[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error scanning directory: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process(
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Number of files to process in batch"),
):
    """Process pending files with document parsing and text cleaning."""
    try:
        sync_manager = SyncManager()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing pending files...", total=None)
            
            stats = sync_manager.process_pending_files(limit=batch_size)
            
            progress.update(task, completed=True)
        
        # Display results
        table = Table(title="Processing Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Processed", str(stats['processed']))
        table.add_row("Succeeded", str(stats['succeeded']))
        table.add_row("Failed", str(stats['failed']))
        table.add_row("Skipped", str(stats['skipped']))
        
        if stats['processed'] > 0:
            success_rate = stats['succeeded'] / stats['processed']
            table.add_row("Success Rate", f"{success_rate:.1%}")
        
        console.print(table)
        
        if stats['succeeded'] > 0:
            console.print(f"\n[green]✓[/green] Successfully processed {stats['succeeded']} files")
        
        if stats['failed'] > 0:
            console.print(f"[yellow]⚠[/yellow] {stats['failed']} files failed processing")
        
    except Exception as e:
        console.print(f"[red]Error processing files: {e}[/red]")
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def file(
    file_path: str = typer.Argument(..., help="Path to file to sync"),
    force: bool = typer.Option(False, "--force", "-f", help="Force sync even if unchanged"),
):
    """Force sync a specific file."""
    file_path = Path(file_path).absolute()
    
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    db = SessionLocal()
    try:
        # Find or create sync state
        sync_state = db.query(FileSyncState).filter_by(file_path=str(file_path)).first()
        
        if sync_state:
            if force or sync_state.sync_status != 'synced':
                sync_state.sync_status = 'pending'
                db.commit()
                console.print(f"[green]✓[/green] File marked for sync: {file_path.name}")
            else:
                console.print(f"[dim]File already synced: {file_path.name}[/dim]")
        else:
            console.print(f"[yellow]File not in database. Run 'sync scan' first.[/yellow]")
    
    except Exception as e:
        db.rollback()
        console.print(f"[red]Error syncing file: {e}[/red]")
        raise typer.Exit(1)
    finally:
        db.close()


@app.command()
def reset():
    """Reset sync state for all files."""
    confirm = typer.confirm("Are you sure you want to reset all sync states?")
    
    if not confirm:
        console.print("[yellow]Operation cancelled[/yellow]")
        return
    
    db = SessionLocal()
    try:
        # Reset all sync states to pending
        count = db.query(FileSyncState).update({'sync_status': 'pending'})
        db.commit()
        
        console.print(f"[green]✓[/green] Reset sync state for {count} files")
        
    except Exception as e:
        db.rollback()
        console.print(f"[red]Error resetting sync state: {e}[/red]")
        raise typer.Exit(1)
    finally:
        db.close()