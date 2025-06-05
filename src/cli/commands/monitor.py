import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from datetime import datetime

from src.core.file_monitor import FileMonitor
from src.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def start(
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
):
    """Start file monitoring daemon."""
    try:
        monitor = FileMonitor()
        
        # Perform initial scan
        console.print("[cyan]Performing initial directory scan...[/cyan]")
        stats = monitor.scan_directory()
        
        # Display scan results
        table = Table(title="Initial Scan Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Total Files", str(stats['total_files']))
        table.add_row("New Files", str(stats['new_files']))
        table.add_row("Modified Files", str(stats['modified_files']))
        table.add_row("Deleted Files", str(stats['deleted_files']))
        table.add_row("Unchanged Files", str(stats['unchanged_files']))
        
        console.print(table)
        
        # Start monitoring
        monitor.start()
        console.print("\n[green]✓[/green] File monitoring started successfully")
        console.print(f"[dim]Monitoring: {monitor.mirror_directory}[/dim]")
        
        if not background:
            console.print("\n[yellow]Press Ctrl+C to stop monitoring[/yellow]")
            try:
                # Keep the program running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[cyan]Stopping file monitor...[/cyan]")
                monitor.stop()
                console.print("[green]✓[/green] File monitoring stopped")
        
    except Exception as e:
        console.print(f"[red]Error starting file monitor: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop file monitoring daemon."""
    # TODO: Implement proper daemon management
    console.print("[yellow]Stop command not yet implemented[/yellow]")
    console.print("[dim]For now, use Ctrl+C to stop the monitor[/dim]")


@app.command()
def restart():
    """Restart file monitoring daemon."""
    # TODO: Implement proper daemon management
    console.print("[yellow]Restart command not yet implemented[/yellow]")


@app.command()
def status():
    """Check file monitoring status."""
    try:
        monitor = FileMonitor()
        status = monitor.get_status()
        
        # Create status panel
        status_text = f"""
[bold]Monitor Status:[/bold] {'[green]Running[/green]' if status['is_running'] else '[red]Stopped[/red]'}
[bold]Directory:[/bold] {status['monitored_directory']}
[bold]Queue Size:[/bold] {status['queue_size']}
[bold]Last Scan:[/bold] {status['last_scan'].strftime('%Y-%m-%d %H:%M:%S') if status['last_scan'] else 'Never'}
        """
        
        panel = Panel(
            status_text.strip(),
            title="File Monitor Status",
            border_style="cyan"
        )
        console.print(panel)
        
        # Create file counts table
        table = Table(title="File Status Summary", box=box.ROUNDED)
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Total Files", str(status['total_files']))
        table.add_row("Pending", str(status['pending_files']))
        table.add_row("Processing", str(status['processing_files']))
        table.add_row("Synced", str(status['synced_files']))
        table.add_row("Errors", str(status['error_files']))
        table.add_row("Deleted", str(status['deleted_files']))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting monitor status: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def progress(
    refresh: int = typer.Option(5, "--refresh", "-r", help="Refresh interval in seconds"),
):
    """Monitor document processing progress in real-time."""
    from src.cli.commands.monitor_progress import monitor_progress
    
    try:
        console.print("[cyan]Starting progress monitoring...[/cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        monitor_progress(refresh_interval=refresh)
    except Exception as e:
        console.print(f"\n[red]Error monitoring progress: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report():
    """Generate detailed processing report."""
    from src.cli.commands.monitor_progress import print_detailed_report
    
    try:
        print_detailed_report()
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(1)