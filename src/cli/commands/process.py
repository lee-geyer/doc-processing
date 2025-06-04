import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich import box
from pathlib import Path
from typing import Optional

from src.core.document_parser import DocumentParser, DocumentParsingError
from src.core.text_cleaner import TextCleaner
from src.utils.context_utils import ContextExtractor
from src.models.database import SessionLocal, FileSyncState
from src.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def file(
    file_path: str = typer.Argument(..., help="Path to file to process"),
    show_content: bool = typer.Option(False, "--show-content", "-c", help="Show processed content"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show processing statistics"),
):
    """Process a single file and show results."""
    file_path = Path(file_path).absolute()
    
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    if not file_path.suffix.lower() in {'.pdf', '.docx', '.doc'}:
        console.print(f"[red]Unsupported file type: {file_path.suffix}[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(console=console) as progress:
            # Parse document
            task = progress.add_task("Parsing document...", total=4)
            
            parser = DocumentParser()
            parsed_doc = parser.parse_document(str(file_path))
            progress.update(task, advance=1)
            
            if not parsed_doc.parsing_success:
                console.print(f"[red]Failed to parse document: {parsed_doc.parsing_error_message}[/red]")
                raise typer.Exit(1)
            
            # Clean text
            progress.update(task, description="Cleaning text...")
            cleaner = TextCleaner()
            cleaning_result = cleaner.clean_text(parsed_doc.raw_text or "")
            progress.update(task, advance=1)
            
            # Extract context
            progress.update(task, description="Extracting context...")
            context_extractor = ContextExtractor()
            context = context_extractor.extract_context(str(file_path))
            progress.update(task, advance=1)
            
            # Validate
            progress.update(task, description="Validating results...")
            validation = context_extractor.validate_context(context)
            progress.update(task, advance=1, description="Complete!")
        
        # Display results
        _display_processing_results(
            file_path, parsed_doc, cleaning_result, context, validation, show_content, show_stats
        )
        
    except DocumentParsingError as e:
        console.print(f"[red]Parsing error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def batch(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of files to process"),
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed results for each file"),
):
    """Process multiple pending files."""
    db = SessionLocal()
    
    try:
        # Get pending files
        pending_files = db.query(FileSyncState).filter_by(sync_status='pending').limit(limit).all()
        
        if not pending_files:
            console.print("[dim]No pending files to process[/dim]")
            return
        
        console.print(f"[cyan]Processing {len(pending_files)} pending files...[/cyan]")
        
        # Initialize processors
        parser = DocumentParser()
        cleaner = TextCleaner()
        context_extractor = ContextExtractor()
        
        # Track statistics
        stats = {
            'total': len(pending_files),
            'successful': 0,
            'failed': 0,
            'total_words': 0,
            'total_pages': 0
        }
        
        with Progress(console=console) as progress:
            main_task = progress.add_task("Processing files...", total=len(pending_files))
            
            for file_state in pending_files:
                file_path = file_state.file_path
                progress.update(main_task, description=f"Processing {Path(file_path).name}...")
                
                try:
                    # Parse document
                    parsed_doc = parser.parse_document(file_path)
                    
                    if parsed_doc.parsing_success:
                        # Clean text
                        cleaning_result = cleaner.clean_text(parsed_doc.raw_text or "")
                        
                        # Extract context
                        context = context_extractor.extract_context(file_path)
                        
                        # Update statistics
                        stats['successful'] += 1
                        stats['total_words'] += parsed_doc.word_count or 0
                        stats['total_pages'] += parsed_doc.page_count or 0
                        
                        if show_details:
                            console.print(f"[green]✓[/green] {Path(file_path).name}")
                            console.print(f"  Words: {parsed_doc.word_count}, Pages: {parsed_doc.page_count}")
                    else:
                        stats['failed'] += 1
                        if show_details:
                            console.print(f"[red]✗[/red] {Path(file_path).name}: {parsed_doc.parsing_error_message}")
                
                except Exception as e:
                    stats['failed'] += 1
                    if show_details:
                        console.print(f"[red]✗[/red] {Path(file_path).name}: {e}")
                
                progress.update(main_task, advance=1)
        
        # Display summary
        _display_batch_results(stats)
        
    except Exception as e:
        console.print(f"[red]Error processing batch: {e}[/red]")
        raise typer.Exit(1)
    finally:
        db.close()


@app.command()
def validate(
    file_path: Optional[str] = typer.Argument(None, help="Path to specific file to validate"),
    sample_size: int = typer.Option(10, "--sample", "-s", help="Number of files to validate (if no specific file)"),
):
    """Validate parsing quality for files."""
    if file_path:
        # Validate single file
        _validate_single_file(file_path)
    else:
        # Validate sample of files
        _validate_sample_files(sample_size)


def _validate_single_file(file_path: str):
    """Validate parsing quality for a single file."""
    file_path = Path(file_path).absolute()
    
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        parser = DocumentParser()
        parsed_doc = parser.parse_document(str(file_path))
        
        if not parsed_doc.parsing_success:
            console.print(f"[red]Parsing failed: {parsed_doc.parsing_error_message}[/red]")
            return
        
        # Validate parsing quality
        quality = parser.validate_parsing_quality(parsed_doc)
        
        # Display validation results
        panel_content = f"""
[bold]File:[/bold] {file_path.name}
[bold]Valid:[/bold] {'[green]Yes[/green]' if quality['is_valid'] else '[red]No[/red]'}
[bold]Word Count:[/bold] {quality['word_count']}
[bold]Character Count:[/bold] {quality['char_count']}
[bold]Content Ratio:[/bold] {quality['content_ratio']:.2f}
[bold]Repetition Ratio:[/bold] {quality['repetition_ratio']:.2f}
        """
        
        if not quality['is_valid']:
            if 'error' in quality:
                panel_content += f"\n[bold red]Error:[/bold red] {quality['error']}"
        
        panel = Panel(
            panel_content.strip(),
            title="Parsing Quality Validation",
            border_style="green" if quality['is_valid'] else "red"
        )
        console.print(panel)
        
    except Exception as e:
        console.print(f"[red]Error validating file: {e}[/red]")
        raise typer.Exit(1)


def _validate_sample_files(sample_size: int):
    """Validate parsing quality for a sample of files."""
    db = SessionLocal()
    
    try:
        # Get sample of files
        files = db.query(FileSyncState).limit(sample_size).all()
        
        if not files:
            console.print("[dim]No files found to validate[/dim]")
            return
        
        console.print(f"[cyan]Validating {len(files)} files...[/cyan]")
        
        parser = DocumentParser()
        validation_results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task("Validating files...", total=len(files))
            
            for file_state in files:
                try:
                    parsed_doc = parser.parse_document(file_state.file_path)
                    
                    if parsed_doc.parsing_success:
                        quality = parser.validate_parsing_quality(parsed_doc)
                        validation_results.append({
                            'file': Path(file_state.file_path).name,
                            'valid': quality['is_valid'],
                            'word_count': quality['word_count'],
                            'content_ratio': quality['content_ratio']
                        })
                    else:
                        validation_results.append({
                            'file': Path(file_state.file_path).name,
                            'valid': False,
                            'word_count': 0,
                            'content_ratio': 0.0
                        })
                
                except Exception as e:
                    validation_results.append({
                        'file': Path(file_state.file_path).name,
                        'valid': False,
                        'word_count': 0,
                        'content_ratio': 0.0
                    })
                
                progress.update(task, advance=1)
        
        # Display results
        _display_validation_results(validation_results)
        
    finally:
        db.close()


def _display_processing_results(file_path, parsed_doc, cleaning_result, context, validation, show_content, show_stats):
    """Display processing results for a single file."""
    # File info
    console.print(f"\n[bold cyan]Processing Results: {file_path.name}[/bold cyan]")
    
    # Parsing results
    if show_stats:
        table = Table(title="Document Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("File Type", parsed_doc.file_type.upper())
        table.add_row("Parsing Method", parsed_doc.parsing_method)
        table.add_row("Page Count", str(parsed_doc.page_count))
        table.add_row("Word Count", str(parsed_doc.word_count))
        table.add_row("Processing Time", f"{parsed_doc.parsing_duration_ms}ms")
        
        console.print(table)
    
    # Context information
    context_content = f"""
[bold]Policy Manual:[/bold] {context.policy_manual or 'Unknown'}
[bold]Policy Acronym:[/bold] {context.policy_acronym or 'Unknown'}
[bold]Section:[/bold] {context.section or 'Unknown'}
[bold]Document Index:[/bold] {context.document_index or 'None'}
[bold]Document Type:[/bold] {context.document_type or 'Unknown'}
[bold]Additional Resource:[/bold] {'Yes' if context.is_additional_resource else 'No'}
    """
    
    panel = Panel(
        context_content.strip(),
        title="Document Context",
        border_style="cyan"
    )
    console.print(panel)
    
    # Cleaning results
    if show_stats:
        stats = cleaning_result['stats']
        cleaning_content = f"""
[bold]Original Length:[/bold] {stats.original_length:,} characters
[bold]Cleaned Length:[/bold] {stats.cleaned_length:,} characters
[bold]Reduction:[/bold] {stats.reduction_ratio:.1%}
[bold]Patterns Removed:[/bold] {len(stats.patterns_removed)}
        """
        
        if stats.excessive_cleaning:
            cleaning_content += "\n[bold red]Warning:[/bold red] Excessive cleaning detected"
        
        panel = Panel(
            cleaning_content.strip(),
            title="Text Cleaning Results",
            border_style="yellow" if stats.excessive_cleaning else "green"
        )
        console.print(panel)
    
    # Validation results
    if validation['warnings'] or not validation['is_valid']:
        validation_content = f"[bold]Valid:[/bold] {'Yes' if validation['is_valid'] else 'No'}\n"
        validation_content += f"[bold]Completeness:[/bold] {validation['context_completeness']:.1%}\n"
        
        if validation['issues']:
            validation_content += f"\n[bold red]Issues:[/bold red]\n"
            for issue in validation['issues']:
                validation_content += f"  • {issue}\n"
        
        if validation['warnings']:
            validation_content += f"\n[bold yellow]Warnings:[/bold yellow]\n"
            for warning in validation['warnings']:
                validation_content += f"  • {warning}\n"
        
        panel = Panel(
            validation_content.strip(),
            title="Validation Results",
            border_style="red" if not validation['is_valid'] else "yellow"
        )
        console.print(panel)
    
    # Content preview
    if show_content:
        cleaned_text = cleaning_result['cleaned_text']
        preview = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
        
        panel = Panel(
            preview,
            title="Cleaned Content Preview",
            border_style="dim"
        )
        console.print(panel)


def _display_batch_results(stats):
    """Display batch processing results."""
    table = Table(title="Batch Processing Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Total Files", str(stats['total']))
    table.add_row("Successful", str(stats['successful']))
    table.add_row("Failed", str(stats['failed']))
    table.add_row("Success Rate", f"{(stats['successful'] / max(stats['total'], 1)):.1%}")
    table.add_row("Total Words", f"{stats['total_words']:,}")
    table.add_row("Total Pages", f"{stats['total_pages']:,}")
    
    if stats['successful'] > 0:
        table.add_row("Avg Words/File", f"{stats['total_words'] // stats['successful']:,}")
        table.add_row("Avg Pages/File", f"{stats['total_pages'] // stats['successful']}")
    
    console.print(table)


def _display_validation_results(results):
    """Display validation results."""
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    # Summary
    table = Table(title="Validation Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Total Files", str(total_count))
    table.add_row("Valid Files", str(valid_count))
    table.add_row("Invalid Files", str(total_count - valid_count))
    table.add_row("Success Rate", f"{(valid_count / max(total_count, 1)):.1%}")
    
    console.print(table)
    
    # Details for invalid files
    invalid_files = [r for r in results if not r['valid']]
    if invalid_files:
        console.print("\n[bold red]Invalid Files:[/bold red]")
        for result in invalid_files[:10]:  # Show first 10
            console.print(f"  • {result['file']}")
        
        if len(invalid_files) > 10:
            console.print(f"  ... and {len(invalid_files) - 10} more")