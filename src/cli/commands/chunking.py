import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from pathlib import Path
from typing import Optional

from src.core.chunking import DocumentChunker
from src.core.document_parser import DocumentParser
from src.core.text_cleaner import TextCleaner
from src.utils.context_utils import ContextExtractor
from src.models.database import SessionLocal, Document, DocumentChunk as DBChunk
from src.config.settings import settings
from src.utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to file to analyze"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", "-s", help="Chunk size in tokens"),
    chunk_overlap: Optional[int] = typer.Option(None, "--overlap", "-o", help="Chunk overlap in tokens"),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Display individual chunks"),
    max_chunks_display: int = typer.Option(3, "--max-display", help="Maximum chunks to display")
):
    """Analyze chunking for a specific file."""
    file_path = Path(file_path).absolute()
    
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing chunking...", total=None)
            
            # Parse document
            parser = DocumentParser()
            parsed_doc = parser.parse_document(str(file_path))
            
            if not parsed_doc.parsing_success:
                console.print(f"[red]Failed to parse document: {parsed_doc.error_message}[/red]")
                raise typer.Exit(1)
            
            # Clean text
            cleaner = TextCleaner()
            cleaning_result = cleaner.clean_text(parsed_doc.raw_text or "")
            cleaned_text = cleaning_result['cleaned_text']
            
            # Extract context
            context_extractor = ContextExtractor(settings.mirror_directory)
            document_context = context_extractor.extract_context(str(file_path))
            
            # Chunk document
            chunker = DocumentChunker()
            chunking_result = chunker.chunk_document(
                text=cleaned_text,
                markdown=parsed_doc.markdown_content or "",
                document_context=document_context,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            progress.update(task, completed=True)
        
        # Display results
        _display_chunking_analysis(chunking_result, parsed_doc, show_chunks, max_chunks_display)
        
    except Exception as e:
        console.print(f"[red]Error analyzing chunking: {e}[/red]")
        logger.error(f"Error in chunking analysis: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def optimize(
    test_sizes: str = typer.Option("500,1000,1500", "--sizes", help="Comma-separated chunk sizes to test"),
    test_overlaps: str = typer.Option("100,200,300", "--overlaps", help="Comma-separated overlaps to test"),
    sample_files: int = typer.Option(5, "--sample", help="Number of sample files to test"),
):
    """Optimize chunking parameters using sample files."""
    try:
        sizes = [int(s.strip()) for s in test_sizes.split(",")]
        overlaps = [int(o.strip()) for o in test_overlaps.split(",")]
        
        console.print(f"[cyan]Testing {len(sizes)} chunk sizes × {len(overlaps)} overlaps on {sample_files} files[/cyan]")
        
        # Get sample files from database
        db = SessionLocal()
        try:
            sample_docs = db.query(Document).filter(
                Document.parsing_success == True,
                Document.cleaned_content.isnot(None)
            ).limit(sample_files).all()
            
            if not sample_docs:
                console.print("[yellow]No processed documents found. Run document processing first.[/yellow]")
                raise typer.Exit(1)
            
            results = []
            
            with Progress(console=console) as progress:
                total_combinations = len(sizes) * len(overlaps)
                task = progress.add_task("Testing combinations...", total=total_combinations)
                
                for chunk_size in sizes:
                    for chunk_overlap in overlaps:
                        if chunk_overlap >= chunk_size:
                            continue  # Skip invalid combinations
                        
                        # Test this combination
                        combination_result = _test_chunking_combination(
                            sample_docs, chunk_size, chunk_overlap
                        )
                        combination_result.update({
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        })
                        results.append(combination_result)
                        
                        progress.advance(task)
            
            # Display optimization results
            _display_optimization_results(results)
            
        finally:
            db.close()
        
    except Exception as e:
        console.print(f"[red]Error optimizing chunking: {e}[/red]")
        logger.error(f"Error in chunking optimization: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def validate(
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Chunk size to validate"),
    chunk_overlap: Optional[int] = typer.Option(None, "--overlap", help="Chunk overlap to validate"),
    sample_count: int = typer.Option(10, "--sample", help="Number of documents to validate"),
):
    """Validate chunk quality across sample documents."""
    try:
        db = SessionLocal()
        try:
            # Get sample documents
            sample_docs = db.query(Document).filter(
                Document.parsing_success == True,
                Document.cleaned_content.isnot(None)
            ).limit(sample_count).all()
            
            if not sample_docs:
                console.print("[yellow]No processed documents found.[/yellow]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Validating chunking quality on {len(sample_docs)} documents[/cyan]")
            
            validation_results = []
            
            with Progress(console=console) as progress:
                task = progress.add_task("Validating chunks...", total=len(sample_docs))
                
                for doc in sample_docs:
                    result = _validate_document_chunking(doc, chunk_size, chunk_overlap)
                    validation_results.append(result)
                    progress.advance(task)
            
            # Display validation results
            _display_validation_results(validation_results)
            
        finally:
            db.close()
        
    except Exception as e:
        console.print(f"[red]Error validating chunking: {e}[/red]")
        logger.error(f"Error in chunking validation: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def stats():
    """Show chunking statistics from processed documents."""
    try:
        db = SessionLocal()
        try:
            # Get chunking statistics
            total_docs = db.query(Document).filter(Document.total_chunks > 0).count()
            total_chunks = db.query(DBChunk).count()
            
            if total_docs == 0:
                console.print("[yellow]No chunked documents found.[/yellow]")
                return
            
            # Aggregate statistics
            from sqlalchemy import func
            
            stats_query = db.query(
                func.avg(Document.total_chunks).label('avg_chunks_per_doc'),
                func.min(Document.total_chunks).label('min_chunks'),
                func.max(Document.total_chunks).label('max_chunks'),
                func.avg(Document.total_tokens).label('avg_tokens_per_doc')
            ).filter(Document.total_chunks > 0).first()
            
            chunk_stats_query = db.query(
                func.avg(DBChunk.chunk_tokens).label('avg_chunk_tokens'),
                func.min(DBChunk.chunk_tokens).label('min_chunk_tokens'),
                func.max(DBChunk.chunk_tokens).label('max_chunk_tokens')
            ).first()
            
            # Display statistics
            table = Table(title="Chunking Statistics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")
            
            table.add_row("Total Documents", str(total_docs))
            table.add_row("Total Chunks", str(total_chunks))
            table.add_row("Avg Chunks per Document", f"{stats_query.avg_chunks_per_doc:.1f}")
            table.add_row("Min/Max Chunks per Document", f"{stats_query.min_chunks}/{stats_query.max_chunks}")
            table.add_row("Avg Tokens per Document", f"{stats_query.avg_tokens_per_doc:.0f}")
            table.add_row("Avg Tokens per Chunk", f"{chunk_stats_query.avg_chunk_tokens:.0f}")
            table.add_row("Min/Max Tokens per Chunk", 
                         f"{chunk_stats_query.min_chunk_tokens}/{chunk_stats_query.max_chunk_tokens}")
            
            console.print(table)
            
            # Show policy breakdown
            _display_policy_chunking_breakdown(db)
            
        finally:
            db.close()
        
    except Exception as e:
        console.print(f"[red]Error getting chunking statistics: {e}[/red]")
        logger.error(f"Error in chunking stats: {e}", exc_info=True)
        raise typer.Exit(1)


def _display_chunking_analysis(chunking_result, parsed_doc, show_chunks: bool, max_display: int):
    """Display comprehensive chunking analysis."""
    # Main statistics table
    table = Table(title="Chunking Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Document Size", f"{len(parsed_doc.raw_text or '')} chars")
    table.add_row("Total Chunks", str(chunking_result.total_chunks))
    table.add_row("Total Tokens", str(chunking_result.total_tokens))
    table.add_row("Average Chunk Size", f"{chunking_result.avg_chunk_size:.1f} tokens")
    table.add_row("Overlap Ratio", f"{chunking_result.overlap_ratio:.1%}")
    table.add_row("Boundary Preservation", f"{chunking_result.boundary_preservation_rate:.1%}")
    table.add_row("Processing Time", f"{chunking_result.processing_time_ms}ms")
    table.add_row("Strategy", chunking_result.chunking_strategy)
    
    console.print(table)
    
    # Show warnings if any
    if chunking_result.warnings:
        warning_panel = Panel(
            "\n".join(chunking_result.warnings),
            title="Warnings",
            border_style="yellow"
        )
        console.print(warning_panel)
    
    # Show individual chunks if requested
    if show_chunks and chunking_result.chunks:
        console.print(f"\n[bold]Sample Chunks (showing {min(max_display, len(chunking_result.chunks))}):[/bold]")
        
        for i, chunk in enumerate(chunking_result.chunks[:max_display]):
            # Create chunk display
            chunk_info = f"Chunk {chunk.index} | {chunk.token_count} tokens | Position: {chunk.document_position:.1%}"
            if chunk.section_title:
                chunk_info += f" | Section: {chunk.section_title}"
            
            # Truncate text for display
            display_text = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            
            chunk_panel = Panel(
                display_text,
                title=chunk_info,
                border_style="blue"
            )
            console.print(chunk_panel)


def _test_chunking_combination(sample_docs, chunk_size: int, chunk_overlap: int) -> dict:
    """Test a specific chunking combination on sample documents."""
    chunker = DocumentChunker()
    context_extractor = ContextExtractor(settings.mirror_directory)
    
    total_chunks = 0
    total_tokens = 0
    boundary_rates = []
    processing_times = []
    
    for doc in sample_docs:
        try:
            # Extract context
            document_context = context_extractor.extract_context(doc.file_path)
            
            # Chunk document
            result = chunker.chunk_document(
                text=doc.cleaned_content or "",
                markdown=doc.markdown_content or "",
                document_context=document_context,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            total_chunks += result.total_chunks
            total_tokens += result.total_tokens
            boundary_rates.append(result.boundary_preservation_rate)
            processing_times.append(result.processing_time_ms)
            
        except Exception as e:
            logger.warning(f"Failed to chunk document {doc.file_path}: {e}")
            continue
    
    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_boundary_preservation": sum(boundary_rates) / len(boundary_rates) if boundary_rates else 0,
        "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
        "docs_processed": len(boundary_rates)
    }


def _validate_document_chunking(doc, chunk_size: Optional[int], chunk_overlap: Optional[int]) -> dict:
    """Validate chunking for a single document."""
    chunker = DocumentChunker()
    context_extractor = ContextExtractor(settings.mirror_directory)
    
    try:
        # Extract context
        document_context = context_extractor.extract_context(doc.file_path)
        
        # Chunk document
        result = chunker.chunk_document(
            text=doc.cleaned_content or "",
            markdown=doc.markdown_content or "",
            document_context=document_context,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Validation checks
        validation_issues = []
        
        # Check for very small chunks
        small_chunks = [c for c in result.chunks if c.token_count < settings.min_chunk_size]
        if small_chunks:
            validation_issues.append(f"{len(small_chunks)} chunks below minimum size")
        
        # Check for very large chunks
        large_chunks = [c for c in result.chunks if c.token_count > settings.max_chunk_size]
        if large_chunks:
            validation_issues.append(f"{len(large_chunks)} chunks above maximum size")
        
        # Check for empty chunks
        empty_chunks = [c for c in result.chunks if not c.text.strip()]
        if empty_chunks:
            validation_issues.append(f"{len(empty_chunks)} empty chunks")
        
        return {
            "file_path": doc.file_path,
            "success": True,
            "total_chunks": result.total_chunks,
            "avg_chunk_size": result.avg_chunk_size,
            "boundary_preservation": result.boundary_preservation_rate,
            "validation_issues": validation_issues,
            "quality_score": _calculate_chunk_quality_score(result, validation_issues)
        }
        
    except Exception as e:
        return {
            "file_path": doc.file_path,
            "success": False,
            "error": str(e),
            "quality_score": 0.0
        }


def _calculate_chunk_quality_score(result, validation_issues: list) -> float:
    """Calculate overall quality score for chunking result."""
    base_score = 1.0
    
    # Penalty for validation issues
    base_score -= len(validation_issues) * 0.1
    
    # Reward for good boundary preservation
    base_score += result.boundary_preservation_rate * 0.2
    
    # Penalty for very uneven chunk sizes
    if result.chunks:
        token_counts = [c.token_count for c in result.chunks]
        if token_counts:
            variance = sum((x - result.avg_chunk_size) ** 2 for x in token_counts) / len(token_counts)
            coefficient_of_variation = (variance ** 0.5) / result.avg_chunk_size if result.avg_chunk_size > 0 else 0
            if coefficient_of_variation > 0.5:  # High variation
                base_score -= 0.2
    
    return max(0.0, min(1.0, base_score))


def _display_optimization_results(results: list):
    """Display optimization results."""
    # Sort by quality score
    results.sort(key=lambda r: (
        r.get("avg_boundary_preservation", 0),
        -r.get("avg_processing_time", float('inf'))
    ), reverse=True)
    
    console.print("\n[bold]Optimization Results (Best First):[/bold]")
    
    table = Table(box=box.ROUNDED)
    table.add_column("Chunk Size", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Avg Chunks", justify="right")
    table.add_column("Boundary Preservation", justify="right")
    table.add_column("Avg Processing Time (ms)", justify="right")
    table.add_column("Docs Processed", justify="right")
    
    for result in results[:10]:  # Show top 10
        table.add_row(
            str(result["chunk_size"]),
            str(result["chunk_overlap"]),
            f"{result['total_chunks'] / result['docs_processed']:.1f}",
            f"{result['avg_boundary_preservation']:.1%}",
            f"{result['avg_processing_time']:.0f}",
            str(result["docs_processed"])
        )
    
    console.print(table)
    
    if results:
        best = results[0]
        console.print(f"\n[green]✓ Recommended: chunk_size={best['chunk_size']}, "
                     f"overlap={best['chunk_overlap']}[/green]")


def _display_validation_results(results: list):
    """Display validation results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if failed:
        console.print(f"[red]Failed to validate {len(failed)} documents[/red]")
    
    if not successful:
        console.print("[yellow]No successful validations[/yellow]")
        return
    
    # Summary statistics
    avg_quality = sum(r["quality_score"] for r in successful) / len(successful)
    avg_chunks = sum(r["total_chunks"] for r in successful) / len(successful)
    avg_chunk_size = sum(r["avg_chunk_size"] for r in successful) / len(successful)
    avg_boundary_preservation = sum(r["boundary_preservation"] for r in successful) / len(successful)
    
    table = Table(title="Validation Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Documents Validated", str(len(successful)))
    table.add_row("Average Quality Score", f"{avg_quality:.2f}")
    table.add_row("Average Chunks per Document", f"{avg_chunks:.1f}")
    table.add_row("Average Chunk Size", f"{avg_chunk_size:.1f} tokens")
    table.add_row("Average Boundary Preservation", f"{avg_boundary_preservation:.1%}")
    
    console.print(table)
    
    # Show issues summary
    all_issues = []
    for result in successful:
        all_issues.extend(result.get("validation_issues", []))
    
    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        
        console.print("\n[bold]Common Issues:[/bold]")
        for issue, count in issue_counts.most_common(5):
            console.print(f"  • {issue}: {count} occurrences")


def _display_policy_chunking_breakdown(db):
    """Display chunking statistics by policy manual."""
    from sqlalchemy import func
    
    # Get policy breakdown
    policy_stats = db.query(
        func.json_extract(Document.context_hierarchy, '$.policy_manual').label('policy_manual'),
        func.count(Document.id).label('doc_count'),
        func.sum(Document.total_chunks).label('total_chunks'),
        func.avg(Document.total_chunks).label('avg_chunks')
    ).filter(
        Document.total_chunks > 0
    ).group_by(
        func.json_extract(Document.context_hierarchy, '$.policy_manual')
    ).all()
    
    if policy_stats:
        console.print("\n")
        table = Table(title="Chunking by Policy Manual", box=box.ROUNDED)
        table.add_column("Policy Manual", style="cyan")
        table.add_column("Documents", justify="right")
        table.add_column("Total Chunks", justify="right")
        table.add_column("Avg Chunks/Doc", justify="right", style="green")
        
        for stat in policy_stats:
            if stat.policy_manual:  # Skip null policy manuals
                table.add_row(
                    str(stat.policy_manual),
                    str(stat.doc_count),
                    str(stat.total_chunks),
                    f"{stat.avg_chunks:.1f}"
                )
        
        console.print(table)