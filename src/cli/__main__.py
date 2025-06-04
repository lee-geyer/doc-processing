#!/usr/bin/env python3
"""
CLI entry point for the Document Processing system.
"""
import typer
from rich.console import Console

from src.cli.commands import monitor, sync
from src.utils.logging import setup_logging

# Create the main CLI app
app = typer.Typer(
    name="doc-processing",
    help="Document Processing & Vector Storage System CLI",
    add_completion=False,
)

# Add command groups
app.add_typer(monitor.app, name="monitor", help="File monitoring commands")
app.add_typer(sync.app, name="sync", help="File synchronization commands")

# Create console for output
console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
):
    """
    Document Processing & Vector Storage System
    
    Monitor and process policy documents for RAG applications.
    """
    # Configure logging based on verbosity
    if quiet:
        setup_logging(log_level="ERROR", console_output=False)
    elif verbose:
        setup_logging(log_level="DEBUG")
    else:
        setup_logging(log_level="INFO")


if __name__ == "__main__":
    app()