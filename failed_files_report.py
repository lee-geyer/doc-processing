#!/usr/bin/env python3
"""
Generate comprehensive failed files report.
"""

from src.models.database import SessionLocal, FileSyncState
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import Counter
import json

def generate_failed_files_report():
    """Generate comprehensive failed files report."""
    console = Console()
    db = SessionLocal()
    
    try:
        # Get all failed files
        failed_files = db.query(FileSyncState).filter_by(sync_status='error').all()
        
        console.print(Panel.fit(
            f"[bold red]Failed Files Report[/bold red]\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Total failed files: {len(failed_files)}"
        ))
        
        if not failed_files:
            console.print("[bold green]🎉 No failed files! 100% success rate![/bold green]")
            return
        
        # Analyze error patterns
        error_patterns = {}
        file_type_errors = Counter()
        
        for file in failed_files:
            filename = file.file_path.split('/')[-1]
            file_ext = filename.split('.')[-1].lower()
            file_type_errors[file_ext] += 1
            
            error_msg = file.last_error_message or 'Unknown error'
            
            # Categorize errors
            if 'not a Word file' in error_msg:
                error_type = 'Corrupted Word Document'
            elif 'not a textpage of this page' in error_msg:
                error_type = 'PDF Table Extraction Error'
            elif 'weakly-referenced object no longer exists' in error_msg:
                error_type = 'PDF Memory Error'
            elif 'Failed to parse' in error_msg:
                error_type = 'Document Parsing Error'
            else:
                error_type = 'Other Error'
            
            if error_type not in error_patterns:
                error_patterns[error_type] = []
            error_patterns[error_type].append({
                'file': file,
                'filename': filename,
                'error': error_msg
            })
        
        # Error summary table
        summary_table = Table(title="Error Summary by Type")
        summary_table.add_column("Error Type", style="yellow")
        summary_table.add_column("Count", justify="right", style="red")
        summary_table.add_column("File Types", style="dim")
        
        for error_type, files in error_patterns.items():
            file_types = list(set(f['filename'].split('.')[-1].lower() for f in files))
            summary_table.add_row(
                error_type,
                str(len(files)),
                ', '.join(file_types)
            )
        
        console.print(summary_table)
        
        # File type breakdown
        console.print(f"\n[bold]File Type Breakdown:[/bold]")
        for ext, count in file_type_errors.most_common():
            console.print(f"  {ext.upper()}: {count} files")
        
        # Detailed error listings
        for error_type, files in error_patterns.items():
            console.print(f"\n[bold yellow]{error_type} ({len(files)} files):[/bold yellow]")
            
            for item in files:
                file = item['file']
                filename = item['filename']
                
                # Get folder context
                path_parts = file.file_path.split('/')
                if 'INTEGRATION FILES TARGETED' in file.file_path:
                    idx = path_parts.index('INTEGRATION FILES TARGETED')
                    if idx + 1 < len(path_parts):
                        policy_folder = path_parts[idx + 1]
                        if idx + 2 < len(path_parts):
                            section_folder = path_parts[idx + 2]
                        else:
                            section_folder = "Root"
                    else:
                        policy_folder = "Unknown"
                        section_folder = "Unknown"
                else:
                    policy_folder = "Unknown"
                    section_folder = "Unknown"
                
                console.print(f"  📄 [cyan]{filename}[/cyan]")
                console.print(f"     Policy: {policy_folder}")
                console.print(f"     Section: {section_folder}")
                console.print(f"     Attempts: {file.sync_error_count}")
                
                # Show error reason based on type
                if error_type == 'Corrupted Word Document':
                    console.print(f"     [red]Issue:[/red] File appears corrupted or has wrong format")
                elif error_type == 'PDF Table Extraction Error':
                    console.print(f"     [red]Issue:[/red] PDF contains complex tables that can't be extracted")
                elif error_type == 'PDF Memory Error':
                    console.print(f"     [red]Issue:[/red] PDF processing memory issue")
                else:
                    console.print(f"     [red]Error:[/red] {item['error'][:100]}...")
                console.print()
        
        # Recommendations
        console.print(Panel.fit(
            "[bold cyan]Recommendations:[/bold cyan]\n\n"
            "1. [yellow]Corrupted Word Documents:[/yellow] Check if files can be opened in Word\n"
            "2. [yellow]PDF Table Errors:[/yellow] These are complex forms/tables - may need manual review\n"
            "3. [yellow]PDF Memory Errors:[/yellow] Large or complex PDFs causing memory issues\n\n"
            f"[green]Success Rate:[/green] {((857 - len(failed_files)) / 857 * 100):.1f}% "
            f"({857 - len(failed_files)}/857 files processed successfully)"
        ))
        
        # Export to JSON for further analysis
        export_data = {
            'total_failed': len(failed_files),
            'success_rate': (857 - len(failed_files)) / 857 * 100,
            'error_summary': {k: len(v) for k, v in error_patterns.items()},
            'file_type_breakdown': dict(file_type_errors),
            'failed_files': [
                {
                    'filename': item['filename'],
                    'full_path': item['file'].file_path,
                    'error_type': error_type,
                    'attempts': item['file'].sync_error_count,
                    'error_message': item['error']
                }
                for error_type, files in error_patterns.items()
                for item in files
            ]
        }
        
        with open('failed_files_report.json', 'w') as f:
            json.dump(export_data, f, indent=2)
        
        console.print(f"\n[dim]Detailed report exported to: failed_files_report.json[/dim]")
        
    finally:
        db.close()

if __name__ == "__main__":
    from datetime import datetime
    generate_failed_files_report()