import hashlib
import os
from pathlib import Path
from typing import Optional, Set


def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file using specified algorithm.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        
    Returns:
        Hex digest of the file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_file_info(file_path: str) -> dict:
    """
    Get basic file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "file_size_bytes": stat.st_size,
        "last_modified": stat.st_mtime,
        "file_type": path.suffix.lower().lstrip('.'),
        "exists": path.exists()
    }


def is_supported_file(file_path: str) -> bool:
    """
    Check if file type is supported for processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file type is supported
    """
    supported_extensions = {'.pdf', '.docx', '.doc'}
    path = Path(file_path)
    return path.suffix.lower() in supported_extensions


def get_relative_path(file_path: str, base_path: str) -> str:
    """
    Get relative path from base path.
    
    Args:
        file_path: Full file path
        base_path: Base path to calculate relative from
        
    Returns:
        Relative path as string
    """
    return str(Path(file_path).relative_to(Path(base_path)))


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def scan_directory(directory: str, extensions: Optional[Set[str]] = None) -> list:
    """
    Recursively scan directory for files with specified extensions.
    
    Args:
        directory: Directory to scan
        extensions: Set of file extensions to include (e.g., {'.pdf', '.docx'})
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = {'.pdf', '.docx', '.doc'}
    
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if Path(file_path).suffix.lower() in extensions:
                files.append(file_path)
    
    return files