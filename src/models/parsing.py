from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with its content and metadata.
    """
    file_path: str
    file_type: str
    parsing_method: str
    parsing_success: bool
    
    # Content
    markdown_content: Optional[str] = None
    raw_text: Optional[str] = None
    
    # Structure
    page_count: int = 0
    word_count: int = 0
    
    # Metadata
    document_metadata: Optional[Dict[str, Any]] = None
    
    # Parsing details
    parsing_error_message: Optional[str] = None
    parsing_duration_ms: Optional[int] = None
    
    # Document structure elements
    headings: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None


@dataclass
class DocumentPage:
    """
    Represents a single page of a document.
    """
    page_number: int
    text_content: str
    markdown_content: str
    
    # Layout elements
    headings: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    
    # Bounding boxes and coordinates
    text_blocks: List[Dict[str, Any]]
    
    # Page metadata
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class TextBlock:
    """
    Represents a block of text with positioning and formatting.
    """
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    
    # Formatting
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    
    # Block type
    block_type: str = "text"  # text, heading, table, list
    heading_level: Optional[int] = None


@dataclass
class ParsingStats:
    """
    Statistics about the parsing process.
    """
    total_files: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    
    # Content statistics
    total_pages: int = 0
    total_words: int = 0
    avg_words_per_page: float = 0.0
    
    # Processing time
    total_processing_time_ms: int = 0
    avg_processing_time_ms: float = 0.0
    
    # File type breakdown
    pdf_files: int = 0
    docx_files: int = 0
    other_files: int = 0