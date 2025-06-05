import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import settings
from src.utils.context_utils import DocumentContext
from src.utils.logging import get_logger
from src.core.form_descriptor import generate_form_description

logger = get_logger(__name__)


@dataclass
class ChunkBoundary:
    """Represents a potential chunking boundary with priority."""
    position: int
    boundary_type: str  # sentence, paragraph, section, table
    priority: int  # Higher number = preferred boundary
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with rich metadata."""
    index: int
    text: str
    markdown: str
    token_count: int
    char_start: int
    char_end: int
    hash: str
    
    # Source attribution
    page_numbers: List[int] = field(default_factory=list)
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    document_position: float = 0.0  # 0.0 to 1.0
    
    # Context metadata
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    surrounding_context: Optional[str] = None
    
    # Structure information
    contains_table: bool = False
    contains_list: bool = False
    contains_heading: bool = False
    structural_elements: List[str] = field(default_factory=list)


@dataclass
class ChunkingResult:
    """Result of chunking operation with statistics."""
    chunks: List[DocumentChunk]
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    overlap_ratio: float
    boundary_preservation_rate: float
    processing_time_ms: int
    chunking_strategy: str
    warnings: List[str] = field(default_factory=list)


class TokenCounter:
    """Simple token counter for chunking."""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using simple whitespace splitting."""
        # This is a simple approximation. For production, consider using
        # tiktoken or similar for more accurate token counting
        return len(text.split())
    
    def split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into segments of approximately max_tokens."""
        words = text.split()
        segments = []
        current_segment = []
        current_count = 0
        
        for word in words:
            if current_count + 1 > max_tokens and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_count = 1
            else:
                current_segment.append(word)
                current_count += 1
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments


class BoundaryDetector:
    """Detects potential chunking boundaries in document text."""
    
    def __init__(self):
        self.token_counter = TokenCounter()
    
    def find_boundaries(self, text: str, markdown: str) -> List[ChunkBoundary]:
        """Find all potential chunking boundaries in text."""
        boundaries = []
        
        # Find sentence boundaries
        boundaries.extend(self._find_sentence_boundaries(text))
        
        # Find paragraph boundaries
        boundaries.extend(self._find_paragraph_boundaries(text))
        
        # Find section boundaries from markdown
        boundaries.extend(self._find_section_boundaries(markdown))
        
        # Find table boundaries
        boundaries.extend(self._find_table_boundaries(markdown))
        
        # Sort by position
        boundaries.sort(key=lambda b: b.position)
        
        return boundaries
    
    def _find_sentence_boundaries(self, text: str) -> List[ChunkBoundary]:
        """Find sentence ending boundaries."""
        boundaries = []
        
        # Simple sentence ending detection
        sentence_endings = re.finditer(r'[.!?]\s+', text)
        
        for match in sentence_endings:
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type="sentence",
                priority=1
            ))
        
        return boundaries
    
    def _find_paragraph_boundaries(self, text: str) -> List[ChunkBoundary]:
        """Find paragraph boundaries."""
        boundaries = []
        
        # Find double newlines (paragraph breaks)
        paragraph_breaks = re.finditer(r'\n\s*\n', text)
        
        for match in paragraph_breaks:
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type="paragraph",
                priority=3
            ))
        
        return boundaries
    
    def _find_section_boundaries(self, markdown: str) -> List[ChunkBoundary]:
        """Find section boundaries from markdown headers."""
        boundaries = []
        
        # Find markdown headers
        headers = re.finditer(r'^#{1,6}\s+.+$', markdown, re.MULTILINE)
        
        for match in headers:
            header_level = len(match.group().split()[0])  # Count # symbols
            priority = 10 - header_level  # Higher level headers = higher priority
            
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type="section",
                priority=priority,
                metadata={"header_level": header_level, "header_text": match.group()}
            ))
        
        return boundaries
    
    def _find_table_boundaries(self, markdown: str) -> List[ChunkBoundary]:
        """Find table boundaries in markdown."""
        boundaries = []
        
        # Find markdown tables
        table_pattern = r'\n\|.*?\|\n(?:\|[-:\s]*\|)?\n(?:\|.*?\|\n)+'
        tables = re.finditer(table_pattern, markdown, re.MULTILINE | re.DOTALL)
        
        for match in tables:
            # Add boundary before table
            boundaries.append(ChunkBoundary(
                position=match.start(),
                boundary_type="table",
                priority=8,
                metadata={"table_start": True}
            ))
            
            # Add boundary after table
            boundaries.append(ChunkBoundary(
                position=match.end(),
                boundary_type="table",
                priority=8,
                metadata={"table_end": True}
            ))
        
        return boundaries


class DocumentChunker:
    """Main chunking engine for document processing."""
    
    def __init__(self):
        self.token_counter = TokenCounter()
        self.boundary_detector = BoundaryDetector()
    
    def chunk_document(
        self,
        text: str,
        markdown: str,
        document_context: DocumentContext,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        file_size_bytes: int = 0
    ) -> ChunkingResult:
        """Chunk document with intelligent boundary detection."""
        start_time = time.time()
        
        # Use provided chunk size or default from settings
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        logger.info(f"Chunking document: {document_context.file_name} "
                   f"(size: {chunk_size}, overlap: {chunk_overlap})")
        
        # Check if document has extractable content
        text_stripped = text.strip() if text else ""
        
        if not text_stripped or len(text_stripped) < 10:
            # Generate synthetic description for forms/templates with no content
            logger.info(f"No extractable content found, generating synthetic description for: {document_context.file_name}")
            chunks = self._create_synthetic_chunk(document_context, file_size_bytes)
            chunking_strategy = "synthetic_description"
        else:
            # Find all potential boundaries
            boundaries = self.boundary_detector.find_boundaries(text, markdown)
            
            # Generate chunks using boundary-aware strategy
            chunks = self._create_chunks_with_boundaries(
                text, markdown, boundaries, chunk_size, chunk_overlap, document_context
            )
            chunking_strategy = "boundary_aware_token_based"
        
        # Add metadata to all chunks
        self._enrich_chunks_with_metadata(chunks, document_context, text_stripped or "")
        
        # Calculate statistics
        processing_time = int((time.time() - start_time) * 1000)
        
        result = ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(chunk.token_count for chunk in chunks),
            avg_chunk_size=sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0,
            overlap_ratio=chunk_overlap / chunk_size if chunk_size > 0 else 0,
            boundary_preservation_rate=self._calculate_boundary_preservation_rate(chunks, boundaries if 'boundaries' in locals() else []),
            processing_time_ms=processing_time,
            chunking_strategy=chunking_strategy
        )
        
        logger.info(f"Chunked into {result.total_chunks} chunks "
                   f"(avg size: {result.avg_chunk_size:.1f} tokens)")
        
        return result
    
    def _create_synthetic_chunk(self, document_context: DocumentContext, file_size_bytes: int) -> List[DocumentChunk]:
        """
        Create a synthetic chunk for forms/templates with no extractable content.
        
        Args:
            document_context: Document context information
            file_size_bytes: File size for confidence calculation
            
        Returns:
            List containing single synthetic chunk
        """
        # Generate synthetic description
        synthetic_text, synthetic_metadata = generate_form_description(document_context, file_size_bytes)
        
        # Create single chunk with synthetic content
        chunk = DocumentChunk(
            index=0,
            text=synthetic_text,
            markdown=f"# {document_context.file_name}\n\n{synthetic_text}",  # Simple markdown wrapper
            token_count=self.token_counter.count_tokens(synthetic_text),
            char_start=0,
            char_end=len(synthetic_text),
            hash=self._generate_chunk_hash(synthetic_text),
            document_position=0.0  # Single chunk = start of document
        )
        
        # Add synthetic metadata to context metadata
        chunk.context_metadata.update(synthetic_metadata)
        
        return [chunk]
    
    def _create_chunks_with_boundaries(
        self,
        text: str,
        markdown: str,
        boundaries: List[ChunkBoundary],
        chunk_size: int,
        chunk_overlap: int,
        document_context: DocumentContext
    ) -> List[DocumentChunk]:
        """Create chunks respecting natural boundaries."""
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Find the best end position for this chunk
            target_end = current_pos + chunk_size * 6  # Rough char estimate
            
            # Find the best boundary near target position
            best_boundary = self._find_best_boundary_near_position(
                boundaries, target_end, chunk_size * 6 // 4  # Search window
            )
            
            if best_boundary:
                chunk_end = min(best_boundary.position, len(text))
            else:
                # Fallback to token-based splitting
                remaining_text = text[current_pos:]
                token_segments = self.token_counter.split_by_tokens(remaining_text, chunk_size)
                if token_segments:
                    chunk_end = current_pos + len(token_segments[0])
                else:
                    chunk_end = len(text)
            
            # Ensure minimum chunk size
            if chunk_end - current_pos < settings.min_chunk_size * 4:  # Rough char estimate
                chunk_end = min(current_pos + settings.min_chunk_size * 4, len(text))
            
            # Create chunk
            chunk_text = text[current_pos:chunk_end].strip()
            chunk_markdown = self._extract_corresponding_markdown(
                markdown, text, current_pos, chunk_end
            )
            
            if chunk_text:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    index=chunk_index,
                    text=chunk_text,
                    markdown=chunk_markdown,
                    token_count=self.token_counter.count_tokens(chunk_text),
                    char_start=current_pos,
                    char_end=chunk_end,
                    hash=self._generate_chunk_hash(chunk_text),
                    document_position=current_pos / len(text) if len(text) > 0 else 0.0
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next position with overlap
            overlap_chars = chunk_overlap * 6  # Rough char estimate
            current_pos = max(chunk_end - overlap_chars, current_pos + 1)
            
            # Break if we're not making progress
            if current_pos >= chunk_end:
                break
        
        return chunks
    
    def _find_best_boundary_near_position(
        self, boundaries: List[ChunkBoundary], target_pos: int, search_window: int
    ) -> Optional[ChunkBoundary]:
        """Find the best boundary near target position."""
        candidates = [
            b for b in boundaries
            if target_pos - search_window <= b.position <= target_pos + search_window
        ]
        
        if not candidates:
            return None
        
        # Sort by priority (higher first), then by distance to target
        candidates.sort(key=lambda b: (
            -b.priority,  # Negative for descending sort
            abs(b.position - target_pos)
        ))
        
        return candidates[0]
    
    def _extract_corresponding_markdown(
        self, markdown: str, text: str, start_pos: int, end_pos: int
    ) -> str:
        """Extract markdown portion corresponding to text chunk."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated mapping between plain text and markdown
        
        if not markdown:
            return ""
        
        # Simple proportional mapping
        markdown_ratio = len(markdown) / len(text) if len(text) > 0 else 1
        markdown_start = int(start_pos * markdown_ratio)
        markdown_end = int(end_pos * markdown_ratio)
        
        return markdown[markdown_start:markdown_end]
    
    def _enrich_chunks_with_metadata(
        self, chunks: List[DocumentChunk], document_context: DocumentContext, full_text: str
    ):
        """Add rich metadata to all chunks."""
        for chunk in chunks:
            # Extract section information
            section_info = self._extract_section_from_position(
                chunk.char_start, chunk.char_end, full_text
            )
            chunk.section_title = section_info.get("section")
            chunk.subsection_title = section_info.get("subsection")
            
            # Analyze structural elements
            chunk.contains_table = "|" in chunk.markdown and "-" in chunk.markdown
            chunk.contains_list = bool(re.search(r'^\s*[-*+]\s', chunk.markdown, re.MULTILINE))
            chunk.contains_heading = bool(re.search(r'^#{1,6}\s', chunk.markdown, re.MULTILINE))
            
            # Build structural elements list
            structural_elements = []
            if chunk.contains_table:
                structural_elements.append("table")
            if chunk.contains_list:
                structural_elements.append("list")
            if chunk.contains_heading:
                structural_elements.append("heading")
            chunk.structural_elements = structural_elements
            
            # Build context metadata (preserve existing metadata from synthetic chunks)
            base_metadata = {
                "file_path": document_context.file_path,
                "file_name": document_context.file_name,
                "policy_manual": document_context.policy_manual,
                "policy_acronym": document_context.policy_acronym,
                "section": document_context.section,
                "document_index": document_context.document_index,
                "document_type": document_context.document_type,
                "is_additional_resource": document_context.is_additional_resource,
                "hierarchy_path": document_context.hierarchy_path,
                "chunk_index": chunk.index,
                "chunk_position": chunk.document_position,
                "structural_elements": chunk.structural_elements,
                "token_count": chunk.token_count
            }
            
            # Merge with existing metadata (preserving synthetic metadata)
            if chunk.context_metadata:
                # Existing metadata takes precedence (for synthetic chunks)
                merged_metadata = base_metadata.copy()
                merged_metadata.update(chunk.context_metadata)
                chunk.context_metadata = merged_metadata
            else:
                chunk.context_metadata = base_metadata
            
            # Add surrounding context if enabled
            if settings.include_surrounding_context:
                chunk.surrounding_context = self._extract_surrounding_context(
                    chunk, chunks, settings.surrounding_context_tokens
                )
    
    def _extract_section_from_position(
        self, start_pos: int, end_pos: int, full_text: str
    ) -> Dict[str, Optional[str]]:
        """Extract section information based on chunk position."""
        # Look backwards from chunk start to find section headers
        preceding_text = full_text[:start_pos]
        
        # Simple header detection (this could be more sophisticated)
        lines = preceding_text.split('\n')
        section = None
        subsection = None
        
        for line in reversed(lines):
            line = line.strip()
            if line and line.isupper() and len(line) < 100:
                if subsection is None:
                    subsection = line
                elif section is None:
                    section = line
                    break
        
        return {"section": section, "subsection": subsection}
    
    def _extract_surrounding_context(
        self, current_chunk: DocumentChunk, all_chunks: List[DocumentChunk], context_tokens: int
    ) -> str:
        """Extract surrounding context from adjacent chunks."""
        context_parts = []
        
        # Get context from previous chunk
        prev_chunk = None
        next_chunk = None
        
        for i, chunk in enumerate(all_chunks):
            if chunk.index == current_chunk.index:
                if i > 0:
                    prev_chunk = all_chunks[i - 1]
                if i < len(all_chunks) - 1:
                    next_chunk = all_chunks[i + 1]
                break
        
        # Add preceding context
        if prev_chunk:
            prev_words = prev_chunk.text.split()
            if len(prev_words) > context_tokens:
                context_words = prev_words[-context_tokens:]
            else:
                context_words = prev_words
            if context_words:
                context_parts.append("..." + " ".join(context_words))
        
        # Add following context
        if next_chunk:
            next_words = next_chunk.text.split()
            if len(next_words) > context_tokens:
                context_words = next_words[:context_tokens]
            else:
                context_words = next_words
            if context_words:
                context_parts.append(" ".join(context_words) + "...")
        
        return " | ".join(context_parts) if context_parts else None
    
    def _generate_chunk_hash(self, text: str) -> str:
        """Generate hash for chunk content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _calculate_boundary_preservation_rate(
        self, chunks: List[DocumentChunk], boundaries: List[ChunkBoundary]
    ) -> float:
        """Calculate how well natural boundaries were preserved."""
        if not boundaries:
            return 1.0
        
        preserved_boundaries = 0
        chunk_boundaries = {chunk.char_start for chunk in chunks} | {chunk.char_end for chunk in chunks}
        
        for boundary in boundaries:
            # Check if chunk boundary is close to natural boundary
            if any(abs(cb - boundary.position) < 20 for cb in chunk_boundaries):
                preserved_boundaries += 1
        
        return preserved_boundaries / len(boundaries) if boundaries else 1.0


# Import time for timing
import time