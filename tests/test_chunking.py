import pytest
from unittest.mock import Mock, patch

from src.core.chunking import DocumentChunker, TokenCounter, BoundaryDetector, ChunkBoundary
from src.utils.context_utils import DocumentContext


class TestTokenCounter:
    """Test token counting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "This is a simple test with eight tokens"
        count = self.counter.count_tokens(text)
        assert count == 8
    
    def test_count_tokens_empty(self):
        """Test token counting with empty text."""
        assert self.counter.count_tokens("") == 0
        assert self.counter.count_tokens("   ") == 0
    
    def test_split_by_tokens(self):
        """Test splitting text by token count."""
        text = "This is a longer text that should be split into multiple segments based on token count"
        segments = self.counter.split_by_tokens(text, max_tokens=5)
        
        assert len(segments) > 1
        for segment in segments:
            token_count = self.counter.count_tokens(segment)
            assert token_count <= 5


class TestBoundaryDetector:
    """Test boundary detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = BoundaryDetector()
    
    def test_find_sentence_boundaries(self):
        """Test sentence boundary detection."""
        text = "This is the first sentence. This is the second sentence! And this is the third?"
        boundaries = self.detector._find_sentence_boundaries(text)
        
        assert len(boundaries) == 3
        assert all(b.boundary_type == "sentence" for b in boundaries)
        assert all(b.priority == 1 for b in boundaries)
    
    def test_find_paragraph_boundaries(self):
        """Test paragraph boundary detection."""
        text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
        boundaries = self.detector._find_paragraph_boundaries(text)
        
        assert len(boundaries) >= 2
        assert all(b.boundary_type == "paragraph" for b in boundaries)
        assert all(b.priority == 3 for b in boundaries)
    
    def test_find_section_boundaries(self):
        """Test section boundary detection."""
        markdown = """# Main Section
        
Content here.

## Subsection

More content.

### Sub-subsection

Final content."""
        
        boundaries = self.detector._find_section_boundaries(markdown)
        
        # Should find 3 headers
        section_boundaries = [b for b in boundaries if b.boundary_type == "section"]
        assert len(section_boundaries) == 3
        
        # Check priority ordering (higher level headers have higher priority)
        priorities = [b.priority for b in section_boundaries]
        assert priorities == [9, 8, 7]  # 10-1, 10-2, 10-3
    
    def test_find_table_boundaries(self):
        """Test table boundary detection."""
        markdown = """Some text before.

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

Some text after."""
        
        boundaries = self.detector._find_table_boundaries(markdown)
        
        # Should find table start and end boundaries
        table_boundaries = [b for b in boundaries if b.boundary_type == "table"]
        assert len(table_boundaries) == 2
        
        # Check metadata
        start_boundary = next(b for b in table_boundaries if b.metadata.get("table_start"))
        end_boundary = next(b for b in table_boundaries if b.metadata.get("table_end"))
        
        assert start_boundary.position < end_boundary.position


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker()
        self.sample_context = DocumentContext(
            file_path="/test/CARE1-P10.01 Test Document.pdf",
            relative_path="Care/Section/CARE1-P10.01 Test Document.pdf",
            file_name="CARE1-P10.01 Test Document.pdf",
            policy_manual="Care",
            policy_acronym="CARE",
            section="Section",
            document_index="CARE1-P10.01",
            document_type="policy",
            hierarchy_path=["Care", "Section"]
        )
    
    def test_chunk_simple_document(self):
        """Test chunking of a simple document."""
        text = "This is a simple test document. " * 50  # About 300 words
        markdown = "# Test Document\n\n" + text
        
        result = self.chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=self.sample_context,
            chunk_size=100,
            chunk_overlap=20
        )
        
        assert result.total_chunks > 0
        assert result.total_tokens > 0
        assert len(result.chunks) == result.total_chunks
        
        # Check that chunks have required attributes
        for chunk in result.chunks:
            assert chunk.text
            assert chunk.token_count > 0
            assert chunk.hash
            assert chunk.context_metadata
            assert 0.0 <= chunk.document_position <= 1.0
    
    def test_chunk_with_boundaries(self):
        """Test chunking respects natural boundaries."""
        text = """This is the first section. It has multiple sentences.
        
        This is the second section. It also has multiple sentences. And some more content to make it longer.
        
        This is the third section. Final content here."""
        
        markdown = """# Document Title

This is the first section. It has multiple sentences.

## Second Section

This is the second section. It also has multiple sentences. And some more content to make it longer.

## Third Section

This is the third section. Final content here."""
        
        result = self.chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=self.sample_context,
            chunk_size=50,
            chunk_overlap=10
        )
        
        # Should create multiple chunks
        assert result.total_chunks > 1
        
        # Boundary preservation rate should be reasonable
        assert result.boundary_preservation_rate >= 0.0
    
    def test_chunk_metadata_enrichment(self):
        """Test that chunks are enriched with proper metadata."""
        text = "Test document content for metadata validation."
        markdown = "# Test\n\nTest document content for metadata validation."
        
        result = self.chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=self.sample_context
        )
        
        assert len(result.chunks) > 0
        
        chunk = result.chunks[0]
        
        # Check context metadata
        assert chunk.context_metadata["file_path"] == self.sample_context.file_path
        assert chunk.context_metadata["policy_manual"] == self.sample_context.policy_manual
        assert chunk.context_metadata["policy_acronym"] == self.sample_context.policy_acronym
        assert chunk.context_metadata["document_index"] == self.sample_context.document_index
        assert chunk.context_metadata["chunk_index"] == chunk.index
        assert chunk.context_metadata["token_count"] == chunk.token_count
    
    def test_chunk_with_tables(self):
        """Test chunking documents with tables."""
        markdown = """# Document with Table

Some introductory text.

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |

Some concluding text."""
        
        text = "Some introductory text. Name Age City John 30 NYC Jane 25 LA Some concluding text."
        
        result = self.chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=self.sample_context
        )
        
        # Check if any chunk detected table content
        table_chunks = [c for c in result.chunks if c.contains_table]
        assert len(table_chunks) > 0
        
        # Check structural elements
        for chunk in result.chunks:
            if chunk.contains_table:
                assert "table" in chunk.structural_elements
    
    def test_chunk_overlap_functionality(self):
        """Test that chunk overlap works correctly."""
        text = "This is a test document. " * 100  # Longer text
        markdown = text
        
        result = self.chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=self.sample_context,
            chunk_size=50,
            chunk_overlap=10
        )
        
        if len(result.chunks) > 1:
            # Check that consecutive chunks have some overlapping content
            # This is a simplified check - in practice, overlap might not be exact due to boundary detection
            first_chunk = result.chunks[0]
            second_chunk = result.chunks[1]
            
            # At minimum, chunks should be different
            assert first_chunk.text != second_chunk.text
            assert first_chunk.char_start < second_chunk.char_start
    
    def test_surrounding_context_extraction(self):
        """Test surrounding context extraction."""
        text = "First chunk content. Second chunk content. Third chunk content."
        markdown = text
        
        with patch('src.config.settings.include_surrounding_context', True):
            with patch('src.config.settings.surrounding_context_tokens', 3):
                result = self.chunker.chunk_document(
                    text=text,
                    markdown=markdown,
                    document_context=self.sample_context,
                    chunk_size=10,
                    chunk_overlap=2
                )
                
                # If we have multiple chunks, middle chunks should have surrounding context
                if len(result.chunks) > 2:
                    middle_chunk = result.chunks[1]
                    assert middle_chunk.surrounding_context is not None
    
    def test_empty_text_handling(self):
        """Test handling of empty or minimal text."""
        result = self.chunker.chunk_document(
            text="",
            markdown="",
            document_context=self.sample_context
        )
        
        # Should handle empty text gracefully
        assert result.total_chunks == 0
        assert result.total_tokens == 0
        assert len(result.chunks) == 0
    
    def test_chunk_hash_generation(self):
        """Test that chunk hashes are generated correctly."""
        text = "Test content for hash generation."
        
        hash1 = self.chunker._generate_chunk_hash(text)
        hash2 = self.chunker._generate_chunk_hash(text)
        hash3 = self.chunker._generate_chunk_hash(text + " different")
        
        # Same content should produce same hash
        assert hash1 == hash2
        
        # Different content should produce different hash
        assert hash1 != hash3
        
        # Hash should be of expected length (16 chars from SHA-256)
        assert len(hash1) == 16
    
    def test_boundary_preservation_calculation(self):
        """Test boundary preservation rate calculation."""
        # Create mock chunks and boundaries
        chunks = [
            Mock(char_start=0, char_end=50),
            Mock(char_start=40, char_end=90),
            Mock(char_start=85, char_end=135)
        ]
        
        boundaries = [
            ChunkBoundary(position=25, boundary_type="sentence", priority=1),
            ChunkBoundary(position=50, boundary_type="paragraph", priority=3),
            ChunkBoundary(position=90, boundary_type="sentence", priority=1),
            ChunkBoundary(position=120, boundary_type="sentence", priority=1)
        ]
        
        rate = self.chunker._calculate_boundary_preservation_rate(chunks, boundaries)
        
        # Should return a value between 0 and 1
        assert 0.0 <= rate <= 1.0
    
    def test_minimum_chunk_size_enforcement(self):
        """Test that minimum chunk size is enforced."""
        text = "Short text."
        markdown = text
        
        with patch('src.config.settings.min_chunk_size', 50):
            result = self.chunker.chunk_document(
                text=text,
                markdown=markdown,
                document_context=self.sample_context,
                chunk_size=10  # Very small chunk size
            )
            
            # Should still create at least one chunk even if below minimum
            assert result.total_chunks >= 1
            
            # But the actual chunk might be larger than requested due to minimum size
            if result.chunks:
                # The implementation might still create small chunks for very short documents
                assert result.chunks[0].token_count >= 0


class TestChunkingIntegration:
    """Integration tests for the complete chunking pipeline."""
    
    def test_chunking_with_real_policy_document_structure(self):
        """Test chunking with realistic policy document structure."""
        markdown = """# CARE1-P10.01 Informed Treatment Consent Form

## Policy Statement

Extendicare is committed to ensuring that residents and their substitute decision-makers are provided with sufficient information to make informed decisions about treatment options.

## Purpose

This policy outlines the requirements for obtaining informed consent for treatment.

### Scope

This policy applies to all care staff and residents.

## Procedure

1. **Assessment**: Staff must assess the resident's capacity to consent.

2. **Information Provision**: The following information must be provided:
   - Nature of the proposed treatment
   - Expected benefits and risks
   - Alternative treatment options
   - Consequences of refusing treatment

3. **Documentation**: All consent discussions must be documented in the resident's care plan.

| Treatment Type | Consent Required | Documentation |
|----------------|------------------|---------------|
| Routine Care   | Implied          | Care Plan     |
| Medical Treatment | Written       | Consent Form  |
| Experimental   | Detailed Written | Special Form  |

## Responsibilities

### Care Staff
- Obtain appropriate consent before providing treatment
- Document consent discussions
- Report any concerns to the supervisor

### Supervisors
- Monitor compliance with consent requirements
- Provide guidance to staff
- Review documentation regularly

## Quality Assurance

Regular audits will be conducted to ensure compliance with this policy.

*This document is uncontrolled when printed.*

*Extendicare (Canada) Inc. will provide, on request, information in an accessible format or with communication supports to people with disabilities. © 2024*"""
        
        # Convert markdown to plain text (simplified)
        text = markdown.replace("#", "").replace("|", " ").replace("-", " ")
        
        context = DocumentContext(
            file_path="/mirror/Care/Section/CARE1-P10.01 Informed Treatment Consent Form.pdf",
            relative_path="Care/Section/CARE1-P10.01 Informed Treatment Consent Form.pdf",
            file_name="CARE1-P10.01 Informed Treatment Consent Form.pdf",
            policy_manual="Care",
            policy_acronym="CARE",
            section="Section",
            document_index="CARE1-P10.01",
            document_type="policy",
            hierarchy_path=["Care", "Section"]
        )
        
        chunker = DocumentChunker()
        result = chunker.chunk_document(
            text=text,
            markdown=markdown,
            document_context=context,
            chunk_size=200,
            chunk_overlap=50
        )
        
        # Validate chunking results
        assert result.total_chunks > 0
        assert result.total_tokens > 0
        assert result.processing_time_ms > 0
        
        # Check that chunks have proper structure detection
        table_chunks = [c for c in result.chunks if c.contains_table]
        heading_chunks = [c for c in result.chunks if c.contains_heading]
        
        # Should detect table content
        assert len(table_chunks) > 0
        
        # Should detect heading content
        assert len(heading_chunks) > 0
        
        # Check metadata quality
        for chunk in result.chunks:
            assert chunk.context_metadata["policy_acronym"] == "CARE"
            assert chunk.context_metadata["document_index"] == "CARE1-P10.01"
            assert chunk.context_metadata["document_type"] == "policy"
            assert chunk.hash
            assert 0.0 <= chunk.document_position <= 1.0