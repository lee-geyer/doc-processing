import pytest
from pathlib import Path

from src.utils.context_utils import ContextExtractor, DocumentContext


class TestContextExtractor:
    """Test context extraction from file paths."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a test mirror directory
        self.extractor = ContextExtractor(mirror_directory="/test/mirror")
    
    def test_extract_context_policy_file(self):
        """Test context extraction for a policy file with document index."""
        file_path = "/test/mirror/Care/Section/CARE1-P10.01 Informed Treatment Consent Form.docx"
        
        context = self.extractor.extract_context(file_path)
        
        assert context.file_path == file_path
        assert context.relative_path == "Care/Section/CARE1-P10.01 Informed Treatment Consent Form.docx"
        assert context.file_name == "CARE1-P10.01 Informed Treatment Consent Form.docx"
        assert context.policy_manual == "Care"
        assert context.policy_acronym == "CARE"
        assert context.section == "Section"
        assert context.document_index == "CARE1-P10.01"
        assert context.is_additional_resource == False
    
    def test_extract_context_additional_resource(self):
        """Test context extraction for additional resource file."""
        file_path = "/test/mirror/Infection Prevention and Control/Additional Resources/HCX Staff follow-up form.docx"
        
        context = self.extractor.extract_context(file_path)
        
        assert context.policy_manual == "Infection Prevention and Control"
        assert context.policy_acronym == "IPC"
        assert context.section == "Additional Resources"
        assert context.is_additional_resource == True
        assert context.document_index == None  # No natural index
    
    def test_extract_document_index(self):
        """Test document index extraction from filenames."""
        test_cases = [
            ("CARE1-P10.01-T3 Informed Treatment Consent Form.docx", "CARE1-P10.01-T3"),
            ("ADM2-P05.02 Policy Manual.pdf", "ADM2-P05.02"),
            ("IPC1-A01.01 Infection Control Checklist.docx", "IPC1-A01.01"),
            ("regular_file_name.pdf", None),
            ("INVALID-P10.01 Test.docx", None),  # Invalid acronym
        ]
        
        for filename, expected_index in test_cases:
            result = self.extractor._extract_document_index(filename)
            assert result == expected_index, f"Failed for {filename}"
    
    def test_classify_document_type(self):
        """Test document type classification."""
        test_cases = [
            ("CARE1-P10.01 Policy Manual.pdf", "policy"),
            ("Form Template.docx", "form"),
            ("Training Guide.pdf", "guide"),
            ("Process Flowchart.pdf", "flowchart"),
            ("Quick Reference.docx", "reference"),
            ("Regular Document.pdf", None),
        ]
        
        for filename, expected_type in test_cases:
            result = self.extractor._classify_document_type(filename)
            assert result == expected_type, f"Failed for {filename}"
    
    def test_generate_synthetic_index(self):
        """Test synthetic index generation."""
        context = DocumentContext(
            file_path="/test/file.pdf",
            relative_path="Care/Additional Resources/form.pdf",
            file_name="form.pdf",
            policy_acronym="CARE",
            is_additional_resource=True
        )
        
        synthetic_index = self.extractor.generate_synthetic_index(context, 5)
        assert synthetic_index == "CARE-ADDL-005"
        
        # Test non-additional resource
        context.is_additional_resource = False
        synthetic_index = self.extractor.generate_synthetic_index(context, 12)
        assert synthetic_index == "CARE-MISC-012"
    
    def test_create_context_metadata(self):
        """Test context metadata creation."""
        context = DocumentContext(
            file_path="/test/mirror/Care/Section/CARE1-P10.01 Form.docx",
            relative_path="Care/Section/CARE1-P10.01 Form.docx",
            file_name="CARE1-P10.01 Form.docx",
            policy_manual="Care",
            policy_acronym="CARE",
            section="Section",
            document_index="CARE1-P10.01",
            document_type="form",
            hierarchy_path=["Care", "Section"]
        )
        
        metadata = self.extractor.create_context_metadata(context)
        
        assert metadata['policy_manual'] == "Care"
        assert metadata['policy_acronym'] == "CARE"
        assert metadata['document_index'] == "CARE1-P10.01"
        assert metadata['document_type'] == "form"
        assert metadata['source_type'] == "policy_document"
        assert 'care' in metadata['tags']
        assert 'form' in metadata['tags']
        assert 'indexed_document' in metadata['tags']
    
    def test_validate_context(self):
        """Test context validation."""
        # Valid context
        valid_context = DocumentContext(
            file_path="/test/file.pdf",
            relative_path="Care/Section/file.pdf",
            file_name="file.pdf",
            policy_manual="Care",
            policy_acronym="CARE",
            section="Section",
            document_index="CARE1-P10.01",
            hierarchy_path=["Care", "Section"]
        )
        
        validation = self.extractor.validate_context(valid_context)
        assert validation['is_valid'] == True
        assert len(validation['issues']) == 0
        assert validation['context_completeness'] > 0.8
        
        # Invalid context (missing policy manual)
        invalid_context = DocumentContext(
            file_path="/test/file.pdf",
            relative_path="file.pdf",
            file_name="file.pdf",
            hierarchy_path=[]
        )
        
        validation = self.extractor.validate_context(invalid_context)
        assert validation['is_valid'] == False
        assert len(validation['issues']) > 0
        assert "Missing policy manual" in validation['issues'][0]