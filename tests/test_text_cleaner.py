import pytest

from src.core.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test text cleaning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
    
    def test_clean_disclaimer_text(self):
        """Test removal of disclaimer patterns."""
        text_with_disclaimer = """
        This is important content.
        
        *This document is uncontrolled when printed.*
        
        More important content here.
        
        *Extendicare (Canada) Inc. will provide, on request, information in an accessible format or*
        *with communication supports to people with disabilities, in a manner that takes into account*
        *their disability. Confidential and Proprietary Information of Extendicare (Canada) Inc. © 2024*
        
        Final content.
        """
        
        result = self.cleaner.clean_text(text_with_disclaimer)
        cleaned_text = result['cleaned_text']
        
        assert "*This document is uncontrolled when printed.*" not in cleaned_text
        assert "Extendicare (Canada) Inc. will provide" not in cleaned_text
        assert "This is important content." in cleaned_text
        assert "More important content here." in cleaned_text
        assert "Final content." in cleaned_text
    
    def test_clean_page_numbers(self):
        """Test removal of page number patterns."""
        text_with_pages = """
        Content here.
        
        Page 5 of 25
        
        More content.
        
        Printed on: 12/31/2024
        
        Final content.
        """
        
        result = self.cleaner.clean_text(text_with_pages)
        cleaned_text = result['cleaned_text']
        
        assert "Page 5 of 25" not in cleaned_text
        assert "Printed on: 12/31/2024" not in cleaned_text
        assert "Content here." in cleaned_text
        assert "More content." in cleaned_text
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text_with_bad_whitespace = """
        Content with    multiple   spaces.
        
        
        
        Multiple blank lines.
        
        
        End content.
        """
        
        result = self.cleaner.clean_text(text_with_bad_whitespace)
        cleaned_text = result['cleaned_text']
        
        # Check that multiple spaces are reduced to single spaces
        assert "multiple   spaces" not in cleaned_text
        assert "multiple spaces" in cleaned_text
        
        # Check that excessive newlines are reduced
        lines = cleaned_text.split('\n')
        consecutive_empty = 0
        max_consecutive_empty = 0
        
        for line in lines:
            if line.strip() == '':
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0
        
        assert max_consecutive_empty <= 1  # At most one blank line
    
    def test_validate_cleaning(self):
        """Test cleaning validation."""
        original = "This is a normal document with reasonable content that should not be over-cleaned."
        
        # Normal cleaning - should be valid
        good_cleaned = "This is a normal document with reasonable content."
        validation = self.cleaner.validate_cleaning(original, good_cleaned)
        assert validation['is_valid'] == True
        assert validation['word_retention'] > 0.3
        
        # Over-cleaning - should be invalid
        bad_cleaned = "This"
        validation = self.cleaner.validate_cleaning(original, bad_cleaned)
        assert validation['is_valid'] == False
        assert validation['word_retention'] < 0.3
        
        # Empty result - should be invalid
        empty_cleaned = ""
        validation = self.cleaner.validate_cleaning(original, empty_cleaned)
        assert validation['is_valid'] == False
    
    def test_cleaning_stats(self):
        """Test cleaning statistics tracking."""
        text = """
        Important content here with some value.
        
        *This document is uncontrolled when printed.*
        
        More content to keep.
        """
        
        result = self.cleaner.clean_text(text)
        stats = result['stats']
        
        assert stats.original_length > 0
        assert stats.cleaned_length > 0
        assert stats.cleaned_length < stats.original_length
        assert stats.reduction_ratio > 0
        assert len(stats.patterns_removed) > 0
        assert not stats.excessive_cleaning  # Should not be excessive for this example
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.cleaner.clean_text("")
        
        assert result['cleaned_text'] == ""
        assert result['stats'].original_length == 0
        assert result['stats'].cleaned_length == 0
        assert result['stats'].reduction_ratio == 0.0
    
    def test_remove_headers_footers(self):
        """Test removal of repetitive headers and footers."""
        text_with_headers = """
        Care Manual
        
        Content here.
        
        Care Manual
        
        More content.
        
        Care Manual
        
        Final content.
        """
        
        result = self.cleaner.clean_text(text_with_headers)
        cleaned_text = result['cleaned_text']
        
        # "Care Manual" appears 3 times, so should be removed as repetitive
        assert cleaned_text.count("Care Manual") < 3
        assert "Content here." in cleaned_text
        assert "More content." in cleaned_text
        assert "Final content." in cleaned_text