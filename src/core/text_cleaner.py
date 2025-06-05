import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import Counter

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningStats:
    """
    Statistics about text cleaning operations.
    """
    original_length: int
    cleaned_length: int
    reduction_ratio: float
    patterns_removed: Dict[str, int]
    excessive_cleaning: bool = False


class TextCleaner:
    """
    Clean document text by removing disclaimers, noise, and repetitive content.
    """
    
    def __init__(self):
        self.disclaimer_patterns = self._get_disclaimer_patterns()
        self.noise_patterns = self._get_noise_patterns()
        self.header_footer_patterns = self._get_header_footer_patterns()
        
        # Frequency-based noise detection
        self.noise_frequency_threshold = settings.noise_frequency_threshold
        self.document_noise_cache: Dict[str, int] = {}
        self.total_documents_processed = 0
    
    def clean_text(self, text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean document text by removing disclaimers and noise.
        
        Args:
            text: Raw text to clean
            document_id: Optional document identifier for tracking
            
        Returns:
            Dictionary with cleaned text and statistics
        """
        if not text:
            return {
                'cleaned_text': '',
                'stats': CleaningStats(0, 0, 0.0, {})
            }
        
        original_text = text
        original_length = len(text)
        patterns_removed = {}
        
        # Step 1: Remove known disclaimer patterns
        if settings.enable_disclaimer_removal:
            text, disclaimer_stats = self._remove_disclaimer_patterns(text)
            patterns_removed.update(disclaimer_stats)
        
        # Step 2: Remove headers and footers
        text, header_footer_stats = self._remove_headers_footers(text)
        patterns_removed.update(header_footer_stats)
        
        # Step 3: Remove general noise patterns
        text, noise_stats = self._remove_noise_patterns(text)
        patterns_removed.update(noise_stats)
        
        # Step 4: Clean up excessive whitespace
        text = self._normalize_whitespace(text)
        
        # Step 5: Update frequency cache for noise detection
        if settings.enable_noise_detection and document_id:
            self._update_noise_frequency(text, document_id)
        
        # Calculate statistics
        cleaned_length = len(text)
        reduction_ratio = (original_length - cleaned_length) / max(original_length, 1)
        excessive_cleaning = cleaned_length < original_length * 0.3  # More than 70% removed
        
        stats = CleaningStats(
            original_length=original_length,
            cleaned_length=cleaned_length,
            reduction_ratio=reduction_ratio,
            patterns_removed=patterns_removed,
            excessive_cleaning=excessive_cleaning
        )
        
        if excessive_cleaning:
            logger.warning(f"Excessive cleaning detected: {reduction_ratio:.2%} of content removed")
        
        return {
            'cleaned_text': text,
            'stats': stats
        }
    
    def _get_disclaimer_patterns(self) -> List[str]:
        """Get regex patterns for known disclaimer text."""
        return [
            # Main Extendicare disclaimer
            r'\*This document is uncontrolled when printed\.\*',
            
            # Accessibility disclaimer
            r'\*Extendicare \(Canada\) Inc\. will provide, on request, information in an accessible format or\*',
            r'\*with communication supports to people with disabilities, in a manner that takes into account\*',
            r'\*their disability\. Confidential and Proprietary Information of Extendicare \(Canada\) Inc\. © \d{4}\*',
            
            # Combined disclaimer pattern
            r'\*Extendicare \(Canada\) Inc\. will provide.*?© \d{4}\*',
            
            # Tagline
            r'Helping people live better\.?',
            
            # Copyright notices
            r'Confidential and Proprietary Information of Extendicare \(Canada\) Inc\. © \d{4}',
            r'© \d{4} Extendicare \(Canada\) Inc\.',
            
            # Document control
            r'Document ID: [A-Z]+-[A-Z0-9\.-]+',
            r'Effective Date: \d{1,2}/\d{1,2}/\d{4}',
            r'Review Date: \d{1,2}/\d{1,2}/\d{4}',
            r'Revision: \d+(\.\d+)?',
            
            # Common footer disclaimers
            r'This document is proprietary and confidential',
            r'Not for distribution outside of Extendicare',
        ]
    
    def _get_noise_patterns(self) -> List[str]:
        """Get regex patterns for general noise text."""
        return [
            # Page numbers
            r'Page \d+ of \d+',
            r'^\d+$',  # Standalone numbers (likely page numbers)
            
            # Date stamps
            r'Printed on: \d{1,2}/\d{1,2}/\d{4}',
            r'Last updated: \d{1,2}/\d{1,2}/\d{4}',
            
            # File paths and URLs
            r'[A-Za-z]:\\[^\\]+\\[^\\]+',  # Windows file paths
            r'https?://[^\s]+',  # URLs
            
            # Empty formatting artifacts
            r'\s*\|\s*\|\s*',  # Empty table cells
            r'_{3,}',  # Multiple underscores
            r'-{3,}',  # Multiple dashes
            
            # Table of contents artifacts
            r'\.{3,}\d+',  # Dots followed by page numbers
            r'\d+\s*\.{3,}',  # Page numbers followed by dots
        ]
    
    def _get_header_footer_patterns(self) -> List[str]:
        """Get patterns for headers and footers."""
        return [
            # Policy manual headers
            r'^[A-Z][a-z]+ Manual\s*$',
            r'^[A-Z]{2,5}\d*-[A-Z]\d+\.\d+',  # Policy codes
            
            # Section headers
            r'^Section \d+:.*$',
            r'^SECTION \d+:.*$',
            
            # Page headers/footers that repeat
            r'^.*Administration.*$',
            r'^.*Care.*$',
            r'^.*Emergency Planning.*$',
            r'^.*Environmental Services.*$',
            r'^.*Infection Prevention.*$',
            r'^.*Legal.*$',
            r'^.*Maintenance.*$',
            r'^.*Privacy.*$',
        ]
    
    def _remove_disclaimer_patterns(self, text: str) -> tuple[str, Dict[str, int]]:
        """Remove disclaimer patterns from text."""
        patterns_removed = {}
        
        for pattern in self.disclaimer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                count = len(matches)
                patterns_removed[f"disclaimer_{pattern[:20]}..."] = count
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text, patterns_removed
    
    def _remove_headers_footers(self, text: str) -> tuple[str, Dict[str, int]]:
        """Remove repetitive headers and footers."""
        patterns_removed = {}
        lines = text.split('\n')
        
        # Track line frequencies to identify repeated headers/footers
        line_counts = Counter(line.strip() for line in lines if line.strip())
        
        # Remove lines that appear frequently (likely headers/footers)
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                filtered_lines.append(line)
                continue
            
            # Check if this line appears too frequently
            frequency = line_counts[stripped_line]
            if frequency > 3:  # Appears more than 3 times
                patterns_removed[f"repeated_line_{stripped_line[:20]}..."] = frequency
                continue
            
            # Check against header/footer patterns
            is_header_footer = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, stripped_line, re.IGNORECASE):
                    patterns_removed[f"header_footer_{pattern[:20]}..."] = 1
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines), patterns_removed
    
    def _remove_noise_patterns(self, text: str) -> tuple[str, Dict[str, int]]:
        """Remove general noise patterns."""
        patterns_removed = {}
        
        for pattern in self.noise_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                count = len(matches)
                patterns_removed[f"noise_{pattern[:20]}..."] = count
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text, patterns_removed
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _update_noise_frequency(self, text: str, document_id: str):
        """Update frequency tracking for noise detection."""
        # Extract potential noise phrases (repeated short phrases)
        words = text.split()
        
        # Look for repeated phrases of 2-5 words
        for phrase_length in range(2, 6):
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i + phrase_length])
                
                # Skip very short or very long phrases
                if len(phrase) < 10 or len(phrase) > 100:
                    continue
                
                # Update frequency count
                if phrase not in self.document_noise_cache:
                    self.document_noise_cache[phrase] = 0
                self.document_noise_cache[phrase] += 1
        
        self.total_documents_processed += 1
    
    def get_frequent_noise(self, min_frequency: Optional[float] = None) -> List[str]:
        """
        Get phrases that appear frequently across documents (likely noise).
        
        Args:
            min_frequency: Minimum frequency ratio (default from settings)
            
        Returns:
            List of phrases that appear frequently
        """
        min_frequency = min_frequency or self.noise_frequency_threshold
        min_count = int(self.total_documents_processed * min_frequency)
        
        frequent_phrases = [
            phrase for phrase, count in self.document_noise_cache.items()
            if count >= min_count
        ]
        
        return sorted(frequent_phrases, key=lambda p: self.document_noise_cache[p], reverse=True)
    
    def validate_cleaning(self, original: str, cleaned: str) -> Dict[str, Any]:
        """
        Validate that cleaning didn't remove too much meaningful content.
        
        Args:
            original: Original text
            cleaned: Cleaned text
            
        Returns:
            Validation results
        """
        if not original:
            return {'is_valid': True, 'reason': 'Empty input'}
        
        original_words = original.split()
        cleaned_words = cleaned.split()
        
        # Calculate retention ratios
        word_retention = len(cleaned_words) / max(len(original_words), 1)
        char_retention = len(cleaned) / max(len(original), 1)
        
        # Check for excessive removal
        if word_retention < 0.3:
            return {
                'is_valid': False,
                'reason': f'Too many words removed: {word_retention:.2%} retained',
                'word_retention': word_retention,
                'char_retention': char_retention
            }
        
        if char_retention < 0.2:
            return {
                'is_valid': False,
                'reason': f'Too many characters removed: {char_retention:.2%} retained',
                'word_retention': word_retention,
                'char_retention': char_retention
            }
        
        # Check for meaningful content
        if len(cleaned_words) < 10:
            return {
                'is_valid': False,
                'reason': f'Too little content remaining: {len(cleaned_words)} words',
                'word_retention': word_retention,
                'char_retention': char_retention
            }
        
        return {
            'is_valid': True,
            'word_retention': word_retention,
            'char_retention': char_retention
        }
    
    def clean_text_fast(self, text: str) -> str:
        """
        Fast text cleaning for optimized processing.
        Skips validation and frequency tracking for speed.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Apply only essential cleaning patterns
        cleaned = text
        
        # Remove only the most common disclaimers (combine patterns for efficiency)
        essential_pattern = r'(\*This document is uncontrolled when printed\.\*|' \
                          r'Helping people live better\.?|' \
                          r'Page \d+ of \d+|' \
                          r'Confidential and Proprietary Information.*?© \d{4})'
        
        cleaned = re.sub(essential_pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Quick whitespace normalization
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        return cleaned.strip()