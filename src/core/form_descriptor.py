"""
Form and Template Description Generator

Creates synthetic descriptions for forms, templates, and checklists
that have no extractable text content but should be discoverable in RAG.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.utils.context_utils import DocumentContext
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FormDescription:
    """Generated description for a form or template."""
    description: str
    keywords: List[str]
    use_cases: List[str]
    document_type: str
    confidence_score: float


class FormDescriptionGenerator:
    """
    Generates synthetic descriptions for forms, templates, and checklists
    based on filename, path context, and document structure patterns.
    """
    
    def __init__(self):
        # Document type patterns with associated descriptions
        self.type_patterns = {
            'checklist': {
                'patterns': [r'checklist', r'check.*list', r'ja.*checklist'],
                'template': "This document is a {title} checklist for {purpose} in the {policy_manual} policy manual.",
                'keywords': ['checklist', 'verification', 'compliance', 'audit', 'steps'],
                'use_cases': ['staff guidance', 'quality assurance', 'procedure verification']
            },
            'template': {
                'patterns': [r'template', r'form.*template', r'planning.*template'],
                'template': "This document is a {title} template for {purpose} as part of {policy_manual} policies.",
                'keywords': ['template', 'form', 'planning', 'documentation'],
                'use_cases': ['documentation', 'planning', 'record keeping']
            },
            'form': {
                'patterns': [r'form(?!at)', r'sheet', r'log', r'record'],
                'template': "This document is a {title} form for {purpose} supporting {policy_manual} procedures.",
                'keywords': ['form', 'documentation', 'record', 'tracking'],
                'use_cases': ['data collection', 'record keeping', 'compliance documentation']
            },
            'audit': {
                'patterns': [r'audit', r'assessment', r'evaluation'],
                'template': "This document is an {title} audit tool for evaluating {purpose} in {policy_manual}.",
                'keywords': ['audit', 'assessment', 'evaluation', 'quality', 'compliance'],
                'use_cases': ['quality assurance', 'compliance monitoring', 'performance evaluation']
            },
            'inventory': {
                'patterns': [r'inventory', r'utilization', r'tracking'],
                'template': "This document is an {title} tracking sheet for monitoring {purpose} in {policy_manual}.",
                'keywords': ['inventory', 'tracking', 'monitoring', 'utilization'],
                'use_cases': ['resource management', 'inventory control', 'utilization tracking']
            },
            'poster': {
                'patterns': [r'poster', r'sign', r'notice'],
                'template': "This document is a {title} poster providing information about {purpose} for {policy_manual}.",
                'keywords': ['poster', 'information', 'guidance', 'reference'],
                'use_cases': ['staff reference', 'patient information', 'policy guidance']
            }
        }
        
        # Policy manual purposes
        self.policy_purposes = {
            'Administration': 'administrative procedures and organizational management',
            'Care': 'resident care delivery and clinical procedures',
            'Emergency Planning and Management': 'emergency response and crisis management',
            'Environmental Services': 'facility maintenance and environmental safety',
            'Infection Prevention and Control': 'infection control and prevention measures',
            'Legal': 'legal compliance and regulatory requirements',
            'Maintenance': 'facility maintenance and equipment management',
            'Privacy and Confidentiality': 'privacy protection and confidentiality procedures',
            'RC': 'restorative care and rehabilitation services'
        }
        
        # Common purpose keywords from filenames
        self.purpose_patterns = {
            'planning': ['planning', 'plan'],
            'assessment': ['assessment', 'evaluation', 'audit'],
            'training': ['training', 'education', 'learning'],
            'safety': ['safety', 'safe', 'protection'],
            'medication': ['medication', 'med', 'drug'],
            'resident care': ['resident', 'patient', 'care'],
            'staff management': ['staff', 'employee', 'personnel'],
            'documentation': ['documentation', 'record', 'log'],
            'equipment': ['equipment', 'device', 'tool'],
            'communication': ['communication', 'report', 'notification'],
            'infection control': ['infection', 'hygiene', 'contamination'],
            'emergency': ['emergency', 'code', 'crisis']
        }
    
    def generate_description(self, context: DocumentContext, file_size_bytes: int = 0) -> FormDescription:
        """
        Generate a synthetic description for a form or template.
        
        Args:
            context: Document context with path and metadata
            file_size_bytes: File size for confidence scoring
            
        Returns:
            FormDescription with generated content
        """
        logger.info(f"Generating synthetic description for: {context.file_name}")
        
        # Detect document type
        doc_type = self._detect_document_type(context.file_name)
        
        # Extract title and purpose
        title = self._extract_title(context.file_name)
        purpose = self._extract_purpose(context.file_name, context)
        
        # Get policy context
        policy_manual = context.policy_manual or "organizational"
        policy_purpose = self.policy_purposes.get(policy_manual, "organizational procedures")
        
        # Generate description
        type_config = self.type_patterns.get(doc_type, self.type_patterns['form'])
        description = type_config['template'].format(
            title=title,
            purpose=purpose,
            policy_manual=policy_manual,
            policy_purpose=policy_purpose
        )
        
        # Add context-specific details
        if context.section:
            description += f" This document is used within the {context.section} section."
        
        if context.document_index:
            description += f" Document reference: {context.document_index}."
        
        # Generate keywords
        keywords = self._generate_keywords(context, doc_type, purpose)
        
        # Generate use cases
        use_cases = self._generate_use_cases(doc_type, purpose, context)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(context, file_size_bytes, doc_type)
        
        return FormDescription(
            description=description,
            keywords=keywords,
            use_cases=use_cases,
            document_type=doc_type,
            confidence_score=confidence
        )
    
    def _detect_document_type(self, filename: str) -> str:
        """Detect document type from filename patterns."""
        filename_lower = filename.lower()
        
        for doc_type, config in self.type_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, filename_lower):
                    return doc_type
        
        return 'form'  # Default fallback
    
    def _extract_title(self, filename: str) -> str:
        """Extract a clean title from filename."""
        # Remove file extension
        title = Path(filename).stem
        
        # Remove document index (e.g., "CARE1-P10.01-T3")
        title = re.sub(r'^[A-Z]+\d*-[A-Z]\d+\.\d+(?:-[A-Z]\d+)?\s*', '', title)
        
        # Clean up the title
        title = re.sub(r'[_-]+', ' ', title)  # Replace underscores/dashes with spaces
        title = re.sub(r'\s+', ' ', title)    # Normalize whitespace
        title = title.strip()
        
        # Capitalize properly
        if title:
            return title.lower()
        else:
            return "document"
    
    def _extract_purpose(self, filename: str, context: DocumentContext) -> str:
        """Extract the likely purpose from filename and context."""
        filename_lower = filename.lower()
        purposes = []
        
        # Check for purpose patterns in filename
        for purpose, patterns in self.purpose_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    purposes.append(purpose)
        
        # Add context-based purposes
        if context.policy_manual:
            if 'care' in context.policy_manual.lower():
                purposes.append('resident care')
            elif 'emergency' in context.policy_manual.lower():
                purposes.append('emergency response')
            elif 'infection' in context.policy_manual.lower():
                purposes.append('infection control')
        
        # Use section information
        if context.section:
            section_lower = context.section.lower()
            if 'medication' in section_lower:
                purposes.append('medication management')
            elif 'safety' in section_lower:
                purposes.append('safety procedures')
        
        # Default based on document type patterns in filename
        if 'checklist' in filename_lower:
            purposes.append('procedure verification')
        elif 'template' in filename_lower:
            purposes.append('standardized documentation')
        elif 'form' in filename_lower:
            purposes.append('information collection')
        
        if purposes:
            # Return the most specific purpose, or combine if multiple
            if len(purposes) == 1:
                return purposes[0]
            else:
                return f"{purposes[0]} and {purposes[1]}" if len(purposes) == 2 else purposes[0]
        
        return "organizational procedures"
    
    def _generate_keywords(self, context: DocumentContext, doc_type: str, purpose: str) -> List[str]:
        """Generate relevant keywords for search optimization."""
        keywords = []
        
        # Add type-specific keywords
        type_config = self.type_patterns.get(doc_type, {})
        keywords.extend(type_config.get('keywords', []))
        
        # Add policy-specific keywords
        if context.policy_acronym:
            keywords.append(context.policy_acronym.lower())
        
        if context.policy_manual:
            keywords.extend(context.policy_manual.lower().split())
        
        # Add purpose keywords
        keywords.extend(purpose.split())
        
        # Add section keywords
        if context.section:
            keywords.extend(context.section.lower().split())
        
        # Add document index
        if context.document_index:
            keywords.append(context.document_index.lower())
        
        # Clean and deduplicate
        keywords = [kw.strip() for kw in keywords if len(kw.strip()) > 2]
        return list(set(keywords))
    
    def _generate_use_cases(self, doc_type: str, purpose: str, context: DocumentContext) -> List[str]:
        """Generate likely use cases for the document."""
        use_cases = []
        
        # Add type-specific use cases
        type_config = self.type_patterns.get(doc_type, {})
        use_cases.extend(type_config.get('use_cases', []))
        
        # Add purpose-specific use cases
        if 'assessment' in purpose:
            use_cases.append('quality improvement')
        if 'training' in purpose:
            use_cases.append('staff education')
        if 'safety' in purpose:
            use_cases.append('risk management')
        if 'resident' in purpose:
            use_cases.append('patient care documentation')
        
        return list(set(use_cases))
    
    def _calculate_confidence(self, context: DocumentContext, file_size_bytes: int, doc_type: str) -> float:
        """Calculate confidence score for the generated description."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on context completeness
        if context.policy_manual:
            confidence += 0.1
        if context.section:
            confidence += 0.1
        if context.document_index:
            confidence += 0.1
        
        # Boost confidence based on filename clarity
        filename_lower = context.file_name.lower()
        if any(pattern in filename_lower for pattern in ['checklist', 'template', 'form']):
            confidence += 0.15
        
        # File size heuristic (reasonable size suggests real content)
        if 10000 < file_size_bytes < 500000:  # 10KB - 500KB
            confidence += 0.1
        elif file_size_bytes > 500000:  # Very large files
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def create_searchable_chunk_text(self, description: FormDescription, context: DocumentContext) -> str:
        """
        Create the final text that will be embedded for search.
        
        Args:
            description: Generated form description
            context: Document context
            
        Returns:
            Searchable text for embedding
        """
        # Start with the main description
        text_parts = [description.description]
        
        # Add additional context for better search
        text_parts.append(f"This {description.document_type} can be used for: {', '.join(description.use_cases)}.")
        
        # Add keywords for search optimization
        if description.keywords:
            keywords_text = ', '.join(description.keywords)
            text_parts.append(f"Related keywords: {keywords_text}.")
        
        # Add file information for exact matching
        text_parts.append(f"Filename: {context.file_name}")
        
        if context.document_index:
            text_parts.append(f"Document ID: {context.document_index}")
        
        return " ".join(text_parts)


def generate_form_description(context: DocumentContext, file_size_bytes: int = 0) -> Tuple[str, Dict]:
    """
    Convenience function to generate description for a form/template.
    
    Args:
        context: Document context
        file_size_bytes: File size for confidence calculation
        
    Returns:
        Tuple of (searchable_text, metadata_dict)
    """
    generator = FormDescriptionGenerator()
    description = generator.generate_description(context, file_size_bytes)
    searchable_text = generator.create_searchable_chunk_text(description, context)
    
    metadata = {
        'synthetic_description': True,
        'document_type': description.document_type,
        'confidence_score': description.confidence_score,
        'generated_keywords': description.keywords,
        'use_cases': description.use_cases,
        'original_description': description.description
    }
    
    return searchable_text, metadata