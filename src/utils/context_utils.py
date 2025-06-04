import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentContext:
    """
    Context information extracted from document file path and content.
    """
    # File information
    file_path: str
    relative_path: str
    file_name: str
    
    # Policy hierarchy
    policy_manual: Optional[str] = None
    policy_acronym: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    
    # Document identification
    document_index: Optional[str] = None
    is_additional_resource: bool = False
    
    # Document type classification
    document_type: Optional[str] = None
    
    # Full hierarchy path
    hierarchy_path: List[str] = None


class ContextExtractor:
    """
    Extract contextual metadata from document file paths and content.
    """
    
    def __init__(self, mirror_directory: Optional[str] = None):
        self.mirror_directory = mirror_directory or settings.mirror_directory
        
        # Policy manual mappings
        self.policy_manual_map = {
            'Administration': 'ADM',
            'Care': 'CARE', 
            'Emergency Planning and Management': 'EPM',
            'Environmental Services': 'EVS',
            'Infection Prevention and Control': 'IPC',
            'Legal': 'LEG',
            'Maintenance': 'MAINT',
            'Privacy and Confidentiality': 'PRV',
            'RC': 'RC'  # Already abbreviated
        }
        
        # Valid policy acronyms for document index extraction
        self.policy_acronyms = set(self.policy_manual_map.values())
        
        # Document type patterns
        self.document_type_patterns = {
            'policy': [r'policy', r'procedure', r'standard'],
            'form': [r'form', r'checklist', r'template'],
            'guide': [r'guide', r'manual', r'handbook'],
            'training': [r'training', r'education', r'learning'],
            'flowchart': [r'flowchart', r'flow\s*chart', r'process\s*flow'],
            'reference': [r'reference', r'quick\s*ref', r'job\s*aid']
        }
    
    def extract_context(self, file_path: str) -> DocumentContext:
        """
        Extract context information from a file path.
        
        Args:
            file_path: Full path to the document file
            
        Returns:
            DocumentContext with extracted information
        """
        path = Path(file_path)
        
        # Calculate relative path from mirror directory
        try:
            relative_path = str(path.relative_to(Path(self.mirror_directory)))
        except ValueError:
            # File is not under mirror directory
            relative_path = str(path)
            logger.warning(f"File not under mirror directory: {file_path}")
        
        # Split path components
        path_parts = Path(relative_path).parts
        
        # Extract basic information
        context = DocumentContext(
            file_path=file_path,
            relative_path=relative_path,
            file_name=path.name,
            hierarchy_path=list(path_parts[:-1])  # Exclude filename
        )
        
        # Extract policy manual from first path component
        if path_parts:
            policy_manual = path_parts[0]
            context.policy_manual = policy_manual
            context.policy_acronym = self.policy_manual_map.get(policy_manual)
        
        # Extract section from second path component
        if len(path_parts) > 1:
            context.section = path_parts[1]
            
            # Check if this is an Additional Resources folder
            if 'additional' in path_parts[1].lower() and 'resource' in path_parts[1].lower():
                context.is_additional_resource = True
        
        # Extract subsection from third path component
        if len(path_parts) > 2:
            context.subsection = path_parts[2]
        
        # Extract document index from filename
        context.document_index = self._extract_document_index(path.name)
        
        # Classify document type
        context.document_type = self._classify_document_type(path.name)
        
        return context
    
    def _extract_document_index(self, filename: str) -> Optional[str]:
        """
        Extract document index from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Document index if found, None otherwise
        """
        # Pattern for policy document indexes: ACRONYM#-X#.##[-T#]
        # Examples: CARE1-P10.01, ADM2-P05.02-T3, IPC1-A01.01
        
        for acronym in self.policy_acronyms:
            # Look for pattern starting with policy acronym
            pattern = rf'^{acronym}\d*-[A-Z]\d+\.\d+(?:-[A-Z]\d+)?'
            match = re.match(pattern, filename, re.IGNORECASE)
            
            if match:
                # Extract until first whitespace or dot (if not part of version)
                index_match = re.match(r'^([A-Z]+\d*-[A-Z]\d+\.\d+(?:-[A-Z]\d+)?)', filename, re.IGNORECASE)
                if index_match:
                    return index_match.group(1).upper()
        
        return None
    
    def _classify_document_type(self, filename: str) -> Optional[str]:
        """
        Classify document type based on filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Document type if detected
        """
        filename_lower = filename.lower()
        
        for doc_type, patterns in self.document_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower):
                    return doc_type
        
        return None
    
    def generate_synthetic_index(self, context: DocumentContext, sequence_number: int) -> str:
        """
        Generate synthetic document index for files without natural indexes.
        
        Args:
            context: Document context
            sequence_number: Sequential number for this file type
            
        Returns:
            Synthetic document index
        """
        # Use policy acronym from context
        acronym = context.policy_acronym or 'UNK'
        
        # Format: ACRONYM-ADDL-### for Additional Resources
        if context.is_additional_resource:
            return f"{acronym}-ADDL-{sequence_number:03d}"
        
        # Format: ACRONYM-MISC-### for other files without indexes
        return f"{acronym}-MISC-{sequence_number:03d}"
    
    def create_context_metadata(self, context: DocumentContext) -> Dict[str, Any]:
        """
        Create comprehensive metadata dictionary for RAG applications.
        
        Args:
            context: Document context
            
        Returns:
            Dictionary with context metadata
        """
        metadata = {
            # File identification
            'file_path': context.file_path,
            'relative_path': context.relative_path,
            'file_name': context.file_name,
            
            # Policy hierarchy
            'policy_manual': context.policy_manual,
            'policy_acronym': context.policy_acronym,
            'section': context.section,
            'subsection': context.subsection,
            'hierarchy_path': context.hierarchy_path,
            
            # Document classification
            'document_index': context.document_index,
            'document_type': context.document_type,
            'is_additional_resource': context.is_additional_resource,
            
            # RAG-specific metadata
            'source_type': 'policy_document',
            'content_domain': 'healthcare_policy',
            'organization': 'Extendicare',
            
            # Searchable tags
            'tags': self._generate_tags(context)
        }
        
        # Add nested structure for hierarchical search
        if context.hierarchy_path:
            metadata['hierarchy'] = {
                'level_0': context.hierarchy_path[0] if len(context.hierarchy_path) > 0 else None,
                'level_1': context.hierarchy_path[1] if len(context.hierarchy_path) > 1 else None,
                'level_2': context.hierarchy_path[2] if len(context.hierarchy_path) > 2 else None,
                'level_3': context.hierarchy_path[3] if len(context.hierarchy_path) > 3 else None,
            }
        
        return metadata
    
    def _generate_tags(self, context: DocumentContext) -> List[str]:
        """
        Generate searchable tags from context.
        
        Args:
            context: Document context
            
        Returns:
            List of tags for search optimization
        """
        tags = []
        
        # Add policy manual tags
        if context.policy_manual:
            tags.append(context.policy_manual.lower().replace(' ', '_'))
        
        if context.policy_acronym:
            tags.append(context.policy_acronym.lower())
        
        # Add section tags
        if context.section:
            tags.append(context.section.lower().replace(' ', '_'))
        
        # Add document type tags
        if context.document_type:
            tags.append(context.document_type)
        
        # Add special tags
        if context.is_additional_resource:
            tags.append('additional_resource')
        
        if context.document_index:
            tags.append('indexed_document')
            tags.append(f"index_{context.document_index.lower()}")
        else:
            tags.append('non_indexed_document')
        
        # Add filename-based tags
        filename_lower = context.file_name.lower()
        if 'form' in filename_lower:
            tags.append('form')
        if 'checklist' in filename_lower:
            tags.append('checklist')
        if 'template' in filename_lower:
            tags.append('template')
        if 'guide' in filename_lower:
            tags.append('guide')
        if 'flowchart' in filename_lower or 'flow chart' in filename_lower:
            tags.append('flowchart')
        
        return list(set(tags))  # Remove duplicates
    
    def validate_context(self, context: DocumentContext) -> Dict[str, Any]:
        """
        Validate extracted context information.
        
        Args:
            context: Document context to validate
            
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Check required fields
        if not context.policy_manual:
            issues.append("Missing policy manual identification")
        
        if not context.policy_acronym:
            warnings.append("Could not map policy manual to acronym")
        
        # Check for document index
        if not context.document_index and not context.is_additional_resource:
            warnings.append("No document index found and not in Additional Resources")
        
        # Check path structure
        if len(context.hierarchy_path) < 2:
            warnings.append("Shallow directory structure - may affect categorization")
        
        # Check for valid policy acronym
        if context.policy_acronym and context.policy_acronym not in self.policy_acronyms:
            issues.append(f"Invalid policy acronym: {context.policy_acronym}")
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'context_completeness': self._calculate_completeness(context)
        }
    
    def _calculate_completeness(self, context: DocumentContext) -> float:
        """
        Calculate how complete the context information is.
        
        Args:
            context: Document context
            
        Returns:
            Completeness score between 0 and 1
        """
        total_fields = 8
        complete_fields = 0
        
        if context.policy_manual:
            complete_fields += 1
        if context.policy_acronym:
            complete_fields += 1
        if context.section:
            complete_fields += 1
        if context.subsection:
            complete_fields += 1
        if context.document_index:
            complete_fields += 1
        if context.document_type:
            complete_fields += 1
        if context.hierarchy_path:
            complete_fields += 1
        if context.file_name:
            complete_fields += 1
        
        return complete_fields / total_fields