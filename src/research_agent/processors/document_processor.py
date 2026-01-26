"""
Document processor for handling PDFs and other academic documents.

This module provides functionality to:
- Extract text and structure from PDFs
- Extract metadata (title, authors, DOI, etc.)
- Chunk documents for embedding and retrieval
- Support multiple document formats
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import optional libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Represents a chunk of a processed document."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str


@dataclass
class ProcessedDocument:
    """Represents a fully processed document."""
    title: str
    authors: List[str]
    content: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    source_path: str
    doc_id: str


class DocumentProcessor:
    """
    Document processor for academic papers and research documents.
    
    Supports PDF, DOCX, and plain text files with intelligent chunking
    and metadata extraction optimized for research content.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        prefer_unstructured: bool = True
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of each chunk in words
            chunk_overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum chunk size in words
            prefer_unstructured: Use unstructured library if available
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.prefer_unstructured = prefer_unstructured
        
        # Check available libraries
        self.has_pymupdf = PYMUPDF_AVAILABLE
        self.has_unstructured = UNSTRUCTURED_AVAILABLE
        self.has_pypdf = PYPDF_AVAILABLE
        self.has_docx = DOCX_AVAILABLE
        
    def process_document(self, file_path: str | Path) -> ProcessedDocument:
        """
        Process a document file and extract structured content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessedDocument with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        # Generate document ID
        doc_id = self._generate_doc_id(file_path)
        
        # Extract content based on file type
        if file_path.suffix.lower() == '.pdf':
            content, metadata = self._extract_pdf_content(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            content, metadata = self._extract_docx_content(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            content, metadata = self._extract_text_content(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Enhance metadata
        metadata.update({
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_type': file_path.suffix.lower(),
            'processed_at': str(file_path.stat().st_mtime),
            'doc_id': doc_id
        })
        
        # Extract title and authors if not already present
        if not metadata.get('title'):
            metadata['title'] = self._extract_title(content)
        if not metadata.get('authors'):
            metadata['authors'] = self._extract_authors(content)
            
        # Create chunks
        chunks = self._create_chunks(content, metadata)
        
        return ProcessedDocument(
            title=metadata.get('title', 'Unknown Title'),
            authors=metadata.get('authors', []),
            content=content,
            chunks=chunks,
            metadata=metadata,
            source_path=str(file_path),
            doc_id=doc_id
        )
    
    def _extract_pdf_content(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from PDF file."""
        metadata = {}
        
        # Try unstructured first if available and preferred
        if self.has_unstructured and self.prefer_unstructured:
            try:
                return self._extract_pdf_unstructured(file_path)
            except Exception as e:
                print(f"Unstructured PDF processing failed: {e}")
        
        # Fall back to PyMuPDF
        if self.has_pymupdf:
            try:
                return self._extract_pdf_pymupdf(file_path)
            except Exception as e:
                print(f"PyMuPDF processing failed: {e}")
        
        # Final fallback to pypdf
        if self.has_pypdf:
            return self._extract_pdf_pypdf(file_path)
            
        raise ImportError("No PDF processing library available")
    
    def _extract_pdf_unstructured(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF content using unstructured library."""
        elements = partition(filename=str(file_path))
        
        # Separate content by type
        text_content = []
        metadata = {'elements': []}
        
        for element in elements:
            text_content.append(str(element))
            element_data = {
                'type': str(getattr(element, 'category', 'Unknown')),
                'text': str(element)[:200]  # Preview
            }
            metadata['elements'].append(element_data)
        
        content = '\n\n'.join(text_content)
        
        # Try to extract title from first title element
        for element in elements:
            if hasattr(element, 'category') and element.category == 'Title':
                metadata['title'] = str(element)
                break
        
        return content, metadata
    
    def _extract_pdf_pymupdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF content using PyMuPDF (fitz)."""
        doc = fitz.open(str(file_path))
        
        # Extract basic metadata
        doc_metadata = doc.metadata if doc.metadata else {}
        metadata = {
            'page_count': len(doc),
            'title': doc_metadata.get('title', ''),
            'author': doc_metadata.get('author', ''),
            'subject': doc_metadata.get('subject', ''),
            'creator': doc_metadata.get('creator', ''),
            'producer': doc_metadata.get('producer', ''),
        }
        
        # Extract text with structure
        full_text = ""
        sections = []
        current_section = {"title": "Introduction", "content": "", "page": 0}
        
        try:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            text = "".join([span["text"] for span in line["spans"]])
                            
                            # Try to detect headers by font size
                            if line["spans"]:
                                font_size = line["spans"][0]["size"]
                                is_bold = line["spans"][0]["flags"] & 2**4 != 0
                                
                                # Header detection: larger font, bold, short text
                                if (font_size > 12 or is_bold) and len(text.strip()) < 100 and text.strip():
                                    if current_section["content"]:
                                        sections.append(current_section)
                                    current_section = {
                                        "title": text.strip(),
                                        "content": "",
                                        "page": page_num + 1
                                    }
                                else:
                                    current_section["content"] += text + " "
                            
                            full_text += text + " "
        except Exception as e:
            print(f"Error extracting structured text: {e}")
            # Fallback to simple text extraction
            full_text = doc.get_text()
        
        if current_section["content"]:
            sections.append(current_section)
        
        metadata['sections'] = sections
        
        # Try to extract DOI
        doi_match = re.search(r'\b10\.\d{4,}/[^\s\]]+', full_text)
        if doi_match:
            metadata['doi'] = doi_match.group()
        
        doc.close()
        return full_text.strip(), metadata
    
    def _extract_pdf_pypdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF content using pypdf (fallback method)."""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            metadata = {
                'page_count': len(pdf_reader.pages),
            }
            
            # Extract metadata if available
            if pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                })
            
            text_content = []
            for page in pdf_reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
            
            content = '\n\n'.join(text_content)
            
            return content, metadata
    
    def _extract_docx_content(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from DOCX file."""
        doc = DocxDocument(str(file_path))
        
        metadata = {}
        
        # Extract paragraphs
        paragraphs = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)
        
        content = '\n\n'.join(paragraphs)
        
        # Try to extract title (first paragraph that looks like a title)
        for para in paragraphs:
            if len(para) < 100 and para:
                metadata['title'] = para
                break
        
        # Extract tables if any
        if doc.tables:
            metadata['has_tables'] = True
            metadata['table_count'] = len(doc.tables)
        
        return content, metadata
    
    def _extract_text_content(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract content from plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        metadata = {
            'word_count': len(content.split()),
            'char_count': len(content)
        }
        
        # Try to extract title (first line that looks like a title)
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 100 and not line.startswith('#'):
                metadata['title'] = line
                break
        
        return content, metadata
    
    def _create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create chunks from document content with overlap.
        
        Args:
            content: Full document content
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        # Split into words
        words = content.split()
        chunks = []
        
        # Create overlapping chunks
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Skip very small chunks unless it's the only one
            if len(chunk_words) < self.min_chunk_size and len(chunks) > 0:
                continue
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
                'start_word': i,
                'end_word': min(i + len(chunk_words), len(words))
            })
            
            # Generate chunk ID
            chunk_id = f"{metadata.get('doc_id', 'unknown')}_chunk_{len(chunks)}"
            
            chunk = DocumentChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                source=metadata.get('file_path', 'unknown')
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _extract_title(self, content: str) -> str:
        """Extract title from document content."""
        lines = content.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            # Look for short lines that might be titles
            if 10 < len(line) < 150 and not line.startswith('http'):
                # Skip if it looks like an email, URL, or reference
                if not any(x in line.lower() for x in ['@', 'http', 'doi:', 'abstract', 'introduction']):
                    return line
        
        return "Untitled Document"
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract author names from document content."""
        authors = []
        
        # Look for patterns that suggest author names
        lines = content.split('\n')[:20]  # Check first 20 lines
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for lines with author-like patterns
            # Skip common non-author lines
            skip_patterns = ['abstract', 'introduction', 'keywords', 'university', 'department', '@', 'http']
            if any(pattern in line.lower() for pattern in skip_patterns):
                continue
            
            # Look for patterns like "Author Name", "Author Name1, Author Name2"
            if 5 < len(line) < 200:
                # Multiple authors separated by commas or "and"
                if ',' in line or ' and ' in line.lower():
                    potential_authors = re.split(r',\s*|\s+and\s+', line)
                    for author in potential_authors:
                        author = author.strip()
                        if 3 < len(author) < 50 and not any(x in author.lower() for x in ['university', 'department', 'institute']):
                            authors.append(author)
                # Single author
                elif len(line.split()) <= 4 and line[0].isupper():
                    authors.append(line)
            
            # Stop after we find some authors or reach abstract
            if authors or 'abstract' in line.lower():
                break
        
        return authors[:5]  # Limit to first 5 authors
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        # Use file path and modification time
        stat = file_path.stat()
        unique_string = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        formats = ['.txt', '.md']
        
        if self.has_pymupdf or self.has_unstructured or self.has_pypdf:
            formats.append('.pdf')
        
        if self.has_docx:
            formats.append('.docx')
        
        return sorted(formats)