"""
Black Sheep Ventures - Document Parsing Utilities
Supports PDF, Markdown, and TXT text extraction with automatic charset detection and chunking.
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Read text file with progressive charset detection fallback:
    1. Try UTF-8 decode.
    2. Try charset_normalizer.
    3. Fallback to chardet.
    4. Ultimate fallback to UTF-8 with character replacement.
    
    Args:
        file_path: Absolute or relative file path.
        
    Returns:
        Decoded text string.
    """
    data = Path(file_path).read_bytes()
    
    # Try UTF-8 first
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Try charset_normalizer
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass
    
    # Try chardet fallback
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass
    
    # Default fallback
    if not encoding:
        encoding = 'utf-8'
    
    return data.decode(encoding, errors='replace')


class FileParser:
    """Multi-format document parsing engine."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check whether the file extension is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Extract clean text content from a file.
        
        Args:
            file_path: Path to target file.
            
        Returns:
            Extracted text content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"Unable to process format: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF parsing: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Extract text from Markdown with automatic encoding detection."""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extract text from TXT with automatic encoding detection."""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Extract text from multiple files and concatenate with headers.
        
        Args:
            file_paths: List of file paths.
            
        Returns:
            Concatenated text string.
        """
        all_texts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Document {i}: {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== Document {i}: {file_path} (Extraction failed: {str(e)}) ===")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks, honoring natural sentence boundaries.
    
    Args:
        text: Input text string.
        chunk_size: Target characters per chunk.
        overlap: Character overlap between chunks.
        
    Returns:
        List of chunk strings.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Look for sentence boundaries
        if end < len(text):
            for sep in ['\n\n', '. ', '! ', '? ', '.\n', '!\n', '?\n', '; ', ',\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
    
    return chunks
