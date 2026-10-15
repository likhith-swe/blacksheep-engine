"""
Black Sheep Ventures - Text Processing Service
"""

from typing import List, Optional
import re
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """Text extraction, chunking, and preprocessing utilities."""
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Extract combined text content from multiple files."""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Raw input text.
            chunk_size: Maximum characters per chunk.
            overlap: Character overlap between consecutive chunks.
            
        Returns:
            List of text chunk strings.
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess and normalize text:
        - Normalize line breaks
        - Collapse redundant empty lines
        - Strip whitespace
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned and normalized text.
        """
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Collapse multiple empty lines (max 2 consecutive newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Get character, line, and word statistics for text."""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }
